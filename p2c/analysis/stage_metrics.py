"""Stage-level analysis, PLAN.md section 9, and the heterogeneity test of section 21.

For stage ``g``, the quantity of interest is::

    Delta_view(g) = L_partial(g) - L_multi(g)

Two things are added here that the plan implies but does not spell out:

**Uncertainty.** With a few dozen demonstrations on a 4 GB GPU, a per-stage difference of
a few percent is easily noise. Every delta therefore carries a bootstrap confidence
interval over validation frames, so a reported gain can be told apart from sampling
scatter rather than being read off a point estimate.

**Heterogeneity.** PLAN.md section 13.5, says the project should stop if
gains are *uniform* across stages. That is a claim about the spread of per-stage deltas,
so :func:`heterogeneity_test` tests it directly with a permutation test on stage labels
instead of leaving it to visual inspection of a table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Delta:
    """One partial-vs-multi comparison with an uncertainty estimate."""

    label: str
    n: int
    mean_partial: float
    mean_multi: float
    delta: float            # positive means the multi-view condition has lower error
    rel_delta: float        # delta / mean_partial
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def significant(self) -> bool:
        """CI excludes zero. Not a multiplicity-corrected claim, just a first filter."""
        return (self.ci_low > 0) or (self.ci_high < 0)

    def as_row(self) -> dict:
        return {
            "stage": self.label,
            "n_frames": self.n,
            "mse_partial": self.mean_partial,
            "mse_multi": self.mean_multi,
            "delta": self.delta,
            "rel_delta_pct": 100.0 * self.rel_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "significant": self.significant,
        }


def _bootstrap_paired(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap of ``mean(a) - mean(b)``. Returns (ci_low, ci_high, p_value).

    Paired because both conditions were evaluated on the *same* validation frames; an
    unpaired interval would discard that and be needlessly wide.
    """
    if a.size == 0:
        return (float("nan"),) * 3
    d = a - b
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Two-sided bootstrap p-value for H0: mean difference == 0.
    centred = boots - boots.mean()
    p = float(np.mean(np.abs(centred) >= abs(d.mean())))
    return float(lo), float(hi), p


def overall_delta(
    err_partial: np.ndarray,
    err_multi: np.ndarray,
    label: str = "all",
    n_boot: int = 2000,
    seed: int = 0,
) -> Delta:
    """Aggregate partial-vs-multi delta over all frames."""
    lo, hi, p = _bootstrap_paired(err_partial, err_multi, n_boot, seed)
    mp, mm = float(err_partial.mean()), float(err_multi.mean())
    d = mp - mm
    return Delta(
        label=label, n=int(err_partial.size), mean_partial=mp, mean_multi=mm,
        delta=d, rel_delta=(d / mp if mp else float("nan")),
        ci_low=lo, ci_high=hi, p_value=p,
    )


def delta_by_stage(
    err_partial: np.ndarray,
    err_multi: np.ndarray,
    stage: np.ndarray,
    stage_names: dict | None = None,
    min_frames: int = 30,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[Delta]:
    """``Delta_view(g)`` for every stage with enough validation frames.

    Stages with fewer than ``min_frames`` frames are skipped rather than reported with a
    meaningless interval.
    """
    out = []
    names = stage_names or {}
    for sid in sorted(set(int(s) for s in np.unique(stage) if s >= 0)):
        m = stage == sid
        if int(m.sum()) < min_frames:
            continue
        label = names.get(str(sid), f"stage{sid}")
        out.append(
            overall_delta(err_partial[m], err_multi[m], label, n_boot, seed)
        )
    return out


def heterogeneity_test(
    err_partial: np.ndarray,
    err_multi: np.ndarray,
    stage: np.ndarray,
    n_perm: int = 2000,
    seed: int = 0,
    min_frames: int = 30,
) -> dict:
    """Test whether the improvement actually varies across stages.

    Statistic: the spread (std) of the per-stage mean improvement. The null distribution
    comes from shuffling stage labels across frames, which breaks any association between
    stage and improvement while keeping the marginal distribution of improvements intact.

    A small p-value means improvement is stage dependent, which is the structured result
    PLAN.md section 9 is looking for. A large p-value is evidence *for* kill criterion 5
    that the effect is not stage dependent, and this function says so in
    ``interpretation`` rather than leaving the direction of the test ambiguous.
    """
    d = err_partial - err_multi
    valid = stage >= 0
    d, st = d[valid], stage[valid]
    sids = [s for s in np.unique(st) if int((st == s).sum()) >= min_frames]
    if len(sids) < 2:
        return {
            "applicable": False,
            "reason": f"need >=2 stages with >={min_frames} frames, found {len(sids)}",
        }

    keep = np.isin(st, sids)
    d, st = d[keep], st[keep]

    def spread(dd: np.ndarray, ss: np.ndarray) -> float:
        means = [dd[ss == s].mean() for s in sids]
        return float(np.std(means))

    obs = spread(d, st)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spread(d, rng.permutation(st))
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))

    per_stage = {int(s): float(d[st == s].mean()) for s in sids}
    return {
        "applicable": True,
        "num_stages": len(sids),
        "observed_spread": obs,
        "null_spread_mean": float(null.mean()),
        "p_value": p,
        "stage_dependent": bool(p < 0.05),
        "per_stage_mean_improvement": per_stage,
        "interpretation": (
            "improvement varies across stages (structured, consistent with PLAN.md "
            "structure)"
            if p < 0.05
            else "improvement is statistically uniform across stages; this is evidence "
            "FOR kill criterion 5 in PLAN.md section 13.5"
        ),
    }


def stage_frame_counts(stage: np.ndarray, stage_names: dict | None = None) -> dict:
    names = stage_names or {}
    out = {}
    for sid in sorted(set(int(s) for s in np.unique(stage) if s >= 0)):
        out[names.get(str(sid), f"stage{sid}")] = int((stage == sid).sum())
    return out
