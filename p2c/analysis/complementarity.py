"""Complementarity metric, SETUP.md section 13.

For a candidate additional view ``v`` given primary view ``p``::

    C(v | p) = L(pi_p) - L(pi_{p+v})

Computed per validation frame from runs that shared a split, then aggregated. This is the
oracle/data-derived target that section 13 says to produce *before* training any
predictor.

Scope note: section 13 is gated on B0-B4 and section 9 says not to implement B5 (learned
view selection) until those produce a convincing phenomenon. So this module computes the
oracle target and measures how much headroom a selector would have — it deliberately does
**not** implement the predictor ``C_phi`` or the selector ``argmax_v C_phi``. The headroom
number is the cheap pre-check for whether B5 is worth building at all: if an oracle
selector barely beats the single best fixed view, a learned selector cannot do better, and
building one would be wasted effort.

Section 13 also warns against reaching for mutual-information estimators before the
empirical signal justifies it; nothing here does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ComplementarityResult:
    """Per-view complementarity scores plus oracle-selection headroom."""

    primary_condition: str
    mean_primary_error: float
    per_view: dict[str, dict]          # candidate condition -> stats
    oracle_select_error: float         # error if the best view were picked per frame
    best_fixed_condition: str
    best_fixed_error: float
    headroom: float                    # best_fixed_error - oracle_select_error
    headroom_pct: float
    n_frames: int
    # Selection-bias null: the same per-frame minimum computed after independently
    # permuting each condition's errors, which destroys any real per-frame structure
    # while preserving each condition's marginal error distribution.
    headroom_noise_floor: float = 0.0
    headroom_noise_floor_pct: float = 0.0

    def as_rows(self) -> list[dict]:
        rows = []
        for cond, s in sorted(self.per_view.items(), key=lambda kv: -kv[1]["mean_C"]):
            rows.append({
                "candidate": cond,
                "mean_C": s["mean_C"],
                "mean_C_pct": s["mean_C_pct"],
                "frac_frames_positive": s["frac_positive"],
                "mean_error": s["mean_error"],
                "oracle_best_share": s["oracle_best_share"],
            })
        return rows

    def summary(self) -> str:
        L = [
            f"complementarity relative to '{self.primary_condition}' "
            f"(L_partial = {self.mean_primary_error:.5f}, n = {self.n_frames} frames)",
            "",
            f"  {'candidate':26s} {'mean C':>10s} {'C %':>8s} "
            f"{'frames C>0':>11s} {'oracle pick':>12s}",
        ]
        for r in self.as_rows():
            L.append(
                f"  {r['candidate']:26s} {r['mean_C']:10.5f} {r['mean_C_pct']:7.1f}% "
                f"{r['frac_frames_positive']*100:10.1f}% {r['oracle_best_share']*100:11.1f}%"
            )
        excess = self.headroom_pct - self.headroom_noise_floor_pct
        L += [
            "",
            f"  best single fixed view : {self.best_fixed_condition} "
            f"(error {self.best_fixed_error:.5f})",
            f"  oracle per-frame select: error {self.oracle_select_error:.5f}",
            f"  raw oracle headroom    : {self.headroom:.5f} "
            f"({self.headroom_pct:.1f}% of the best fixed view)",
            f"  selection-bias floor   : {self.headroom_noise_floor:.5f} "
            f"({self.headroom_noise_floor_pct:.1f}%) - what the same per-frame minimum",
            "                           yields from permuted errors, i.e. from noise alone",
            f"  structure above noise  : {excess:+.1f} percentage points",
        ]
        L.append("")
        if excess < 2.0:
            L.append(
                "  -> The raw headroom is mostly selection bias: taking a per-frame minimum\n"
                "     over several noisy models looks good even when the models carry no\n"
                "     complementary structure. On this evidence a learned selector (B5)\n"
                "     has little real headroom. Do NOT read the raw number as attainable."
            )
        else:
            L.append(
                "  -> There is per-frame structure beyond selection bias, so which view\n"
                "     helps genuinely varies by frame. Note the raw headroom is still an\n"
                "     upper bound: a learned selector can only capture the part of\n"
                "     C(v | p) that is predictable from the images."
            )
        return "\n".join(L)


def _selection_bias_floor(
    stack: np.ndarray, best_fixed_err: float, n_rep: int = 200, seed: int = 0
) -> float:
    """Headroom obtainable from noise alone, by permuting each condition independently.

    A per-frame minimum over K noisy error series is biased downwards even when no
    condition is genuinely better on any particular frame: with K candidates you are
    taking the best of K draws per frame. Permuting each condition's errors across frames
    keeps its marginal distribution but destroys any frame-level association, so the
    resulting apparent headroom is pure selection bias and forms the null to compare the
    observed headroom against.
    """
    rng = np.random.default_rng(seed)
    floors = np.empty(n_rep)
    for i in range(n_rep):
        permuted = np.stack([rng.permutation(row) for row in stack])
        floors[i] = best_fixed_err - float(permuted.min(axis=0).mean())
    return float(floors.mean())


def complementarity(
    errors: dict[str, np.ndarray],
    primary_condition: str,
    candidate_conditions: list[str] | None = None,
    n_null: int = 200,
    seed: int = 0,
) -> ComplementarityResult:
    """Compute ``C(v | p)`` for each candidate condition.

    Parameters
    ----------
    errors:
        condition -> per-frame squared error, all aligned on the same frames (use
        :func:`p2c.analysis.results.join_per_sample`).
    primary_condition:
        The partial-observation baseline, e.g. ``"single_primary"`` (B0).
    candidate_conditions:
        Conditions that add exactly one view to the primary. Defaults to every other
        condition present, which is fine for reading the table but means the caller
        should not interpret a 3-view condition's ``C`` as a single-view contribution.
    """
    if primary_condition not in errors:
        raise KeyError(
            f"primary condition '{primary_condition}' not among results: {sorted(errors)}"
        )
    base = errors[primary_condition]
    cands = candidate_conditions or [c for c in errors if c != primary_condition]
    cands = [c for c in cands if c in errors]
    if not cands:
        raise ValueError("no candidate conditions to compare against the primary")

    # Oracle per-frame selection over the candidates.
    stack = np.stack([errors[c] for c in cands])          # [C, N]
    oracle_err = stack.min(axis=0)
    oracle_arg = stack.argmin(axis=0)

    mean_base = float(base.mean())
    per_view: dict[str, dict] = {}
    for i, c in enumerate(cands):
        C = base - errors[c]
        per_view[c] = {
            "mean_C": float(C.mean()),
            "mean_C_pct": float(100.0 * C.mean() / mean_base) if mean_base else float("nan"),
            "frac_positive": float((C > 0).mean()),
            "mean_error": float(errors[c].mean()),
            "oracle_best_share": float((oracle_arg == i).mean()),
        }

    best_fixed = min(cands, key=lambda c: float(errors[c].mean()))
    best_fixed_err = float(errors[best_fixed].mean())
    oracle_mean = float(oracle_err.mean())
    headroom = best_fixed_err - oracle_mean
    floor = _selection_bias_floor(stack, best_fixed_err, n_rep=n_null, seed=seed)

    return ComplementarityResult(
        primary_condition=primary_condition,
        mean_primary_error=mean_base,
        per_view=per_view,
        oracle_select_error=oracle_mean,
        best_fixed_condition=best_fixed,
        best_fixed_error=best_fixed_err,
        headroom=headroom,
        headroom_pct=(100.0 * headroom / best_fixed_err) if best_fixed_err else float("nan"),
        n_frames=int(base.size),
        headroom_noise_floor=floor,
        headroom_noise_floor_pct=(
            100.0 * floor / best_fixed_err if best_fixed_err else float("nan")
        ),
    )


def per_frame_complementarity(
    errors: dict[str, np.ndarray], primary_condition: str, candidate: str
) -> np.ndarray:
    """Raw per-frame ``C(v | p)``, the oracle target of section 13."""
    return errors[primary_condition] - errors[candidate]


def most_improved_frames(
    keys: np.ndarray,
    errors: dict[str, np.ndarray],
    primary_condition: str,
    candidate: str,
    top_k: int = 16,
) -> list[dict]:
    """Frames where the extra view helps most (SETUP.md section 11 asks to visualise these).

    Returns ``(episode, frame)`` decoded from the join keys, so the caller can go back to
    the cache and render exactly those timesteps.
    """
    C = per_frame_complementarity(errors, primary_condition, candidate)
    order = np.argsort(-C)[:top_k]
    out = []
    for i in order:
        k = int(keys[i])
        out.append({
            "episode": k // 1_000_000,
            "frame": k % 1_000_000,
            "C": float(C[i]),
            "err_partial": float(errors[primary_condition][i]),
            "err_candidate": float(errors[candidate][i]),
        })
    return out
