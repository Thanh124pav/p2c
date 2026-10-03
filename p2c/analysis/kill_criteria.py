"""Automated evaluation of the kill criteria in SETUP.md section 21.

The plan is explicit that P2C should be falsifiable and that a negative result is useful
(sections 21 and 25). Leaving the criteria to be eyeballed off a table invites reading a
2% wobble as a phenomenon, so each computable criterion is checked here against a stated
threshold with a paired bootstrap interval, and the verdict is reported either way.

Criteria 1-5 are computable from the local ablation. Criterion 6 needs a camera-pose
conditioned baseline, which SETUP.md section 15 defers to after the MVP; criterion 7 is a
judgement about the final method, not a measurement. Both are reported as not evaluated
rather than quietly passed.

Thresholds are arguments, not constants, and whatever is used is recorded in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from p2c.analysis.stage_metrics import _bootstrap_paired, heterogeneity_test


@dataclass
class Verdict:
    criterion: int
    name: str
    status: str             # "triggered" | "not_triggered" | "not_evaluated"
    detail: str
    numbers: dict = field(default_factory=dict)

    @property
    def triggered(self) -> bool:
        return self.status == "triggered"

    def line(self) -> str:
        mark = {
            "triggered": "KILL",
            "not_triggered": "ok  ",
            "not_evaluated": "n/a ",
        }[self.status]
        return f"  [{mark}] ({self.criterion}) {self.name}\n         {self.detail}"


def _rel(a: float, b: float) -> float:
    """Relative reduction from a to b, as a fraction of a."""
    return (a - b) / a if a else float("nan")


def check_validity(
    errors: dict[str, np.ndarray],
    trivial_baseline: float,
    num_instructions: int = 1,
) -> Verdict:
    """Precondition: did *any* condition learn anything at all?

    This is separate from the kill criteria on purpose. If no arm beats the
    predict-the-mean baseline, the ablation has no signal, and that is a statement about
    the experiment — underpowered, or a task whose goal is not visually determined — not
    evidence against P2C. Collapsing the two would let a broken setup masquerade as a
    falsified hypothesis, which is the opposite of what SETUP.md section 21 is for.
    """
    if trivial_baseline != trivial_baseline:  # NaN
        return Verdict(
            0, "experiment validity", "not_evaluated",
            "no predict-the-mean baseline recorded; re-run to capture it",
        )

    best = min(errors, key=lambda c: float(errors[c].mean())) if errors else None
    best_mse = float(errors[best].mean()) if best else float("nan")
    margin = (trivial_baseline - best_mse) / trivial_baseline if trivial_baseline else 0.0

    detail = (
        f"best condition '{best}' mse={best_mse:.5f} vs predict-the-mean "
        f"{trivial_baseline:.5f} ({margin*100:+.1f}%)"
    )
    if num_instructions > 1:
        detail += (
            f"; cache spans {num_instructions} language instructions, so part of the "
            f"action is unpredictable from the image alone"
        )

    if margin < 0.05:
        return Verdict(
            0, "experiment validity", "triggered",
            detail + ".\n         NO ARM MEANINGFULLY BEATS THE TRIVIAL BASELINE. The "
            "kill criteria below are NOT interpretable as evidence about P2C: fix the "
            "experiment (more data, a visually determined task, or goal conditioning) "
            "before drawing any conclusion.",
            {"best": best, "best_mse": best_mse, "trivial_baseline": trivial_baseline,
             "margin": margin, "num_instructions": num_instructions},
        )
    return Verdict(
        0, "experiment validity", "not_triggered",
        detail + " - at least one condition has learned a useful mapping, so the "
        "criteria below are interpretable",
        {"best": best, "best_mse": best_mse, "trivial_baseline": trivial_baseline,
         "margin": margin, "num_instructions": num_instructions},
    )


def evaluate(
    errors: dict[str, np.ndarray],
    stage: np.ndarray | None = None,
    *,
    capacity_matched: dict[str, bool] | None = None,
    primary: str = "single_primary",
    complementary: str = "primary+wrist",
    min_rel_gain: float = 0.05,
    equivalence_margin: float = 0.25,
    seed: int = 0,
    trivial_baseline: float | None = None,
    num_instructions: int = 1,
) -> list[Verdict]:
    """Check every computable kill criterion.

    Parameters
    ----------
    errors:
        condition -> per-frame squared error, aligned across conditions.
    stage:
        per-frame stage id, or None when the task has no stage annotations.
    capacity_matched:
        condition -> whether that run's parameter count was view-count independent.
    min_rel_gain:
        A multi-view gain below this relative reduction counts as "no meaningful gain"
        (criterion 1). Default 5%.
    equivalence_margin:
        For criteria 2 and 3, a control is treated as performing "as well as" the
        complementary pair when it recovers at least ``1 - equivalence_margin`` of the
        pair's gain. Default: a control reaching 75% of the gain triggers the criterion.
    """
    out: list[Verdict] = []
    have = set(errors)
    thresholds = {
        "min_rel_gain": min_rel_gain,
        "equivalence_margin": equivalence_margin,
    }

    if trivial_baseline is not None:
        out.append(check_validity(errors, trivial_baseline, num_instructions))

    def mean(c: str) -> float:
        return float(errors[c].mean())

    # ---- criterion 1: no meaningful gain over the strongest single view ----
    singles = [c for c in have if c.startswith("single_")]
    multis = [c for c in have if not c.startswith("single_")]
    if singles and multis:
        best_single = min(singles, key=mean)
        best_multi = min(multis, key=mean)
        gain = _rel(mean(best_single), mean(best_multi))
        lo, hi, p = _bootstrap_paired(errors[best_single], errors[best_multi], seed=seed)
        trig = (gain < min_rel_gain) or (lo <= 0)
        out.append(Verdict(
            1, "multi-view gives no meaningful gain over the strongest single view",
            "triggered" if trig else "not_triggered",
            f"best single '{best_single}' mse={mean(best_single):.5f}; "
            f"best multi '{best_multi}' mse={mean(best_multi):.5f}; "
            f"relative gain {gain*100:.1f}% (threshold {min_rel_gain*100:.0f}%), "
            f"paired 95% CI [{lo:.5f}, {hi:.5f}], p={p:.3f}"
            + ("" if not trig else "  <- gain too small or CI includes zero"),
            {"best_single": best_single, "best_multi": best_multi,
             "rel_gain": gain, "ci": [lo, hi], "p_value": p, **thresholds},
        ))
    else:
        out.append(Verdict(
            1, "multi-view gives no meaningful gain over the strongest single view",
            "not_evaluated",
            f"need at least one single-view and one multi-view condition; have {sorted(have)}",
        ))

    # ---- criterion 2: random extra view performs as well as the complementary pair ----
    out.append(_control_criterion(
        errors, 2,
        "random additional view performs as well as the complementary view",
        primary, complementary, "random_two", equivalence_margin, seed, thresholds,
        note="this is SETUP.md section 12 Control A; if triggered, the effect is generic "
             "multi-view scaling, not complementarity",
    ))

    # ---- criterion 3: duplicate view performs as well as a real second view ----
    out.append(_control_criterion(
        errors, 3,
        "duplicate-view control performs similarly to a real second view",
        primary, complementary, "primary+duplicate", equivalence_margin, seed, thresholds,
        note="this is SETUP.md section 12 Control D; if triggered, the gain comes from "
             "input/compute scaling rather than new information",
    ))

    # ---- criterion 4: gains disappear once capacity is matched ----
    if capacity_matched:
        unmatched = sorted(c for c, ok in capacity_matched.items() if not ok)
        relevant = [c for c in (primary, complementary) if c in have]
        if unmatched:
            out.append(Verdict(
                4, "gains disappear after matching model capacity",
                "not_evaluated",
                f"these runs were not capacity matched: {unmatched}. Re-run them with a "
                f"view-count-independent fusion (mean/attn) before trusting any gain.",
                {"unmatched": unmatched},
            ))
        elif len(relevant) == 2:
            out.append(Verdict(
                4, "gains disappear after matching model capacity",
                "not_triggered",
                "all compared runs used a shared encoder and view-count-independent "
                "fusion, so the parameter count is identical across conditions; any gain "
                "cannot be attributed to extra capacity",
                {"capacity_matched": True},
            ))
        else:
            out.append(Verdict(
                4, "gains disappear after matching model capacity", "not_evaluated",
                f"need both '{primary}' and '{complementary}'; have {sorted(have)}",
            ))
    else:
        out.append(Verdict(
            4, "gains disappear after matching model capacity", "not_evaluated",
            "no capacity metadata supplied",
        ))

    # ---- criterion 5: gains uniform across stages ----
    if stage is not None and primary in have and complementary in have and (stage >= 0).any():
        h = heterogeneity_test(errors[primary], errors[complementary], stage, seed=seed)
        if not h.get("applicable"):
            out.append(Verdict(
                5, "gains are uniform across all task stages", "not_evaluated",
                str(h.get("reason")), h,
            ))
        else:
            trig = not h["stage_dependent"]
            out.append(Verdict(
                5, "gains are uniform across all task stages",
                "triggered" if trig else "not_triggered",
                f"permutation test over {h['num_stages']} stages: observed spread of "
                f"per-stage improvement {h['observed_spread']:.5f} vs null "
                f"{h['null_spread_mean']:.5f}, p={h['p_value']:.3f}. {h['interpretation']}",
                h,
            ))
    else:
        out.append(Verdict(
            5, "gains are uniform across all task stages", "not_evaluated",
            "no per-frame stage annotations for this task. RoboCasa365 ships these only "
            "for target *composite* tasks, so this criterion needs a composite dataset.",
        ))

    # ---- criteria 6 and 7: outside the MVP ----
    out.append(Verdict(
        6, "a camera-pose-conditioned single-view baseline closes almost all of the gap",
        "not_evaluated",
        "requires the camera-pose conditioning baseline, which SETUP.md section 15 lists "
        "as Priority 3 after the MVP. Not implemented yet, so this remains an open risk.",
    ))
    out.append(Verdict(
        7, "the only successful version requires full 3D/4D reconstruction",
        "not_evaluated",
        "a judgement about the eventual method, not measurable from the data study. "
        "Revisit when a P2C method exists beyond real-camera oracle selection.",
    ))
    return out


def _control_criterion(
    errors: dict[str, np.ndarray],
    number: int,
    name: str,
    primary: str,
    complementary: str,
    control: str,
    margin: float,
    seed: int,
    thresholds: dict,
    note: str = "",
) -> Verdict:
    """Shared logic for the random-view and duplicate-view controls."""
    need = [primary, complementary, control]
    missing = [c for c in need if c not in errors]
    if missing:
        return Verdict(
            number, name, "not_evaluated",
            f"missing condition(s) {missing}; run them to evaluate this criterion",
        )

    e_p = float(errors[primary].mean())
    e_c = float(errors[complementary].mean())
    e_x = float(errors[control].mean())
    gain_pair = e_p - e_c
    gain_ctrl = e_p - e_x

    if gain_pair <= 0:
        return Verdict(
            number, name, "not_evaluated",
            f"the complementary pair '{complementary}' does not beat the primary "
            f"(mse {e_c:.5f} vs {e_p:.5f}), so there is no gain for the control to match. "
            f"Criterion 1 is the relevant one here.",
            {"gain_pair": gain_pair, "gain_control": gain_ctrl},
        )

    share = gain_ctrl / gain_pair
    lo, hi, p = _bootstrap_paired(errors[control], errors[complementary], seed=seed)
    trig = share >= (1.0 - margin)
    detail = (
        f"primary mse={e_p:.5f}, '{complementary}' mse={e_c:.5f} (gain {gain_pair:.5f}), "
        f"'{control}' mse={e_x:.5f} (gain {gain_ctrl:.5f}) = {share*100:.0f}% of the pair's "
        f"gain; control-minus-pair 95% CI [{lo:.5f}, {hi:.5f}], p={p:.3f}"
    )
    if trig:
        detail += f"  <- control recovers >={(1-margin)*100:.0f}% of the gain"
    if note:
        detail += f"\n         {note}"
    return Verdict(
        number, name, "triggered" if trig else "not_triggered", detail,
        {"gain_pair": gain_pair, "gain_control": gain_ctrl, "share": share,
         "ci": [lo, hi], "p_value": p, **thresholds},
    )


def report(verdicts: list[Verdict]) -> str:
    L = ["=" * 78, "KILL CRITERIA (SETUP.md section 21)", "=" * 78]
    for v in verdicts:
        L.append(v.line())

    validity = next((v for v in verdicts if v.criterion == 0), None)
    if validity is not None and validity.triggered:
        L.append("")
        L.append(
            "VERDICT: the experiment itself is not valid (criterion 0). Nothing below is\n"
            "         evidence for or against P2C. Do not record a kill on this run."
        )
        return "\n".join(L)

    trig = [v for v in verdicts if v.triggered and v.criterion != 0]
    na = [v for v in verdicts if v.status == "not_evaluated" and v.criterion != 0]
    L.append("")
    if trig:
        L.append(
            f"VERDICT: {len(trig)} criterion/criteria triggered "
            f"({', '.join(str(v.criterion) for v in trig)}). SETUP.md section 21 says to "
            f"stop or substantially reframe P2C rather than force a method."
        )
    else:
        L.append(
            "VERDICT: no kill criterion triggered on the evidence available."
        )
    if na:
        L.append(
            f"         {len(na)} criterion/criteria not evaluated "
            f"({', '.join(str(v.criterion) for v in na)}) - these remain open risks, "
            f"not passes."
        )
    return "\n".join(L)
