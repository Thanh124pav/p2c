"""Tests for the analysis layer: stage metrics, complementarity and kill criteria.

These matter as much as the data tests. The kill-criteria code decides whether the project
continues, so it is tested against synthetic error arrays with *known* answers — including
the negative cases, where the correct behaviour is to trigger a kill rather than report a
phenomenon.
"""

from __future__ import annotations

import numpy as np
import pytest

from p2c.analysis.complementarity import (
    complementarity,
    most_improved_frames,
    per_frame_complementarity,
)
from p2c.analysis.kill_criteria import check_validity, evaluate, report
from p2c.analysis.stage_metrics import (
    delta_by_stage,
    heterogeneity_test,
    overall_delta,
    stage_frame_counts,
)

N = 600


def rng(seed=0):
    return np.random.default_rng(seed)


def errs(base: float, n: int = N, scale: float = 0.05, seed: int = 0) -> np.ndarray:
    return np.abs(rng(seed).normal(base, scale, n))


# ---------------------------------------------------------------- stage metrics


def test_overall_delta_detects_a_real_improvement():
    a = errs(1.0, seed=1)
    d = overall_delta(a, a * 0.5, seed=0)
    assert d.delta > 0
    assert d.rel_delta == pytest.approx(0.5, abs=0.05)
    assert d.significant
    assert d.ci_low > 0


def test_overall_delta_on_identical_errors_is_not_significant():
    a = errs(1.0, seed=2)
    d = overall_delta(a, a.copy(), seed=0)
    assert d.delta == pytest.approx(0.0, abs=1e-12)
    assert not d.significant


def test_overall_delta_handles_a_regression():
    a = errs(1.0, seed=3)
    d = overall_delta(a, a * 1.5, seed=0)
    assert d.delta < 0
    assert d.ci_high < 0


def test_delta_by_stage_splits_by_label():
    stage = np.repeat([0, 1, 2], N // 3)
    partial = errs(1.0, seed=4)
    # Only stage 2 improves.
    multi = partial.copy()
    multi[stage == 2] *= 0.4
    deltas = delta_by_stage(partial, multi, stage, {"0": "nav", "1": "pick", "2": "place"})
    by = {d.label: d for d in deltas}
    assert set(by) == {"nav", "pick", "place"}
    assert by["place"].delta > 0 and by["place"].significant
    assert by["nav"].delta == pytest.approx(0.0, abs=1e-12)
    assert not by["nav"].significant


def test_delta_by_stage_skips_tiny_stages():
    stage = np.zeros(N, int)
    stage[:5] = 1  # too few frames to say anything
    deltas = delta_by_stage(errs(1.0), errs(0.8), stage, min_frames=30)
    assert [d.label for d in deltas] == ["stage0"]


def test_stage_frame_counts():
    stage = np.array([0, 0, 1, -1, 1, 1])
    assert stage_frame_counts(stage, {"0": "a", "1": "b"}) == {"a": 2, "b": 3}


# ---------------------------------------------------------------- heterogeneity


def test_heterogeneity_detects_stage_dependent_gains():
    """The structured result SETUP.md section 10 is looking for."""
    stage = np.repeat([0, 1, 2], N // 3)
    partial = errs(1.0, seed=5)
    multi = partial.copy()
    multi[stage == 1] *= 0.5     # big gain
    multi[stage == 2] *= 0.85    # modest gain
    h = heterogeneity_test(partial, multi, stage, n_perm=500, seed=0)
    assert h["applicable"]
    assert h["stage_dependent"] is True
    assert h["p_value"] < 0.05
    assert "varies across stages" in h["interpretation"]


def test_heterogeneity_reports_uniform_gains_as_kill_evidence():
    """A uniform gain must be reported as evidence FOR kill criterion 5."""
    stage = np.repeat([0, 1, 2], N // 3)
    partial = errs(1.0, seed=6)
    multi = partial * 0.7        # identical relative gain everywhere
    h = heterogeneity_test(partial, multi, stage, n_perm=500, seed=0)
    assert h["applicable"]
    assert h["stage_dependent"] is False
    assert "kill criterion 5" in h["interpretation"]


def test_heterogeneity_needs_at_least_two_stages():
    h = heterogeneity_test(errs(1.0), errs(0.8), np.zeros(N, int), n_perm=50)
    assert h["applicable"] is False
    assert "need >=2 stages" in h["reason"]


def test_heterogeneity_ignores_unlabelled_frames():
    stage = np.full(N, -1)
    h = heterogeneity_test(errs(1.0), errs(0.8), stage, n_perm=50)
    assert h["applicable"] is False


# ---------------------------------------------------------------- complementarity


def test_complementarity_ranks_candidates():
    base = errs(1.0, seed=7)
    errors = {
        "single_primary": base,
        "primary+wrist": base * 0.6,
        "primary+secondary": base * 0.9,
    }
    c = complementarity(errors, "single_primary")
    rows = c.as_rows()
    assert rows[0]["candidate"] == "primary+wrist"   # sorted by mean C, descending
    assert c.per_view["primary+wrist"]["mean_C"] > c.per_view["primary+secondary"]["mean_C"]
    assert c.per_view["primary+wrist"]["frac_positive"] == 1.0
    assert rows[0]["frac_frames_positive"] == 1.0  # the as_rows() alias
    assert c.best_fixed_condition == "primary+wrist"
    assert c.summary()


def test_selection_bias_floor_exposes_a_spurious_headroom():
    """Independent noise alone produces a large apparent oracle headroom.

    A per-frame minimum over K noisy conditions is biased downwards even with no real
    per-frame structure. Without this null, that bias reads as evidence that a learned
    view selector (B5) would help.
    """
    rg = rng(40)
    base = np.abs(rg.normal(1.0, 0.3, N))
    # Three conditions with the same mean error and no shared per-frame structure.
    errors = {
        "single_primary": base,
        "a": np.abs(rg.normal(1.0, 0.3, N)),
        "b": np.abs(rg.normal(1.0, 0.3, N)),
        "c": np.abs(rg.normal(1.0, 0.3, N)),
    }
    c = complementarity(errors, "single_primary", n_null=100, seed=0)
    assert c.headroom_pct > 10, "pure noise should still show a large raw headroom"
    assert c.headroom_noise_floor_pct > 0
    excess = c.headroom_pct - c.headroom_noise_floor_pct
    assert excess < 2.0, f"noise-only data must show no structure, got {excess:+.1f}pp"
    assert "mostly selection bias" in c.summary()


def test_genuine_per_frame_structure_exceeds_the_noise_floor():
    """When different views really do win on different frames, the excess is positive."""
    n = N
    base = np.full(n, 1.0)
    half = np.zeros(n, bool)
    half[: n // 2] = True
    a, b = base.copy(), base.copy()
    a[half] = 0.05          # view a is decisive on one half
    b[~half] = 0.05         # view b on the other
    c = complementarity({"single_primary": base, "a": a, "b": b}, "single_primary",
                        n_null=100, seed=0)
    excess = c.headroom_pct - c.headroom_noise_floor_pct
    assert excess > 2.0, f"real structure should clear the null, got {excess:+.1f}pp"
    assert "structure beyond selection bias" in c.summary()


def test_complementarity_headroom_is_zero_when_one_view_always_wins():
    """If one fixed view dominates every frame, a selector has nothing to add."""
    base = errs(1.0, seed=8)
    errors = {
        "single_primary": base,
        "primary+wrist": base * 0.5,
        "primary+secondary": base * 0.9,
    }
    c = complementarity(errors, "single_primary")
    assert c.headroom == pytest.approx(0.0, abs=1e-12)
    assert c.headroom_pct - c.headroom_noise_floor_pct < 2.0
    assert "little real headroom" in c.summary()


def test_complementarity_headroom_is_positive_when_the_best_view_alternates():
    base = errs(1.0, n=N, seed=9)
    half = np.zeros(N, bool)
    half[: N // 2] = True
    a, b = base.copy(), base.copy()
    a[half] *= 0.4
    b[~half] *= 0.4
    c = complementarity({"single_primary": base, "primary+wrist": a, "primary+secondary": b},
                        "single_primary")
    assert c.headroom > 0
    assert c.headroom_pct > 2.0
    shares = [v["oracle_best_share"] for v in c.per_view.values()]
    assert all(s > 0.3 for s in shares), "both views should win on some frames"


def test_complementarity_requires_the_primary_condition():
    with pytest.raises(KeyError, match="not among results"):
        complementarity({"primary+wrist": errs(1.0)}, "single_primary")


def test_complementarity_requires_a_candidate():
    with pytest.raises(ValueError, match="no candidate"):
        complementarity({"single_primary": errs(1.0)}, "single_primary")


def test_per_frame_complementarity_is_a_difference():
    a, b = errs(1.0, seed=10), errs(0.5, seed=11)
    C = per_frame_complementarity({"p": a, "q": b}, "p", "q")
    np.testing.assert_allclose(C, a - b)


def test_most_improved_frames_decodes_keys_and_sorts():
    keys = np.array([1_000_005, 2_000_010, 3_000_000], dtype=np.int64)
    errors = {"p": np.array([1.0, 1.0, 1.0]), "q": np.array([0.9, 0.1, 0.99])}
    top = most_improved_frames(keys, errors, "p", "q", top_k=2)
    assert top[0]["episode"] == 2 and top[0]["frame"] == 10
    assert top[0]["C"] > top[1]["C"]


# ---------------------------------------------------------------- kill criteria


def base_errors(seed=20):
    """A clean positive phenomenon: the complementary pair wins, controls do not."""
    base = errs(1.0, seed=seed)
    return {
        "single_primary": base,
        "single_wrist": base * 1.1,
        "primary+wrist": base * 0.6,       # real complementary gain
        "random_two": base * 0.95,         # control recovers little
        "primary+duplicate": base * 0.99,  # duplicate adds nearly nothing
        "all_views": base * 0.55,
    }


def verdicts_by_number(vs):
    return {v.criterion: v for v in vs}


def test_no_criteria_trigger_on_a_clean_phenomenon():
    vs = evaluate(base_errors(), capacity_matched={k: True for k in base_errors()})
    by = verdicts_by_number(vs)
    assert by[1].status == "not_triggered"
    assert by[2].status == "not_triggered"
    assert by[3].status == "not_triggered"
    assert by[4].status == "not_triggered"
    assert "no kill criterion triggered" in report(vs)


def test_criterion_1_triggers_without_a_meaningful_gain():
    base = errs(1.0, seed=21)
    e = {"single_primary": base, "primary+wrist": base * 0.99}
    by = verdicts_by_number(evaluate(e, capacity_matched={k: True for k in e}))
    assert by[1].triggered
    assert "relative gain" in by[1].detail


def test_criterion_2_triggers_when_random_matches_complementary():
    """SETUP.md section 12 Control A: the result would be generic multi-view scaling."""
    base = errs(1.0, seed=22)
    e = {
        "single_primary": base,
        "primary+wrist": base * 0.6,
        "random_two": base * 0.62,   # recovers ~95% of the gain
    }
    by = verdicts_by_number(evaluate(e))
    assert by[2].triggered
    assert "generic multi-view scaling" in by[2].detail


def test_criterion_3_triggers_when_duplicate_matches_complementary():
    base = errs(1.0, seed=23)
    e = {
        "single_primary": base,
        "primary+wrist": base * 0.6,
        "primary+duplicate": base * 0.63,
    }
    by = verdicts_by_number(evaluate(e))
    assert by[3].triggered
    assert "input/compute scaling" in by[3].detail


def test_controls_not_evaluated_when_missing():
    base = errs(1.0, seed=24)
    e = {"single_primary": base, "primary+wrist": base * 0.6}
    by = verdicts_by_number(evaluate(e))
    assert by[2].status == "not_evaluated"
    assert by[3].status == "not_evaluated"
    assert "missing condition" in by[2].detail


def test_control_criteria_skip_when_the_pair_has_no_gain():
    """With no gain to match, criterion 1 is the relevant one, not 2 or 3."""
    base = errs(1.0, seed=25)
    e = {
        "single_primary": base,
        "primary+wrist": base * 1.2,     # pair is worse
        "random_two": base * 1.1,
        "primary+duplicate": base * 1.15,
    }
    by = verdicts_by_number(evaluate(e))
    assert by[2].status == "not_evaluated"
    assert "does not beat the primary" in by[2].detail


def test_criterion_4_not_evaluated_when_capacity_is_unmatched():
    e = base_errors()
    cap = {k: True for k in e}
    cap["primary+wrist"] = False
    by = verdicts_by_number(evaluate(e, capacity_matched=cap))
    assert by[4].status == "not_evaluated"
    assert "not capacity matched" in by[4].detail


def test_criterion_5_triggers_on_uniform_stage_gains():
    stage = np.repeat([0, 1, 2], N // 3)
    base = errs(1.0, seed=26)
    e = {"single_primary": base, "primary+wrist": base * 0.6}
    by = verdicts_by_number(evaluate(e, stage))
    assert by[5].triggered
    assert "kill criterion 5" in by[5].detail


def test_criterion_5_does_not_trigger_on_stage_dependent_gains():
    stage = np.repeat([0, 1, 2], N // 3)
    base = errs(1.0, seed=27)
    multi = base.copy()
    multi[stage == 1] *= 0.4
    multi[stage == 2] *= 0.8
    by = verdicts_by_number(evaluate({"single_primary": base, "primary+wrist": multi}, stage))
    assert by[5].status == "not_triggered"


def test_criterion_5_not_evaluated_without_stage_labels():
    by = verdicts_by_number(evaluate(base_errors(), None))
    assert by[5].status == "not_evaluated"
    assert "composite" in by[5].detail


def test_criteria_6_and_7_are_always_reported_as_open_risks():
    """They must not be silently treated as passes."""
    vs = evaluate(base_errors())
    by = verdicts_by_number(vs)
    assert by[6].status == "not_evaluated"
    assert by[7].status == "not_evaluated"
    assert "open risks" in report(vs)


def test_report_names_triggered_criteria():
    base = errs(1.0, seed=28)
    e = {"single_primary": base, "primary+wrist": base * 0.995}
    txt = report(evaluate(e))
    assert "KILL" in txt
    assert "stop or substantially reframe" in txt


def test_thresholds_are_recorded_in_the_output():
    e = base_errors()
    by = verdicts_by_number(evaluate(e, min_rel_gain=0.2, equivalence_margin=0.1))
    assert by[1].numbers["min_rel_gain"] == 0.2
    assert by[2].numbers["equivalence_margin"] == 0.1


# ---------------------------------------------------------------- validity (criterion 0)


def test_validity_triggers_when_nothing_beats_the_trivial_baseline():
    """The NavigateKitchen failure mode: every arm sits at predict-the-mean."""
    base = errs(1.0, seed=30)
    e = {"single_primary": base * 1.03, "single_wrist": base * 0.98}
    v = check_validity(e, trivial_baseline=float((base).mean()))
    assert v.triggered
    assert "NO ARM MEANINGFULLY BEATS" in v.detail


def test_validity_passes_when_a_condition_clearly_learns():
    base = errs(1.0, seed=31)
    e = {"single_primary": base * 0.5, "primary+wrist": base * 0.3}
    v = check_validity(e, trivial_baseline=float(base.mean()))
    assert v.status == "not_triggered"
    assert "interpretable" in v.detail


def test_validity_mentions_language_ambiguity():
    base = errs(1.0, seed=32)
    v = check_validity({"single_primary": base}, float(base.mean()), num_instructions=14)
    assert "14 language instructions" in v.detail


def test_validity_not_evaluated_without_a_baseline():
    v = check_validity({"a": errs(1.0)}, float("nan"))
    assert v.status == "not_evaluated"


def test_invalid_experiment_suppresses_the_kill_verdict():
    """A broken experiment must not be reported as a falsified hypothesis."""
    base = errs(1.0, seed=33)
    e = {"single_primary": base, "primary+wrist": base * 0.995}
    txt = report(evaluate(e, trivial_baseline=float(base.mean())))
    assert "not valid (criterion 0)" in txt
    assert "Do not record a kill on this run" in txt
    # The ordinary "stop or reframe" verdict must NOT appear.
    assert "stop or substantially reframe" not in txt


def test_valid_experiment_still_reports_kills():
    base = errs(1.0, seed=34)
    e = {"single_primary": base * 0.5, "primary+wrist": base * 0.499}
    txt = report(evaluate(e, trivial_baseline=float(base.mean())))
    assert "not valid (criterion 0)" not in txt
    assert "KILL" in txt


def test_evaluate_without_baseline_omits_criterion_zero():
    vs = evaluate(base_errors())
    assert all(v.criterion != 0 for v in vs)


def test_min_rel_gain_threshold_changes_the_verdict():
    base = errs(1.0, seed=29)
    e = {"single_primary": base, "primary+wrist": base * 0.92}  # 8% gain
    assert not verdicts_by_number(evaluate(e, min_rel_gain=0.05))[1].triggered
    assert verdicts_by_number(evaluate(e, min_rel_gain=0.15))[1].triggered
