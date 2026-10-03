"""Tests for the information tiers of PLAN.md section 9.

These guard the property the plan says carries the central claim: the P2C method must not
consume privileged simulator state, because cheap Internet video cannot supply it. A leak
here would be invisible in a results table — the numbers would simply look better — so it
is worth testing rather than documenting.
"""

from __future__ import annotations

import pytest

from p2c.data.tiers import (
    FIELD_TIERS,
    Tier,
    TieredSample,
    TierViolation,
    check_access,
    fields_at,
    tier_of,
)


# ---------------------------------------------------------------- tier ordering


def test_tiers_are_ordered_from_restricted_to_permissive():
    assert Tier.METHOD < Tier.POLICY < Tier.ORACLE


def test_each_tier_describes_itself():
    for t in Tier:
        assert t.description


# ---------------------------------------------------------------- classification


def test_rgb_is_method_tier():
    """What an action-free Internet clip supplies."""
    assert tier_of("images") is Tier.METHOD
    assert tier_of("language") is Tier.METHOD


def test_actions_and_proprioception_are_policy_tier():
    """Robot demonstrations are downstream supervision (PLAN.md section 3.2)."""
    assert tier_of("action") is Tier.POLICY
    assert tier_of("state") is Tier.POLICY


def test_stage_labels_are_oracle_tier():
    """Legitimate for per-stage analysis, illegitimate as policy input."""
    assert tier_of("stage") is Tier.ORACLE


def test_privileged_simulator_fields_are_oracle_tier():
    for f in ("object_pose", "segmentation", "contact", "sim_state", "reward", "success"):
        assert tier_of(f) is Tier.ORACLE, f


def test_withheld_complement_is_oracle_tier():
    """The ground-truth complement is the target, so feeding it in would be circular."""
    assert tier_of("withheld") is Tier.ORACLE


def test_unknown_fields_fail_closed():
    """A field added without thought must not silently become method input."""
    assert tier_of("some_new_privileged_thing") is Tier.ORACLE


# ---------------------------------------------------------------- access control


def test_method_tier_may_read_rgb():
    check_access("images", Tier.METHOD)


def test_method_tier_may_not_read_actions():
    with pytest.raises(TierViolation, match="POLICY-tier"):
        check_access("action", Tier.METHOD)


def test_method_tier_may_not_read_stage():
    with pytest.raises(TierViolation, match="ORACLE-tier"):
        check_access("stage", Tier.METHOD)


def test_policy_tier_may_read_actions_but_not_stage():
    check_access("action", Tier.POLICY)
    check_access("state", Tier.POLICY)
    with pytest.raises(TierViolation):
        check_access("stage", Tier.POLICY)


def test_oracle_tier_may_read_everything_known():
    for f in FIELD_TIERS:
        check_access(f, Tier.ORACLE)


def test_violation_message_points_at_the_plan():
    with pytest.raises(TierViolation, match="PLAN.md section 9"):
        check_access("sim_state", Tier.METHOD)


def test_fields_at_grows_with_tier():
    m, p, o = fields_at(Tier.METHOD), fields_at(Tier.POLICY), fields_at(Tier.ORACLE)
    assert set(m) < set(p) < set(o)
    assert "images" in m
    assert "action" in p and "action" not in m
    assert "stage" in o and "stage" not in p


# ---------------------------------------------------------------- TieredSample


def sample(tier: Tier) -> TieredSample:
    return TieredSample(
        {"images": 1, "state": 2, "action": 3, "stage": 4, "episode": 5}, tier
    )


def test_tiered_sample_allows_permitted_reads():
    s = sample(Tier.POLICY)
    assert s["images"] == 1
    assert s["action"] == 3


def test_tiered_sample_blocks_reads_above_its_tier():
    s = sample(Tier.POLICY)
    with pytest.raises(TierViolation):
        s["stage"]


def test_tiered_sample_blocks_get_too():
    """`.get` must not be an escape hatch around the check."""
    s = sample(Tier.METHOD)
    with pytest.raises(TierViolation):
        s.get("action")


def test_restricted_to_drops_higher_tier_fields():
    s = sample(Tier.ORACLE)
    method_view = s.restricted_to(Tier.METHOD)
    assert set(method_view) == {"images", "episode"}
    policy_view = s.restricted_to(Tier.POLICY)
    assert set(policy_view) == {"images", "episode", "state", "action"}


def test_restricted_view_is_a_plain_dict():
    """So it collates in a dataloader without carrying the guard into worker processes."""
    s = sample(Tier.ORACLE)
    out = s.restricted_to(Tier.METHOD)
    assert type(out) is dict
    assert out["images"] == 1  # no guard, no exception


def test_oracle_sample_still_exposes_everything():
    s = sample(Tier.ORACLE)
    assert s["stage"] == 4
