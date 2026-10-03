"""Tests for the camera-subset logic (SETUP.md section 18, dataset tests).

These run without a dataset or a GPU: the module under test is pure logic over camera
names, which is deliberate, because this is where a silent mistake would invalidate every
experiment in the study.
"""

from __future__ import annotations

import pytest

from p2c.data.camera_subset import (
    CONDITION_NAMES,
    CameraRoles,
    Slot,
    ViewCondition,
    build_condition,
)

# The real camera names discovered in RoboCasa365 target datasets.
ROBOCASA_CAMS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def hint(cams):
    wrist = [c for c in cams if "eye_in_hand" in c or "wrist" in c]
    third = [c for c in cams if c not in wrist]
    return {"wrist": wrist, "third_person": third}


@pytest.fixture
def roles():
    return CameraRoles.from_discovered(ROBOCASA_CAMS, hint(ROBOCASA_CAMS))


# ---------------------------------------------------------------- role resolution


def test_roles_assigned_from_discovered_names(roles):
    assert roles.wrist == "robot0_eye_in_hand"
    assert roles.primary == "robot0_agentview_left"       # first third-person, sorted
    assert roles.secondary == "robot0_agentview_right"
    assert set(roles.all_cameras) == set(ROBOCASA_CAMS)


def test_roles_are_deterministic_regardless_of_input_order():
    """Role assignment must not depend on how the dataset happened to list cameras."""
    a = CameraRoles.from_discovered(ROBOCASA_CAMS, hint(ROBOCASA_CAMS))
    b = CameraRoles.from_discovered(list(reversed(ROBOCASA_CAMS)), hint(ROBOCASA_CAMS))
    assert (a.primary, a.secondary, a.wrist) == (b.primary, b.secondary, b.wrist)


def test_primary_can_be_overridden():
    r = CameraRoles.from_discovered(
        ROBOCASA_CAMS, hint(ROBOCASA_CAMS), primary="robot0_agentview_right"
    )
    assert r.primary == "robot0_agentview_right"
    assert r.secondary == "robot0_agentview_left"


def test_unknown_primary_is_rejected():
    with pytest.raises(ValueError, match="not in dataset cameras"):
        CameraRoles.from_discovered(ROBOCASA_CAMS, hint(ROBOCASA_CAMS), primary="nope")


def test_missing_role_raises_rather_than_substituting():
    """A dataset without a wrist camera must fail loudly, not silently pick another view."""
    cams = ["cam_a", "cam_b"]
    r = CameraRoles.from_discovered(cams, hint(cams))
    assert r.wrist is None
    with pytest.raises(ValueError, match="no camera for role 'wrist'"):
        r.resolve("wrist")


def test_no_cameras_is_an_error():
    with pytest.raises(ValueError, match="no cameras"):
        CameraRoles.from_discovered([], {})


# ---------------------------------------------------------------- view counts


@pytest.mark.parametrize(
    "name,expected_views",
    [
        ("single_primary", 1),
        ("single_wrist", 1),
        ("single_secondary", 1),
        ("primary+wrist", 2),
        ("primary+secondary", 2),
        ("random_two", 2),
        ("primary+duplicate", 2),
        ("primary+wrist_dropout", 2),
        ("all_views", 3),
    ],
)
def test_condition_returns_correct_number_of_views(roles, name, expected_views):
    cond = build_condition(name, roles)
    assert cond.num_views == expected_views
    assert len(cond.select(0, 0, 0)) == expected_views


def test_every_registered_condition_builds(roles):
    for name in CONDITION_NAMES:
        cond = build_condition(name, roles)
        assert cond.num_views >= 1
        assert cond.describe()


def test_explicit_camera_list_is_accepted(roles):
    cond = build_condition("primary,wrist", roles)
    assert cond.select(0, 0, 0) == [roles.primary, roles.wrist]


def test_unknown_condition_is_rejected(roles):
    with pytest.raises(ValueError, match="unknown view condition"):
        build_condition("not_a_condition", roles)


# ---------------------------------------------------------------- determinism


def test_random_camera_sampling_is_deterministic_under_fixed_seed(roles):
    """SETUP.md section 7, requirement 6."""
    cond = build_condition("random_two", roles)
    first = [cond.select(ep, fr, seed=0) for ep in range(5) for fr in range(10)]
    second = [cond.select(ep, fr, seed=0) for ep in range(5) for fr in range(10)]
    assert first == second


def test_random_camera_sampling_changes_with_seed(roles):
    cond = build_condition("random_two", roles)
    a = [cond.select(ep, fr, seed=0) for ep in range(8) for fr in range(12)]
    b = [cond.select(ep, fr, seed=1) for ep in range(8) for fr in range(12)]
    assert a != b, "a different seed must produce a different camera draw"


def test_random_slot_never_duplicates_the_primary(roles):
    """B4 must add a *second* view, not re-use the primary."""
    cond = build_condition("random_two", roles)
    for ep in range(10):
        for fr in range(20):
            cams = cond.select(ep, fr, 0)
            assert cams[0] == roles.primary
            assert cams[1] != roles.primary


def test_random_slot_covers_its_whole_pool(roles):
    """A random condition must actually spread over candidates, not collapse to one."""
    cond = build_condition("random_two", roles)
    seen = {cond.select(ep, fr, 0)[1] for ep in range(20) for fr in range(30)}
    assert seen == {roles.secondary, roles.wrist}


def test_episode_scope_is_constant_within_an_episode(roles):
    cond = ViewCondition(
        "ep_scope", build_condition("random_two", roles).slots, random_scope="episode"
    )
    picks = {cond.select(7, fr, 0)[1] for fr in range(50)}
    assert len(picks) == 1, "episode scope must hold the camera fixed within an episode"


def test_run_scope_is_constant_everywhere(roles):
    cond = ViewCondition(
        "run_scope", build_condition("random_two", roles).slots, random_scope="run"
    )
    picks = {cond.select(ep, fr, 0)[1] for ep in range(10) for fr in range(10)}
    assert len(picks) == 1


def test_bad_random_scope_is_rejected(roles):
    cond = ViewCondition("bad", (Slot(choices=("a", "b")),), random_scope="weekly")
    with pytest.raises(ValueError, match="bad random_scope"):
        cond.select(0, 0, 0)


# ---------------------------------------------------------------- controls


def test_duplicate_control_feeds_the_same_camera_twice(roles):
    """Control D of SETUP.md section 12 must carry no new information."""
    cond = build_condition("primary+duplicate", roles)
    cams = cond.select(3, 4, 0)
    assert cams == [roles.primary, roles.primary]


def test_dropout_mask_keeps_at_least_one_view(roles):
    """A fully dropped observation would be meaningless, so one view always survives."""
    cond = build_condition("all_views", roles, camera_dropout=0.99)
    for ep in range(20):
        for fr in range(20):
            assert any(cond.dropout_mask(ep, fr, 0))


def test_dropout_is_off_by_default(roles):
    cond = build_condition("primary+wrist", roles)
    assert cond.camera_dropout == 0.0
    assert cond.dropout_mask(0, 0, 0) == [True, True]
    assert not cond.is_stochastic


def test_dropout_condition_actually_drops(roles):
    cond = build_condition("primary+wrist_dropout", roles)
    assert cond.camera_dropout > 0
    masks = [cond.dropout_mask(ep, fr, 0) for ep in range(20) for fr in range(20)]
    assert any(not all(m) for m in masks), "dropout must sometimes drop a view"


def test_dropout_mask_is_deterministic(roles):
    cond = build_condition("primary+wrist_dropout", roles)
    a = [cond.dropout_mask(ep, fr, 0) for ep in range(10) for fr in range(10)]
    b = [cond.dropout_mask(ep, fr, 0) for ep in range(10) for fr in range(10)]
    assert a == b


# ---------------------------------------------------------------- slot invariants


def test_slot_requires_exactly_one_of_fixed_or_choices():
    with pytest.raises(ValueError):
        Slot()
    with pytest.raises(ValueError):
        Slot(fixed="a", choices=("b", "c"))
    assert Slot(fixed="a").is_random is False
    assert Slot(choices=("a", "b")).is_random is True


def test_view_count_is_fixed_even_for_stochastic_conditions(roles):
    """The guard against random-view runs silently changing the sample shape."""
    for name in ("random_two", "primary+wrist_dropout"):
        cond = build_condition(name, roles)
        counts = {len(cond.select(ep, fr, 0)) for ep in range(10) for fr in range(10)}
        assert counts == {cond.num_views}
