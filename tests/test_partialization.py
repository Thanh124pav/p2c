"""Tests for synthetic partialization (PLAN.md sections 1, 6.1, Stage 2).

The invariants that matter: every strategy is deterministic in (seed, episode, frame) so
dataloader workers agree; every strategy records what it withheld, because that record is
the oracle target that makes "complementary evidence" measurable; and no strategy can
remove everything, which would leave nothing to learn from.
"""

from __future__ import annotations

import numpy as np
import pytest

from p2c.data.partialization import (
    PARTIALIZATION_AXES,
    ComposedPartial,
    OcclusionPartial,
    PhaseDropPartial,
    TemporalCropPartial,
    TrajectoryContext,
    ViewpointPartial,
    build_partialization,
)

CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def ctx(frame=10, length=100, with_stages=True, episode=3):
    stages = np.repeat([0, 1, 2], [30, 40, 30]) if with_stages else None
    return TrajectoryContext(
        episode=episode, frame=frame, episode_length=length, cameras=list(CAMERAS),
        resolution=96, stages=stages,
    )


# ---------------------------------------------------------------- viewpoint


def test_viewpoint_keeps_requested_count_and_records_the_rest():
    s = ViewpointPartial(n_keep=1).apply(ctx(), seed=0)
    assert len(s.kept_cameras) == 1
    assert len(s.withheld_cameras) == 2
    assert set(s.kept_cameras) | set(s.withheld_cameras) == set(CAMERAS)
    assert s.is_partial


def test_viewpoint_keeping_everything_is_not_partial():
    s = ViewpointPartial(n_keep=3).apply(ctx(), seed=0)
    assert s.withheld_cameras == []
    assert not s.is_partial


def test_viewpoint_explicit_cameras():
    s = ViewpointPartial(keep=[CAMERAS[2]]).apply(ctx(), seed=0)
    assert s.kept_cameras == [CAMERAS[2]]
    assert set(s.withheld_cameras) == set(CAMERAS[:2])


def test_viewpoint_rejects_unknown_camera():
    with pytest.raises(ValueError, match="not in trajectory"):
        ViewpointPartial(keep=["nope"]).apply(ctx(), seed=0)


def test_viewpoint_never_keeps_zero_cameras():
    s = ViewpointPartial(n_keep=0).apply(ctx(), seed=0)
    assert len(s.kept_cameras) >= 1


# ---------------------------------------------------------------- temporal


def test_temporal_crop_keeps_a_window_and_records_the_remainder():
    s = TemporalCropPartial(keep_fraction=0.4).apply(ctx(length=100), seed=0)
    start, stop = s.kept_frames
    assert stop - start == 40
    covered = sum(b - a for a, b in s.withheld_frames)
    assert covered == 60, "withheld spans must account for everything outside the window"
    assert s.is_partial


@pytest.mark.parametrize("anchor,expected_start", [("start", 0), ("end", 60)])
def test_temporal_anchors(anchor, expected_start):
    s = TemporalCropPartial(0.4, anchor=anchor).apply(ctx(length=100), seed=0)
    assert s.kept_frames[0] == expected_start


def test_temporal_anchor_around_frame_contains_that_frame():
    c = ctx(frame=70, length=100)
    s = TemporalCropPartial(0.2, anchor="around_frame").apply(c, seed=0)
    start, stop = s.kept_frames
    assert start <= c.frame < stop


def test_temporal_window_stays_inside_the_episode():
    for f in (0, 50, 99):
        s = TemporalCropPartial(0.3, anchor="around_frame").apply(ctx(frame=f), seed=0)
        start, stop = s.kept_frames
        assert 0 <= start < stop <= 100


def test_temporal_keep_everything_is_not_partial():
    s = TemporalCropPartial(1.0).apply(ctx(), seed=0)
    assert not s.is_partial
    assert s.withheld_frames == []


def test_temporal_rejects_bad_fraction():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="keep_fraction"):
            TemporalCropPartial(bad).apply(ctx(), seed=0)


def test_temporal_rejects_unknown_anchor():
    with pytest.raises(ValueError, match="unknown anchor"):
        TemporalCropPartial(0.5, anchor="sideways").apply(ctx(), seed=0)


# ---------------------------------------------------------------- phase


def test_phase_drop_withholds_a_stage_and_keeps_the_rest():
    s = PhaseDropPartial(n_drop=1).apply(ctx(), seed=0)
    assert len(s.withheld_stages) == 1
    assert set(s.kept_stages) | set(s.withheld_stages) == {0, 1, 2}
    assert s.is_partial


def test_phase_drop_never_drops_every_stage():
    s = PhaseDropPartial(n_drop=99).apply(ctx(), seed=0)
    assert len(s.kept_stages) >= 1, "dropping all phases would leave nothing to learn from"


def test_phase_drop_explicit_stages():
    s = PhaseDropPartial(drop_stages=[1]).apply(ctx(), seed=0)
    assert s.withheld_stages == [1]
    assert s.kept_stages == [0, 2]


def test_phase_drop_frame_mask_matches_the_spec():
    c = ctx()
    p = PhaseDropPartial(drop_stages=[1])
    s = p.apply(c, seed=0)
    mask = p.frame_mask(c, s)
    assert mask.sum() == 60          # stages 0 and 2 are 30 frames each
    assert not mask[30:70].any()     # stage 1 fully removed


def test_phase_drop_requires_stage_labels():
    """Atomic RoboCasa tasks have none, so this must fail loudly rather than no-op."""
    with pytest.raises(ValueError, match="stage labels"):
        PhaseDropPartial().apply(ctx(with_stages=False), seed=0)


# ---------------------------------------------------------------- occlusion


def test_occlusion_produces_in_bounds_boxes():
    s = OcclusionPartial(n_boxes=2, box_fraction=0.25).apply(ctx(), seed=0)
    assert len(s.occlusion_boxes) == 2
    for (y, x, h, w) in s.occlusion_boxes:
        assert 0 <= y and y + h <= 96
        assert 0 <= x and x + w <= 96
        assert h == w == 24
    assert s.is_partial


def test_occlusion_blanks_the_image_region():
    img = np.full((96, 96, 3), 255, np.uint8)
    s = OcclusionPartial(n_boxes=1, box_fraction=0.5).apply(ctx(), seed=0)
    out = OcclusionPartial.apply_to_image(img, s)
    y, x, h, w = s.occlusion_boxes[0]
    assert out[y : y + h, x : x + w].sum() == 0
    assert out.sum() < img.sum()


def test_occlusion_does_not_mutate_a_read_only_input():
    """Cache frames are read-only memmaps."""
    img = np.full((96, 96, 3), 255, np.uint8)
    img.setflags(write=False)
    s = OcclusionPartial().apply(ctx(), seed=0)
    out = OcclusionPartial.apply_to_image(img, s)
    assert img.sum() == 96 * 96 * 3 * 255   # untouched
    assert out.sum() < img.sum()


def test_occlusion_per_episode_is_stable_across_frames():
    p = OcclusionPartial(per_frame=False)
    boxes = {tuple(p.apply(ctx(frame=f), 0).occlusion_boxes[0]) for f in range(10)}
    assert len(boxes) == 1


def test_occlusion_per_frame_varies():
    p = OcclusionPartial(per_frame=True)
    boxes = {tuple(p.apply(ctx(frame=f), 0).occlusion_boxes[0]) for f in range(20)}
    assert len(boxes) > 1


def test_occlusion_rejects_bad_fraction():
    for bad in (0.0, 1.0, 2.0):
        with pytest.raises(ValueError, match="box_fraction"):
            OcclusionPartial(box_fraction=bad).apply(ctx(), seed=0)


# ---------------------------------------------------------------- determinism


@pytest.mark.parametrize("strategy", [
    ViewpointPartial(n_keep=1),
    TemporalCropPartial(0.5),
    PhaseDropPartial(n_drop=1),
    OcclusionPartial(per_frame=True),
])
def test_every_strategy_is_deterministic_under_a_fixed_seed(strategy):
    a = [strategy.apply(ctx(frame=f), seed=0).as_dict() for f in range(8)]
    b = [strategy.apply(ctx(frame=f), seed=0).as_dict() for f in range(8)]
    assert a == b


@pytest.mark.parametrize("strategy_cls,kwargs", [
    (ViewpointPartial, {"n_keep": 1}),
    (TemporalCropPartial, {"keep_fraction": 0.5}),
    (OcclusionPartial, {"per_frame": True}),
])
def test_strategies_vary_with_seed(strategy_cls, kwargs):
    """Vary the episode, not just the frame.

    Viewpoint and temporal draws are keyed on the episode, so sweeping frames within one
    episode is a single draw — and with only three cameras two seeds collide one time in
    three. Sweeping episodes samples the actual randomness.
    """
    s = strategy_cls(**kwargs)
    a = [s.apply(ctx(episode=e, frame=e), seed=0).as_dict() for e in range(16)]
    b = [s.apply(ctx(episode=e, frame=e), seed=1).as_dict() for e in range(16)]
    assert a != b


# ---------------------------------------------------------------- composition


def test_composed_applies_every_axis():
    c = ComposedPartial([
        ViewpointPartial(n_keep=1), TemporalCropPartial(0.5), OcclusionPartial(),
    ])
    s = c.apply(ctx(), seed=0)
    assert s.withheld_cameras          # viewpoint fired
    assert s.kept_frames is not None   # temporal fired
    assert s.occlusion_boxes           # occlusion fired
    assert s.axis == "viewpoint+temporal+occlusion"
    assert s.is_partial


def test_composed_is_deterministic():
    c = ComposedPartial([ViewpointPartial(n_keep=1), OcclusionPartial(per_frame=True)])
    assert c.apply(ctx(), 0).as_dict() == c.apply(ctx(), 0).as_dict()


# ---------------------------------------------------------------- registry


def test_every_named_axis_builds():
    for axis in PARTIALIZATION_AXES:
        if axis == "composed":
            p = build_partialization(axis, parts=[ViewpointPartial()])
        else:
            p = build_partialization(axis)
        assert p.axis


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="unknown partialization"):
        build_partialization("telepathy")


def test_composed_without_parts_is_rejected():
    with pytest.raises(ValueError, match="needs parts"):
        build_partialization("composed")


def test_spec_round_trips_through_a_dict():
    s = ComposedPartial([ViewpointPartial(n_keep=1), TemporalCropPartial(0.5)]).apply(
        ctx(), seed=0
    )
    d = s.as_dict()
    assert d["is_partial"] is True
    assert isinstance(d["withheld_cameras"], list)
    assert isinstance(d["kept_frames"], list)
