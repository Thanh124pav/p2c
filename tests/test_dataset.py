"""Dataset tests from SETUP.md section 18.

The listed cases are: image keys exist, camera subset returns the correct number of views,
timestamps/indices remain aligned, the action target is unchanged across view subsets, and
random-camera sampling is deterministic under a fixed seed.

The action-target test and the shared-split test are the load-bearing ones. If either
fails, the conditions are not comparable and every number in the study is meaningless —
which is exactly the failure mode SETUP.md section 7 is written to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from p2c.data.frame_cache import FrameCache
from p2c.data.robocasa_dataset import CameraSubsetDataset, build_train_val

from .conftest import CAMERAS, RES, camera_fill

CONDITIONS = [
    "single_primary", "single_wrist", "primary+wrist", "primary+secondary",
    "all_views", "random_two", "primary+duplicate",
]


# ---------------------------------------------------------------- cache integrity


def test_cache_loads_and_reports_schema(synthetic_cache):
    c = synthetic_cache
    assert c.cameras == CAMERAS
    assert c.resolution == RES
    assert c.num_frames == 120
    assert c.action_dim == 12
    assert c.state_dim == 16
    assert c.summary()


def test_image_keys_exist_for_every_camera(synthetic_cache):
    for cam in synthetic_cache.cameras:
        img = synthetic_cache.image(cam, 0)
        assert img.shape == (RES, RES, 3)
        assert img.dtype == np.uint8


def test_unknown_camera_raises_a_clear_error(synthetic_cache):
    with pytest.raises(KeyError, match="not in cache"):
        synthetic_cache.image("no_such_camera", 0)


def test_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no frame cache"):
        FrameCache(tmp_path / "nope")


def test_corrupt_cache_is_detected(synthetic_cache_path):
    """A row-count mismatch must fail at load, not silently mis-align views."""
    np.save(synthetic_cache_path / "actions.npy", np.zeros((7, 12), np.float32))
    with pytest.raises(ValueError, match="actions has 7 rows"):
        FrameCache(synthetic_cache_path)


def test_action_groups_are_read_from_the_index(synthetic_cache):
    g = synthetic_cache.action_groups
    assert g["end_effector_rotation"] == (8, 11)
    assert g["gripper_close"] == (11, 12)


# ---------------------------------------------------------------- splits


def test_split_is_disjoint_and_covers_everything(synthetic_cache):
    train, val = synthetic_cache.split_episodes(val_fraction=0.2, split_seed=0)
    assert set(train) & set(val) == set()
    assert sorted(train + val) == sorted(s.episode_index for s in synthetic_cache.spans)
    assert len(val) == 2  # 10 episodes, 20%


def test_split_is_deterministic(synthetic_cache):
    a = synthetic_cache.split_episodes(0.2, 0)
    b = synthetic_cache.split_episodes(0.2, 0)
    assert a == b


def test_split_changes_with_split_seed(synthetic_cache):
    a = synthetic_cache.split_episodes(0.2, 0)
    b = synthetic_cache.split_episodes(0.2, 7)
    assert a != b


def test_split_rejects_degenerate_fractions(synthetic_cache):
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            synthetic_cache.split_episodes(bad, 0)


def test_split_is_identical_across_view_conditions(synthetic_cache):
    """SETUP.md section 7, requirement 2: the same split for every ablation arm."""
    splits = {}
    for cond in CONDITIONS:
        tr, va = build_train_val(synthetic_cache, cond, val_fraction=0.2, split_seed=0)
        splits[cond] = (tuple(tr.episodes), tuple(va.episodes))
    assert len(set(splits.values())) == 1, f"splits differ across conditions: {splits}"


# ---------------------------------------------------------------- view counts and shapes


@pytest.mark.parametrize(
    "cond,views",
    [("single_primary", 1), ("single_wrist", 1), ("primary+wrist", 2),
     ("random_two", 2), ("primary+duplicate", 2), ("all_views", 3)],
)
def test_camera_subset_returns_correct_number_of_views(synthetic_cache, cond, views):
    tr, _ = build_train_val(synthetic_cache, cond)
    s = tr[0]
    assert s["images"].shape == (views, 3, RES, RES)
    assert tr.num_views == views
    assert s["camera_ids"].shape == (views,)


def test_sample_count_is_identical_across_conditions(synthetic_cache):
    """A random-view run must not silently change the number of samples (section 7)."""
    counts = {}
    for cond in CONDITIONS:
        tr, va = build_train_val(synthetic_cache, cond)
        counts[cond] = (len(tr), len(va))
    assert len(set(counts.values())) == 1, f"sample counts differ: {counts}"


def test_history_adds_a_time_axis(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "primary+wrist", n_obs_steps=3)
    assert tr[0]["images"].shape == (2, 3, 3, RES, RES)


def test_action_horizon_shapes(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "single_primary", action_horizon=4)
    assert tr[0]["action"].shape == (4, 12)


def test_empty_episode_list_is_rejected(synthetic_cache):
    with pytest.raises(ValueError, match="empty episode list"):
        CameraSubsetDataset(synthetic_cache, "single_primary", [])


# ---------------------------------------------------------------- the critical invariant


def test_action_target_is_unchanged_across_view_subsets(synthetic_cache):
    """SETUP.md section 18: the action target must not depend on the camera subset.

    Without this, a difference between conditions could come from different supervision
    rather than different observations.
    """
    ref = None
    for cond in CONDITIONS:
        tr, _ = build_train_val(synthetic_cache, cond)
        acts = np.stack([tr[i]["action"].numpy() for i in range(len(tr))])
        if ref is None:
            ref = acts
        else:
            np.testing.assert_allclose(
                acts, ref, rtol=0, atol=0,
                err_msg=f"condition '{cond}' changed the action targets",
            )


def test_state_target_is_unchanged_across_view_subsets(synthetic_cache):
    ref = None
    for cond in CONDITIONS:
        tr, _ = build_train_val(synthetic_cache, cond)
        st = np.stack([tr[i]["state"].numpy() for i in range(len(tr))])
        if ref is None:
            ref = st
        else:
            np.testing.assert_allclose(st, ref, rtol=0, atol=0)


def test_sample_order_is_unchanged_across_conditions(synthetic_cache):
    """Same (episode, frame) at every index, so per-sample errors can be joined later."""
    ref = None
    for cond in CONDITIONS:
        tr, _ = build_train_val(synthetic_cache, cond)
        keys = [(int(tr[i]["episode"]), int(tr[i]["frame"])) for i in range(len(tr))]
        if ref is None:
            ref = keys
        else:
            assert keys == ref, f"condition '{cond}' reordered the samples"


# ---------------------------------------------------------------- alignment


def test_indices_remain_aligned_with_the_cache(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "single_primary")
    for i in range(0, len(tr), 7):
        s = tr[i]
        ep, fr = int(s["episode"]), int(s["frame"])
        span = synthetic_cache.span_of_episode(ep)
        g = span.start + fr
        assert int(synthetic_cache.episode_index[g]) == ep
        assert int(synthetic_cache.frame_index[g]) == fr


def test_correct_camera_pixels_land_in_each_slot(synthetic_cache):
    """Verifies the view axis is not permuted: slot k must hold camera k's pixels."""
    tr, _ = build_train_val(synthetic_cache, "all_views")
    cam_order = tr.condition.select(tr.episodes[0], 0, tr.seed)
    s = tr[0]
    frame_in_ep = int(s["frame"])
    for slot, cam in enumerate(cam_order):
        expected = camera_fill(synthetic_cache.cameras.index(cam), frame_in_ep) / 255.0
        got = float(s["images"][slot].mean())
        assert abs(got - expected) < 1e-3, (
            f"slot {slot} should hold {cam} (expected {expected:.4f}, got {got:.4f})"
        )


def test_duplicate_condition_yields_two_identical_views(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "primary+duplicate")
    s = tr[3]
    assert np.allclose(s["images"][0].numpy(), s["images"][1].numpy())


def test_history_never_crosses_an_episode_boundary(synthetic_cache):
    """At frame 0 the history must repeat frame 0, not borrow the previous episode."""
    tr, _ = build_train_val(synthetic_cache, "single_primary", n_obs_steps=3)
    first = next(i for i in range(len(tr)) if int(tr[i]["frame"]) == 0)
    imgs = tr[first]["images"][0]  # [T, 3, H, W]
    assert np.allclose(imgs[0].numpy(), imgs[-1].numpy()), (
        "history at frame 0 should be clamped to frame 0"
    )


def test_action_horizon_clamps_at_the_episode_end(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "single_primary", action_horizon=5)
    ep = tr.episodes[0]
    span = synthetic_cache.span_of_episode(ep)
    last = next(
        i for i in range(len(tr))
        if int(tr[i]["episode"]) == ep and int(tr[i]["frame"]) == span.length - 1
    )
    act = tr[last]["action"].numpy()
    for k in range(1, 5):
        np.testing.assert_allclose(act[k], act[0], rtol=0, atol=0)


# ---------------------------------------------------------------- determinism


def test_deterministic_random_camera_sampling_under_fixed_seed(synthetic_cache):
    """SETUP.md section 18."""
    a, _ = build_train_val(synthetic_cache, "random_two", seed=0)
    b, _ = build_train_val(synthetic_cache, "random_two", seed=0)
    for i in range(0, len(a), 5):
        assert a[i]["camera_ids"].tolist() == b[i]["camera_ids"].tolist()


def test_random_camera_sampling_differs_across_seeds(synthetic_cache):
    a, _ = build_train_val(synthetic_cache, "random_two", seed=0)
    b, _ = build_train_val(synthetic_cache, "random_two", seed=1)
    ids_a = [a[i]["camera_ids"].tolist() for i in range(len(a))]
    ids_b = [b[i]["camera_ids"].tolist() for i in range(len(b))]
    assert ids_a != ids_b


def test_repeated_access_to_the_same_index_is_stable(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "random_two", seed=0)
    first = tr[11]["camera_ids"].tolist()
    for _ in range(5):
        assert tr[11]["camera_ids"].tolist() == first


def test_random_condition_actually_uses_both_candidates(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "random_two", seed=0)
    usage = tr.camera_usage()
    non_primary = {c: n for c, n in usage.items() if c != tr.roles.primary}
    assert all(n > 0 for n in non_primary.values()), f"pool not covered: {usage}"


# ---------------------------------------------------------------- normalisation


def test_action_stats_come_from_training_episodes_only(synthetic_cache):
    train, val = synthetic_cache.split_episodes(0.2, 0)
    s_train = synthetic_cache.action_stats(train)
    s_all = synthetic_cache.action_stats(train + val)
    assert not np.allclose(s_train["mean"], s_all["mean"]), (
        "statistics computed over train+val would leak validation data"
    )


def test_normalised_actions_are_roughly_standardised(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "single_primary")
    acts = np.stack([tr[i]["action"].numpy() for i in range(len(tr))])
    assert abs(float(acts.mean())) < 0.25
    assert 0.5 < float(acts.std()) < 2.0


def test_constant_action_dimensions_do_not_blow_up(synthetic_cache_path):
    """A near-constant dimension (e.g. control_mode) must not divide by ~0."""
    acts = np.load(synthetic_cache_path / "actions.npy")
    acts[:, 4] = 1.0  # make control_mode constant
    np.save(synthetic_cache_path / "actions.npy", acts)
    cache = FrameCache(synthetic_cache_path)
    train, _ = cache.split_episodes(0.2, 0)
    stats = cache.action_stats(train)
    assert stats["std"][4] == 1.0
    tr, _ = build_train_val(cache, "single_primary")
    assert np.isfinite(tr[0]["action"].numpy()).all()


def test_images_are_scaled_to_unit_range(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "all_views")
    imgs = tr[0]["images"]
    assert imgs.dtype.is_floating_point
    assert 0.0 <= float(imgs.min()) and float(imgs.max()) <= 1.0


# ---------------------------------------------------------------- dropout and stages


def test_dropped_views_are_zeroed_but_keep_their_slot(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "primary+wrist_dropout", seed=0)
    found = False
    for i in range(len(tr)):
        s = tr[i]
        assert s["images"].shape[0] == 2  # shape never changes
        if not bool(s["view_kept"].all()):
            dropped = (~s["view_kept"]).nonzero()[0].item()
            assert float(s["images"][dropped].abs().sum()) == 0.0
            found = True
            break
    assert found, "no dropped view encountered; dropout is not firing"


def test_stage_labels_are_exposed_when_available(synthetic_cache_with_stages):
    tr, _ = build_train_val(synthetic_cache_with_stages, "single_primary")
    stages = {int(tr[i]["stage"]) for i in range(len(tr))}
    assert stages == {0, 1, 2}
    assert synthetic_cache_with_stages.stage_name(1) == "pick"


def test_stage_is_minus_one_without_annotations(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "single_primary")
    assert int(tr[0]["stage"]) == -1
    assert synthetic_cache.has_stages is False


# ---------------------------------------------------------------- logging metadata


def test_describe_records_what_section_16_requires(synthetic_cache):
    tr, _ = build_train_val(synthetic_cache, "random_two", seed=3)
    d = tr.describe()
    assert d["condition"] == "random_two"
    assert d["num_views"] == 2
    assert d["seed"] == 3
    assert d["is_stochastic"] is True
    assert d["roles"]["primary"] in CAMERAS
    assert d["num_samples"] == len(tr)
    assert len(d["slots"]) == 2
