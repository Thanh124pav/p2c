"""Tests for LeRobot schema discovery.

Written against a synthetic ``meta/`` directory mirroring what RoboCasa365 actually ships,
so they run without the downloaded datasets.

The stage-key tests encode two bugs found while building this: a filter that passed
episode-level language off as per-frame stage labels, and a suffix test that then excluded
the genuine ``subtask_name`` key because it happens to end with ``task_name``.
"""

from __future__ import annotations

import json

import pytest

from p2c.data.lerobot_meta import DatasetMeta


def write_dataset(root, features: dict, tasks: list[dict] | None = None, episodes=None):
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "PandaOmron",
        "fps": 20,
        "total_episodes": len(episodes or []),
        "total_frames": sum(e["length"] for e in (episodes or [])),
        "chunks_size": 1000,
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info))
    if tasks is not None:
        (meta / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
    if episodes is not None:
        (meta / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
    return root


IMG = {"shape": [256, 256, 3], "dtype": "video"}


def base_features(extra: dict | None = None) -> dict:
    f = {
        "observation.images.robot0_agentview_left": dict(IMG),
        "observation.images.robot0_agentview_right": dict(IMG),
        "observation.images.robot0_eye_in_hand": dict(IMG),
        "observation.state": {"shape": [16]},
        "action": {"shape": [12]},
        "task_index": {"shape": [1]},
        "annotation.human.task_description": {"shape": [1]},
        "annotation.human.task_name": {"shape": [1]},
    }
    f.update(extra or {})
    return f


@pytest.fixture
def atomic(tmp_path):
    return write_dataset(
        tmp_path / "atomic", base_features(),
        tasks=[{"task_index": 0, "task": "Navigate to the sink."}],
        episodes=[{"episode_index": 0, "length": 177}],
    )


@pytest.fixture
def composite(tmp_path):
    extra = {
        "annotation.human.subtask": {"shape": [1]},
        "annotation.human.subtask_name": {"shape": [1]},
        "annotation.human.subtask_stage": {"shape": [1]},
        "subtask_idx": {"shape": [1]},
    }
    return write_dataset(
        tmp_path / "composite", base_features(extra),
        tasks=[
            {"task_index": 0, "task": "Stack the bowls."},
            {"task_index": 11, "task": "done"},
            {"task_index": 12, "task": "place"},
            {"task_index": 13, "task": "pick"},
            {"task_index": 15, "task": "navigate"},
        ],
        episodes=[{"episode_index": 0, "length": 291}],
    )


# ---------------------------------------------------------------- basics


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="meta/info.json missing"):
        DatasetMeta.load(tmp_path / "nope")


def test_discovers_cameras_without_hard_coding(atomic):
    m = DatasetMeta.load(atomic)
    assert m.camera_names == [
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand",
    ]
    roles = m.classify_cameras()
    assert roles["wrist"] == ["robot0_eye_in_hand"]
    assert len(roles["third_person"]) == 2


def test_action_and_state_shapes(atomic):
    m = DatasetMeta.load(atomic)
    assert m.action_key == "action"
    assert m.action_shape == (12,)
    assert m.state_keys == ["observation.state"]
    assert m.feature_shape("observation.state") == (16,)


def test_image_shape_normalisation(atomic):
    m = DatasetMeta.load(atomic)
    assert m.image_hwc("observation.images.robot0_eye_in_hand") == (256, 256, 3)


def test_image_shape_normalisation_handles_chw(tmp_path):
    root = write_dataset(
        tmp_path / "chw",
        {"observation.images.cam": {"shape": [3, 128, 128]}, "action": {"shape": [7]}},
        tasks=[], episodes=[],
    )
    m = DatasetMeta.load(root)
    assert m.image_hwc("observation.images.cam") == (128, 128, 3)


def test_episode_lengths_and_counts(atomic):
    m = DatasetMeta.load(atomic)
    assert m.num_episodes == 1
    assert m.episode_lengths() == [177]
    assert m.fps == 20


# ---------------------------------------------------------------- stage annotations


def test_atomic_task_reports_no_stage_annotations(atomic):
    """Episode-level language must not be mistaken for per-frame stage labels."""
    m = DatasetMeta.load(atomic)
    assert m.annotation_keys  # it does have annotation.* keys
    assert m.stage_annotation_keys == []
    assert m.has_stage_annotations is False


def test_composite_task_reports_stage_annotations(composite):
    m = DatasetMeta.load(composite)
    assert m.has_stage_annotations
    assert set(m.stage_annotation_keys) == {
        "annotation.human.subtask",
        "annotation.human.subtask_name",
        "annotation.human.subtask_stage",
        "subtask_idx",
    }


def test_subtask_name_is_not_excluded_by_the_task_name_filter(composite):
    """Regression: 'subtask_name'.endswith('task_name') is True, so a naive suffix
    test dropped a genuine per-frame key."""
    m = DatasetMeta.load(composite)
    assert "annotation.human.subtask_name" in m.stage_annotation_keys
    assert "annotation.human.task_name" not in m.stage_annotation_keys
    assert "task_index" not in m.stage_annotation_keys


# ---------------------------------------------------------------- vocabulary


def test_task_vocabulary_resolves_stage_indices(composite):
    """The stage column holds ints indexing meta/tasks.jsonl, not readable labels."""
    m = DatasetMeta.load(composite)
    vocab = m.task_vocabulary()
    assert vocab[11] == "done"
    assert vocab[12] == "place"
    assert vocab[13] == "pick"
    assert vocab[15] == "navigate"


def test_task_vocabulary_is_empty_without_tasks_file(tmp_path):
    root = write_dataset(tmp_path / "bare", base_features(), tasks=None, episodes=[])
    assert DatasetMeta.load(root).task_vocabulary() == {}


# ---------------------------------------------------------------- modality groups


def test_modality_groups_are_read_not_guessed(tmp_path, atomic):
    modality = {
        "action": {
            "base_motion": {"start": 0, "end": 4},
            "control_mode": {"start": 4, "end": 5},
            "end_effector_position": {"start": 5, "end": 8},
            "end_effector_rotation": {"start": 8, "end": 11},
            "gripper_close": {"start": 11, "end": 12},
        },
        "state": {"gripper_qpos": {"start": 14, "end": 16}},
    }
    (atomic / "meta" / "modality.json").write_text(json.dumps(modality))
    m = DatasetMeta.load(atomic)
    groups = m.action_groups
    assert groups["end_effector_rotation"] == (8, 11)
    assert groups["gripper_close"] == (11, 12)
    # Ordered by start offset, which keeps printed tables readable.
    assert list(groups) == [
        "base_motion", "control_mode", "end_effector_position",
        "end_effector_rotation", "gripper_close",
    ]
    assert m.state_groups["gripper_qpos"] == (14, 16)


def test_modality_groups_absent_is_not_an_error(atomic):
    assert DatasetMeta.load(atomic).action_groups == {}
