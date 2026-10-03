"""Shared fixtures: a tiny synthetic frame cache.

Building a synthetic cache rather than depending on the downloaded RoboCasa365 data keeps
the test suite runnable on any machine and fast enough to run on every change. The layout
matches exactly what ``scripts/build_frame_cache.py`` writes, so the tests exercise the
real reader.

Pixels encode their own identity: frame ``i`` of camera ``c`` is filled with a value
derived from both, so a test can assert that the right camera's pixels reached the right
observation slot.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
ACTION_DIM = 12
STATE_DIM = 16
RES = 16  # tiny: these tests check plumbing, not learning

ACTION_GROUPS = {
    "base_motion": [0, 4],
    "control_mode": [4, 5],
    "end_effector_position": [5, 8],
    "end_effector_rotation": [8, 11],
    "gripper_close": [11, 12],
}


def camera_fill(cam_idx: int, frame: int) -> int:
    """A per-(camera, frame) pixel value, so provenance is checkable."""
    return (cam_idx * 61 + frame * 7) % 256


def build_synthetic_cache(
    path,
    num_episodes: int = 10,
    ep_len: int = 12,
    with_stages: bool = False,
    cameras: list[str] | None = None,
):
    cameras = cameras or CAMERAS
    path.mkdir(parents=True, exist_ok=True)
    total = num_episodes * ep_len

    episodes = []
    ep_idx = np.zeros(total, np.int32)
    fr_idx = np.zeros(total, np.int32)
    actions = np.zeros((total, ACTION_DIM), np.float32)
    states = np.zeros((total, STATE_DIM), np.float32)
    stages = np.zeros(total, np.int32)
    images = {c: np.zeros((total, RES, RES, 3), np.uint8) for c in cameras}

    rng = np.random.default_rng(0)
    cursor = 0
    for ep in range(num_episodes):
        episodes.append({"episode_index": ep, "start": cursor, "length": ep_len})
        for t in range(ep_len):
            g = cursor + t
            ep_idx[g] = ep
            fr_idx[g] = t
            # Actions vary per frame so a model has something to fit and so the
            # "action target unchanged across view subsets" test is meaningful.
            actions[g] = rng.normal(size=ACTION_DIM).astype(np.float32)
            states[g] = rng.normal(size=STATE_DIM).astype(np.float32)
            if with_stages:
                stages[g] = min(2, (t * 3) // ep_len)  # 3 stages across the episode
            for ci, c in enumerate(cameras):
                images[c][g, :, :, :] = camera_fill(ci, t)
        cursor += ep_len

    for c in cameras:
        np.save(path / f"images_{c}.npy", images[c])
    np.save(path / "actions.npy", actions)
    np.save(path / "states.npy", states)
    np.save(path / "episode_index.npy", ep_idx)
    np.save(path / "frame_index.npy", fr_idx)

    index = {
        "schema_version": 2,
        "source_dataset": "synthetic",
        "task": "SyntheticTask",
        "cameras": cameras,
        "image_keys": [f"observation.images.{c}" for c in cameras],
        "resolution": RES,
        "num_frames": total,
        "num_episodes": num_episodes,
        "action_key": "action",
        "action_dim": ACTION_DIM,
        "action_groups": ACTION_GROUPS,
        "state_keys": ["observation.state"],
        "state_dim": STATE_DIM,
        "state_groups": {},
        "fps": 20,
        "episodes": episodes,
        "stage_key": "annotation.stage" if with_stages else None,
        "stage_names": {"0": "navigate", "1": "pick", "2": "place"} if with_stages else None,
    }
    if with_stages:
        np.save(path / "stage.npy", stages)
        with open(path / "stage_names.json", "w") as f:
            json.dump(index["stage_names"], f)

    with open(path / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    return path


@pytest.fixture
def synthetic_cache_path(tmp_path):
    return build_synthetic_cache(tmp_path / "cache")


@pytest.fixture
def synthetic_cache_with_stages_path(tmp_path):
    return build_synthetic_cache(tmp_path / "cache_stages", with_stages=True)


@pytest.fixture
def synthetic_cache(synthetic_cache_path):
    from p2c.data.frame_cache import FrameCache

    return FrameCache(synthetic_cache_path)


@pytest.fixture
def synthetic_cache_with_stages(synthetic_cache_with_stages_path):
    from p2c.data.frame_cache import FrameCache

    return FrameCache(synthetic_cache_with_stages_path)
