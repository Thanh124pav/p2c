"""Schema discovery for RoboCasa365 LeRobot datasets.

PLAN.md Stage 2 requires that dataset properties be verified programmatically rather
than assumed, and camera names are never hard-coded. Everything in this module
is therefore read off disk: nothing about camera names, resolutions or action dimensions
is baked in.

A RoboCasa365 LeRobot dataset on disk looks like::

    <root>/
      meta/info.json              features, fps, episode/frame counts
      meta/tasks.jsonl            language instructions
      meta/episodes.jsonl         per-episode index, instruction, length
      meta/modality.json          how the flat state/action vectors are split
      data/chunk-*/episode_*.parquet
      videos/chunk-*/observation.images.<camera>/episode_*.mp4
      extras/                     MuJoCo replay state (not for training)

This module deliberately depends only on the standard library plus pyarrow, so Stage 0
can run before robosuite/robocasa finish installing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_KEY_PREFIX = "observation.images."


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class DatasetMeta:
    """Discovered schema of one LeRobot dataset."""

    root: Path
    info: dict
    modality: dict | None
    tasks: list[dict]
    episodes: list[dict]

    # discovered, never assumed
    image_keys: list[str] = field(default_factory=list)
    camera_names: list[str] = field(default_factory=list)
    annotation_keys: list[str] = field(default_factory=list)

    # ---------------- construction ----------------

    @classmethod
    def load(cls, root: str | Path) -> "DatasetMeta":
        root = Path(root)
        meta_dir = root / "meta"
        if not (meta_dir / "info.json").exists():
            raise FileNotFoundError(
                f"{root} does not look like a LeRobot dataset: meta/info.json missing"
            )

        info = _read_json(meta_dir / "info.json")
        modality = (
            _read_json(meta_dir / "modality.json")
            if (meta_dir / "modality.json").exists()
            else None
        )
        tasks = (
            _read_jsonl(meta_dir / "tasks.jsonl")
            if (meta_dir / "tasks.jsonl").exists()
            else []
        )
        episodes = (
            _read_jsonl(meta_dir / "episodes.jsonl")
            if (meta_dir / "episodes.jsonl").exists()
            else []
        )

        self = cls(root=root, info=info, modality=modality, tasks=tasks, episodes=episodes)
        self.image_keys = self._discover_image_keys()
        self.camera_names = [k[len(IMAGE_KEY_PREFIX) :] for k in self.image_keys]
        self.annotation_keys = self._discover_annotation_keys()
        return self

    def _discover_image_keys(self) -> list[str]:
        """Image keys from info.json features, cross-checked against videos/ on disk."""
        feats = self.info.get("features", {})
        declared = sorted(k for k in feats if k.startswith(IMAGE_KEY_PREFIX))

        # Cross-check: a declared key with no video directory is unusable.
        present = []
        for key in declared:
            if not self.video_dirs(key):
                continue
            present.append(key)

        if not present and declared:
            # videos/ may use a different layout; fall back to the declaration so the
            # caller can see the mismatch rather than silently getting an empty list.
            return declared
        return present

    def _discover_annotation_keys(self) -> list[str]:
        """All annotation-ish keys, including episode-level language ones.

        Note this is deliberately broad and includes things like
        ``annotation.human.task_description`` and ``task_index``, which every dataset has.
        For the stage analysis of PLAN.md section 9, use
        :attr:`stage_annotation_keys` instead — a non-empty value here does *not* mean
        the dataset carries per-frame stage labels.
        """
        feats = self.info.get("features", {})
        keys = [
            k
            for k in feats
            if k.startswith("annotation.")
            or any(
                tok in k.lower()
                for tok in ("subtask", "stage", "skill", "instruction", "task_index")
            )
        ]
        return sorted(set(keys))

    @property
    def stage_annotation_keys(self) -> list[str]:
        """Keys that carry genuine per-frame subtask/stage/skill labels.

        RoboCasa365 ships these only for *target composite* task datasets (README update
        7/7/2026: subtask index, atomic-skill name, stage, instruction). Atomic datasets
        have episode-level language only, so they cannot support the stage-dependence
        analysis of PLAN.md section 9.

        Episode-level language keys (``task_description``, ``task_name``, ``task_index``)
        are explicitly excluded: they describe the whole episode, not the timestep.
        """
        feats = self.info.get("features", {})
        # Compare the final dotted component exactly. A substring test would wrongly
        # exclude 'annotation.human.subtask_name', since "subtask_name" happens to end
        # with "task_name" — and that key *is* a genuine per-frame label.
        episode_level = {"task_index", "task_description", "task_name", "validity"}
        out = []
        for k in feats:
            leaf = k.lower().rsplit(".", 1)[-1]
            if leaf in episode_level:
                continue
            if any(tok in leaf for tok in ("subtask", "stage", "skill")):
                out.append(k)
        return sorted(out)

    @property
    def has_stage_annotations(self) -> bool:
        return bool(self.stage_annotation_keys)

    # ---------------- derived properties ----------------

    @property
    def fps(self) -> float | None:
        return self.info.get("fps")

    @property
    def num_episodes(self) -> int:
        n = self.info.get("total_episodes")
        return int(n) if n is not None else len(self.episodes)

    @property
    def num_frames(self) -> int | None:
        n = self.info.get("total_frames")
        return int(n) if n is not None else None

    @property
    def chunks_size(self) -> int:
        return int(self.info.get("chunks_size", 1000))

    def feature_shape(self, key: str) -> tuple[int, ...] | None:
        feat = self.info.get("features", {}).get(key)
        if feat is None:
            return None
        shape = feat.get("shape")
        return tuple(shape) if shape is not None else None

    @property
    def action_key(self) -> str | None:
        for cand in ("action", "actions"):
            if cand in self.info.get("features", {}):
                return cand
        return None

    @property
    def action_shape(self) -> tuple[int, ...] | None:
        key = self.action_key
        return self.feature_shape(key) if key else None

    @property
    def state_keys(self) -> list[str]:
        """Proprioception keys: observation.* that are not images."""
        feats = self.info.get("features", {})
        return sorted(
            k
            for k in feats
            if k.startswith("observation.") and not k.startswith(IMAGE_KEY_PREFIX)
        )

    def image_resolution(self, key: str) -> tuple[int, ...] | None:
        """Resolution as declared in info.json for one image key.

        Note the declared shape may be (H, W, C) or (C, H, W) depending on writer
        version; callers should use :meth:`image_hwc` which normalises it.
        """
        return self.feature_shape(key)

    def image_hwc(self, key: str) -> tuple[int, int, int] | None:
        """Normalise a declared image shape to (H, W, C)."""
        shape = self.feature_shape(key)
        if shape is None or len(shape) != 3:
            return None
        a, b, c = shape
        if c in (1, 3, 4):  # already (H, W, C)
            return (a, b, c)
        if a in (1, 3, 4):  # (C, H, W)
            return (b, c, a)
        return (a, b, c)

    # ---------------- on-disk paths ----------------

    def chunk_of(self, ep_idx: int) -> int:
        return ep_idx // self.chunks_size

    def parquet_path(self, ep_idx: int) -> Path:
        """Episode parquet path, honouring info.json's template when present."""
        tmpl = self.info.get("data_path")
        if tmpl:
            return self.root / tmpl.format(
                episode_chunk=self.chunk_of(ep_idx),
                episode_index=ep_idx,
                chunk_index=self.chunk_of(ep_idx),
            )
        return (
            self.root
            / f"data/chunk-{self.chunk_of(ep_idx):03d}/episode_{ep_idx:06d}.parquet"
        )

    def video_dirs(self, key: str) -> list[Path]:
        base = self.root / "videos"
        if not base.exists():
            return []
        return sorted(p for p in base.glob(f"chunk-*/{key}") if p.is_dir())

    def video_path(self, ep_idx: int, key: str) -> Path:
        tmpl = self.info.get("video_path")
        if tmpl:
            return self.root / tmpl.format(
                episode_chunk=self.chunk_of(ep_idx),
                chunk_index=self.chunk_of(ep_idx),
                video_key=key,
                episode_index=ep_idx,
            )
        return (
            self.root
            / f"videos/chunk-{self.chunk_of(ep_idx):03d}/{key}/episode_{ep_idx:06d}.mp4"
        )

    def episode_lengths(self) -> list[int]:
        return [int(e["length"]) for e in self.episodes if "length" in e]

    def task_vocabulary(self) -> dict[int, str]:
        """``task_index`` -> string, from ``meta/tasks.jsonl``.

        LeRobot stores *every* annotation vocabulary in this one table, not just the
        episode instruction. A per-frame ``annotation.human.subtask_stage`` column holds
        int indices into it, so without this lookup the stage labels come out as bare
        numbers like 11/12/13/15 instead of done/place/pick/navigate.
        """
        out = {}
        for row in self.tasks:
            if "task_index" in row:
                for field_name in ("task", "task_description", "name"):
                    if field_name in row:
                        out[int(row["task_index"])] = str(row[field_name])
                        break
        return out

    def task_names(self) -> list[str]:
        out = []
        for t in self.tasks:
            for field_name in ("task", "task_description", "name"):
                if field_name in t:
                    out.append(str(t[field_name]))
                    break
        return out

    # ---------------- modality groups ----------------

    def modality_groups(self, kind: str) -> dict[str, tuple[int, int]]:
        """Named ``[start, end)`` slices of the flat action or state vector.

        Read from ``meta/modality.json``. For RoboCasa365's PandaOmron this yields, for
        ``kind="action"``: base_motion [0,4), control_mode [4,5),
        end_effector_position [5,8), end_effector_rotation [8,11), gripper_close [11,12).

        These groups are what make the per-component metrics of PLAN.md Stage 2
        (rotation error, gripper error) possible without guessing the layout.
        """
        if not self.modality or kind not in self.modality:
            return {}
        out = {}
        for name, spec in self.modality[kind].items():
            if "start" in spec and "end" in spec:
                out[name] = (int(spec["start"]), int(spec["end"]))
        return dict(sorted(out.items(), key=lambda kv: kv[1][0]))

    @property
    def action_groups(self) -> dict[str, tuple[int, int]]:
        return self.modality_groups("action")

    @property
    def state_groups(self) -> dict[str, tuple[int, int]]:
        return self.modality_groups("state")

    # ---------------- camera role resolution ----------------

    def classify_cameras(self) -> dict[str, list[str]]:
        """Group discovered camera names into roles by naming convention.

        Returns a dict with keys ``wrist`` and ``third_person``. This reads the *actual*
        discovered names; it never invents one. Used by the camera-subset layer to build
        the named view conditions of the harness contract (docs/harness_contract.md)
        without hard-coding.
        """
        wrist, third = [], []
        for name in self.camera_names:
            low = name.lower()
            if any(tok in low for tok in ("eye_in_hand", "wrist", "hand", "gripper")):
                wrist.append(name)
            else:
                third.append(name)
        return {"wrist": wrist, "third_person": third}
