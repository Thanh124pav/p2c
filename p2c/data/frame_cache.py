"""Reader for the frame cache written by ``scripts/build_frame_cache.py``.

Opens every array as a memmap so a cache larger than the machine's 5.8 GB of RAM is
still usable: only the touched pages are resident.

The deterministic train/validation split lives here rather than in the Dataset, because
the harness contract (docs/harness_contract.md) requires *identical* splits across every
camera ablation. Splitting by
episode (never by frame) also stops frames from one demonstration appearing on both
sides, which would leak and flatter every condition equally but invalidate the
comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EpisodeSpan:
    episode_index: int
    start: int
    length: int

    @property
    def stop(self) -> int:
        return self.start + self.length


class FrameCache:
    """Memmap-backed view of one cached task."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        index_file = self.path / "index.json"
        if not index_file.exists():
            raise FileNotFoundError(
                f"no frame cache at {self.path}. Build one with scripts/build_frame_cache.py"
            )
        with open(index_file) as f:
            self.index = json.load(f)

        self.cameras: list[str] = list(self.index["cameras"])
        self.resolution: int = int(self.index["resolution"])
        self.num_frames: int = int(self.index["num_frames"])
        self.action_dim: int = int(self.index["action_dim"])
        self.state_dim: int = int(self.index.get("state_dim") or 0)
        self.task: str = self.index.get("task", self.path.name)
        self.stage_names: dict | None = self.index.get("stage_names")
        self.stage_key: str | None = self.index.get("stage_key")
        self.action_groups: dict[str, tuple[int, int]] = {
            k: (int(v[0]), int(v[1]))
            for k, v in (self.index.get("action_groups") or {}).items()
        }
        self.state_groups: dict[str, tuple[int, int]] = {
            k: (int(v[0]), int(v[1]))
            for k, v in (self.index.get("state_groups") or {}).items()
        }

        self.spans = [
            EpisodeSpan(int(e["episode_index"]), int(e["start"]), int(e["length"]))
            for e in self.index["episodes"]
        ]

        self._images = {
            c: np.load(self.path / f"images_{c}.npy", mmap_mode="r") for c in self.cameras
        }
        self.actions = np.load(self.path / "actions.npy", mmap_mode="r")
        self.states = np.load(self.path / "states.npy", mmap_mode="r")
        self.episode_index = np.load(self.path / "episode_index.npy", mmap_mode="r")
        self.frame_index = np.load(self.path / "frame_index.npy", mmap_mode="r")
        stage_file = self.path / "stage.npy"
        self.stage = np.load(stage_file, mmap_mode="r") if stage_file.exists() else None
        # Optional: which language instruction each frame belongs to. Caches built before
        # this was recorded simply lack the file.
        ti_file = self.path / "task_index.npy"
        self.task_index = np.load(ti_file, mmap_mode="r") if ti_file.exists() else None

        self._validate()

    def _validate(self) -> None:
        n = self.num_frames
        for name, arr in [
            ("actions", self.actions),
            ("states", self.states),
            ("episode_index", self.episode_index),
            ("frame_index", self.frame_index),
        ]:
            if arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} rows, index.json says {n}")
        for c, arr in self._images.items():
            if arr.shape[0] != n:
                raise ValueError(f"images_{c} has {arr.shape[0]} rows, expected {n}")
            if arr.shape[1:3] != (self.resolution, self.resolution):
                raise ValueError(f"images_{c} is {arr.shape[1:3]}, expected square {self.resolution}")
        if self.stage is not None and self.stage.shape[0] != n:
            raise ValueError("stage.npy length mismatch")
        total = sum(s.length for s in self.spans)
        if total != n:
            raise ValueError(f"episode spans sum to {total}, index.json says {n}")

    # ---------------- access ----------------

    @property
    def has_stages(self) -> bool:
        return self.stage is not None

    @property
    def num_instructions(self) -> int:
        """Distinct language instructions spanned by this cache.

        A vision-only policy cannot beat the conditional mean on a task whose goal is
        carried by the instruction rather than the image: the same view then maps to
        different actions. When this is greater than 1, check that the instruction is
        recoverable from the observation before reading a flat validation loss as
        evidence about view sufficiency.
        """
        if self.task_index is None:
            return int(self.index.get("num_distinct_instructions", -1))
        return int(np.unique(np.asarray(self.task_index)).size)

    def image(self, camera: str, i: int) -> np.ndarray:
        """uint8 [H, W, 3] for one camera at global frame index ``i``."""
        try:
            return self._images[camera][i]
        except KeyError:
            raise KeyError(
                f"camera '{camera}' not in cache. available: {self.cameras}"
            ) from None

    def stage_name(self, stage_id: int) -> str:
        if not self.stage_names:
            return str(stage_id)
        return self.stage_names.get(str(int(stage_id)), str(stage_id))

    def span_of_episode(self, ep: int) -> EpisodeSpan:
        for s in self.spans:
            if s.episode_index == ep:
                return s
        raise KeyError(f"episode {ep} not in cache")

    # ---------------- splits ----------------

    def split_episodes(
        self, val_fraction: float = 0.2, split_seed: int = 0
    ) -> tuple[list[int], list[int]]:
        """Deterministic episode-level train/val split.

        Identical for every camera condition because it depends only on the cache
        contents and ``split_seed`` — not on the view condition, model seed, or
        iteration order (harness contract C5).
        """
        eps = sorted(s.episode_index for s in self.spans)
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")

        # Hash-based assignment: stable under a change in the number of episodes, so
        # adding data later does not reshuffle the existing split.
        def rank(ep: int) -> int:
            raw = f"{split_seed}|{ep}".encode()
            return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")

        ordered = sorted(eps, key=rank)
        n_val = max(1, int(round(len(ordered) * val_fraction)))
        if n_val >= len(ordered):
            raise ValueError(
                f"val_fraction={val_fraction} leaves no training episodes "
                f"({len(ordered)} available)"
            )
        val = sorted(ordered[:n_val])
        train = sorted(ordered[n_val:])
        return train, val

    def frame_indices(self, episodes: list[int]) -> np.ndarray:
        """Global frame indices belonging to the given episodes, in episode order."""
        wanted = set(episodes)
        chunks = [
            np.arange(s.start, s.stop, dtype=np.int64)
            for s in self.spans
            if s.episode_index in wanted
        ]
        if not chunks:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(chunks)

    # ---------------- normalisation stats ----------------

    def action_stats(self, episodes: list[int]) -> dict[str, np.ndarray]:
        """Per-dimension action mean/std over the given episodes.

        Computed from the *training* episodes only, so validation loss is not scaled by
        statistics that saw validation data. Because the split is identical across
        conditions, so are these statistics — a precondition for comparing losses
        between view conditions at all.
        """
        idx = self.frame_indices(episodes)
        if idx.size == 0:
            raise ValueError("no frames for the given episodes")
        arr = np.asarray(self.actions[idx], dtype=np.float64)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        # Guard constant dimensions: RoboCasa's 12-D action has near-constant entries
        # (e.g. control mode) whose tiny std would otherwise blow up the normalised loss.
        std = np.where(std < 1e-4, 1.0, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    def state_stats(self, episodes: list[int]) -> dict[str, np.ndarray]:
        idx = self.frame_indices(episodes)
        arr = np.asarray(self.states[idx], dtype=np.float64)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < 1e-4, 1.0, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    def trivial_baseline_mse(
        self, train_episodes: list[int], val_episodes: list[int]
    ) -> float:
        """MSE of predicting the train-set mean action, evaluated on validation frames.

        The reference line every learned condition must clear. In normalised action space
        this sits near 1.0 by construction, so a condition scoring ~1.0 has learned
        nothing — and two such conditions differing by a few percent are both at chance,
        not meaningfully ranked. Reporting this stops that misreading.
        """
        stats = self.action_stats(train_episodes)
        idx = self.frame_indices(val_episodes)
        if idx.size == 0:
            return float("nan")
        val = np.asarray(self.actions[idx], dtype=np.float64)
        val_n = (val - stats["mean"]) / stats["std"]
        # The normalised prediction of the train mean is exactly zero.
        return float(np.mean(val_n**2))

    def summary(self) -> str:
        tr, va = self.split_episodes()
        s = [
            f"FrameCache {self.path.name}",
            f"  task        {self.task}",
            f"  cameras     {self.cameras}",
            f"  resolution  {self.resolution}x{self.resolution}",
            f"  frames      {self.num_frames} in {len(self.spans)} episodes",
            f"  action_dim  {self.action_dim}   state_dim {self.state_dim}",
            f"  stages      {len(self.stage_names) if self.stage_names else 0}"
            f" (key={self.stage_key})",
            f"  split       {len(tr)} train / {len(va)} val episodes",
        ]
        return "\n".join(s)
