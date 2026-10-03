"""Synthetic partialization of full trajectories, PLAN.md sections 1, 6.1 and 11 (Stage 2).

PLAN.md section 1 says a sample is "partial" when it holds only incomplete visual evidence
of a larger manipulation process, and names the ways that happens in real Internet video:

    temporal cropping, occlusion, missing task phases, viewpoint limitations,
    incomplete instructional clips

Section 6.1 lists *synthetic partialization* as a legitimate use of the existing RoboCasa
data, and Stage 2 requires creating partial visual samples from full trajectories. This
module provides those axes as interchangeable strategies over one trajectory.

Why they are interchangeable rather than one hard-coded notion: PLAN.md section 13.6 warns
against letting the earlier multi-view implementation lock the research problem into view
selection. Viewpoint is one axis here, not the axis.

Every strategy returns both the kept sample **and a record of what was withheld**. That
record is the oracle target for "complementary evidence": because partialization is
synthetic, the complement is known exactly, which is what makes recovery measurable at all.
The record is ORACLE-tier (PLAN.md section 9) and must never reach the method as input.

These operate on cached RGB frames and frame indices only — no simulator, no privileged
state — so the core stays simulator-agnostic (section 13.1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def _digest(key: tuple) -> int:
    raw = "|".join(str(k) for k in key).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _unit(key: tuple) -> float:
    return _digest(key) / float(1 << 64)


def _choice(key: tuple, n: int) -> int:
    return _digest(key) % n


@dataclass
class PartialSpec:
    """What a partialization kept and what it withheld, for one sample.

    ``withheld_*`` fields are ORACLE-tier: they describe the ground-truth complement and
    exist so an experiment can ask whether a method recovers it. Passing them to the
    method would be circular.
    """

    kept_cameras: list[str] = field(default_factory=list)
    withheld_cameras: list[str] = field(default_factory=list)
    kept_frames: tuple[int, int] | None = None     # [start, stop) within the episode
    withheld_frames: list[tuple[int, int]] = field(default_factory=list)
    kept_stages: list[int] = field(default_factory=list)
    withheld_stages: list[int] = field(default_factory=list)
    occlusion_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    axis: str = "none"

    @property
    def is_partial(self) -> bool:
        return bool(
            self.withheld_cameras
            or self.withheld_frames
            or self.withheld_stages
            or self.occlusion_boxes
        )

    def as_dict(self) -> dict:
        return {
            "axis": self.axis,
            "kept_cameras": list(self.kept_cameras),
            "withheld_cameras": list(self.withheld_cameras),
            "kept_frames": list(self.kept_frames) if self.kept_frames else None,
            "withheld_frames": [list(w) for w in self.withheld_frames],
            "kept_stages": list(self.kept_stages),
            "withheld_stages": list(self.withheld_stages),
            "occlusion_boxes": [list(b) for b in self.occlusion_boxes],
            "is_partial": self.is_partial,
        }


class Partialization(Protocol):
    """A way of making a full trajectory sample partial.

    Implementations must be deterministic in ``(seed, episode, frame)`` so a dataloader
    with several workers, or a resumed run, produces the same partial samples.
    """

    axis: str

    def apply(self, ctx: "TrajectoryContext", seed: int) -> PartialSpec: ...


@dataclass
class TrajectoryContext:
    """What a partialization strategy needs to know about one sample's trajectory."""

    episode: int
    frame: int                  # index within the episode
    episode_length: int
    cameras: list[str]
    resolution: int
    stages: np.ndarray | None = None   # per-frame stage ids for this episode, if known


# ---------------------------------------------------------------- viewpoint


@dataclass
class ViewpointPartial:
    """Keep a subset of camera streams; withhold the rest.

    The axis the earlier multi-view study explored. Kept as one option among several, not
    as the definition of partiality (PLAN.md section 13.6).
    """

    keep: list[str] | None = None       # explicit cameras, or None to keep ``n_keep``
    n_keep: int = 1
    axis: str = "viewpoint"

    def apply(self, ctx: TrajectoryContext, seed: int) -> PartialSpec:
        if self.keep is not None:
            kept = [c for c in self.keep if c in ctx.cameras]
            unknown = [c for c in self.keep if c not in ctx.cameras]
            if unknown:
                raise ValueError(f"cameras {unknown} not in trajectory {ctx.cameras}")
        else:
            n = max(1, min(self.n_keep, len(ctx.cameras)))
            start = _choice(("view", seed, ctx.episode), len(ctx.cameras))
            kept = [ctx.cameras[(start + i) % len(ctx.cameras)] for i in range(n)]
        withheld = [c for c in ctx.cameras if c not in kept]
        return PartialSpec(kept_cameras=kept, withheld_cameras=withheld, axis=self.axis)


# ---------------------------------------------------------------- temporal


@dataclass
class TemporalCropPartial:
    """Keep a contiguous window of the episode; withhold what falls outside.

    The closest synthetic analogue of an incomplete instructional clip: the viewer sees a
    span of the manipulation and never sees how it began or ended.
    """

    keep_fraction: float = 0.5
    anchor: str = "random"          # "random" | "start" | "end" | "around_frame"
    axis: str = "temporal"

    def apply(self, ctx: TrajectoryContext, seed: int) -> PartialSpec:
        if not 0.0 < self.keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1]")
        T = max(1, ctx.episode_length)
        win = max(1, int(round(T * self.keep_fraction)))
        if win >= T:
            return PartialSpec(kept_frames=(0, T), axis=self.axis)

        if self.anchor == "start":
            start = 0
        elif self.anchor == "end":
            start = T - win
        elif self.anchor == "around_frame":
            start = int(np.clip(ctx.frame - win // 2, 0, T - win))
        elif self.anchor == "random":
            start = _choice(("tcrop", seed, ctx.episode), T - win + 1)
        else:
            raise ValueError(f"unknown anchor '{self.anchor}'")

        stop = start + win
        withheld = []
        if start > 0:
            withheld.append((0, start))
        if stop < T:
            withheld.append((stop, T))
        return PartialSpec(
            kept_frames=(start, stop), withheld_frames=withheld, axis=self.axis
        )


# ---------------------------------------------------------------- phase


@dataclass
class PhaseDropPartial:
    """Withhold every frame belonging to one or more task stages.

    This is the "missing task phases" case of PLAN.md section 1, and the sharpest probe of
    complementarity: if a whole phase is unobserved, can anything recover what it
    contained? It needs per-frame stage labels, which are ORACLE-tier, so it is a
    *controlled-experiment* tool — the labels construct the condition, they are never fed
    to the method.
    """

    drop_stages: list[int] | None = None   # explicit stage ids, or None to drop one
    n_drop: int = 1
    axis: str = "phase"

    def apply(self, ctx: TrajectoryContext, seed: int) -> PartialSpec:
        if ctx.stages is None:
            raise ValueError(
                "PhaseDropPartial needs per-frame stage labels. RoboCasa365 ships these "
                "only for target composite tasks."
            )
        present = sorted({int(s) for s in np.unique(ctx.stages) if s >= 0})
        if not present:
            return PartialSpec(axis=self.axis)

        if self.drop_stages is not None:
            dropped = [s for s in self.drop_stages if s in present]
        else:
            n = max(0, min(self.n_drop, max(0, len(present) - 1)))  # never drop all
            dropped = []
            for i in range(n):
                remaining = [s for s in present if s not in dropped]
                if not remaining:
                    break
                dropped.append(remaining[_choice(("phase", seed, ctx.episode, i), len(remaining))])

        kept = [s for s in present if s not in dropped]
        return PartialSpec(kept_stages=kept, withheld_stages=sorted(dropped), axis=self.axis)

    def frame_mask(self, ctx: TrajectoryContext, spec: PartialSpec) -> np.ndarray:
        """Boolean mask over the episode's frames: True where the frame is kept."""
        if ctx.stages is None:
            raise ValueError("no stage labels")
        dropped = set(spec.withheld_stages)
        return np.array([int(s) not in dropped for s in ctx.stages], dtype=bool)


# ---------------------------------------------------------------- occlusion


@dataclass
class OcclusionPartial:
    """Blank rectangular regions of the image, simulating occlusion or framing loss.

    Unlike the other axes this removes evidence *within* a frame rather than removing
    frames or streams, so it probes whether a method can use spatial context rather than
    another view or another timestep.
    """

    n_boxes: int = 1
    box_fraction: float = 0.3      # side length as a fraction of the image
    per_frame: bool = False        # resample the box per frame, or hold it per episode
    axis: str = "occlusion"

    def apply(self, ctx: TrajectoryContext, seed: int) -> PartialSpec:
        if not 0.0 < self.box_fraction < 1.0:
            raise ValueError("box_fraction must be in (0, 1)")
        R = ctx.resolution
        side = max(1, int(round(R * self.box_fraction)))
        scope = (ctx.episode, ctx.frame) if self.per_frame else (ctx.episode,)

        boxes = []
        for i in range(max(0, self.n_boxes)):
            y = _choice(("occy", seed, *scope, i), max(1, R - side + 1))
            x = _choice(("occx", seed, *scope, i), max(1, R - side + 1))
            boxes.append((int(y), int(x), side, side))
        return PartialSpec(occlusion_boxes=boxes, axis=self.axis)

    @staticmethod
    def apply_to_image(img: np.ndarray, spec: PartialSpec) -> np.ndarray:
        """Blank the specified boxes. Returns a copy; the input may be a read-only memmap."""
        out = np.array(img)
        for (y, x, h, w) in spec.occlusion_boxes:
            out[y : y + h, x : x + w] = 0
        return out


# ---------------------------------------------------------------- composition


@dataclass
class ComposedPartial:
    """Apply several axes to the same sample.

    Useful for the realistic case: an Internet clip is typically temporally cropped *and*
    shot from one viewpoint *and* partly occluded, not partial along a single axis.
    """

    parts: list[Partialization]
    axis: str = "composed"

    def apply(self, ctx: TrajectoryContext, seed: int) -> PartialSpec:
        merged = PartialSpec(axis="+".join(p.axis for p in self.parts))
        for i, p in enumerate(self.parts):
            s = p.apply(ctx, seed + 1000 * i)
            if s.kept_cameras:
                merged.kept_cameras = s.kept_cameras
                merged.withheld_cameras = s.withheld_cameras
            if s.kept_frames is not None:
                merged.kept_frames = s.kept_frames
                merged.withheld_frames.extend(s.withheld_frames)
            if s.kept_stages or s.withheld_stages:
                merged.kept_stages = s.kept_stages
                merged.withheld_stages.extend(s.withheld_stages)
            merged.occlusion_boxes.extend(s.occlusion_boxes)
        return merged


#: Axes named by PLAN.md section 1. ``incomplete clip`` is temporal cropping.
PARTIALIZATION_AXES = ("viewpoint", "temporal", "phase", "occlusion", "composed")


def build_partialization(name: str, **kwargs) -> Partialization:
    """Construct a strategy by axis name."""
    table = {
        "viewpoint": ViewpointPartial,
        "temporal": TemporalCropPartial,
        "phase": PhaseDropPartial,
        "occlusion": OcclusionPartial,
    }
    if name == "composed":
        parts = kwargs.get("parts")
        if not parts:
            raise ValueError("composed partialization needs parts=[...]")
        return ComposedPartial(parts=parts)
    if name not in table:
        raise ValueError(f"unknown partialization '{name}'. known: {PARTIALIZATION_AXES}")
    return table[name](**kwargs)
