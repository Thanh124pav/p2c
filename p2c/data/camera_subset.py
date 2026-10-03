"""Camera-subset logic: the one reusable abstraction of the harness contract
(docs/harness_contract.md).

The policy must not care whether an observation holds 1, 2 or 3 cameras, and the
ablations must differ *only* in which camera streams are fed. This module therefore
separates two things:

* **roles** (``primary``, ``secondary``, ``wrist``) resolved against the camera names
  actually discovered in the dataset — never hard-coded (PLAN.md section 6.1);
* **view conditions** (``single_primary``, ``primary+wrist``, ``random_two``, ...) defined
  as a fixed number of *slots*, each slot filled by either a fixed camera or a
  deterministic random draw.

Fixing the slot count per condition is what keeps the "more pixels" controls honest: a
random-view condition has the same number of image tensors per sample as the fixed
complementary pair it is compared against, and the sample count never changes
silently (harness contract C3).

This module is pure Python over camera-name lists, so it is testable without a dataset
or a GPU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# ---------------------------------------------------------------- roles


@dataclass(frozen=True)
class CameraRoles:
    """Mapping from P2C role names to concrete discovered camera names."""

    primary: str
    wrist: str | None = None
    secondary: str | None = None
    others: tuple[str, ...] = ()

    @property
    def all_cameras(self) -> list[str]:
        """Every camera, in a deterministic order: primary, secondary, wrist, rest."""
        out = [self.primary]
        for c in (self.secondary, self.wrist):
            if c and c not in out:
                out.append(c)
        for c in self.others:
            if c not in out:
                out.append(c)
        return out

    def resolve(self, role: str) -> str:
        """Resolve a role name or a literal camera name to a camera name."""
        if role in ("primary", "secondary", "wrist"):
            val = getattr(self, role)
            if val is None:
                raise ValueError(
                    f"dataset has no camera for role '{role}'. "
                    f"available: {self.all_cameras}"
                )
            return val
        if role in self.all_cameras:
            return role
        raise ValueError(f"unknown camera/role '{role}'. available: {self.all_cameras}")

    @classmethod
    def from_discovered(
        cls,
        camera_names: list[str],
        roles_hint: dict[str, list[str]] | None = None,
        primary: str | None = None,
    ) -> "CameraRoles":
        """Assign roles from discovered camera names.

        ``roles_hint`` is the output of :meth:`DatasetMeta.classify_cameras`. The primary
        view is the first third-person camera in sorted order unless given explicitly,
        which makes the assignment reproducible across runs and machines.
        """
        if not camera_names:
            raise ValueError("no cameras discovered in dataset")

        hint = roles_hint or {}
        wrists = [c for c in hint.get("wrist", []) if c in camera_names]
        thirds = [c for c in hint.get("third_person", []) if c in camera_names]
        if not wrists and not thirds:
            thirds = sorted(camera_names)

        if primary is not None:
            if primary not in camera_names:
                raise ValueError(
                    f"requested primary '{primary}' not in dataset cameras {camera_names}"
                )
            prim = primary
        elif thirds:
            prim = sorted(thirds)[0]
        else:
            prim = sorted(camera_names)[0]

        remaining_thirds = [c for c in sorted(thirds) if c != prim]
        secondary = remaining_thirds[0] if remaining_thirds else None
        wrist = sorted(wrists)[0] if wrists else None

        used = {prim, secondary, wrist} - {None}
        others = tuple(c for c in sorted(camera_names) if c not in used)
        return cls(primary=prim, wrist=wrist, secondary=secondary, others=others)


# ---------------------------------------------------------------- slots


@dataclass(frozen=True)
class Slot:
    """One image position in the observation.

    Exactly one of ``fixed`` or ``choices`` is set. A slot with ``choices`` draws one
    camera deterministically from that pool, so the observation shape is unchanged.
    """

    fixed: str | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.fixed is None) == (not self.choices):
            raise ValueError("Slot needs exactly one of fixed= or choices=")

    @property
    def is_random(self) -> bool:
        return self.fixed is None


# ---------------------------------------------------------------- conditions


@dataclass(frozen=True)
class ViewCondition:
    """A named camera subset: a fixed number of slots plus optional dropout."""

    name: str
    slots: tuple[Slot, ...]
    camera_dropout: float = 0.0
    random_scope: str = "sample"  # "sample" | "episode" | "run"
    notes: str = ""

    @property
    def num_views(self) -> int:
        return len(self.slots)

    @property
    def is_stochastic(self) -> bool:
        return any(s.is_random for s in self.slots) or self.camera_dropout > 0.0

    def select(self, episode_index: int, frame_index: int, seed: int) -> list[str]:
        """Pick the camera for every slot. Deterministic in (seed, episode, frame).

        Determinism does not go through global RNG state, so a dataloader with several
        workers, or a resumed run, yields exactly the same cameras (the harness contract,
        requirement 6).
        """
        out = []
        for i, slot in enumerate(self.slots):
            if not slot.is_random:
                out.append(slot.fixed)
                continue
            if self.random_scope == "run":
                key = (seed, self.name, i)
            elif self.random_scope == "episode":
                key = (seed, self.name, i, episode_index)
            elif self.random_scope == "sample":
                key = (seed, self.name, i, episode_index, frame_index)
            else:
                raise ValueError(f"bad random_scope: {self.random_scope}")
            out.append(slot.choices[_stable_index(key, len(slot.choices))])
        return out

    def dropout_mask(
        self, episode_index: int, frame_index: int, seed: int
    ) -> list[bool]:
        """Per-slot keep mask for camera dropout (harness contract C9).

        At least one view is always kept, so the policy never sees an empty observation.
        """
        if self.camera_dropout <= 0.0:
            return [True] * self.num_views
        keep = []
        for i in range(self.num_views):
            key = ("drop", seed, self.name, i, episode_index, frame_index)
            u = _stable_unit(key)
            keep.append(u >= self.camera_dropout)
        if not any(keep):
            forced = _stable_index(
                ("force", seed, self.name, episode_index, frame_index), self.num_views
            )
            keep[forced] = True
        return keep

    def describe(self, episode_index: int = 0, frame_index: int = 0, seed: int = 0) -> str:
        parts = []
        for s in self.slots:
            parts.append(s.fixed if not s.is_random else f"rand({'|'.join(s.choices)})")
        txt = f"{self.name}: [{', '.join(parts)}]"
        if self.camera_dropout:
            txt += f" dropout={self.camera_dropout}"
        if self.is_stochastic:
            txt += f" scope={self.random_scope}"
        return txt


def _stable_digest(key: tuple) -> int:
    raw = "|".join(str(k) for k in key).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _stable_index(key: tuple, n: int) -> int:
    return _stable_digest(key) % n


def _stable_unit(key: tuple) -> float:
    return _stable_digest(key) / float(1 << 64)


# ---------------------------------------------------------------- registry

#: Condition names required by harness contract C1-C9.
CONDITION_NAMES = (
    "single_primary",
    "single_wrist",
    "single_secondary",
    "primary+wrist",
    "primary+secondary",
    "all_views",
    "random_two",
    "primary+duplicate",
    "primary+wrist_dropout",
)


def build_condition(
    name: str,
    roles: CameraRoles,
    camera_dropout: float | None = None,
    random_scope: str = "sample",
) -> ViewCondition:
    """Build a named view condition against the dataset's real camera names.

    Named conditions, with the role each plays in a controlled comparison:

    ``single_primary``      B0, the partial observation
    ``single_wrist``        B1, the alternative single view
    ``primary+wrist``       B2, the fixed complementary pair
    ``all_views``           B3, the full-view oracle
    ``random_two``          B4, primary plus a random second view
    ``primary+duplicate``   control: the same camera twice, no new information
    ``primary+wrist_dropout`` control: complementary pair with stochastic view dropout
    """
    drop = 0.0 if camera_dropout is None else camera_dropout

    def fixed(*role_names: str) -> tuple[Slot, ...]:
        return tuple(Slot(fixed=roles.resolve(r)) for r in role_names)

    if name == "single_primary":
        return ViewCondition(name, fixed("primary"), drop, random_scope,
                             "B0 partial observation")
    if name == "single_wrist":
        return ViewCondition(name, fixed("wrist"), drop, random_scope,
                             "B1 alternative single view")
    if name == "single_secondary":
        return ViewCondition(name, fixed("secondary"), drop, random_scope,
                             "alternative third-person single view")
    if name == "primary+wrist":
        return ViewCondition(name, fixed("primary", "wrist"), drop, random_scope,
                             "B2 fixed complementary pair")
    if name == "primary+secondary":
        return ViewCondition(name, fixed("primary", "secondary"), drop, random_scope,
                             "fixed third-person pair")
    if name == "all_views":
        slots = tuple(Slot(fixed=c) for c in roles.all_cameras)
        return ViewCondition(name, slots, drop, random_scope, "B3 full-view oracle")
    if name == "random_two":
        pool = tuple(c for c in roles.all_cameras if c != roles.primary)
        if not pool:
            raise ValueError("random_two needs at least 2 cameras")
        return ViewCondition(
            name,
            (Slot(fixed=roles.primary), Slot(choices=pool)),
            drop,
            random_scope,
            # Because the draw is deterministic in (seed, episode, frame), a given frame
            # keeps the same second view across epochs: this is a random *assignment*,
            # not per-epoch resampling. That is what the harness contract requirement 6
            # asks for, and it is the right semantics for the random-view control, which asks whether
            # *which* view is added matters while holding the view count fixed. Setting
            # random_scope="sample" with a per-epoch reseed would instead make this a
            # view-augmentation condition, a different experiment.
            "B4 primary + random second view (deterministic assignment)",
        )
    if name == "primary+duplicate":
        return ViewCondition(
            name,
            (Slot(fixed=roles.primary), Slot(fixed=roles.primary)),
            drop,
            random_scope,
            "control: duplicate view, carries no new information",
        )
    if name == "primary+wrist_dropout":
        d = 0.5 if camera_dropout is None else camera_dropout
        return ViewCondition(name, fixed("primary", "wrist"), d, random_scope,
                             "control: complementary pair with stochastic view dropout")

    # Fall back to an explicit comma-separated camera/role list, e.g. "primary,wrist".
    if "," in name or name in ("primary", "secondary", "wrist"):
        role_names = [r.strip() for r in name.split(",") if r.strip()]
        return ViewCondition(name, fixed(*role_names), drop, random_scope, "explicit list")

    raise ValueError(
        f"unknown view condition '{name}'. known: {CONDITION_NAMES} "
        f"(or a comma-separated list of roles/camera names)"
    )
