"""Information tiers, PLAN.md section 9.

The simulator exposes far more than the P2C method may consume. The plan is explicit that
this separation carries the central claim: a method that depends on privileged simulator
state would not transfer to cheap Internet video, which is the whole target distribution
(PLAN.md sections 1 and 3.1).

Three tiers, from most to least restricted:

``METHOD``
    What the P2C method itself may read: RGB video, optionally weak language. Nothing
    else. This is the only tier that an Internet video clip could also supply.

``POLICY``
    What the downstream robot policy may read: a visual representation, robot actions as
    supervision, optionally proprioception (PLAN.md section 3.2). Robot demonstrations are
    downstream supervision, not source data.

``ORACLE``
    Analysis and controlled experiments only: stage labels, object poses, segmentation,
    contact state, full simulator state, reward, success flags.

Stage labels sit in ``ORACLE`` deliberately. They are legitimate for the per-stage
analysis — that is exactly what PLAN.md section 9 permits them for — and illegitimate as
policy input. Keeping that boundary in code rather than in a comment is the point: it is
the kind of leak that is invisible in a results table.
"""

from __future__ import annotations

from enum import IntEnum


class Tier(IntEnum):
    """Access tiers, ordered so a higher value permits strictly more."""

    METHOD = 0
    POLICY = 1
    ORACLE = 2

    @property
    def description(self) -> str:
        return {
            Tier.METHOD: "RGB video + optional weak language (what Internet video supplies)",
            Tier.POLICY: "METHOD + robot actions as supervision + optional proprioception",
            Tier.ORACLE: "POLICY + privileged simulator state, stage labels, success",
        }[self]


#: Which tier each field a Dataset may emit belongs to.
FIELD_TIERS: dict[str, Tier] = {
    # --- METHOD: what a cheap action-free video could also provide ---
    "images": Tier.METHOD,
    "language": Tier.METHOD,
    "episode": Tier.METHOD,      # bookkeeping, carries no privileged content
    "frame": Tier.METHOD,
    "camera_ids": Tier.METHOD,
    "view_kept": Tier.METHOD,
    "partial_mask": Tier.METHOD,
    # --- POLICY: robot demonstration supervision ---
    "action": Tier.POLICY,
    "state": Tier.POLICY,        # proprioception
    # --- ORACLE: privileged, analysis only ---
    "stage": Tier.ORACLE,
    "object_pose": Tier.ORACLE,
    "segmentation": Tier.ORACLE,
    "contact": Tier.ORACLE,
    "sim_state": Tier.ORACLE,
    "reward": Tier.ORACLE,
    "success": Tier.ORACLE,
    "withheld": Tier.ORACLE,     # what partialization removed: the oracle target
}


class TierViolation(RuntimeError):
    """Raised when a consumer reads a field above the tier it declared."""


def tier_of(field: str) -> Tier:
    """Tier of a field name. Unknown fields are treated as ORACLE.

    Defaulting unknown fields to the most restricted tier is deliberate: a new field
    added without thought should fail closed, not quietly become method input.
    """
    return FIELD_TIERS.get(field, Tier.ORACLE)


def check_access(field: str, tier: Tier) -> None:
    """Raise if ``field`` is above ``tier``."""
    needed = tier_of(field)
    if needed > tier:
        raise TierViolation(
            f"field '{field}' is {needed.name}-tier but the consumer declared "
            f"{tier.name}. {needed.name} means: {needed.description}. "
            f"If this is an analysis or oracle experiment, declare Tier.ORACLE "
            f"explicitly (PLAN.md section 9)."
        )


def fields_at(tier: Tier) -> list[str]:
    """Every known field a consumer at ``tier`` may read."""
    return sorted(f for f, t in FIELD_TIERS.items() if t <= tier)


class TieredSample(dict):
    """A sample dict that refuses reads above its declared tier.

    Behaves as an ordinary mapping for permitted fields, so it drops into a dataloader
    unchanged; ``collate`` sees a plain dict of tensors.
    """

    def __init__(self, data: dict, tier: Tier = Tier.ORACLE):
        super().__init__(data)
        self._tier = tier

    @property
    def tier(self) -> Tier:
        return self._tier

    def __getitem__(self, key):
        check_access(key, self._tier)
        return super().__getitem__(key)

    def get(self, key, default=None):
        check_access(key, self._tier)
        return super().get(key, default)

    def restricted_to(self, tier: Tier) -> dict:
        """A plain dict holding only the fields permitted at ``tier``.

        Use this to hand the P2C method exactly what an Internet video would supply,
        instead of trusting the method not to look at the rest.
        """
        return {k: v for k, v in self.items() if tier_of(k) <= tier}
