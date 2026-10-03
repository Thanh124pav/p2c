"""``CameraSubsetDataset``: a torch Dataset over cached RoboCasa trajectories.

The consumer receives an image stack of shape ``[V, 3, H, W]`` and never learns which
cameras produced it, so one model and one training loop serve the 1-, 2- and 3-view
conditions. Viewpoint is one partialization axis among several
(:mod:`p2c.data.partialization`); PLAN.md section 13.6 warns against letting it define the
research problem.

**Information tiers.** Construct with a ``tier`` (PLAN.md section 9) and the sample dict
contains only fields that tier may read. This is enforcement, not documentation: a dataset
built at ``Tier.METHOD`` physically cannot hand out actions or stage labels, so a method
meant to work on action-free Internet video cannot quietly come to depend on robot
supervision or privileged simulator state. The default is ``Tier.ORACLE`` because the
analysis path legitimately needs everything.

What is held fixed across conditions, per the harness contract
(docs/harness_contract.md):

1. same demonstrations  - the episode list comes from the shared cache;
2. same train/val split - computed by :meth:`FrameCache.split_episodes`, which depends
   only on the cache and ``split_seed``;
3. same preprocessing   - pixels were decoded once by the cache builder; this class only
   casts and scales, identically for every camera;
4. same action targets  - actions are read at the same global frame index regardless of
   which cameras are selected, and normalised with train-split statistics;
5. same sample count    - ``len()`` depends only on the split, never on the condition, so
   a random-view run cannot silently change the number of samples;
6. deterministic sampling - random slots hash ``(seed, episode, frame)`` instead of
   drawing from global RNG state, so dataloader workers and resumed runs agree;
7. camera IDs are returned with every sample so a run can log exactly what it saw.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from p2c.data.camera_subset import CameraRoles, ViewCondition, build_condition
from p2c.data.frame_cache import FrameCache
from p2c.data.tiers import Tier, tier_of


class CameraSubsetDataset(Dataset):
    """Frames from a :class:`FrameCache`, restricted to one camera subset.

    Parameters
    ----------
    cache:
        Open frame cache, or a path to one.
    condition:
        A :class:`ViewCondition`, or the name of one (e.g. ``"primary+wrist"``).
    episodes:
        Episode indices to draw from. Pass the output of
        :meth:`FrameCache.split_episodes` so train and val stay disjoint.
    camera_dropout:
        Stochastic view dropout, harness contract C9. Only applied when the
        condition itself does not already set it.
    n_obs_steps:
        Frames of history per view. Clamped at the episode start, never crossing an
        episode boundary.
    action_horizon:
        Number of future actions to predict. Clamped at the episode end by repeating the
        final action, which keeps the target shape fixed.
    """

    def __init__(
        self,
        cache: FrameCache | str,
        condition: ViewCondition | str,
        episodes: list[int],
        *,
        action_stats: dict[str, np.ndarray] | None = None,
        state_stats: dict[str, np.ndarray] | None = None,
        camera_dropout: float | None = None,
        n_obs_steps: int = 1,
        action_horizon: int = 1,
        seed: int = 0,
        primary_camera: str | None = None,
        normalize_actions: bool = True,
        tier: Tier = Tier.ORACLE,
    ):
        self.cache = cache if isinstance(cache, FrameCache) else FrameCache(cache)
        self.roles = CameraRoles.from_discovered(
            self.cache.cameras,
            roles_hint=_roles_hint(self.cache.cameras),
            primary=primary_camera,
        )
        self.condition = (
            condition
            if isinstance(condition, ViewCondition)
            else build_condition(condition, self.roles, camera_dropout=camera_dropout)
        )
        self.episodes = sorted(episodes)
        self.seed = int(seed)
        self.n_obs_steps = max(1, int(n_obs_steps))
        self.action_horizon = max(1, int(action_horizon))
        self.normalize_actions = normalize_actions
        self.tier = Tier(tier)

        if not self.episodes:
            raise ValueError("CameraSubsetDataset got an empty episode list")

        # Global frame index per sample. Depends only on the episode list, so every
        # condition built from the same split has the same length and the same order.
        self._frames = self.cache.frame_indices(self.episodes)

        # Episode bounds for clamping history and action horizon.
        self._ep_start = {}
        self._ep_stop = {}
        for ep in self.episodes:
            span = self.cache.span_of_episode(ep)
            self._ep_start[ep] = span.start
            self._ep_stop[ep] = span.stop

        self.action_stats = action_stats
        self.state_stats = state_stats
        self._cam_to_id = {c: i for i, c in enumerate(self.cache.cameras)}

    # ---------------- torch Dataset ----------------

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def __getitem__(self, i: int) -> dict:
        g = int(self._frames[i])
        ep = int(self.cache.episode_index[g])
        fr = int(self.cache.frame_index[g])
        start, stop = self._ep_start[ep], self._ep_stop[ep]

        cams = self.condition.select(ep, fr, self.seed)
        keep = self.condition.dropout_mask(ep, fr, self.seed)

        # history indices, clamped inside the episode (no boundary crossing)
        hist = [max(start, g - k) for k in reversed(range(self.n_obs_steps))]

        views = []
        for cam, keep_it in zip(cams, keep):
            frames = [self.cache.image(cam, j) for j in hist]
            # np.array copies: the cache arrays are read-only memmaps, and handing torch
            # a non-writable buffer is undefined behaviour if anything later writes
            # in place. The copy is unavoidable anyway, since the pixels must leave the
            # memmap to be scaled.
            arr = np.array(np.stack(frames) if self.n_obs_steps > 1 else frames[0][None])
            t = torch.from_numpy(arr)  # [T, H, W, 3] uint8
            t = t.permute(0, 3, 1, 2).float().div_(255.0)  # [T, 3, H, W]
            if not keep_it:
                t = torch.zeros_like(t)  # dropped view: zeros, shape preserved
            views.append(t)

        images = torch.stack(views)  # [V, T, 3, H, W]
        if self.n_obs_steps == 1:
            images = images[:, 0]  # [V, 3, H, W]

        # actions: same global index for every condition, clamped at the episode end
        a_idx = [min(stop - 1, g + k) for k in range(self.action_horizon)]
        act = np.array(self.cache.actions[a_idx], dtype=np.float32)
        if self.normalize_actions and self.action_stats is not None:
            act = (act - self.action_stats["mean"]) / self.action_stats["std"]
        action = torch.from_numpy(np.ascontiguousarray(act, dtype=np.float32))
        if self.action_horizon == 1:
            action = action[0]

        state = np.array(self.cache.states[g], dtype=np.float32)
        if self.state_stats is not None:
            state = (state - self.state_stats["mean"]) / self.state_stats["std"]

        sample = {
            "images": images,
            "state": torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)),
            "action": action,
            "episode": torch.tensor(ep, dtype=torch.int32),
            "frame": torch.tensor(fr, dtype=torch.int32),
            "camera_ids": torch.tensor(
                [self._cam_to_id[c] for c in cams], dtype=torch.int16
            ),
            "view_kept": torch.tensor(keep, dtype=torch.bool),
        }
        if self.cache.has_stages:
            sample["stage"] = torch.tensor(int(self.cache.stage[g]), dtype=torch.int32)
        else:
            sample["stage"] = torch.tensor(-1, dtype=torch.int32)

        # Drop anything above the declared tier. Filtering here rather than trusting the
        # consumer is the point: a Tier.METHOD dataset cannot leak actions or stage labels
        # into something that is supposed to work on action-free video (PLAN.md section 9).
        if self.tier is not Tier.ORACLE:
            sample = {k: v for k, v in sample.items() if tier_of(k) <= self.tier}
        return sample

    # ---------------- introspection / logging ----------------

    @property
    def num_views(self) -> int:
        return self.condition.num_views

    @property
    def image_shape(self) -> tuple[int, ...]:
        r = self.cache.resolution
        if self.n_obs_steps == 1:
            return (self.num_views, 3, r, r)
        return (self.num_views, self.n_obs_steps, 3, r, r)

    def describe(self) -> dict:
        """Everything a run needs to log about its camera configuration (contract C12)."""
        return {
            "condition": self.condition.name,
            "notes": self.condition.notes,
            "num_views": self.num_views,
            "slots": [
                {"fixed": s.fixed, "choices": list(s.choices)}
                for s in self.condition.slots
            ],
            "camera_dropout": self.condition.camera_dropout,
            "random_scope": self.condition.random_scope,
            "is_stochastic": self.condition.is_stochastic,
            "roles": {
                "primary": self.roles.primary,
                "secondary": self.roles.secondary,
                "wrist": self.roles.wrist,
            },
            "cache_cameras": self.cache.cameras,
            "num_episodes": len(self.episodes),
            "num_samples": len(self),
            "image_shape": list(self.image_shape),
            "action_horizon": self.action_horizon,
            "n_obs_steps": self.n_obs_steps,
            "seed": self.seed,
            "tier": self.tier.name,
            "fields_emitted": sorted(self[0].keys()) if len(self) else [],
        }

    def camera_usage(self, max_samples: int = 2000) -> dict[str, int]:
        """Count how often each camera is actually selected.

        Used to verify that a stochastic condition really does spread over its pool, and
        to record the realised camera mix for every experiment (contract C3).
        """
        counts = {c: 0 for c in self.cache.cameras}
        step = max(1, len(self) // max_samples)
        for i in range(0, len(self), step):
            g = int(self._frames[i])
            ep = int(self.cache.episode_index[g])
            fr = int(self.cache.frame_index[g])
            for c in self.condition.select(ep, fr, self.seed):
                counts[c] += 1
        return counts


def _roles_hint(camera_names: list[str]) -> dict[str, list[str]]:
    """Classify cached camera names into wrist / third-person by naming convention."""
    wrist, third = [], []
    for name in camera_names:
        low = name.lower()
        if any(tok in low for tok in ("eye_in_hand", "wrist", "hand", "gripper")):
            wrist.append(name)
        else:
            third.append(name)
    return {"wrist": wrist, "third_person": third}


def build_train_val(
    cache: FrameCache | str,
    condition: str | ViewCondition,
    *,
    val_fraction: float = 0.2,
    split_seed: int = 0,
    seed: int = 0,
    **kwargs,
) -> tuple[CameraSubsetDataset, CameraSubsetDataset]:
    """Build a train/val pair sharing one split and one set of normalisation statistics.

    Both datasets get statistics computed from the *training* episodes only.
    """
    cache = cache if isinstance(cache, FrameCache) else FrameCache(cache)
    train_eps, val_eps = cache.split_episodes(val_fraction, split_seed)
    a_stats = cache.action_stats(train_eps)
    s_stats = cache.state_stats(train_eps) if cache.state_dim else None

    common = dict(action_stats=a_stats, state_stats=s_stats, seed=seed, **kwargs)
    train = CameraSubsetDataset(cache, condition, train_eps, **common)
    val = CameraSubsetDataset(cache, condition, val_eps, **common)
    return train, val
