#!/usr/bin/env python
"""Closed-loop rollout: run a trained tiny BC checkpoint inside the RoboCasa simulator.

This closes the last gap in the pipeline. Training reads pre-rendered mp4 and never touches
MuJoCo; the simulator check builds environments but runs no policy. Neither says whether a
*trained* policy can actually drive the robot, which is what closed-loop evaluation needs
(PLAN.md section 4, Stage C).

The bridge is narrow and worth stating, because getting it wrong produces a policy that
runs but acts on nonsense:

* **Observation assembly.** The gym wrapper exposes ``state.*`` and ``video.*`` keys with
  the same names the LeRobot dataset uses, so the flat 16-D state vector is rebuilt by
  concatenating groups **in the order recorded in the frame cache**, not in a hard-coded
  order. If the cache and the environment ever disagree, that is a bug worth crashing on
  rather than silently feeding a permuted state.
* **Normalisation must be inverted.** The policy was trained on actions normalised with
  training-split statistics; the environment expects raw actions. The checkpoint carries
  those statistics, and state normalisation is recomputed from the same cache and split.
* **Camera order.** Images go into view slots in the order the training condition used,
  recovered from the checkpoint, and are resized to the cache resolution the encoder was
  built for.

Usage::

    python scripts/rollout_local_bc.py --checkpoint outputs/runs_composite/<run>/best.pt
    python scripts/rollout_local_bc.py --checkpoint <path> --episodes 5 --video out.mp4

Expect weak behaviour: the policy is a ~1.8M-parameter sanity model trained for a few
epochs on a 4 GB GPU. The question here is whether the loop runs end to end, not whether
the robot succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# MuJoCo reads this once at import, so it must be set before anything imports mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2c.data.frame_cache import FrameCache  # noqa: E402
from p2c.utils.seeding import seed_everything  # noqa: E402

VIDEO_PREFIX = "video."
STATE_PREFIX = "state."


def build_state(obs: dict, state_groups: dict[str, tuple[int, int]], dim: int) -> np.ndarray:
    """Rebuild the flat proprioception vector the policy was trained on.

    ``state_groups`` comes from the cache, which read it from the dataset's own
    ``modality.json``. Each group is written at its recorded offset, so a mismatch in
    naming or width fails loudly instead of shifting every downstream value.
    """
    if not state_groups:
        raise RuntimeError(
            "the frame cache has no state_groups; rebuild it so the state layout is known "
            "rather than guessed"
        )
    out = np.zeros(dim, dtype=np.float32)
    written = np.zeros(dim, dtype=bool)
    for name, (lo, hi) in state_groups.items():
        key = f"{STATE_PREFIX}{name}"
        if key not in obs:
            raise RuntimeError(
                f"environment has no '{key}'. Available state keys: "
                f"{sorted(k for k in obs if k.startswith(STATE_PREFIX))}"
            )
        v = np.asarray(obs[key], dtype=np.float32).ravel()
        if v.size != hi - lo:
            raise RuntimeError(
                f"'{key}' is {v.size}-D but the cache records slice [{lo},{hi}) "
                f"({hi - lo}-D). Cache and environment disagree about the state layout."
            )
        out[lo:hi] = v
        written[lo:hi] = True
    if not written.all():
        raise RuntimeError(
            f"state slots {np.flatnonzero(~written).tolist()} were never filled; "
            f"the cache's state_groups do not cover the full {dim}-D vector"
        )
    return out


def build_images(obs: dict, cameras: list[str], res: int) -> np.ndarray:
    """Stack the requested camera streams into ``[V, 3, res, res]`` in ``[0, 1]``.

    Uses INTER_AREA to match the cache builder: a different interpolation would shift the
    input statistics away from what the encoder saw in training.
    """
    import cv2

    views = []
    for cam in cameras:
        key = f"{VIDEO_PREFIX}{cam}"
        if key not in obs:
            raise RuntimeError(
                f"environment has no '{key}'. Available video keys: "
                f"{sorted(k for k in obs if k.startswith(VIDEO_PREFIX))}"
            )
        img = np.asarray(obs[key])
        if img.shape[0] != res or img.shape[1] != res:
            img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        views.append(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
    return np.stack(views)


def split_action(flat: np.ndarray, action_groups: dict[str, tuple[int, int]]) -> dict:
    """Turn the flat 12-D action back into the environment's Dict action space."""
    out = {}
    for name, (lo, hi) in action_groups.items():
        out[f"action.{name}"] = flat[lo:hi].astype(np.float32)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default=None, help="defaults to the task the cache was built from")
    p.add_argument("--split", default="target", choices=["target", "pretrain"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", type=str, default=None, help="write a rollout mp4 here")
    p.add_argument("--device", default="auto")
    p.add_argument("--cache", default=None, help="override the cache in the checkpoint")
    args = p.parse_args()

    import gymnasium as gym
    import torch

    import robocasa  # noqa: F401  (registers the gym environments)
    from p2c.models.tiny_bc import TinyBCPolicy

    seed_everything(args.seed)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cache = FrameCache(args.cache or cfg["cache"])
    view = ckpt["view_condition"]
    task = args.task or cache.task

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )

    # Cameras in the slot order the policy was trained with. A stochastic condition has no
    # single answer, so fall back to the first choice of each slot and say so.
    cameras, stochastic = [], False
    for slot in view["slots"]:
        if slot.get("fixed"):
            cameras.append(slot["fixed"])
        else:
            cameras.append(slot["choices"][0])
            stochastic = True

    model = TinyBCPolicy(
        action_dim=cache.action_dim,
        state_dim=cache.state_dim,
        num_views=len(cameras),
        n_obs_steps=int(cfg["n_obs_steps"]),
        action_horizon=int(cfg["action_horizon"]),
        encoder=cfg["encoder"],
        fusion=cfg["fusion"],
        feat_dim=int(cfg["feat_dim"]),
        width=int(cfg["width"]),
        hidden_dim=int(cfg["hidden_dim"]),
        input_res=cache.resolution,
        use_state=bool(cfg["use_state"]),
        # Dropout must be reconstructed even though it is inert at eval time: the modules
        # occupy positions in the head's Sequential, so omitting them shifts every
        # subsequent layer index and the state_dict no longer matches.
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    a_mean = np.asarray(ckpt["action_stats"]["mean"], dtype=np.float32)
    a_std = np.asarray(ckpt["action_stats"]["std"], dtype=np.float32)
    train_eps, _ = cache.split_episodes(
        float(cfg["val_fraction"]), int(cfg["split_seed"])
    )
    s_stats = cache.state_stats(train_eps) if cache.state_dim else None

    print(f"checkpoint : {ckpt_path}")
    print(f"condition  : {view['condition']}  cameras={cameras}"
          + ("  (stochastic condition: using the first choice per slot)" if stochastic else ""))
    print(f"task       : {task} (split={args.split})")
    print(f"cache      : {cache.path}  res={cache.resolution}  action_dim={cache.action_dim}")
    print(f"device     : {device}   MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"episodes   : {args.episodes} x {args.max_steps} steps")
    print()

    env = gym.make(f"robocasa/{task}", split=args.split, seed=args.seed)
    writer = None
    if args.video:
        import imageio

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=20)

    successes, lengths, t0 = 0, [], time.time()
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done, steps, success = False, 0, False
            while not done and steps < args.max_steps:
                imgs = build_images(obs, cameras, cache.resolution)
                state = build_state(obs, cache.state_groups, cache.state_dim)
                if s_stats is not None:
                    state = (state - s_stats["mean"]) / s_stats["std"]

                with torch.no_grad():
                    pred = model(
                        torch.from_numpy(imgs)[None].to(device),
                        torch.from_numpy(state)[None].to(device),
                        torch.ones(1, len(cameras), dtype=torch.bool, device=device),
                    ).float().cpu().numpy()[0]

                flat = pred[0] if pred.ndim == 2 else pred     # drop the horizon axis
                flat = flat * a_std + a_mean                   # undo training normalisation
                obs, rew, term, trunc, info = env.step(split_action(flat, cache.action_groups))

                if writer is not None:
                    writer.append_data(np.asarray(obs[f"{VIDEO_PREFIX}{cameras[0]}"]))
                success = success or bool(term) or bool(info.get("is_success", False))
                done = bool(term) or bool(trunc)
                steps += 1

            successes += int(success)
            lengths.append(steps)
            print(f"  episode {ep}: {steps:4d} steps  success={success}")
    finally:
        if writer is not None:
            writer.close()
        env.close()

    rate = successes / max(args.episodes, 1)
    print()
    print(f"success rate : {successes}/{args.episodes} = {rate:.0%}")
    print(f"mean length  : {np.mean(lengths):.1f} steps")
    print(f"wall time    : {time.time() - t0:.0f}s")
    if args.video:
        print(f"video        : {args.video}")
    print()
    print("Note: a ~1.8M-parameter sanity policy trained for a few epochs is not expected "
          "to succeed.\nWhat this verifies is that the closed loop runs: observations "
          "assemble, the policy\nacts, and the simulator steps.")

    out = ckpt_path.parent / "rollout.json"
    with open(out, "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path), "task": task, "split": args.split,
                "condition": view["condition"], "cameras": cameras,
                "stochastic_condition": stochastic,
                "episodes": args.episodes, "max_steps": args.max_steps,
                "successes": successes, "success_rate": rate,
                "mean_length": float(np.mean(lengths)), "seed": args.seed,
            },
            f, indent=2,
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
