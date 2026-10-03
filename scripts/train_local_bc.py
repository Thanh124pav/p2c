#!/usr/bin/env python
"""Train the tiny BC baseline for one camera condition (PLAN.md Stage 2).

This is the fast falsification test, not a paper result. One invocation trains exactly one
view condition; ``scripts/run_view_ablation.sh`` sweeps them with everything else held
fixed.

What the run writes (harness contract (docs/harness_contract.md) C12): ``meta.json``
with git hash, config, dataset
identity, camera subset, parameter count and resolution; ``metrics.jsonl`` / ``.csv`` with
the loss history; ``val_per_sample.npz`` with per-frame validation errors keyed by
(episode, frame) — the raw material for the per-stage breakdown and the complementarity
metric; ``best.pt``; and ``console.log``.

Usage::

    python scripts/train_local_bc.py --config configs/local_debug.yaml --views primary
    python scripts/train_local_bc.py --config configs/local_debug.yaml --views primary,wrist
    python scripts/train_local_bc.py --config configs/local_debug.yaml --views all_views
    python scripts/train_local_bc.py --config configs/overfit.yaml --views primary --overfit 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2c.data.frame_cache import FrameCache  # noqa: E402
from p2c.data.robocasa_dataset import build_train_val  # noqa: E402
from p2c.data.transforms import augment_batch  # noqa: E402
from p2c.models.tiny_bc import (  # noqa: E402
    TinyBCPolicy,
    action_metrics,
    bc_loss,
    per_sample_squared_error,
)
from p2c.utils.seeding import (  # noqa: E402
    nondeterminism_report,
    seed_everything,
    worker_init_fn,
)
from p2c.utils.tracking import RunLogger, capture_console, run_name  # noqa: E402

DEFAULTS = {
    "cache": None,
    "views": "single_primary",
    "seed": 0,
    "split_seed": 0,
    "val_fraction": 0.2,
    "epochs": 10,
    "max_steps": None,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "loss": "l2",
    "encoder": "tiny_cnn",
    "fusion": "mean",
    "feat_dim": 256,
    "width": 32,
    "hidden_dim": 512,
    "dropout": 0.0,
    "use_state": True,
    "n_obs_steps": 1,
    "action_horizon": 1,
    "camera_dropout": None,
    "random_scope": "sample",
    "num_workers": 2,
    "augment": True,
    "aug_pad": 4,
    "amp": False,
    "device": "auto",
    "eval_every": 1,
    "grad_clip": 1.0,
    "output_root": "outputs/runs",
    "policy": "bc",
    "wandb": False,
    "wandb_project": "p2c",
    "overfit": None,
}


def load_config(path: str | None, overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        import yaml

        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def pick_device(spec: str):
    import torch

    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, device, action_groups, amp: bool) -> tuple[dict, dict]:
    """Validation pass. Returns (aggregate metrics, per-sample arrays).

    Per-sample errors are kept with their (episode, frame, stage) keys so the same
    validation frames can be compared across view conditions later.
    """
    import torch

    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    eps, frs, stages, errs = [], [], [], []

    use_amp = amp and device.type == "cuda"

    def amp_ctx():
        # A fresh context per batch rather than re-entering one instance.
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp else _NullCtx()
        )

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            mask = batch["view_kept"].to(device, non_blocking=True)
            with amp_ctx():
                pred = model(images, state, mask)
            pred = pred.float()

            m = action_metrics(pred, target, action_groups)
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1

            errs.append(per_sample_squared_error(pred, target).cpu().numpy())
            eps.append(batch["episode"].numpy())
            frs.append(batch["frame"].numpy())
            stages.append(batch["stage"].numpy())

    agg = {k: v / max(n_batches, 1) for k, v in sums.items()}
    per_sample = {
        "episode": np.concatenate(eps) if eps else np.zeros(0, np.int32),
        "frame": np.concatenate(frs) if frs else np.zeros(0, np.int32),
        "stage": np.concatenate(stages) if stages else np.zeros(0, np.int32),
        "sq_error": np.concatenate(errs) if errs else np.zeros(0, np.float32),
    }
    model.train(was_training)  # restore, rather than assuming we came from train mode
    return agg, per_sample


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--cache", type=str, default=None, help="frame cache dir")
    p.add_argument("--views", type=str, default=None, help="view condition name or list")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--fusion", type=str, default=None, choices=["mean", "attn", "concat"])
    p.add_argument("--encoder", type=str, default=None)
    p.add_argument("--camera-dropout", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--no-augment", dest="augment", action="store_false", default=None,
                   help="disable training-time random-shift augmentation")
    p.add_argument("--aug-pad", type=int, default=None, help="random shift in pixels")
    p.add_argument("--overfit", type=int, default=None,
                   help="train on N samples only (PLAN.md Stage 2 overfit test)")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--tag", type=str, default=None, help="suffix for the run name")
    p.add_argument("--wandb", action="store_true", default=None)
    args = p.parse_args()

    cfg = load_config(args.config, {
        "cache": args.cache, "views": args.views, "seed": args.seed,
        "epochs": args.epochs, "max_steps": args.max_steps,
        "batch_size": args.batch_size, "lr": args.lr, "fusion": args.fusion,
        "encoder": args.encoder, "camera_dropout": args.camera_dropout,
        "num_workers": args.num_workers, "device": args.device, "amp": args.amp,
        "overfit": args.overfit, "output_root": args.output_root, "wandb": args.wandb,
        "augment": args.augment, "aug_pad": args.aug_pad,
    })
    if not cfg["cache"]:
        print("error: --cache (or cache: in the config) is required", file=sys.stderr)
        return 2

    import torch
    from torch.utils.data import DataLoader, Subset

    seed_report = seed_everything(int(cfg["seed"]))
    cache = FrameCache(cfg["cache"])
    device = pick_device(cfg["device"])

    train_ds, val_ds = build_train_val(
        cache,
        cfg["views"],
        val_fraction=float(cfg["val_fraction"]),
        split_seed=int(cfg["split_seed"]),
        seed=int(cfg["seed"]),
        camera_dropout=cfg["camera_dropout"],
        n_obs_steps=int(cfg["n_obs_steps"]),
        action_horizon=int(cfg["action_horizon"]),
    )

    # Overfit mode: a contiguous slice, so the same frames are used for
    # training and validation and the loss must approach zero if the model can learn.
    overfit_n = cfg["overfit"]
    if overfit_n:
        idx = list(range(min(int(overfit_n), len(train_ds))))
        train_sub = Subset(train_ds, idx)
        val_sub = Subset(train_ds, idx)
    else:
        train_sub, val_sub = train_ds, val_ds

    name = run_name(cache.task, cfg["policy"], cfg["views"], int(cfg["seed"]))
    if overfit_n:
        name += f"_overfit{overfit_n}"
    if args.tag:
        name += f"_{args.tag}"
    logger = RunLogger(cfg["output_root"], name, config=cfg)
    restore = capture_console(logger.dir)

    try:
        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_sub, batch_size=int(cfg["batch_size"]), shuffle=True,
            num_workers=int(cfg["num_workers"]), pin_memory=pin, drop_last=False,
            worker_init_fn=worker_init_fn, persistent_workers=int(cfg["num_workers"]) > 0,
        )
        val_loader = DataLoader(
            val_sub, batch_size=int(cfg["batch_size"]), shuffle=False,
            num_workers=int(cfg["num_workers"]), pin_memory=pin,
            worker_init_fn=worker_init_fn, persistent_workers=int(cfg["num_workers"]) > 0,
        )

        model = TinyBCPolicy(
            action_dim=cache.action_dim,
            state_dim=cache.state_dim,
            num_views=train_ds.num_views,
            n_obs_steps=int(cfg["n_obs_steps"]),
            action_horizon=int(cfg["action_horizon"]),
            encoder=cfg["encoder"],
            fusion=cfg["fusion"],
            feat_dim=int(cfg["feat_dim"]),
            width=int(cfg["width"]),
            hidden_dim=int(cfg["hidden_dim"]),
            input_res=cache.resolution,
            use_state=bool(cfg["use_state"]),
            dropout=float(cfg["dropout"]),
        ).to(device)

        cap = model.capacity_report()
        view_desc = train_ds.describe()
        usage = train_ds.camera_usage()

        print("=" * 78)
        print(f"RUN       {name}")
        print("=" * 78)
        print(cache.summary())
        print(f"condition : {train_ds.condition.describe()}")
        print(f"cameras   : realised usage over train split -> {usage}")
        print(f"samples   : {len(train_sub)} train / {len(val_sub)} val")
        print(f"device    : {device}")
        print(f"params    : {cap['total_trainable']:,} trainable "
              f"(encoder {cap['params_encoder']:,}, fusion {cap['params_fusion']:,}, "
              f"head {cap['params_head']:,})")
        print(f"capacity  : view_count_independent={cap['view_count_independent']} "
              f"(fusion={cap['fusion_name']})")
        if not cap["view_count_independent"]:
            print("  WARNING: this fusion scales parameters with the view count, so this "
                  "run cannot satisfy harness contract C2.")
        print(f"actions   : dim={cache.action_dim} groups={list(cache.action_groups)}")

        # Reference line: predicting the training-set mean action. A run that does not
        # clear this has learned nothing, and comparing two such runs is meaningless.
        trivial = cache.trivial_baseline_mse(train_ds.episodes, val_ds.episodes)
        n_instr = cache.num_instructions
        print(f"baseline  : predict-the-mean val_mse = {trivial:.5f} "
              f"(any condition near or above this has learned nothing)")
        print(f"instructions in cache: {n_instr}"
              + ("  <- WARNING: the goal varies by language. If it is not recoverable "
                 "from the image, a vision-only policy cannot beat the baseline and this "
                 "ablation cannot test view sufficiency." if n_instr > 1 else ""))
        print()

        logger.update_meta(
            dataset={
                "cache": str(cache.path),
                "source_dataset": cache.index.get("source_dataset"),
                "task": cache.task,
                "resolution": cache.resolution,
                "num_episodes_cached": len(cache.spans),
                "num_frames_cached": cache.num_frames,
                "action_dim": cache.action_dim,
                "state_dim": cache.state_dim,
                "action_groups": {k: list(v) for k, v in cache.action_groups.items()},
                "stage_key": cache.stage_key,
                "stage_names": cache.stage_names,
                "num_instructions": n_instr,
            },
            trivial_baseline_mse=trivial,
            split={
                "val_fraction": cfg["val_fraction"],
                "split_seed": cfg["split_seed"],
                "train_episodes": train_ds.episodes,
                "val_episodes": val_ds.episodes,
                "num_train_samples": len(train_sub),
                "num_val_samples": len(val_sub),
            },
            view_condition=view_desc,
            realised_camera_usage=usage,
            capacity=cap,
            seeding=seed_report,
            nondeterminism=nondeterminism_report(bool(cfg["amp"])),
            device=str(device),
        )
        logger.maybe_init_wandb(cfg["wandb_project"], bool(cfg["wandb"]))

        opt = torch.optim.AdamW(
            model.parameters(), lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        use_amp = bool(cfg["amp"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best = {"val_mse": float("inf"), "step": -1, "epoch": -1}
        step = 0
        max_steps = cfg["max_steps"]
        t0 = time.time()
        stop = False

        for epoch in range(int(cfg["epochs"])):
            run_loss, nb = 0.0, 0
            for batch in train_loader:
                images = batch["images"].to(device, non_blocking=True)
                state = batch["state"].to(device, non_blocking=True)
                target = batch["action"].to(device, non_blocking=True)
                mask = batch["view_kept"].to(device, non_blocking=True)

                # Training-only augmentation, identical in kind and magnitude for every
                # view condition, so it regularises all arms equally. Validation is never
                # augmented, which keeps per-sample errors comparable across conditions.
                images = augment_batch(images, int(cfg["aug_pad"]), bool(cfg["augment"]))

                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp else _NullCtx()
                ):
                    pred = model(images, state, mask)
                    loss = bc_loss(pred.float(), target, cfg["loss"])

                opt.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if cfg["grad_clip"]:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg["grad_clip"])
                        )
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg["grad_clip"]:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg["grad_clip"])
                        )
                    opt.step()

                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

                run_loss += loss.item()
                nb += 1
                step += 1
                if max_steps and step >= int(max_steps):
                    stop = True
                    break

            train_loss = run_loss / max(nb, 1)

            if (epoch + 1) % int(cfg["eval_every"]) == 0 or stop or epoch + 1 == int(cfg["epochs"]):
                val, per_sample = evaluate(
                    model, val_loader, device, cache.action_groups, use_amp
                )
                row = {"epoch": epoch + 1, "train_loss": train_loss}
                row.update({f"val_{k}": v for k, v in val.items()})
                logger.log(step, **row)
                print(
                    f"epoch {epoch+1:3d}/{cfg['epochs']}  step {step:6d}  "
                    f"train {train_loss:.5f}  val_mse {val['mse']:.5f}  "
                    f"val_l1 {val['l1']:.5f}  ({time.time()-t0:.0f}s)",
                    flush=True,
                )
                if val["mse"] < best["val_mse"]:
                    best = {
                        "val_mse": val["mse"], "val_l1": val["l1"],
                        "step": step, "epoch": epoch + 1,
                        "metrics": val,
                    }
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": cfg,
                            "capacity": cap,
                            "view_condition": view_desc,
                            "epoch": epoch + 1,
                            "step": step,
                            "val": val,
                            "action_stats": {
                                "mean": train_ds.action_stats["mean"],
                                "std": train_ds.action_stats["std"],
                            },
                        },
                        logger.dir / "best.pt",
                    )
                    np.savez_compressed(
                        logger.dir / "val_per_sample.npz",
                        **per_sample,
                        stage_names=json.dumps(cache.stage_names or {}),
                        condition=cfg["views"],
                        epoch=epoch + 1,
                    )
            else:
                logger.log(step, epoch=epoch + 1, train_loss=train_loss)

            if stop:
                break

        result = {
            "best_val_mse": best["val_mse"],
            "best_val_l1": best.get("val_l1"),
            "best_epoch": best["epoch"],
            "best_step": best["step"],
            "total_steps": step,
            "training_steps": step,
            "per_group": {
                k: v for k, v in (best.get("metrics") or {}).items() if "/" in k
            },
            "condition": cfg["views"],
            "num_views": train_ds.num_views,
            "params_trainable": cap["total_trainable"],
            "task": cache.task,
            "seed": cfg["seed"],
            "overfit_n": overfit_n,
            "trivial_baseline_mse": trivial,
            "beats_trivial_baseline": bool(best["val_mse"] < trivial),
        }
        logger.finish(result)
        print()
        print(f"best val_mse {best['val_mse']:.6f} at epoch {best['epoch']}")
        print(f"run dir      {logger.dir}")
        return 0
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(main())
