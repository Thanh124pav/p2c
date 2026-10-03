#!/usr/bin/env python
"""Evaluate a trained tiny BC checkpoint (the harness contract (docs/harness_contract.md)).

Re-runs validation from a saved checkpoint and reports overall, per-action-group and
per-stage errors. Useful for two things the training loop does not cover:

* evaluating a checkpoint under a **different** view condition than it was trained on,
  which is how you test whether a policy trained on a complementary pair degrades when a
  view is removed at test time;
* recomputing per-sample errors without retraining, e.g. after adding stage annotations.

Usage::

    python scripts/eval_local_bc.py --checkpoint outputs/runs/<run>/best.pt
    python scripts/eval_local_bc.py --checkpoint <path> --views single_primary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2c.data.frame_cache import FrameCache  # noqa: E402
from p2c.data.robocasa_dataset import build_train_val  # noqa: E402
from p2c.models.tiny_bc import TinyBCPolicy  # noqa: E402
from p2c.utils.seeding import seed_everything  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache", type=str, default=None,
                   help="override the cache recorded in the checkpoint")
    p.add_argument("--views", type=str, default=None,
                   help="evaluate under a different view condition than training")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--split", type=str, default="val", choices=["val", "train"])
    p.add_argument("--out", type=str, default=None, help="write a JSON report here")
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    trained_views = cfg["views"]
    views = args.views or trained_views
    cache_path = args.cache or cfg["cache"]

    seed_everything(int(cfg.get("seed", 0)))
    cache = FrameCache(cache_path)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )

    train_ds, val_ds = build_train_val(
        cache, views,
        val_fraction=float(cfg["val_fraction"]),
        split_seed=int(cfg["split_seed"]),
        seed=int(cfg["seed"]),
        camera_dropout=cfg.get("camera_dropout"),
        n_obs_steps=int(cfg["n_obs_steps"]),
        action_horizon=int(cfg["action_horizon"]),
    )
    ds = val_ds if args.split == "val" else train_ds

    model = TinyBCPolicy(
        action_dim=cache.action_dim,
        state_dim=cache.state_dim,
        num_views=ds.num_views,
        n_obs_steps=int(cfg["n_obs_steps"]),
        action_horizon=int(cfg["action_horizon"]),
        encoder=cfg["encoder"],
        fusion=cfg["fusion"],
        feat_dim=int(cfg["feat_dim"]),
        width=int(cfg["width"]),
        hidden_dim=int(cfg["hidden_dim"]),
        input_res=cache.resolution,
        use_state=bool(cfg["use_state"]),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    print(f"checkpoint     : {ckpt_path}")
    print(f"trained views  : {trained_views}")
    print(f"evaluated views: {views}"
          + ("   (cross-condition evaluation)" if views != trained_views else ""))
    print(f"cache          : {cache.path}")
    print(f"split          : {args.split} ({len(ds)} samples, "
          f"{len(ds.episodes)} episodes)")
    print(f"device         : {device}")
    if missing or unexpected:
        # With a shared encoder and view-count-independent fusion the weights transfer
        # across view counts, so this is expected to be empty; report it if not.
        print(f"state_dict     : missing={list(missing)} unexpected={list(unexpected)}")
    print()

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    from p2c.models.tiny_bc import action_metrics, per_sample_squared_error

    sums: dict[str, float] = {}
    nb = 0
    eps, frs, stages, errs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["images"].to(device),
                batch["state"].to(device),
                batch["view_kept"].to(device),
            ).float()
            target = batch["action"].to(device)
            for k, v in action_metrics(pred, target, cache.action_groups).items():
                sums[k] = sums.get(k, 0.0) + v
            nb += 1
            errs.append(per_sample_squared_error(pred, target).cpu().numpy())
            eps.append(batch["episode"].numpy())
            frs.append(batch["frame"].numpy())
            stages.append(batch["stage"].numpy())

    agg = {k: v / max(nb, 1) for k, v in sums.items()}
    stage_arr = np.concatenate(stages)
    err_arr = np.concatenate(errs)

    print(f"{'metric':34s} {'value':>12s}")
    for k in sorted(agg):
        print(f"  {k:32s} {agg[k]:12.6f}")

    per_stage = {}
    if (stage_arr >= 0).any():
        print()
        print(f"{'per-stage mse':34s} {'value':>12s} {'frames':>8s}")
        for sid in sorted(set(int(s) for s in np.unique(stage_arr) if s >= 0)):
            m = stage_arr == sid
            nm = cache.stage_name(sid)
            per_stage[nm] = {"mse": float(err_arr[m].mean()), "n": int(m.sum())}
            print(f"  {nm:32s} {per_stage[nm]['mse']:12.6f} {per_stage[nm]['n']:8d}")

    report = {
        "checkpoint": str(ckpt_path),
        "trained_views": trained_views,
        "evaluated_views": views,
        "cross_condition": views != trained_views,
        "cache": str(cache.path),
        "split": args.split,
        "num_samples": len(ds),
        "metrics": agg,
        "per_stage": per_stage,
        "view_condition": ds.describe(),
    }
    out = Path(args.out) if args.out else ckpt_path.parent / f"eval_{views.replace('+','-')}_{args.split}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
