#!/usr/bin/env python
"""Decode a RoboCasa365 LeRobot dataset once into a compact frame cache.

Why this exists
---------------
The datasets ship 256x256 mp4 per camera. Random-access decoding during training means
seeking inside mp4 for every sample, which is slow and, on a 5.8 GB RAM machine, awkward
to buffer. More importantly for P2C, SETUP.md section 7 requires *identical* image
preprocessing across every camera ablation. Decoding once into a shared cache makes that
a structural guarantee rather than something to remember: all view conditions then read
the same uint8 pixels at the same global frame index.

The cache also makes camera synchronisation true by construction. Every camera's frames
are written to the same global frame index, so view condition "primary+wrist" at index i
cannot accidentally mix timesteps.

Layout written to ``--out``::

    index.json              schema, cameras, resolution, episode table, provenance
    images_<camera>.npy     uint8  [N, H, W, 3]   (memmap-friendly, one per camera)
    actions.npy             float32[N, A]
    states.npy              float32[N, S]
    episode_index.npy       int32  [N]            episode each frame belongs to
    frame_index.npy         int32  [N]            index within its episode
    stage.npy               int32  [N]            per-frame stage id, if available
    stage_names.json        stage id -> name,     if available

Usage::

    python scripts/build_frame_cache.py --task NavigateKitchen --num-episodes 40 --res 84
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2c.data.lerobot_meta import DatasetMeta  # noqa: E402
from p2c.utils.paths import resolve_dataset_path, task_of  # noqa: E402


def _git_hash(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _episode_num_rows(meta: DatasetMeta, ep: int) -> int:
    """Row count for one episode, from parquet metadata only.

    Reading the footer rather than the data matters here: this machine has 5.8 GB of RAM
    and a composite task has ~175k frames, so materialising every episode's columns just
    to learn their lengths would waste hundreds of megabytes for nothing.
    """
    import pyarrow.parquet as pq

    return pq.ParquetFile(str(meta.parquet_path(ep))).metadata.num_rows


def _read_episode_table(meta: DatasetMeta, ep: int):
    """Read one episode's parquet. Returns (num_rows, dict of column -> list)."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(str(meta.parquet_path(ep)))
    cols = {name: tbl.column(name).to_pylist() for name in tbl.schema.names}
    return tbl.num_rows, cols


def _to_2d(col: list, n: int) -> np.ndarray:
    """Normalise a parquet column into float32 [n, D]."""
    arr = np.asarray(col, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    assert arr.shape[0] == n, f"column length {arr.shape[0]} != {n}"
    return arr


def _decode_video(path: Path, res: int, expected: int) -> np.ndarray:
    """Sequentially decode an mp4 to uint8 [T, res, res, 3] RGB.

    Sequential reads only: no seeking, which keeps this fast and memory-flat.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[0] != res or bgr.shape[1] != res:
            # INTER_AREA is the correct choice for downscaling; it avoids the aliasing
            # that INTER_LINEAR introduces, which would otherwise differ across cameras.
            bgr = cv2.resize(bgr, (res, res), interpolation=cv2.INTER_AREA)
        frames.append(bgr[:, :, ::-1])  # BGR -> RGB
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}")
    out = np.stack(frames).astype(np.uint8)
    if expected is not None and out.shape[0] != expected:
        raise RuntimeError(
            f"{path.name}: decoded {out.shape[0]} frames but parquet has {expected}. "
            f"Refusing to build a cache with misaligned views."
        )
    return out


def _resolve_stage(cols: dict, stage_keys: list[str]) -> tuple[list | None, str | None]:
    """Pick the per-frame stage column, preferring an explicit 'stage' label."""
    if not stage_keys:
        return None, None
    ranked = sorted(
        stage_keys,
        key=lambda k: (
            0 if "stage" in k.lower() else 1 if "subtask" in k.lower() else 2,
            k,
        ),
    )
    for key in ranked:
        if key in cols:
            return cols[key], key
    return None, None


def build(
    root: Path,
    out: Path,
    num_episodes: int | None,
    res: int,
    episode_stride: int,
    seed: int,
) -> dict:
    meta = DatasetMeta.load(root)
    cams = meta.camera_names
    if not cams:
        raise RuntimeError("no cameras discovered; cannot build cache")

    all_eps = sorted(int(e["episode_index"]) for e in meta.episodes)
    eps = all_eps[::episode_stride]
    if num_episodes is not None:
        eps = eps[:num_episodes]
    if not eps:
        raise RuntimeError("no episodes selected")

    print(f"dataset   : {root}")
    print(f"cameras   : {cams}")
    print(f"episodes  : {len(eps)} of {len(all_eps)} (stride={episode_stride})")
    print(f"resolution: {res}x{res}")

    # ---- pass 1: frame layout from parquet footers only (cheap in RAM) ----
    stage_key_used = None
    lengths = {ep: _episode_num_rows(meta, ep) for ep in eps}
    total = sum(lengths.values())
    print(f"total frames: {total}")

    act_key = meta.action_key
    if act_key is None:
        raise RuntimeError("no action column found in info.json features")
    state_keys = meta.state_keys

    # Probe the first episode only, to fix the action and state widths.
    n0, cols0 = _read_episode_table(meta, eps[0])
    action_dim = _to_2d(cols0[act_key], n0).shape[1]
    state_dim = sum(
        _to_2d(cols0[k], n0).shape[1] for k in state_keys if k in cols0
    )
    del cols0

    out.mkdir(parents=True, exist_ok=True)
    actions = np.lib.format.open_memmap(
        out / "actions.npy", mode="w+", dtype=np.float32, shape=(total, action_dim)
    )
    states = np.lib.format.open_memmap(
        out / "states.npy", mode="w+", dtype=np.float32, shape=(total, max(state_dim, 1))
    )
    ep_idx_arr = np.lib.format.open_memmap(
        out / "episode_index.npy", mode="w+", dtype=np.int32, shape=(total,)
    )
    fr_idx_arr = np.lib.format.open_memmap(
        out / "frame_index.npy", mode="w+", dtype=np.int32, shape=(total,)
    )
    # Which language instruction each frame belongs to. Recorded because a task whose
    # goal is specified only in language (NavigateKitchen has 14 instructions over the
    # same kitchen) is not a function of the image, so a vision-only policy cannot do
    # better than the conditional mean there. Keeping this lets a later analysis check
    # or control for it instead of mistaking ambiguity for a negative P2C result.
    task_idx_arr = np.lib.format.open_memmap(
        out / "task_index.npy", mode="w+", dtype=np.int32, shape=(total,)
    )
    img_mm = {
        c: np.lib.format.open_memmap(
            out / f"images_{c}.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total, res, res, 3),
        )
        for c in cams
    }

    stage_vals: list = []
    episode_table = []
    cursor = 0
    t0 = time.time()

    for i, ep in enumerate(eps):
        # Pass 2: read, write and release one episode at a time, so peak RAM is one
        # episode's columns plus one episode's decoded frames.
        n, cols = _read_episode_table(meta, ep)
        if n != lengths[ep]:
            raise RuntimeError(
                f"episode {ep}: footer said {lengths[ep]} rows, read {n}"
            )
        sl = slice(cursor, cursor + n)

        actions[sl] = _to_2d(cols[act_key], n)
        if state_dim:
            parts = [_to_2d(cols[k], n) for k in state_keys if k in cols]
            states[sl] = np.concatenate(parts, axis=1)
        ep_idx_arr[sl] = ep
        fr_idx_arr[sl] = np.arange(n, dtype=np.int32)
        if "task_index" in cols:
            task_idx_arr[sl] = np.asarray(cols["task_index"], dtype=np.int32)
        else:
            task_idx_arr[sl] = -1

        svals, skey = _resolve_stage(cols, meta.stage_annotation_keys)
        if svals is not None:
            stage_key_used = skey
            stage_vals.extend(svals)
        elif stage_key_used is not None:
            raise RuntimeError(f"episode {ep} missing stage column {stage_key_used}")

        for c in cams:
            key = f"observation.images.{c}"
            img_mm[c][sl] = _decode_video(meta.video_path(ep, key), res, expected=n)

        episode_table.append({"episode_index": ep, "start": cursor, "length": n})
        cursor += n
        if (i + 1) % 5 == 0 or i + 1 == len(eps):
            el = time.time() - t0
            print(
                f"  [{i+1:4d}/{len(eps)}] frames={cursor}/{total} "
                f"{el:.0f}s ({cursor/max(el,1e-9):.0f} fr/s)",
                flush=True,
            )

    for mm in list(img_mm.values()) + [actions, states, ep_idx_arr, fr_idx_arr,
                                       task_idx_arr]:
        mm.flush()

    # ---- stages ----
    stage_names = None
    if stage_vals:
        if len(stage_vals) != total:
            raise RuntimeError(f"stage column has {len(stage_vals)} rows, expected {total}")

        # The stage column holds int indices into meta/tasks.jsonl, so resolve them to
        # readable labels (11/12/13/15 -> done/place/pick/navigate). Without this the
        # per-stage table of SETUP.md section 11 is a list of bare numbers.
        vocab = meta.task_vocabulary()

        def label(v) -> str:
            try:
                return vocab.get(int(v), str(v))
            except (TypeError, ValueError):
                return str(v)

        # Order stage ids by the *raw* index, not by label spelling: the raw index is
        # stable across datasets and rebuilds, whereas sorting by label would silently
        # renumber every stage if a vocabulary entry were reworded.
        def raw_key(v):
            try:
                return (0, int(v))
            except (TypeError, ValueError):
                return (1, str(v))

        raw_uniq = sorted({v for v in stage_vals}, key=raw_key)
        raw_to_id = {v: i for i, v in enumerate(raw_uniq)}
        arr = np.array([raw_to_id[v] for v in stage_vals], dtype=np.int32)
        np.save(out / "stage.npy", arr)
        stage_names = {str(i): label(v) for v, i in raw_to_id.items()}
        uniq = [stage_names[str(i)] for i in range(len(raw_uniq))]
        with open(out / "stage_names.json", "w") as f:
            json.dump(stage_names, f, indent=2)
        raw = sorted({str(v) for v in stage_vals})
        print(f"stages    : {len(uniq)} distinct -> {uniq}")
        print(f"            (raw indices {raw} resolved via meta/tasks.jsonl)")
    else:
        print("stages    : none (atomic task; per-stage analysis unavailable)")

    index = {
        "schema_version": 2,
        "source_dataset": str(root),
        "task": task_of(root),
        "cameras": cams,
        "image_keys": meta.image_keys,
        "resolution": res,
        "num_frames": total,
        "num_episodes": len(eps),
        "episode_stride": episode_stride,
        "action_key": act_key,
        "action_dim": action_dim,
        "action_groups": {k: list(v) for k, v in meta.action_groups.items()},
        "state_keys": state_keys,
        "state_dim": state_dim,
        "state_groups": {k: list(v) for k, v in meta.state_groups.items()},
        "fps": meta.fps,
        "episodes": episode_table,
        "num_distinct_instructions": int(len(set(np.asarray(task_idx_arr).tolist()))),
        "stage_key": stage_key_used,
        "stage_names": stage_names,
        "selection_seed": seed,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "p2c_git_hash": _git_hash(Path(__file__).resolve().parents[1]),
        "resize_interpolation": "INTER_AREA",
        "color_order": "RGB",
    }
    with open(out / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    nbytes = sum(p.stat().st_size for p in out.glob("*.npy"))
    print(f"\ncache     : {out}")
    print(f"size      : {nbytes/1e9:.2f} GB")
    return index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--split", type=str, default="target", choices=["target", "pretrain"])
    p.add_argument("--out", type=str, default=None, help="cache dir (auto if omitted)")
    p.add_argument("--num-episodes", type=int, default=40)
    p.add_argument("--all-episodes", action="store_true")
    p.add_argument("--episode-stride", type=int, default=1)
    p.add_argument("--res", type=int, default=84, help="square output resolution")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="rebuild if cache exists")
    args = p.parse_args()

    root = resolve_dataset_path(dataset=args.dataset, task=args.task, split=args.split)
    task = task_of(root)
    n_eps = None if args.all_episodes else args.num_episodes

    if args.out:
        out = Path(args.out)
    else:
        tag = f"{task}_{args.split}_r{args.res}_e{'all' if n_eps is None else n_eps}"
        if args.episode_stride != 1:
            tag += f"_s{args.episode_stride}"
        out = Path(__file__).resolve().parents[1] / "outputs" / "cache" / tag

    if (out / "index.json").exists() and not args.force:
        print(f"cache already exists at {out} (use --force to rebuild)")
        return 0

    build(root, out, n_eps, args.res, args.episode_stride, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
