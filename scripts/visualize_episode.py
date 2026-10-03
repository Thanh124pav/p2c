#!/usr/bin/env python
"""Visualise one episode across all camera views (PLAN.md Stage 2).

Produces a side-by-side strip of every synchronised view so the acceptance criterion
"all camera frames for timestep t correspond to the same timestep" can be checked by eye,
not just by comparing frame counts. Stage/subtask labels are burnt in when the dataset
has them.

Two output modes:

``--mode video``  one mp4 with all views tiled horizontally, labelled per frame.
``--mode grid``   a PNG contact sheet sampling N timesteps down the rows, views across
                  the columns — easier to eyeball alignment than scrubbing a video.

Usage::

    python scripts/visualize_episode.py --task NavigateKitchen --episode 0
    python scripts/visualize_episode.py --cache outputs/cache/... --episode 0 --mode grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2c.data.lerobot_meta import DatasetMeta  # noqa: E402
from p2c.utils.paths import resolve_dataset_path  # noqa: E402

LABEL_H = 22


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a caption bar above an RGB image."""
    import cv2

    h, w = img.shape[:2]
    bar = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
    scale = max(0.32, min(0.5, w / 420.0))
    cv2.putText(
        bar, text[: int(w / (7 * scale))], (3, LABEL_H - 7),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([bar, img])


def _read_views_from_dataset(meta: DatasetMeta, ep: int) -> dict[str, np.ndarray]:
    """Decode every camera's mp4 for one episode. Returns camera -> [T, H, W, 3] RGB."""
    import cv2

    out = {}
    for key in meta.image_keys:
        cam = key.split("observation.images.")[-1]
        path = meta.video_path(ep, key)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr[:, :, ::-1].copy())
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        out[cam] = np.stack(frames)
    return out


def _read_views_from_cache(cache_path: Path, ep: int) -> tuple[dict, dict]:
    from p2c.data.frame_cache import FrameCache

    cache = FrameCache(cache_path)
    span = cache.span_of_episode(ep)
    views = {
        c: np.asarray(cache._images[c][span.start : span.stop]) for c in cache.cameras
    }
    extras = {
        "stages": (
            np.asarray(cache.stage[span.start : span.stop])
            if cache.has_stages else None
        ),
        "stage_names": cache.stage_names,
        "actions": np.asarray(cache.actions[span.start : span.stop]),
        "fps": cache.index.get("fps", 20),
    }
    return views, extras


def _stage_text(extras: dict, t: int) -> str:
    st = extras.get("stages")
    if st is None:
        return ""
    sid = int(st[t])
    names = extras.get("stage_names") or {}
    return names.get(str(sid), str(sid))


def write_video(views: dict, extras: dict, out: Path, fps: float) -> Path:
    import cv2

    cams = list(views)
    T = min(v.shape[0] for v in views.values())
    tiles = [_label(views[c][0], c) for c in cams]
    H, W = tiles[0].shape[0], sum(t.shape[1] for t in tiles)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + LABEL_H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {out}")

    for t in range(T):
        row = np.hstack([_label(views[c][t], c) for c in cams])
        stage = _stage_text(extras, t)
        caption = f"t={t}/{T-1}" + (f"   stage={stage}" if stage else "")
        frame = _label(row, caption)
        writer.write(frame[:, :, ::-1])  # RGB -> BGR
    writer.release()
    return out


def write_grid(views: dict, extras: dict, out: Path, n_rows: int) -> Path:
    import cv2

    cams = list(views)
    T = min(v.shape[0] for v in views.values())
    steps = np.linspace(0, T - 1, min(n_rows, T)).round().astype(int)

    rows = []
    for t in steps:
        stage = _stage_text(extras, t)
        tag = f"t={t}" + (f" {stage}" if stage else "")
        tiles = [_label(views[c][t], f"{c}  {tag}" if i == 0 else c)
                 for i, c in enumerate(cams)]
        rows.append(np.hstack(tiles))
    grid = np.vstack(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid[:, :, ::-1])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--split", type=str, default="target", choices=["target", "pretrain"])
    p.add_argument("--cache", type=str, default=None,
                   help="read from a frame cache instead of the raw dataset")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--mode", type=str, default="grid", choices=["grid", "video", "both"])
    p.add_argument("--rows", type=int, default=6, help="timesteps sampled in grid mode")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    if args.cache:
        views, extras = _read_views_from_cache(Path(args.cache), args.episode)
        label = Path(args.cache).name
        fps = float(extras.get("fps") or 20)
    else:
        root = resolve_dataset_path(
            dataset=args.dataset, task=args.task, split=args.split
        )
        meta = DatasetMeta.load(root)
        views = _read_views_from_dataset(meta, args.episode)
        # Raw-dataset mode shows pixels only; per-frame stage labels live in the parquet
        # and are surfaced via the cache (use --cache to see them burnt in).
        extras = {"stages": None, "stage_names": None}
        label = root.parent.parent.name
        fps = float(meta.fps or 20)

    T = min(v.shape[0] for v in views.values())
    counts = {c: int(v.shape[0]) for c, v in views.items()}
    print(f"episode {args.episode}: {len(views)} views, frame counts {counts}")
    if len(set(counts.values())) != 1:
        print("WARNING: views have different frame counts; they are NOT synchronised",
              file=sys.stderr)
    print(f"resolution: {views[list(views)[0]].shape[1:3]}  frames used: {T}")

    out_dir = Path(args.out) if args.out else root_dir / "outputs" / "viz"
    base = f"{label}_ep{args.episode}"

    if args.mode in ("grid", "both"):
        p_grid = write_grid(views, extras, out_dir / f"{base}_grid.png", args.rows)
        print(f"wrote {p_grid}")
    if args.mode in ("video", "both"):
        p_vid = write_video(views, extras, out_dir / f"{base}.mp4", fps)
        print(f"wrote {p_vid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
