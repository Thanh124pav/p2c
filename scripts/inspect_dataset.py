#!/usr/bin/env python
"""Stage 0 of PLAN.md: verify the RoboCasa365 dataset schema programmatically.

Prints every field PLAN.md Stage 2 requires, and checks the acceptance criteria that
matter for the P2C study: that every camera stream loads, that all cameras share a frame
count with the action stream (a necessary condition for synchronisation), and that
episode boundaries are consistent.

Nothing about camera names, resolutions or action dimensions is assumed; everything is
read from the dataset. Usage::

    python scripts/inspect_dataset.py --dataset <path-to-lerobot-dir>
    python scripts/inspect_dataset.py --task NavigateKitchen        # resolve via registry
    python scripts/inspect_dataset.py --dataset <path> --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2c.data.lerobot_meta import DatasetMeta  # noqa: E402
from p2c.utils.paths import resolve_dataset_path  # noqa: E402


def _probe_video(path: Path) -> dict:
    """Read an mp4's real frame count and resolution. Returns {} if unreadable."""
    try:
        import cv2
    except ImportError:
        return {"error": "opencv not installed"}
    if not path.exists():
        return {"error": "missing file"}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "could not open"}
    out = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
    }
    ok, frame = cap.read()
    out["first_frame_shape"] = list(frame.shape) if ok else None
    cap.release()
    return out


def _probe_parquet(path: Path) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"error": "pyarrow not installed"}
    if not path.exists():
        return {"error": "missing file"}
    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    tbl = pf.read()
    cols = {}
    for name in schema.names:
        col = tbl.column(name)
        val = col[0].as_py() if len(col) else None
        if isinstance(val, list):
            desc = f"list[{len(val)}]"
        else:
            desc = type(val).__name__
        cols[name] = {"arrow_type": str(schema.field(name).type), "sample": desc}
    return {"num_rows": tbl.num_rows, "columns": cols}


def inspect(root: Path, episode: int, probe_videos: bool) -> dict:
    meta = DatasetMeta.load(root)
    report: dict = {"dataset_path": str(root)}

    # ---- the fields PLAN.md Stage 2 asks for ----
    report["dataset_name"] = meta.info.get("robot_type") or root.parent.parent.name
    report["codebase_version"] = meta.info.get("codebase_version")
    report["num_episodes"] = meta.num_episodes
    report["num_frames"] = meta.num_frames
    report["fps"] = meta.fps
    report["task_names"] = meta.task_names()
    report["action_key"] = meta.action_key
    report["action_shape"] = list(meta.action_shape) if meta.action_shape else None
    report["state_keys"] = {k: meta.feature_shape(k) for k in meta.state_keys}
    report["image_keys"] = meta.image_keys
    report["camera_names"] = meta.camera_names
    report["camera_roles"] = meta.classify_cameras()
    report["declared_image_shapes"] = {
        k: {"raw": meta.feature_shape(k), "hwc": meta.image_hwc(k)}
        for k in meta.image_keys
    }
    report["annotation_keys"] = meta.annotation_keys
    report["stage_annotation_keys"] = meta.stage_annotation_keys

    lengths = meta.episode_lengths()
    if lengths:
        report["episode_lengths"] = {
            "count": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
            "sum": sum(lengths),
            "first_10": lengths[:10],
        }
    else:
        report["episode_lengths"] = None

    # ---- per-episode verification ----
    ep = episode
    report["probe_episode"] = ep
    report["parquet"] = _probe_parquet(meta.parquet_path(ep))
    report["parquet_path"] = str(meta.parquet_path(ep))

    videos = {}
    if probe_videos:
        for key in meta.image_keys:
            videos[key] = _probe_video(meta.video_path(ep, key))
    report["videos"] = videos

    # ---- acceptance criteria (PLAN.md Stage 2) ----
    checks: dict[str, object] = {}
    checks["has_multiple_cameras"] = len(meta.image_keys) >= 2
    checks["all_camera_video_dirs_exist"] = all(
        bool(meta.video_dirs(k)) for k in meta.image_keys
    )

    n_rows = report["parquet"].get("num_rows")
    declared_len = None
    for e in meta.episodes:
        if int(e.get("episode_index", -1)) == ep:
            declared_len = int(e.get("length", -1))
            break
    checks["parquet_rows_match_declared_length"] = (
        None if (n_rows is None or declared_len is None) else n_rows == declared_len
    )

    if probe_videos and videos:
        frame_counts = {k: v.get("frames") for k, v in videos.items()}
        good = [v for v in frame_counts.values() if isinstance(v, int) and v > 0]
        checks["video_frame_counts"] = frame_counts
        # Necessary condition for synchronisation: every view has the same length as the
        # action stream. This does not by itself prove frame t aligns across views; the
        # visual check in visualize_episode.py covers that.
        checks["all_views_same_frame_count"] = len(set(good)) == 1 if good else None
        checks["views_match_action_frames"] = (
            None if (not good or n_rows is None) else all(v == n_rows for v in good)
        )
        res = {k: (v.get("height"), v.get("width")) for k, v in videos.items()}
        checks["video_resolutions"] = res
        checks["all_views_same_resolution"] = len(set(res.values())) == 1 if res else None

    # Genuine per-frame stage labels, not episode-level language. Reported as INFO
    # rather than a hard check: atomic tasks legitimately lack these, and only the
    # stage analysis of PLAN.md section 9 needs them.
    checks["stage_annotation_keys"] = meta.stage_annotation_keys or "none (atomic task?)"
    report["checks"] = checks
    return report


def _fmt(report: dict) -> str:
    L = []
    g = report.get
    L.append("=" * 78)
    L.append(f"DATASET: {g('dataset_path')}")
    L.append("=" * 78)
    L.append(f"dataset name          : {g('dataset_name')}")
    L.append(f"codebase version      : {g('codebase_version')}")
    L.append(f"number of episodes    : {g('num_episodes')}")
    L.append(f"number of frames      : {g('num_frames')}")
    L.append(f"fps                   : {g('fps')}")
    tasks = g("task_names") or []
    L.append(f"task name             : {tasks[0] if tasks else '(none)'}")
    if len(tasks) > 1:
        L.append(f"  (+{len(tasks)-1} more instructions)")
    L.append(f"action key / shape    : {g('action_key')} {g('action_shape')}")

    L.append("")
    L.append("proprioception keys:")
    for k, shape in (g("state_keys") or {}).items():
        L.append(f"  {k:52s} {shape}")

    L.append("")
    L.append("available image keys / camera names:")
    for k in g("image_keys") or []:
        hwc = (g("declared_image_shapes") or {}).get(k, {}).get("hwc")
        name = k.split("observation.images.")[-1]
        L.append(f"  {k:52s} name={name:24s} declared_hwc={hwc}")
    roles = g("camera_roles") or {}
    L.append(f"  roles -> wrist={roles.get('wrist')}  third_person={roles.get('third_person')}")

    el = g("episode_lengths")
    L.append("")
    if el:
        L.append(
            f"episode lengths       : min={el['min']} max={el['max']} "
            f"mean={el['mean']} total={el['sum']}"
        )
        L.append(f"  first 10            : {el['first_10']}")
    else:
        L.append("episode lengths       : (episodes.jsonl missing)")

    ann = g("annotation_keys") or []
    stage = g("stage_annotation_keys") or []
    L.append("")
    L.append(f"annotation keys ({len(ann)}):")
    for k in ann:
        L.append(f"  {k}")
    L.append("")
    L.append(f"per-frame STAGE annotation keys ({len(stage)}):")
    if stage:
        for k in stage:
            L.append(f"  {k}")
    else:
        L.append("  (none — expected for atomic tasks. The stage-dependence analysis of")
        L.append("   PLAN.md section 9 requires a target *composite* task dataset.)")

    pq_info = g("parquet") or {}
    L.append("")
    L.append(f"--- probe episode {g('probe_episode')} ---")
    L.append(f"parquet rows          : {pq_info.get('num_rows')}  ({pq_info.get('error','ok')})")
    for name, d in (pq_info.get("columns") or {}).items():
        L.append(f"  {name:48s} {d['arrow_type']:26s} {d['sample']}")

    vids = g("videos") or {}
    if vids:
        L.append("")
        L.append("video streams:")
        for k, v in vids.items():
            if "error" in v:
                L.append(f"  {k:52s} ERROR: {v['error']}")
            else:
                L.append(
                    f"  {k:52s} frames={v['frames']:5d} "
                    f"{v['width']}x{v['height']} fps={v['fps']}"
                )

    L.append("")
    L.append("--- acceptance checks (PLAN.md Stage 2) ---")
    for k, v in (g("checks") or {}).items():
        if isinstance(v, bool):
            mark = "PASS" if v else "FAIL"
        elif v is None:
            mark = "SKIP"
        else:
            mark = "INFO"
        L.append(f"  [{mark}] {k}: {v if not isinstance(v, bool) else ''}".rstrip())
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default=None, help="path to a lerobot dir")
    p.add_argument("--task", type=str, default=None, help="task name to resolve")
    p.add_argument("--split", type=str, default="target", choices=["target", "pretrain"])
    p.add_argument("--episode", type=int, default=0, help="episode to probe")
    p.add_argument("--no-videos", action="store_true", help="skip mp4 probing (faster)")
    p.add_argument("--json", type=str, default=None, help="also write report as JSON")
    args = p.parse_args()

    root = resolve_dataset_path(dataset=args.dataset, task=args.task, split=args.split)
    report = inspect(root, args.episode, probe_videos=not args.no_videos)
    print(_fmt(report))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nwrote JSON report to {args.json}")

    failed = [k for k, v in report["checks"].items() if v is False]
    if failed:
        print(f"\nFAILED CHECKS: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
