"""Dataset path resolution.

Keeps the P2C scripts usable either with an explicit ``--dataset`` path or with a task
name, without requiring robocasa to be importable (useful on a machine where only the
LeRobot data has been downloaded). When robocasa *is* importable we defer to its registry
so paths never drift from upstream.
"""

from __future__ import annotations

import os
from pathlib import Path

# Where fetch/download put the LeRobot trees. Override with P2C_DATASET_BASE.
DEFAULT_DATASET_BASE = Path(__file__).resolve().parents[2] / "datasets"


def dataset_base() -> Path:
    return Path(os.environ.get("P2C_DATASET_BASE", DEFAULT_DATASET_BASE))


def resolve_dataset_path(
    dataset: str | None = None,
    task: str | None = None,
    split: str = "target",
    source: str = "human",
) -> Path:
    """Resolve a dataset directory from an explicit path or a task name.

    Resolution order:
      1. explicit ``dataset`` path;
      2. robocasa's own registry, if importable (authoritative);
      3. a glob under :func:`dataset_base`, which tolerates the undated layout.
    """
    if dataset:
        p = Path(dataset).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"dataset path does not exist: {p}")
        return p

    if not task:
        raise ValueError("pass either --dataset <path> or --task <TaskName>")

    try:
        from robocasa.utils.dataset_registry import get_ds_meta  # noqa: F401

        meta = get_ds_meta(task=task, split=split, source=source)
        if meta is not None:
            p = Path(meta["path"])
            if p.exists():
                return p
    except Exception:
        pass  # fall through to glob; robocasa may not be installed yet

    base = dataset_base()
    hits = sorted(base.glob(f"v1.0/{split}/*/{task}/*/lerobot"))
    if not hits:
        raise FileNotFoundError(
            f"no dataset for task={task} split={split} under {base}. "
            f"Download it first, or pass --dataset explicitly."
        )
    if len(hits) > 1:
        raise ValueError(
            f"multiple datasets matched task={task}: {[str(h) for h in hits]}. "
            f"Disambiguate with --dataset."
        )
    return hits[0]


def list_local_datasets(split: str | None = None) -> list[Path]:
    """Every downloaded LeRobot dataset, for summary scripts and tests."""
    base = dataset_base()
    pattern = f"v1.0/{split or '*'}/*/*/*/lerobot"
    return sorted(base.glob(pattern))


def task_of(path: str | Path) -> str:
    """Recover the task name from a dataset path (…/<split>/<kind>/<Task>/<date>/lerobot)."""
    parts = Path(path).parts
    if parts and parts[-1].startswith("lerobot"):
        return parts[-3]
    return Path(path).name
