"""Load and join experiment results across view conditions.

Every run writes per-sample validation errors keyed by ``(episode, frame)``. Because all
conditions share one train/validation split (SETUP.md section 7, requirement 2), the same
validation frames appear in every run, so the errors can be joined frame by frame rather
than only compared as averages.

That join is what makes the two analyses SETUP.md asks for possible:

* the per-stage difference of section 11, which needs error attributed to timesteps;
* the complementarity metric of section 13, ``C(v | p) = L(pi_p) - L(pi_{p+v})``, which is
  defined per frame before being aggregated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunResult:
    """One training run: its metadata plus per-sample validation errors."""

    path: Path
    meta: dict
    episode: np.ndarray
    frame: np.ndarray
    stage: np.ndarray
    sq_error: np.ndarray
    stage_names: dict = field(default_factory=dict)

    # ---------- identity ----------

    @property
    def condition(self) -> str:
        return (
            self.meta.get("view_condition", {}).get("condition")
            or self.meta.get("config", {}).get("views")
            or self.path.name
        )

    @property
    def task(self) -> str:
        return self.meta.get("dataset", {}).get("task", "unknown")

    @property
    def seed(self) -> int:
        return int(self.meta.get("config", {}).get("seed", 0))

    @property
    def num_views(self) -> int:
        return int(self.meta.get("view_condition", {}).get("num_views", 0))

    @property
    def params(self) -> int:
        return int(self.meta.get("capacity", {}).get("total_trainable", 0))

    @property
    def capacity_matched(self) -> bool:
        return bool(self.meta.get("capacity", {}).get("view_count_independent", False))

    @property
    def best_val_mse(self) -> float:
        r = self.meta.get("result") or {}
        v = r.get("best_val_mse")
        return float(v) if v is not None else float("nan")

    @property
    def best_val_l1(self) -> float:
        r = self.meta.get("result") or {}
        v = r.get("best_val_l1")
        return float(v) if v is not None else float("nan")

    @property
    def is_overfit_run(self) -> bool:
        return bool((self.meta.get("config") or {}).get("overfit"))

    @property
    def trivial_baseline_mse(self) -> float:
        """MSE of predicting the training-set mean action. None for older runs."""
        v = self.meta.get("trivial_baseline_mse")
        if v is None:
            v = (self.meta.get("result") or {}).get("trivial_baseline_mse")
        return float(v) if v is not None else float("nan")

    @property
    def beats_trivial_baseline(self) -> bool | None:
        t = self.trivial_baseline_mse
        if t != t:  # NaN
            return None
        return bool(self.best_val_mse < t)

    @property
    def num_instructions(self) -> int:
        return int(self.meta.get("dataset", {}).get("num_instructions", -1))

    @property
    def has_stages(self) -> bool:
        return self.stage.size > 0 and bool((self.stage >= 0).any())

    def keys(self) -> np.ndarray:
        """Stable per-sample key combining episode and frame."""
        return self.episode.astype(np.int64) * 1_000_000 + self.frame.astype(np.int64)

    def stage_name(self, sid: int) -> str:
        return self.stage_names.get(str(int(sid)), str(int(sid)))


def load_run(run_dir: str | Path) -> RunResult | None:
    """Load one run directory. Returns None if it has no usable results."""
    run_dir = Path(run_dir)
    meta_file = run_dir / "meta.json"
    npz_file = run_dir / "val_per_sample.npz"
    if not meta_file.exists():
        return None
    with open(meta_file) as f:
        meta = json.load(f)

    if not npz_file.exists():
        # A run that trained but never evaluated: keep the summary metrics only.
        return RunResult(
            path=run_dir, meta=meta,
            episode=np.zeros(0, np.int32), frame=np.zeros(0, np.int32),
            stage=np.zeros(0, np.int32), sq_error=np.zeros(0, np.float32),
        )

    d = np.load(npz_file, allow_pickle=False)
    stage_names = {}
    if "stage_names" in d:
        try:
            stage_names = json.loads(str(d["stage_names"].item()))
        except Exception:
            stage_names = {}
    return RunResult(
        path=run_dir,
        meta=meta,
        episode=d["episode"].astype(np.int64),
        frame=d["frame"].astype(np.int64),
        stage=d["stage"].astype(np.int64),
        sq_error=d["sq_error"].astype(np.float64),
        stage_names=stage_names or {},
    )


def load_runs(root: str | Path, include_overfit: bool = False) -> list[RunResult]:
    """Load every run under ``root``, newest last. Overfit runs are excluded by default."""
    root = Path(root)
    out = []
    for meta_file in sorted(root.rglob("meta.json")):
        r = load_run(meta_file.parent)
        if r is None:
            continue
        if r.is_overfit_run and not include_overfit:
            continue
        out.append(r)
    return out


def join_per_sample(runs: list[RunResult]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Align per-sample errors across runs on their shared validation frames.

    Returns ``(keys, {condition: sq_error}, stage)`` restricted to the frames present in
    *every* run. Refusing to pad missing frames is deliberate: a partial overlap means the
    runs did not share a split, and comparing them would be invalid.
    """
    usable = [r for r in runs if r.sq_error.size > 0]
    if not usable:
        return np.zeros(0, np.int64), {}, np.zeros(0, np.int64)

    common: set[int] | None = None
    for r in usable:
        ks = set(r.keys().tolist())
        common = ks if common is None else (common & ks)
    shared = np.array(sorted(common or []), dtype=np.int64)
    if shared.size == 0:
        return shared, {}, np.zeros(0, np.int64)

    errors: dict[str, np.ndarray] = {}
    stage_ref = None
    for r in usable:
        k = r.keys()
        order = np.argsort(k)
        k_sorted = k[order]
        pos = np.searchsorted(k_sorted, shared)
        sel = order[pos]
        label = _label_of(r)
        if label in errors:
            # Several seeds of the same condition: average them per frame.
            errors[label] = (errors[label] + r.sq_error[sel]) / 2.0
        else:
            errors[label] = r.sq_error[sel]
        if stage_ref is None and r.has_stages:
            stage_ref = r.stage[sel]
    return shared, errors, (stage_ref if stage_ref is not None else np.full(shared.shape, -1))


def _label_of(r: RunResult) -> str:
    return r.condition


def group_by_condition(runs: list[RunResult]) -> dict[str, list[RunResult]]:
    out: dict[str, list[RunResult]] = {}
    for r in runs:
        out.setdefault(r.condition, []).append(r)
    return out


def group_by_task(runs: list[RunResult]) -> dict[str, list[RunResult]]:
    out: dict[str, list[RunResult]] = {}
    for r in runs:
        out.setdefault(r.task, []).append(r)
    return out
