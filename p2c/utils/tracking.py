"""Experiment tracking, per SETUP.md section 16.

Every run writes a self-contained directory holding the full list section 16 requires:
git commit hash, config, dataset identity, task, seed, camera subset, demo count, image
resolution, parameter count, training steps, validation losses, success metrics,
checkpoint and captured stdout/stderr.

Structured JSON/CSV is the default so the project does not depend on Weights & Biases
being configured; W&B is used additionally when available and requested.

Run naming follows section 16::

    p2c_<task>_<policy>_<views>_seed<seed>
"""

from __future__ import annotations

import csv
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path


def git_info(repo: Path | None = None) -> dict:
    repo = repo or Path(__file__).resolve().parents[2]

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=repo, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def run_name(task: str, policy: str, views: str, seed: int) -> str:
    """``p2c_<task>_<policy>_<views>_seed<seed>`` with path-safe view names."""
    safe = views.replace("+", "-").replace(",", "-").replace("/", "-")
    return f"p2c_{task}_{policy}_{safe}_seed{seed}"


class RunLogger:
    """Writes one run directory: metadata, metric history, checkpoint, console log."""

    def __init__(self, root: str | Path, name: str, config: dict | None = None):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.t0 = time.time()
        self._rows: list[dict] = []
        self._meta: dict = {
            "run_name": name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git": git_info(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "argv": sys.argv,
        }
        if config:
            self._meta["config"] = config
        self._wandb = None

    # ---------------- metadata ----------------

    def update_meta(self, **kw) -> None:
        self._meta.update(kw)
        self.save_meta()

    def save_meta(self) -> None:
        with open(self.dir / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2, default=str)

    @property
    def meta(self) -> dict:
        return self._meta

    # ---------------- metrics ----------------

    def log(self, step: int, **metrics) -> None:
        row = {"step": step, "elapsed_s": round(time.time() - self.t0, 2)}
        row.update({k: _plain(v) for k, v in metrics.items()})
        self._rows.append(row)
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def write_csv(self) -> Path | None:
        if not self._rows:
            return None
        keys: list[str] = []
        for r in self._rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        path = self.dir / "metrics.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._rows)
        return path

    def save_json(self, name: str, obj) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return path

    # ---------------- final result ----------------

    def finish(self, result: dict | None = None) -> None:
        self._meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._meta["duration_s"] = round(time.time() - self.t0, 1)
        if result:
            self._meta["result"] = {k: _plain(v) for k, v in result.items()}
        self.save_meta()
        self.write_csv()
        if self._wandb is not None:
            self._wandb.finish()

    # ---------------- optional W&B ----------------

    def maybe_init_wandb(self, project: str, enable: bool) -> bool:
        """Mirror to W&B when requested and importable. Never fatal."""
        if not enable:
            return False
        try:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=project,
                name=self.name,
                config=self._meta.get("config", {}),
                dir=str(self.dir),
            )
            return True
        except Exception as e:  # keep the run going; JSON/CSV is the source of truth
            print(f"[tracking] W&B disabled ({e})")
            self._wandb = None
            return False


class Tee:
    """Duplicate stdout/stderr into the run directory (section 16: stdout/stderr)."""

    def __init__(self, path: Path, stream):
        self.file = open(path, "a", buffering=1)
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def capture_console(run_dir: Path):
    """Redirect stdout/stderr into ``console.log``; returns a restore callable."""
    out, err = sys.stdout, sys.stderr
    t_out = Tee(run_dir / "console.log", out)
    t_err = Tee(run_dir / "console.log", err)
    sys.stdout, sys.stderr = t_out, t_err

    def restore():
        sys.stdout, sys.stderr = out, err
        t_out.close()
        t_err.close()

    return restore


def _plain(v):
    """Convert tensors/arrays/numpy scalars to plain Python for JSON."""
    if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
        try:
            return v.item()
        except Exception:
            return v
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            return v
    return v
