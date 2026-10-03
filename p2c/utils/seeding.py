"""Seeding and determinism, per harness contract (docs/harness_contract.md) C6.

Seeds ``random``, ``numpy``, ``torch`` and ``torch.cuda``. Dataset sampling and camera
randomisation are *not* seeded here: they are made deterministic structurally, by hashing
``(seed, episode, frame)`` in :mod:`p2c.data.camera_subset`, so they do not depend on
global RNG state or on dataloader worker scheduling.

:func:`nondeterminism_report` records what remains nondeterministic, rather than
silently assuming bit-exactness.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic: bool = True) -> dict:
    """Seed every RNG this project touches. Returns a report for the run metadata."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    report: dict = {"seed": seed, "deterministic_requested": deterministic}

    try:
        import numpy as np

        np.random.seed(seed)
        report["numpy"] = np.__version__
    except ImportError:
        report["numpy"] = None

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        report["torch"] = torch.__version__
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        report["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except ImportError:
        report["torch"] = None

    return report


def worker_init_fn(worker_id: int) -> None:
    """Give each dataloader worker a distinct but reproducible numpy/random seed."""
    import numpy as np
    import torch

    base = torch.initial_seed() % 2**31
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def nondeterminism_report(mixed_precision: bool = False) -> list[str]:
    """Operations that may still vary run to run, recorded in each run's metadata."""
    notes = [
        "cuDNN convolution backward kernels may be nondeterministic unless "
        "torch.use_deterministic_algorithms(True) is set; we set cudnn.deterministic "
        "only, which covers the common conv algorithms but is not a hard guarantee.",
        "Multi-worker dataloading changes the order of floating-point accumulation in "
        "metric averaging; per-sample errors are unaffected.",
    ]
    if mixed_precision:
        notes.append(
            "Mixed precision (AMP) introduces run-to-run variation through dynamic loss "
            "scaling; the headline comparisons are therefore repeated across seeds."
        )
    return notes
