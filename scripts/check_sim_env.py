#!/usr/bin/env python
"""Verify the RoboCasa simulator path: env creation, stepping, and offscreen rendering.

The data pipeline reads pre-rendered mp4 and never touches MuJoCo, so a working data path
says nothing about whether the simulator works. Closed-loop evaluation does need it
(PLAN.md section 7.2 assigns that to the remote server, but it should at least be known
whether it runs locally).

This probes the three MuJoCo GL backends in turn and reports, for each, how far it got:

    import -> env created -> reset -> step -> offscreen frame rendered

A backend that creates an environment but cannot render is a real and easily missed state:
training on cached video would still work while rollout evaluation silently could not.

Usage::

    python scripts/check_sim_env.py
    python scripts/check_sim_env.py --task PickPlaceCounterToCabinet --backends egl osmesa
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROBE = r'''
import os, sys, json
os.environ["MUJOCO_GL"] = {backend!r}
# PYOPENGL_PLATFORM must be unset or "egl"; setting it to a MuJoCo backend name such as
# "glfw" makes PyOpenGL refuse to load and looks like a backend failure that it is not.
if {backend!r} == "egl":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
else:
    os.environ.pop("PYOPENGL_PLATFORM", None)

result = {{"backend": {backend!r}, "stage": "start", "error": None}}
try:
    import numpy as np
    import mujoco
    result["mujoco_version"] = mujoco.__version__
    result["stage"] = "mujoco_imported"

    import gymnasium as gym
    import robocasa
    result["stage"] = "robocasa_imported"

    env = gym.make("robocasa/{task}", split={split!r}, seed=0)
    result["stage"] = "env_created"

    env.reset()
    result["stage"] = "reset"

    env.step(env.action_space.sample())
    result["stage"] = "stepped"

    # Render explicitly through the underlying MuJoCo sim. The gym observation does not
    # carry camera images by default, so checking obs for image keys would report "no
    # rendering" on a machine where rendering works perfectly.
    sim = env.unwrapped.sim
    frames = {{}}
    for cam in ["robot0_agentview_left", "robot0_eye_in_hand"]:
        arr = np.asarray(sim.render(width=128, height=128, camera_name=cam))
        frames[cam] = {{
            "shape": list(arr.shape),
            # An all-zero frame means the renderer ran but produced nothing — the failure
            # mode that otherwise looks like success.
            "nonzero": bool(arr.any()),
            "mean": float(arr.mean()),
        }}
    result["frames"] = frames
    result["frame_nonzero"] = all(f["nonzero"] for f in frames.values())
    result["stage"] = "rendered"
    env.close()
except Exception as e:
    result["error"] = f"{{type(e).__name__}}: {{str(e)[:300]}}"
print("P2C_PROBE_RESULT " + json.dumps(result))
'''


def probe(backend: str, task: str, split: str, python: str, timeout: int) -> dict:
    code = PROBE.format(backend=backend, task=task, split=split)
    try:
        out = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"backend": backend, "stage": "timeout", "error": f"timed out after {timeout}s"}

    for line in (out.stdout or "").splitlines():
        if line.startswith("P2C_PROBE_RESULT "):
            import json

            return json.loads(line[len("P2C_PROBE_RESULT ") :])
    tail = (out.stderr or out.stdout or "").strip().splitlines()[-3:]
    return {
        "backend": backend,
        "stage": "crashed",
        "error": " | ".join(tail)[:300] or f"exit {out.returncode}",
    }


STAGES = [
    "mujoco_imported", "robocasa_imported", "env_created", "reset", "stepped", "rendered",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="PickPlaceCounterToCabinet")
    p.add_argument("--split", default="target", choices=["target", "pretrain"])
    p.add_argument("--backends", nargs="+", default=["egl", "osmesa", "glfw"])
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    # Each backend runs in a fresh interpreter: MUJOCO_GL is read once at import, so
    # probing several in one process would silently test the first one three times.
    print(f"task    : {args.task} (split={args.split})")
    print(f"python  : {args.python}")
    print()

    results = []
    for b in args.backends:
        print(f"--- {b} ---", flush=True)
        r = probe(b, args.task, args.split, args.python, args.timeout)
        results.append(r)
        reached = r.get("stage")
        idx = STAGES.index(reached) if reached in STAGES else -1
        trail = " -> ".join(
            s if STAGES.index(s) <= idx else f"({s})" for s in STAGES
        )
        print(f"  reached : {reached}")
        print(f"  path    : {trail}")
        for cam, f in (r.get("frames") or {}).items():
            print(f"  {cam:24s} shape={f['shape']} nonzero={f['nonzero']} "
                  f"mean={f['mean']:.2f}")
        if r.get("error"):
            print("  error   :")
            print(textwrap.indent(textwrap.fill(r["error"], 86), "            "))
        print()

    working = [r for r in results if r.get("stage") == "rendered" and r.get("frame_nonzero")]
    print("=" * 78)
    if working:
        print(f"OFFSCREEN RENDERING WORKS with: {', '.join(r['backend'] for r in working)}")
        print("Closed-loop rollout evaluation is possible on this machine.")
    else:
        got_env = [r for r in results if r.get("stage") in ("reset", "stepped", "env_created")]
        if got_env:
            print("Environments build and step, but no backend produced a usable frame.")
            print("Training from cached video is unaffected; rollout evaluation is not")
            print("possible locally and must run where a GL backend works.")
        else:
            print("No backend reached a working environment. See the errors above.")
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
