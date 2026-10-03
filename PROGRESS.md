# P2C — Progress and Verified Facts

Last updated 2026-10-03. This file records what has been **verified on this machine**, so a
later session does not re-investigate. Claims here were checked by running something;
where something is assumed or unverified, it says so.

---

## 1. Machine (verified)

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.2 LTS on WSL2 (kernel 6.6.87.2) |
| GPU | GTX 1650, 4 GB VRAM, driver 572.70, CUDA 12.8 |
| nvcc | 12.0 |
| CPU | 8 cores |
| **RAM** | **5.8 GB total, ~4.4 GB available** |
| Disk free | ~790 GB on `/` |
| conda | 25.9.1 at `~/miniconda3` (not on the PATH of a non-interactive bash) |
| EGL | `libEGL.so.1`, `libEGL_mesa.so.0` present |
| Vulkan | loader + mesa ICDs present, **no NVIDIA ICD** — treat as unusable |
| OSMesa | **absent** (`libosmesa6` not installed) |

**RAM, not VRAM, is the binding constraint.** Every data path is memmap-backed for this
reason. Use conda via `export PATH="$HOME/miniconda3/bin:$PATH"`.

## 2. Environment `robocasa` (verified working)

Python 3.11.16 at `/home/pavt1024/miniconda3/envs/robocasa`. Verified by import:

| Package | Version | Note |
|---|---|---|
| torch | 2.6.0+cu124 | **CUDA available: True**, GTX 1650, sm_75, 4.29 GB |
| robocasa | 1.0.1 | editable from `external/robocasa`, installed `--no-deps` |
| robosuite | 1.5.2 | master branch |
| mujoco | 3.3.1 | exact version robocasa asserts |
| numpy | 2.2.5 | exact version robocasa asserts |
| pyarrow / opencv / pyyaml / matplotlib / pandas / h5py | current | |

`robocasa.utils.dataset_registry.get_ds_meta` resolves all three downloaded tasks
(`exists=True`), because `DATASET_BASE_PATH` in
`external/robocasa/robocasa/macros_private.py` points at `p2c/datasets`.

### How robocasa was installed, and why plain `pip install -e .` hangs

`pip install -e external/robocasa` ran ~14 minutes at 60%+ CPU with no output and nothing
written to `site-packages`, then was killed. Root cause, confirmed from PyPI metadata:
`robocasa/setup.py` pins **`tianshou==0.4.10`**, which requires **`gym>=0.23.1`** — the old
`gym` package. pip walks many `gym` releases, most of which fail to build on Python 3.11
with numpy 2.x, so the resolver backtracks effectively forever.

`tianshou` is imported by exactly one file, `robocasa/scripts/bench_speed.py`. The working
install is therefore:

```bash
pip install pyarrow pyyaml matplotlib pandas h5py lxml imageio pygame hidapi gymnasium
pip install --no-deps -e external/robocasa
pip install "mujoco==3.3.1" "numpy==2.2.5"   # robocasa asserts these exact versions
python -m robocasa.scripts.setup_macros
```

Known gaps from this approach, all harmless for P2C:
- `tianshou` absent → `robocasa/scripts/bench_speed.py` will not run.
- `lerobot` absent → robocasa's own `download_datasets.py` / `playback_dataset.py` will
  not run. P2C reads the LeRobot files directly, so nothing in this repo needs it.
- `numba` 0.68.0 and `scipy` 1.17.1 differ from robocasa's pins; neither is asserted and
  nothing has failed because of it.

## 3. Key finding: the MVP needs no renderer

RoboCasa365 datasets are LeRobot format and **ship mp4 already rendered for three
synchronised cameras**. Verified by reading `meta/info.json` and decoding the mp4 directly:

| Camera | Role in P2C | Verified shape |
|---|---|---|
| `robot0_agentview_left` | primary | 256x256x3, 20 fps |
| `robot0_agentview_right` | secondary | 256x256x3, 20 fps |
| `robot0_eye_in_hand` | wrist | 256x256x3, 20 fps |

Consequences:

- Stages 0–2 of SETUP.md read parquet + mp4 and **never call MuJoCo**, so the missing
  Vulkan/OSMesa does not block the data study.
- Rendering is needed only for policy *rollout* evaluation (`eval_robocasa.py`), which is
  the remote Diffusion Policy stage anyway.
- The three real cameras cover every view condition in SETUP.md section 7.

## 4. Datasets downloaded (verified)

Stored under `datasets/` (gitignored), 1.7 GB total. Downloaded with a standalone script
replicating `robocasa.scripts.download_datasets` path logic, because robocasa was not
installed. Box blocks `HEAD`, so sizes were measured with `GET` + `Range: bytes=0-0`.

| Task | Type | Tar | Episodes | Frames | Mean ep. length | Stage labels |
|---|---|---|---|---|---|---|
| `NavigateKitchen` | target atomic | 0.47 GB | 500 | 72,786 | 145.6 | no |
| `PickPlaceCounterToCabinet` | target atomic | 0.56 GB | — | — | — | no |
| `StackBowlsCabinet` | target composite | 0.74 GB | 515 | 175,620 | 341.0 | **yes** |

Other measured sizes, for planning: `TurnOnMicrowave` 0.34 GB, `OpenDrawer` 0.48 GB,
`SteamInMicrowave` 2.06 GB, `StoreLeftoversInBowl` 2.01 GB.

### Verified schema (`PandaOmron`, codebase v2.1)

- action: `[12]`; state: `(16,)`; 20 fps.
- Action groups from the dataset's own `meta/modality.json`:
  `base_motion [0,4)`, `control_mode [4,5)`, `end_effector_position [5,8)`,
  `end_effector_rotation [8,11)`, `gripper_close [11,12)`.
- State groups: `base_position [0,3)`, `base_rotation [3,7)`,
  `end_effector_position_relative [7,10)`, `end_effector_rotation_relative [10,14)`,
  `gripper_qpos [14,16)`.
- Episode 0 of `NavigateKitchen`: all three cameras decode to **177 frames each** —
  frame counts agree across views.

### Stage annotations: composite only (verified)

`StackBowlsCabinet` (composite) carries per-frame keys:
`annotation.human.subtask`, `annotation.human.subtask_name`,
`annotation.human.subtask_stage`, `subtask_idx`.

`NavigateKitchen` (atomic) carries **none** — only episode-level
`annotation.human.task_description` / `task_name` / `task_index`.

**So the stage-dependence analysis of SETUP.md sections 10–11 requires a target composite
task.** Atomic tasks can show the overall phenomenon but not its stage structure.

## 5. Code written

All of it reads the schema from disk; no camera name, resolution or action dimension is
hard-coded (SETUP.md sections 3.1 and 6).

```
p2c/data/lerobot_meta.py        LeRobot schema discovery, modality groups, camera roles
p2c/data/frame_cache.py         memmap cache reader, deterministic split, norm statistics
p2c/data/camera_subset.py       roles, slots, view conditions, hash-based determinism
p2c/data/robocasa_dataset.py    CameraSubsetDataset + build_train_val
p2c/models/view_encoder.py      shared tiny CNN; optional frozen ResNet
p2c/models/fusion.py            mean / attn (view-count independent), concat (warns)
p2c/models/tiny_bc.py           policy, losses, per-group and per-sample metrics
p2c/analysis/results.py         load runs, join per-sample errors on shared frames
p2c/analysis/stage_metrics.py   Delta_view(g), paired bootstrap, heterogeneity test
p2c/analysis/complementarity.py C(v|p), oracle selection headroom
p2c/analysis/kill_criteria.py   automated verdict on SETUP.md section 21
p2c/analysis/visualization.py   improved-frame contact sheet, stage and histogram plots
p2c/utils/paths.py              dataset resolution (registry if available, else glob)
p2c/utils/seeding.py            seeding + explicit nondeterminism report
p2c/utils/tracking.py           run dirs, meta.json, metrics.jsonl/csv, console capture
scripts/inspect_dataset.py      Stage 0 schema verification
scripts/visualize_episode.py    synchronised multi-view grid / video
scripts/build_frame_cache.py    decode mp4 once into a uint8 memmap cache
scripts/train_local_bc.py       train one view condition
scripts/eval_local_bc.py        evaluate a checkpoint, incl. cross-condition
scripts/run_view_ablation.sh    sweep B0-B4 + controls with everything else fixed
scripts/summarize_view_ablation.py  tables, CSVs, figures, complementarity, kill criteria
configs/local_debug.yaml        shared config; only `views` differs between arms
configs/overfit.yaml            SETUP.md section 18 overfit test
tests/                          conftest synthetic cache + 3 test modules
```

Two design decisions worth keeping in mind:

1. **The frame cache is not just a speed optimisation.** Decoding mp4 once makes
   "identical preprocessing across conditions" (section 7.3) structural, and makes camera
   synchronisation true by construction, since all cameras write to the same global frame
   index.
2. **Capacity matching is automatic.** One shared encoder plus `mean`/`attn` fusion makes
   the trainable parameter count independent of the view count, so section 12 Control B
   holds by construction rather than by later adjustment. `concat` fusion is available but
   warns and is recorded as violating the control.

## 6. Tests (verified passing)

63 passed: `tests/test_camera_subset.py` (32) and `tests/test_analysis.py` (31).
`tests/test_dataset.py` and `tests/test_model_forward.py` need torch and have **not been
run yet**.

Run them with:

```bash
/home/pavt1024/miniconda3/envs/robocasa/bin/python -m pytest tests -q
```

## 7. Bugs found and fixed while building

- `has_stage_annotations` initially passed on atomic tasks, because it matched
  `task_index` and `task_description`. Split into a broad `annotation_keys` and a strict
  `stage_annotation_keys`, and demoted the hard check to INFO (atomic tasks legitimately
  lack stage labels).
- The strict filter then wrongly *excluded* `annotation.human.subtask_name`, because
  `"subtask_name".endswith("task_name")` is true. Fixed to compare the final dotted
  component exactly.
- `capacity_report()` had a duplicate `"fusion"` key, so the parameter count was silently
  overwritten by the fusion name.

## 8. Open items

1. Finish the torch install; then run `tests/test_dataset.py` and
   `tests/test_model_forward.py`.
2. Install `pyarrow`, `pyyaml`, `matplotlib`, `pandas` (needed by the cache builder,
   configs and figures). `pyarrow` is what currently makes `inspect_dataset.py` skip the
   parquet probe.
3. Build a frame cache and run the SETUP.md section 18 overfit test **before** any
   ablation.
4. Run the ablation on an atomic task first (cheap), then on `StackBowlsCabinet` for the
   stage analysis.
5. Not yet started, and correctly gated by SETUP.md: B5 learned view selection, the VISTA
   baseline, the camera-pose-conditioned baseline (kill criterion 6 cannot be evaluated
   without it), and the remote Diffusion Policy configs.
