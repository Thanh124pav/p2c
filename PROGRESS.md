# P2C — Progress and Verified Facts

Last updated 2026-10-03. [PLAN.md](PLAN.md) section 10 names this file the source of truth
for machine-specific facts: verified package versions, downloaded datasets, passing tests,
known environment limits. It records what has been **verified on this machine**, so a later
session does not re-investigate. Claims here were checked by running something; where
something is assumed or unverified, it says so.

The research question lives in [PLAN.md](PLAN.md), not here. Results below from the earlier
multi-view study are kept because they are measured facts about this machine and this data,
not because they define the direction — PLAN.md section 13.6 is explicit that the multi-view
implementation must not lock the research problem into view selection.

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

## 3. Key finding: the local data path needs no renderer

RoboCasa365 datasets are LeRobot format and **ship mp4 already rendered for three
synchronised cameras**. Verified by reading `meta/info.json` and decoding the mp4 directly:

| Camera | Role in the view-partialization axis | Verified shape |
|---|---|---|
| `robot0_agentview_left` | primary | 256x256x3, 20 fps |
| `robot0_agentview_right` | secondary | 256x256x3, 20 fps |
| `robot0_eye_in_hand` | wrist | 256x256x3, 20 fps |

Consequences:

- The local data path reads parquet + mp4 and **never calls MuJoCo**, so the missing
  Vulkan/OSMesa does not block Stage 2 work (PLAN.md section 6.1 says the same).
- Rendering is needed only for closed-loop *rollout* evaluation, which PLAN.md section 7.2
  assigns to the remote server anyway.
- The three real cameras support the viewpoint partialization axis end to end.

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

**So any per-stage analysis requires a target composite task.** Stage labels are
ORACLE-tier (PLAN.md section 9): legitimate for analysis and for constructing controlled
conditions such as phase-drop partialization, never as method input.

## 5. Code written

All of it reads the schema from disk; no camera name, resolution or action dimension is
hard-coded (PLAN.md section 6.1).

```
p2c/data/lerobot_meta.py        LeRobot schema discovery, modality groups, camera roles
p2c/data/frame_cache.py         memmap cache reader, deterministic split, norm statistics
p2c/data/camera_subset.py       roles, slots, view conditions, hash-based determinism
p2c/data/partialization.py      viewpoint / temporal / phase / occlusion partialization
p2c/data/tiers.py               PLAN.md section 9 information tiers, enforced
p2c/data/transforms.py          training-only random-shift augmentation
p2c/data/robocasa_dataset.py    CameraSubsetDataset + build_train_val, tier-filtered
p2c/models/view_encoder.py      shared tiny CNN; optional frozen ResNet
p2c/models/fusion.py            mean / attn (view-count independent), concat (warns)
p2c/models/tiny_bc.py           policy, losses, per-group and per-sample metrics
p2c/analysis/results.py         load runs, join per-sample errors on shared frames
p2c/analysis/stage_metrics.py   Delta_view(g), paired bootstrap, heterogeneity test
p2c/analysis/complementarity.py C(v|p), oracle selection headroom
p2c/analysis/kill_criteria.py   automated falsification verdict + validity precondition
p2c/analysis/visualization.py   improved-frame contact sheet, stage and histogram plots
p2c/utils/paths.py              dataset resolution (registry if available, else glob)
p2c/utils/seeding.py            seeding + explicit nondeterminism report
p2c/utils/tracking.py           run dirs, meta.json, metrics.jsonl/csv, console capture
scripts/inspect_dataset.py      Stage 0 schema verification
scripts/visualize_episode.py    synchronised multi-view grid / video
scripts/build_frame_cache.py    decode mp4 once into a uint8 memmap cache
scripts/train_local_bc.py       train one view condition
scripts/eval_local_bc.py        evaluate a checkpoint, incl. cross-condition
scripts/run_view_ablation.sh    sweep conditions with everything else held fixed
scripts/summarize_view_ablation.py  tables, CSVs, figures, complementarity, verdict
scripts/make_dp_configs.py      generate Diffusion Policy camera-subset configs
configs/local_debug.yaml        shared config; only the condition differs between arms
configs/composite.yaml          composite-task setting (stage labels available)
configs/overfit.yaml            PLAN.md Stage 2 overfit test
docs/harness_contract.md        C1-C12: the controlled-comparison rules, enforced in code
tests/                          synthetic-cache conftest + 6 test modules
```

Three design decisions worth keeping in mind:

1. **The frame cache is not just a speed optimisation.** Decoding mp4 once makes
   "identical preprocessing across conditions" structural (harness contract C8), and makes
   camera synchronisation true by construction, since all cameras write to the same global
   frame index.
2. **Capacity matching is automatic.** One shared encoder plus `mean`/`attn` fusion makes
   the trainable parameter count independent of the view count, so contract C2 holds by
   construction rather than by later adjustment. `concat` fusion is available but warns and
   is recorded as violating it.
3. **Tiers are enforced, not documented.** `CameraSubsetDataset(..., tier=Tier.METHOD)`
   physically omits actions, proprioception and stage labels, so a method intended for
   action-free video cannot quietly come to depend on robot supervision or privileged
   simulator state.

## 6. Tests (verified passing)

**221 passed** across `test_camera_subset`, `test_analysis`, `test_dataset`,
`test_model_forward`, `test_lerobot_meta`, `test_tiers` and `test_partialization`. They run
against a synthetic cache, so they need neither the downloaded datasets nor a GPU.

Run them with:

```bash
/home/pavt1024/miniconda3/envs/robocasa/bin/python -m pytest tests -q
```

## 7. Bugs found and fixed while building

Recorded because each produced a plausible-looking wrong answer rather than an error.

**Schema discovery**

- `has_stage_annotations` initially passed on atomic tasks, because it matched
  `task_index` and `task_description`. Split into a broad `annotation_keys` and a strict
  `stage_annotation_keys`.
- The strict filter then wrongly *excluded* `annotation.human.subtask_name`, because
  `"subtask_name".endswith("task_name")` is true. Fixed to compare the final dotted
  component exactly.
- Stage labels came out as bare indices (11/12/13/15). They index `meta/tasks.jsonl`;
  resolved to `done` / `place` / `pick` / `navigate`.
- `capacity_report()` had a duplicate `"fusion"` key, so the parameter count was silently
  overwritten by the fusion name.

**Statistics — the two that mattered**

- *Assumed* the predict-the-mean baseline for normalised actions was exactly 1.0, and
  reported that models "learned nothing" at val_mse ~0.95. The **measured** baseline was
  1.123, so those models were 14% better than chance. Baselines are now computed from the
  cache, never assumed, and reported with every run.
- Oracle per-frame selection headroom read **32%**, which looked like strong evidence for
  building a view selector. Taking a per-frame minimum over K noisy models is biased
  downwards even with no real structure: a permutation null put the selection-bias floor at
  **90%**, above the observed value. The entire headroom was selection bias. Now reported
  against that null (harness contract C11).

## 8. Experiments run (earlier multi-view study)

Kept as measured facts. PLAN.md section 13.6 is explicit that these must not define the
research problem.

**Gate passed.** Overfit test: 0.444 -> 0.0054 on 32 samples, so the pipeline learns.

**`NavigateKitchen`** (atomic, 40 episodes, 84x84, 1 seed). Predict-the-mean baseline
1.123. Best condition `primary+wrist` 0.962. Falsification criterion 1 **triggered**: the
best multi-view condition beat the best *single* view by only 1.9%, CI [-0.021, 0.059],
p=0.347. The apparent 11.9% gain over the primary view was not "two views beat one" — it
was "the wrist view is simply better for this task". Controls behaved correctly: a random
second view recovered 42% of the pair's gain (p=0.001), and a duplicated view was *worse*
than the single view.

**`StackBowlsCabinet`** (composite, 120 episodes, 96x96, 1 seed, 12 epochs, all 8 arms).
Predict-the-mean baseline 0.577; every arm beats it (best 0.413, **28.6%** better), so the
experiment has signal and the verdict below is interpretable.

| condition | val_mse | vs primary |
|---|---|---|
| `random_two` (control) | 0.4137 | +2.4% |
| `primary+wrist_dropout` (control) | 0.4211 | +0.7% |
| `all_views` | 0.4221 | +0.4% |
| `single_primary` | 0.4239 | — |
| `primary+secondary` | 0.4252 | -0.3% |
| `primary+duplicate` (control) | 0.4262 | -0.6% |
| `primary+wrist` | 0.4322 | **-1.9%** |
| `single_wrist` | 0.4392 | -3.6% |

Falsification criterion 1 **triggered**: the best multi-view arm beats the best single view
by 2.3%, under the 5% threshold. More pointedly, the fixed complementary pair
`primary+wrist` is **significantly worse** than the single primary view (paired delta
-2.6%, CI [-0.019, -0.004], p=0.004). Per stage, the harm is largest and only significant
on `pick` (-3.8%); the permutation test says the effect genuinely varies by stage
(p=0.027), but what varies is harm, not benefit.

The only arms that helped at all were the two *stochastic* ones (`random_two`,
`primary+wrist_dropout`). That is the signature of regularisation rather than information,
which is exactly what the dropout control exists to detect.

**A confound I introduced, not yet ruled out.** Mean fusion averages view features, so an
uninformative view dilutes a useful one — the architecture can make a second view harmful
whether or not it carries complementary information. Mean fusion was chosen because it
makes capacity matching exact (C2), but `attn` fusion is equally capacity-matched and can
*ignore* a view. **These negative results are therefore partly confounded with the fusion
choice, and an `attn` rerun is needed before treating them as evidence about
complementarity.**

Oracle per-frame selection again showed nothing: raw headroom 33.1% against a
selection-bias floor of 86.0%, so the apparent headroom is entirely noise.

**Task choice is a real confound.** `NavigateKitchen` spreads 500 episodes over **14**
language instructions ("navigate to the sink" / "to the stove" / ...) that are not
recoverable from the image, so a vision-only policy there cannot beat the conditional mean
however many cameras it gets. `StackBowlsCabinet` has **1** instruction for all 515
episodes. Every run now prints its instruction count and warns when it exceeds one.

## 9. Open items

1. **Method design** — PLAN.md section 14 puts the P2C-v0 specification before any further
   implementation, and section 13.5 forbids scaling before a falsifiable local signal.
   Nothing below should start first.
2. Partialization along the temporal, phase and occlusion axes is implemented and tested
   but has not been run end-to-end through training; only the viewpoint axis has.
3. Multi-seed runs, if the viewpoint axis is ever revisited. One seed cannot separate the
   conditions.
4. Diffusion Policy configs are generated from upstream but **never executed**; PLAN.md
   section 7.2 assigns that to the remote server (local GPU is 4 GB, guidance is 24 GB+).
5. LIBERO is **not installed**, deliberately. PLAN.md section 13.2 forbids installing it
   into this environment, and its pins (`numpy==1.22.4`, `robosuite==1.4.0`, `gym==0.25.2`,
   Python 3.8) are incompatible. It needs a separate env when Stage 3 arrives.
6. `tianshou` and `lerobot` are absent from the robocasa env by choice; see section 2.
