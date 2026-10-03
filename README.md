# P2C — Partial-to-Complementary Visual Data for Robot Learning

A controlled data study, not a method. The implementation plan is [SETUP.md](SETUP.md);
this README covers how to run what has been built.

## The question

Does complementary visual evidence measurably improve robot learning compared with a
partial observation, and is the improvement task/stage dependent rather than merely caused
by adding more pixels?

The headline comparison is deliberately not "more views help". It is:

```
Performance(partial + complementary)  >  Performance(partial + random)
```

If that inequality does not hold, the result reduces to generic multi-view scaling, and
[SETUP.md section 21](SETUP.md) says to stop rather than force a method.

## Why this runs on a 4 GB GPU without a working renderer

RoboCasa365 ships its datasets in LeRobot format with **mp4 already rendered for three
synchronised cameras** (`robot0_agentview_left`, `robot0_agentview_right`,
`robot0_eye_in_hand`, all 256x256 at 20 fps). The MVP therefore reads parquet + mp4
directly and never calls MuJoCo, so it needs neither Vulkan nor EGL. Offscreen rendering
is only required for policy *rollout* evaluation, which belongs to the remote Diffusion
Policy stage.

The three real cameras cover every view condition the plan asks for, with
`agentview_left` as primary, `agentview_right` as secondary and `eye_in_hand` as wrist.

## Setup

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Datasets are downloaded per task (roughly 0.3–0.6 GB for an atomic task, 0.7–2.1 GB for a
composite one, each with 500 demonstrations):

```bash
python -m robocasa.scripts.download_datasets --tasks NavigateKitchen --split target
```

`robocasa` is only needed for that download helper and for the simulator stages. See
[requirements.txt](requirements.txt) for why it is installed separately.

## Pipeline

### Stage 0 — verify the dataset schema

Nothing about camera names, resolutions or action dimensions is hard-coded; it is all read
off disk and checked.

```bash
python scripts/inspect_dataset.py --task NavigateKitchen
python scripts/visualize_episode.py --task NavigateKitchen --episode 0 --mode grid
```

### Stage 1 — build a frame cache

Decodes every camera's mp4 **once** into a uint8 memmap. Two reasons beyond speed: it makes
"identical preprocessing across conditions" ([SETUP.md section 7](SETUP.md)) a structural
guarantee rather than a convention, and it makes camera synchronisation true by
construction, since every camera writes to the same global frame index.

```bash
python scripts/build_frame_cache.py --task NavigateKitchen --num-episodes 40 --res 84
```

### Stage 2 — sanity-check, then ablate

```bash
# Overfit test (SETUP.md section 18): must reach near-zero loss or stop here.
python scripts/train_local_bc.py --config configs/overfit.yaml --views primary --overfit 32

# One condition.
python scripts/train_local_bc.py --config configs/local_debug.yaml --views primary+wrist

# The full ablation with everything else held fixed.
CACHE=outputs/cache/NavigateKitchen_target_r84_e40 bash scripts/run_view_ablation.sh

# Tables, figure, complementarity metric and kill-criteria verdict.
python scripts/summarize_view_ablation.py --input outputs/runs
```

## Choosing a task: the goal must be visually determined

This matters more than it looks, and it is easy to get wrong.

A vision-only behaviour-cloning policy can only beat the predict-the-mean baseline if the
action is a function of the observation. On a task whose goal is carried by the *language
instruction* rather than the image, the same view maps to different actions, so the best
achievable prediction is the conditional mean — and every camera condition sits at that
floor regardless of how much visual evidence it gets.

Measured instruction diversity in the three downloaded target datasets:

| Task | Instructions | Episodes | Verdict |
|---|---|---|---|
| `NavigateKitchen` | **14** (sink / stove / fridge / …) | 500 | goal not visually determined — unusable for this study without goal conditioning |
| `PickPlaceCounterToCabinet` | 106 (one per object) | 502 | target object is visible on the counter, so largely visually determined |
| `StackBowlsCabinet` | **1** | 515 | unambiguous, **and** carries per-frame stage labels |

`StackBowlsCabinet` is therefore the primary task: single instruction, stage annotations,
and manipulation-heavy, which is where occlusion and fine alignment — the mechanisms P2C
is about — actually bite.

Every run prints its predict-the-mean baseline and the number of instructions in the
cache, and `summarize_view_ablation.py` reports **criterion 0, experiment validity**,
before the kill criteria. If no arm clears the baseline, the summary says the ablation has
no signal and explicitly refuses to record a kill. A broken experiment and a falsified
hypothesis look identical in a table of numbers; they are not the same thing, and
conflating them is the easiest way to kill a project for the wrong reason.

## View conditions

Mapped to the baselines of [SETUP.md section 9](SETUP.md):

| Condition | Baseline | Meaning |
|---|---|---|
| `single_primary` | B0 | partial observation — the reference everything is measured against |
| `single_wrist` | B1 | alternative single view |
| `primary+wrist` | B2 | fixed complementary pair |
| `all_views` | B3 | full-view oracle, the dataset's upper bound |
| `random_two` | B4 | primary + random second view — **Control A** |
| `primary+duplicate` | — | the same camera twice — **Control D** |
| `primary+wrist_dropout` | — | pair with stochastic view dropout — **Control C** |

`B5` (learned view selection) is deliberately not implemented: section 9 gates it on
B0–B4 first producing a convincing phenomenon. What *is* implemented is the oracle
complementarity target and a measurement of how much headroom a selector would have — if
an oracle per-frame selector barely beats the best fixed pair, a learned one cannot do
better, and B5 is not worth building.

## What keeps the comparison honest

The controls are the point of the study, so they are enforced in code rather than left to
discipline:

- **Capacity matching (Control B).** One encoder is shared across views and the default
  fusion is view-count independent, so a 1-view and a 3-view model have *identical*
  parameter counts. `tests/test_model_forward.py` asserts this. `concat` fusion is
  available but warns and is recorded as violating the control.
- **Fixed sample count and order.** Every condition yields the same number of samples in
  the same `(episode, frame)` order, so a random-view arm cannot quietly change the
  dataset. Asserted in `tests/test_dataset.py`.
- **Identical targets.** Actions and states are read at the same frame index regardless of
  camera subset, and normalised with statistics from the training episodes only.
- **Shared split.** The train/validation split is hashed from the cache and a split seed,
  never from the view condition, so per-sample errors can be joined frame by frame across
  conditions.
- **Determinism.** Random camera draws hash `(seed, episode, frame)` instead of touching
  global RNG state, so dataloader workers and resumed runs agree.

## Analysis

`scripts/summarize_view_ablation.py` reports:

- overall, per-task and per-stage tables plus CSVs and a comparison figure;
- **paired** bootstrap confidence intervals, since all conditions are evaluated on the
  same validation frames — a point estimate alone cannot separate a real gain from scatter
  at this data scale;
- `C(v | p) = L(pi_p) - L(pi_{p+v})` per candidate view ([section 13](SETUP.md));
- the frames where the extra view helps most, for inspection;
- an automated verdict on the seven kill criteria of
  [section 21](SETUP.md). Criteria 1–5 are computed; 6 needs the camera-pose baseline that
  section 15 defers past the MVP, and 7 is a judgement, so both are reported as *not
  evaluated* rather than as passes.

Stage-dependence (`criterion 5`) is tested with a permutation test on stage labels, not by
reading a table — "gains are uniform across stages" is a claim about spread and deserves a
test. Stage labels exist only for **target composite** tasks; atomic tasks carry
episode-level language only.

## Status

See [PROGRESS.md](PROGRESS.md) for what has been verified on this machine and what is
still open.

## Layout

```
configs/      run configurations; only `views` differs between ablation arms
p2c/data/     schema discovery, frame cache, camera-subset dataset
p2c/models/   tiny BC policy, shared view encoder, fusion
p2c/analysis/ result joining, stage metrics, complementarity, kill criteria
scripts/      inspect, visualize, build cache, train, ablate, summarize
tests/        dataset, camera-subset and model tests (run against a synthetic cache)
external/     upstream RoboCasa / robosuite checkouts (gitignored)
datasets/     downloaded LeRobot data (gitignored)
outputs/      caches, runs, summaries (gitignored)
```
