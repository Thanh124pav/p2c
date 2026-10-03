# P2C — Partial-to-Complementary Visual Learning for Robot Manipulation

Infrastructure for the research programme in [PLAN.md](PLAN.md). The plan is the source of
truth for the research question; this README covers what is built and how to run it.
[PROGRESS.md](PROGRESS.md) is the source of truth for machine-specific facts: verified
package versions, downloaded datasets, passing tests, environment limits.

## What this repository is, and is not

The target ([PLAN.md §1](PLAN.md)) is **partial, cheap, action-free human manipulation
video** — RGB plus at most weak text, with no action labels, no proprioception, no reward
and no simulator state. The question is whether incomplete visual evidence from such video
can be turned into complementary manipulation knowledge that improves embodied learning.

Simulation is therefore **evaluation apparatus and a controlled data source**, not the
research problem ([§2](PLAN.md)). Nothing here should be read as "we improve policy
learning on RoboCasa".

**The P2C method itself is not implemented yet.** [§14](PLAN.md) puts method design before
further implementation, and [§13.5](PLAN.md) forbids scaling before a falsifiable local
signal. What exists today is the infrastructure [§10](PLAN.md) says to keep: RoboCasa365
readers, a frame cache, partialization, small encoders, a tiny BC policy, analysis and
tracking.

## The information tiers ([§9](PLAN.md))

The simulator exposes far more than the method may consume, and conflating the tiers would
undermine the central claim — a method that needs privileged simulator state does not
transfer to cheap Internet video. The code enforces the separation rather than relying on
discipline:

| Tier | May use | Enforced by |
|---|---|---|
| **Main method input** | RGB video, optional weak language | `Tier.METHOD` views of a sample |
| **Downstream policy** | RGB representation, robot actions as supervision, optional proprioception | `Tier.POLICY` |
| **Analysis / oracle only** | stage labels, object poses, segmentation, contact, full sim state, reward, success | `Tier.ORACLE` |

`p2c.data.tiers` defines these, and `CameraSubsetDataset` tags every field it emits. Asking
a sample for a field above the tier you declared raises rather than silently returning it.
Stage labels in particular are **oracle-only**: they are legitimate for the per-stage
analysis, and illegitimate as policy input.

## Partialization ([§1](PLAN.md), [§6.1](PLAN.md))

"Partial" arises from temporal cropping, occlusion, missing task phases, viewpoint limits,
or incomplete clips. `p2c.data.partialization` treats these as interchangeable axes over a
full trajectory, so an experiment can vary *how* a sample is made partial without the code
assuming a particular answer. This matters because [§13.6](PLAN.md) warns against letting
the earlier multi-view implementation lock the research problem into view selection.

Available axes:

| Axis | What it removes | Class |
|---|---|---|
| viewpoint | camera streams | `ViewpointPartial` |
| temporal | frames outside a window | `TemporalCropPartial` |
| phase | frames belonging to whole task stages | `PhaseDropPartial` |
| occlusion | image regions | `OcclusionPartial` |

Each returns the kept sample plus a record of what was withheld, which is what makes
"complementary evidence" measurable against a known ground truth.

## Setup

The verified environment is recorded in [PROGRESS.md](PROGRESS.md) and
[PLAN.md §6.1](PLAN.md). It **must not be destabilized**.

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

RoboCasa itself needs a specific install order; plain `pip install -e .` hangs. See
[PROGRESS.md §2](PROGRESS.md) for the working sequence and why.

LIBERO goes in **its own environment** — [PLAN.md §13.2](PLAN.md) forbids installing it
into the RoboCasa environment, and its pins (`numpy==1.22.4`, `robosuite==1.4.0`,
`gym==0.25.2`, Python 3.8) are incompatible with this one.

## Pipeline

```bash
# Verify the dataset schema. Nothing about cameras or shapes is hard-coded.
python scripts/inspect_dataset.py --task StackBowlsCabinet
python scripts/visualize_episode.py --task StackBowlsCabinet --episode 0 --mode grid

# Decode mp4 once into a uint8 memmap cache.
python scripts/build_frame_cache.py --task StackBowlsCabinet --num-episodes 120 --res 96

# Sanity gate: must reach near-zero loss or stop.
python scripts/train_local_bc.py --config configs/overfit.yaml --views primary --overfit 32

# A controlled experiment over partialization conditions.
CACHE=outputs/cache/StackBowlsCabinet_target_r96_e120 CONFIG=configs/composite.yaml \
  bash scripts/run_view_ablation.sh

python scripts/summarize_view_ablation.py --input outputs/runs_composite
```

RoboCasa365 ships mp4 already rendered for three synchronised cameras, so the data path
reads parquet + mp4 and never calls MuJoCo. It needs no GPU renderer, which is what makes
it runnable here without a usable Vulkan ICD.

## What keeps a comparison honest

These are properties of the harness, not of any particular hypothesis, so they survive the
method still being undesigned:

- **Capacity matching.** One encoder is shared across views and the default fusion is
  view-count independent, so 1-, 2- and 3-view models have *identical* parameter counts.
  Asserted in `tests/test_model_forward.py`. `concat` fusion is available but warns and is
  recorded as violating the property.
- **Fixed sample count and order.** Every condition yields the same samples in the same
  `(episode, frame)` order, so a stochastic condition cannot quietly change the dataset.
- **Identical targets.** Actions and states are read at the same frame index regardless of
  condition, normalised with training-split statistics only.
- **Shared split.** Train/validation is hashed from the cache and a split seed, never from
  the condition, so per-sample errors can be joined frame by frame across conditions.
- **Determinism.** Random draws hash `(seed, episode, frame)` instead of touching global
  RNG state, so dataloader workers and resumed runs agree.
- **Augmentation is training-only**, applied in the loop rather than the Dataset, so
  validation frames stay byte-identical across conditions.

## Analysis

`scripts/summarize_view_ablation.py` reports overall, per-task and per-stage tables with
CSVs and figures, **paired** bootstrap intervals (all conditions see the same validation
frames), the complementarity metric, and an automated falsification verdict.

Two guards exist because the first runs produced misleading numbers:

- **Experiment validity.** If no condition beats the predict-the-mean baseline, the summary
  says the experiment has no signal and refuses to record a falsification. A broken
  experiment and a falsified hypothesis look identical in a table of numbers.
- **Selection-bias null for oracle headroom.** A per-frame minimum over K noisy models is
  biased downwards even with no real structure. On the first task the raw headroom read
  32% while the permutation noise floor was 90% — the apparent headroom was entirely
  selection bias, and without the null it would have read as evidence to build a selector.

## Status

Running experiments and verified facts: [PROGRESS.md](PROGRESS.md).
Research direction and stage gates: [PLAN.md](PLAN.md).

## Layout

```
PLAN.md        research programme — the source of truth for the question
PROGRESS.md    machine-specific verified facts, environment, results
configs/       run configurations; only the condition differs between arms
p2c/data/      schema discovery, frame cache, tiers, partialization, dataset
p2c/models/    tiny BC policy, shared view encoder, fusion
p2c/analysis/  result joining, stage metrics, complementarity, falsification
scripts/       inspect, visualize, cache, train, summarize
tests/         run against a synthetic cache; no dataset or GPU needed
external/      upstream RoboCasa / robosuite / diffusion_policy (gitignored)
datasets/      downloaded LeRobot data (gitignored)
outputs/       caches, runs, summaries (gitignored)
```
