# P2C Setup Plan
## Partial-to-Complementary Visual Data for Robot Learning

This document is an execution plan for setting up the initial P2C codebase and running the first controlled experiments.

The immediate goal is **not** to implement novel-view synthesis, 3D reconstruction, or Internet-video processing.

The immediate goal is to answer one question:

> **Does complementary visual evidence measurably improve robot learning compared with a partial observation, and is the improvement task/stage dependent rather than merely caused by adding more pixels?**

If this phenomenon is not clearly observed, stop before implementing a new method.

---

# 1. Research Hypothesis

We assume a robot demonstration contains a latent task-relevant physical state \(s_t\), while an individual camera provides only a projection:

\[
I_t^{(v)} = P_v(s_t).
\]

A single observation can therefore be **task-insufficient** even if it looks visually informative.

P2C targets:

\[
\text{partial visual observation}
\rightarrow
\text{detect missing task-relevant evidence}
\rightarrow
\text{recover/select complementary evidence}
\rightarrow
\text{robot-fit training sample}.
\]

For the MVP, we do not recover missing evidence synthetically. We use real synchronized multi-view observations as an oracle source of complementary evidence.

The first phenomenon to establish is:

\[
\text{Performance}(V_\text{partial}+V_\text{complementary})
>
\text{Performance}(V_\text{partial})
\]

and, more importantly,

\[
\text{Performance}(V_\text{partial}+V_\text{complementary})
>
\text{Performance}(V_\text{partial}+V_\text{random}).
\]

The second inequality is essential. Otherwise the result may simply mean “more views / more pixels help.”

---

# 2. Scope

## In scope for the first implementation

1. Install and verify RoboCasa.
2. Inspect synchronized multi-view demonstrations.
3. Build a configurable camera-subset dataset interface.
4. Train/debug a lightweight behavior-cloning baseline locally.
5. Prepare Diffusion Policy experiments for remote GPU execution.
6. Run controlled single-view vs multi-view ablations.
7. Measure results by task and, where available, by manipulation stage.
8. Build a minimal complementary-view selector only after the phenomenon is validated.

## Explicitly out of scope for now

Do **not** implement any of the following before the basic phenomenon is validated:

- Internet-video retrieval.
- 3D Gaussian Splatting.
- NeRF.
- 4D reconstruction.
- video diffusion.
- novel-view synthesis.
- depth completion as a new model.
- OpenVLA / π0 / GR00T training.
- custom VLA architectures.
- cross-video consensus.
- new policy architecture.

Existing pretrained vision modules may be used later, but the first milestone is a controlled data study.

---

# 3. Primary Codebases

## 3.1 RoboCasa

Official repository:

```bash
git clone https://github.com/robocasa/robocasa.git
cd robocasa
```

Use the current official installation instructions from the repository.

RoboCasa365 is preferred because it provides large-scale demonstrations, multiple camera observations, policy benchmarking support, and per-frame annotations for target composite datasets.

Important dataset properties to verify programmatically rather than assuming:

- available camera names;
- image tensor shapes;
- whether streams are synchronized;
- action shape;
- proprioceptive state;
- task metadata;
- per-frame subtask / stage annotations;
- episode boundaries.

Do not hard-code camera names until they are inspected from the installed dataset.

## 3.2 RoboCasa Diffusion Policy

Official RoboCasa fork:

```bash
git clone https://github.com/robocasa-benchmark/diffusion_policy.git
cd diffusion_policy
pip install -e .
```

Official entry points currently include:

```text
train.py
eval_robocasa.py
diffusion_policy/scripts/get_eval_stats.py
```

Do not modify the official policy architecture in the first experiments except where necessary to support configurable observation cameras.

---

# 4. Hardware Strategy

The local machine is intended for:

- installation;
- dataset inspection;
- visualization;
- unit tests;
- preprocessing;
- camera-subset logic;
- small forward/backward tests;
- overfitting tiny subsets;
- lightweight BC sanity experiments.

Do not attempt full Diffusion Policy training on the local GTX 1650 4 GB GPU.

The official RoboCasa Diffusion Policy documentation recommends at least 24 GB GPU memory for training and at least 8 GB for inference. Full experiments should therefore run on a remote GPU.

Use the local GPU only to prove that code is correct.

---

# 5. Repository Layout

Create a separate project repository rather than heavily modifying upstream RoboCasa.

Suggested structure:

```text
p2c/
├── README.md
├── SETUP.md
├── requirements.txt
├── configs/
│   ├── local_debug.yaml
│   ├── bc_single_view.yaml
│   ├── bc_multi_view.yaml
│   └── diffusion_policy/
├── p2c/
│   ├── data/
│   │   ├── robocasa_dataset.py
│   │   ├── camera_subset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── tiny_bc.py
│   │   ├── view_encoder.py
│   │   └── fusion.py
│   ├── analysis/
│   │   ├── stage_metrics.py
│   │   ├── complementarity.py
│   │   └── visualization.py
│   └── utils/
├── scripts/
│   ├── inspect_dataset.py
│   ├── visualize_episode.py
│   ├── train_local_bc.py
│   ├── eval_local_bc.py
│   ├── run_view_ablation.sh
│   └── summarize_view_ablation.py
├── tests/
│   ├── test_dataset.py
│   ├── test_camera_subset.py
│   └── test_model_forward.py
└── outputs/
```

Keep RoboCasa and Diffusion Policy as external sibling repositories or git submodules if convenient.

---

# 6. Stage 0 — Environment Verification

Before changing any training code, create:

```bash
python scripts/inspect_dataset.py
```

It must print:

```text
dataset name
number of episodes
number of frames
task name
action shape
proprioception shape
available image keys
camera names
image resolution
episode lengths
available per-frame annotation keys
```

Also create:

```bash
python scripts/visualize_episode.py --episode <id>
```

It should produce a synchronized visualization of all available views.

Acceptance criteria:

- every camera stream can be loaded;
- all camera frames for timestep \(t\) correspond to the same timestep;
- actions are aligned with frames;
- episode boundaries are correct;
- stage/subtask annotations are visible when available.

Do not proceed until this works.

---

# 7. Stage 1 — Camera-Subset Dataset Interface

Implement one reusable abstraction:

```python
CameraSubsetDataset(
    dataset,
    cameras=[...],
    camera_dropout=0.0,
    camera_order=...,
)
```

The policy must not care whether the observation contains 1, 2, or 3 cameras.

Support at least:

```text
single_primary
single_wrist
primary+wrist
primary+secondary
all_views
random_two
```

The actual camera names must be resolved from the dataset inspection.

Requirements:

1. Same demonstrations for all ablations.
2. Same train/validation split.
3. Same image preprocessing.
4. Same action targets.
5. Same optimization settings.
6. Deterministic camera sampling under a fixed seed.
7. Log selected camera IDs for every experiment.

Do not let random-view experiments silently change sample count.

---

# 8. Stage 2 — Local Tiny BC Sanity Baseline

Before Diffusion Policy, create a deliberately small model that fits on GTX 1650.

Suggested baseline:

```text
camera image
→ small pretrained/frozen visual encoder or tiny CNN
→ per-view feature
→ simple mean/concat fusion
→ MLP or tiny transformer
→ action regression
```

Keep this model simple. It is only a phenomenon detector.

Recommended local debugging configuration:

```text
one task
small trajectory subset
64x64 / 84x84 / 96x96 images
batch size: as large as fits
mixed precision if stable
short history
short action horizon or one-step action prediction
few epochs
```

Run at least:

```text
L:      primary camera only
W:      wrist camera only
L+W:    primary + wrist
ALL:    all available views
RAND2:  primary + random second view
```

Metrics:

```text
validation action MSE / L1
rotation error if actions separate orientation
gripper classification/error if applicable
task-stage-conditioned validation loss
```

The local experiment is not a paper result. It is a fast falsification test.

---

# 9. Stage 3 — Main Controlled Ablations

After the local pipeline works, prepare full training using the official RoboCasa Diffusion Policy implementation.

Required baselines:

## B0 — Partial observation

```text
primary third-person camera only
```

## B1 — Alternative single view

```text
wrist only
```

## B2 — Fixed complementary pair

```text
primary + wrist
```

## B3 — Full-view oracle

```text
all available synchronized views
```

This defines the approximate upper bound available from the dataset.

## B4 — Random second view

```text
primary + randomly selected additional view
```

This is crucial to separate complementarity from generic multi-view scaling.

## B5 — Learned view selection

Do not implement B5 until B0–B4 produce a convincing phenomenon.

Initial B5 may select one extra real view from available synchronized cameras:

\[
v^* = \arg\max_v C(v \mid I_\text{primary}, \text{task})
\]

where \(C\) is a learned complementarity score.

No image generation is necessary yet.

---

# 10. Task Selection

Do not start with the full 365-task benchmark.

Select approximately 3–5 tasks with qualitatively different visual requirements.

Prefer tasks that approximately cover:

```text
A. gross motion / navigation-like phase
B. reaching / approach
C. fine alignment
D. grasp/contact
E. placement/insertion
```

The goal is to test whether complementary information is **stage dependent**.

An ideal result would look qualitatively like:

```text
navigation       +0–2%
approach         +1–4%
alignment        +10–20%
grasp/contact    +10–20%
placement        +5–15%
```

Exact numbers are not expected. The important point is a structured difference across stages.

If every stage improves equally, investigate whether the effect is merely increased input capacity.

---

# 11. Stage-Level Analysis

RoboCasa365 target composite datasets include per-frame annotations such as subtask index, atomic skill, stage, and natural-language instruction.

If available for the selected dataset, compute:

\[
\Delta_{\text{view}}(g)
=
L_{\text{partial}}(g)
-
L_{\text{multi}}(g)
\]

for stage \(g\).

Also compute policy success when possible:

\[
\Delta SR(g)
=
SR_{\text{multi}}(g)
-
SR_{\text{partial}}(g).
\]

Create plots/tables for:

```text
task
stage
camera subset
validation loss
success rate
```

Add visualization of frames where multi-view improves the prediction most.

These samples are especially important for understanding whether P2C is a real problem.

---

# 12. Controls Against “More Pixels = Better”

This is mandatory.

At minimum include:

## Control A — Random extra camera

Compare:

\[
\text{primary+wrist}
\quad\text{vs}\quad
\text{primary+random}.
\]

## Control B — Capacity-matched encoder

Ensure the single-view baseline is not trivially weaker because the multi-view model has more trainable parameters.

Possible implementation:

- use the same visual encoder weights across cameras;
- share encoder across views;
- keep fusion lightweight;
- optionally add an equally sized projection head to the single-view baseline.

## Control C — Camera dropout

Train a multi-view policy with stochastic view dropout.

This tests whether performance is caused by regularization rather than true complementary information.

## Control D — Duplicate view

Feed the same camera twice:

```text
primary + duplicate(primary)
```

If duplicated information gives the same gain as real complementary views, the current hypothesis is weak.

---

# 13. Complementarity Metric — MVP

Only after B0–B4.

Start with a simple operational definition.

For candidate additional view \(v\):

\[
C(v \mid p)
=
L(\pi_p)
-
L(\pi_{p+v}),
\]

where \(p\) denotes the primary view.

This gives an oracle/data-derived complementarity target.

Use it to train a predictor:

\[
\hat C_\phi(I_p, I_v, \text{task})
\approx
C(v \mid p).
\]

Then select:

\[
v^*
=
\arg\max_v \hat C_\phi(I_p,I_v,\text{task}).
\]

This is only an initial formulation.

Do not introduce mutual information or complicated information-theoretic estimators unless the empirical signal justifies them.

---

# 14. Potential P2C Method After MVP

If the phenomenon is validated, evolve toward:

```text
primary observation
        |
        v
task-sufficiency estimator
        |
        +---- sufficient ----> use sample directly
        |
        +---- insufficient --> identify missing evidence
                                  |
                                  v
                         complementary evidence selector
                                  |
                                  v
                           robot-fit training sample
```

Longer-term versions may replace real additional cameras with:

```text
novel-view synthesis
depth
de-occlusion
object-centric reconstruction
temporal evidence
other recovered modalities
```

But these are later stages.

---

# 15. Baselines to Prepare After the MVP

Do not port all of these immediately.

## Priority 1 — VISTA

Reason:

```text
single-view demonstration
→ novel-view synthesis
→ viewpoint augmentation
→ policy
```

This is the most important baseline against a claim that P2C is more than generic view augmentation.

Repository:

```text
https://github.com/s-tian/VISTA
```

Initial goal:

- install;
- reproduce one provided task;
- understand camera sampling and generated-view interface.

Do not port it to RoboCasa until the P2C phenomenon is validated.

## Priority 2 — Task-Aware Virtual View Exploration / TVVE

Initial goal:

- install official implementation if available;
- reproduce one experiment;
- understand how task-aware viewpoint selection is defined;
- document exactly how P2C differs.

P2C should be framed around **training-data sufficiency and recovery**, not active viewpoint selection at execution time.

## Priority 3 — Camera-pose conditioning baseline

Use an existing camera-conditioning baseline if implementation is available.

Research question:

> Is a single view actually sufficient once the policy knows the camera pose?

If camera-pose conditioning closes the full gap, P2C's missing-evidence story becomes weaker.

---

# 16. Experiment Tracking

Every run must save:

```text
git commit hash
config file
dataset identifier/version
task
seed
camera subset
number of demonstrations
image resolution
policy parameter count
training steps
validation losses
success metrics
checkpoint
stdout/stderr
```

Prefer Weights & Biases if already configured; otherwise save structured JSON/CSV.

All experiment names should follow:

```text
p2c_<task>_<policy>_<views>_seed<seed>
```

Example:

```text
p2c_pick_bc_primary_seed0
p2c_pick_bc_primary-wrist_seed0
p2c_pick_bc_random2_seed0
p2c_pick_bc_all_seed0
```

---

# 17. Reproducibility

Set seeds for:

```python
random
numpy
torch
torch.cuda
dataset sampling
camera randomization
```

Where practical:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Record any operation that remains nondeterministic.

Use identical data splits across camera ablations.

---

# 18. Tests

At minimum implement:

## Dataset tests

- image keys exist;
- camera subset returns correct number of views;
- timestamps/indices remain aligned;
- action target unchanged across view subsets;
- deterministic random-camera sampling under fixed seed.

## Model tests

- single-view forward pass;
- multi-view forward pass;
- different camera counts;
- loss backward;
- no NaNs;
- batch size 1 works.

## Overfit test

Train on approximately 16–64 samples until near-zero training loss.

If this fails, do not start full experiments.

---

# 19. Required Scripts

Claude should finish the following runnable interfaces.

```bash
python scripts/inspect_dataset.py \
    --dataset <dataset>

python scripts/visualize_episode.py \
    --dataset <dataset> \
    --episode 0

python scripts/train_local_bc.py \
    --config configs/local_debug.yaml \
    --views primary

python scripts/train_local_bc.py \
    --config configs/local_debug.yaml \
    --views primary,wrist

python scripts/eval_local_bc.py \
    --checkpoint <path>

bash scripts/run_view_ablation.sh

python scripts/summarize_view_ablation.py \
    --input outputs/
```

The summarization script should generate at minimum:

```text
overall comparison table
per-task table
per-stage table
CSV output
one figure comparing view subsets
```

---

# 20. Remote Diffusion Policy Stage

Do this only after local sanity tests pass.

Official RoboCasa workflow currently follows the form:

```bash
python train.py \
  --config-name=train_diffusion_transformer_bs192 \
  task=robocasa/<dataset-soup>
```

Evaluation:

```bash
python eval_robocasa.py \
  --checkpoint <checkpoint-path> \
  --task_set <task-set> \
  --split <split>
```

Statistics:

```bash
python diffusion_policy/scripts/get_eval_stats.py \
  --dir <outputs-dir>
```

Do not use the default large batch size blindly. Adjust batch size for the actual GPU while keeping the effective batch size documented.

For the first remote experiment, train only:

```text
B0 primary
B2 primary+wrist
B3 all views
B4 primary+random second view
```

on one selected task.

Only scale to multiple tasks/seeds if this produces a meaningful difference.

---

# 21. Kill Criteria

This project should be falsifiable.

Stop or substantially reframe P2C if, after controlled experiments:

1. Multi-view provides no meaningful gain over the strongest single view.
2. Random additional views perform as well as supposedly complementary views.
3. Duplicate-view control performs similarly to real second views.
4. Gains disappear after matching model capacity.
5. Gains are uniform across all task stages and cannot be associated with missing visual evidence.
6. A simple camera-pose-conditioned single-view baseline closes almost all of the gap.
7. The only successful version requires full 3D/4D reconstruction, making the project indistinguishable from existing view-generation/reconstruction work.

A negative result here is useful: stop early rather than forcing a method.

---

# 22. Success Criteria for MVP

The MVP is successful when all of the following hold:

- [ ] RoboCasa dataset loads correctly.
- [ ] Multiple synchronized camera views are visualized.
- [ ] Camera-subset interface is implemented.
- [ ] Tiny BC can overfit a small subset.
- [ ] Single-view and multi-view experiments run from the same code.
- [ ] At least one task shows a reproducible single-view vs complementary-view gap.
- [ ] Random/duplicate-view controls are implemented.
- [ ] Complementary improvement varies meaningfully by task or stage.
- [ ] Results are summarized automatically.
- [ ] Full Diffusion Policy experiments are ready to launch remotely.

Only after these boxes are checked should development move to the actual P2C algorithm.

---

# 23. Immediate Claude Task

Start with the following exact objective:

> Build a reproducible RoboCasa multi-view ablation framework for P2C. Inspect the actual dataset schema first rather than assuming camera names. Implement configurable synchronized camera subsets, visualization, unit tests, and a tiny local BC baseline that can run on a 4 GB GPU. Support single-view, fixed complementary-view, random extra-view, duplicate-view, and all-view oracle conditions. Keep data splits, actions, preprocessing, and model capacity controlled across experiments. Do not implement novel-view synthesis or a new policy architecture. Once the tiny baseline works, prepare but do not launch full RoboCasa Diffusion Policy training configs for a remote GPU.

Prioritize correctness, reproducibility, and simple falsification of the P2C hypothesis over feature completeness.

---

# 24. Useful Official References

RoboCasa:

```text
https://github.com/robocasa/robocasa
```

RoboCasa datasets documentation:

```text
https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md
```

RoboCasa policy-learning documentation:

```text
https://github.com/robocasa/robocasa/blob/main/docs/benchmarking/policy_learning_algorithms.md
```

RoboCasa Diffusion Policy fork:

```text
https://github.com/robocasa-benchmark/diffusion_policy
```

VISTA:

```text
https://github.com/s-tian/VISTA
```

---

# 25. Final Principle

The first paper claim should not be:

> More camera views improve robot learning.

The intended claim is:

> **Robot training samples can be visually insufficient for the supervision they carry, and targeted complementary evidence can make those samples task-sufficient.**

The initial experiments should be designed to falsify or support this claim as cheaply as possible.
