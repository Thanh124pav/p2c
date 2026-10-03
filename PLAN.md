# P2C Research and Execution Plan

## Partial-to-Complementary Visual Learning for Robot Manipulation

Last updated: 2026-10-03

This document defines the current research scope and execution plan for P2C. It replaces the previous `SETUP.md`, which was centered on an early multi-view RoboCasa MVP.

The target is a **CVPR-style vision-centric paper**. The central problem is not to build a better RoboCasa policy. The central problem is to learn useful manipulation knowledge from **cheap, partial, action-free human videos**, then test whether that visual knowledge transfers to embodied manipulation.

The immediate deadline is to freeze a concrete method by **Monday, 2026-10-05**, then begin experiments.

---

# 1. Core Research Target

P2C targets:

> **Partial, cheap, action-free human manipulation video from the Internet.**

A source sample should primarily contain:

- RGB video frames;
- optional weak text such as title, caption, or task description;
- no robot action labels;
- no proprioception;
- no reward;
- no calibrated simulator state.

The target domain is **object-centric human manipulation**, not generic Internet video.

Examples include:

- opening a drawer or cabinet;
- grasping and moving an object;
- inserting an object into a container;
- pouring;
- cutting or tool use;
- assembly;
- rearrangement;
- cleaning or wiping.

"Partial" means that the observed clip contains only incomplete visual evidence of a larger manipulation process. This may arise naturally from temporal cropping, occlusion, missing task phases, viewpoint limitations, or incomplete instructional clips.

The final method is intentionally **not specified in this file yet**. Method design is the next research step.

---

# 2. Paper Framing

The paper should remain vision-centric.

The desired high-level structure is:

```text
cheap partial human video
        |
        v
       P2C
        |
        v
transferable visual / temporal manipulation knowledge
        |
        v
robot policy learning
        |
        v
simulation + real-robot evaluation
```

Simulation environments are therefore **evaluation apparatus and controlled data sources**, not the research problem itself.

The main paper claim should not be framed as:

> We improve policy learning on RoboCasa.

Instead, the intended framing is closer to:

> Can incomplete visual evidence from cheap human videos be transformed into complementary manipulation knowledge that improves embodied learning?

---

# 3. Data Hierarchy

## 3.1 Target source data: Internet human manipulation video

Primary target distribution:

- action-free human manipulation videos;
- cheap / scalable;
- naturally partial and noisy;
- no robot state or action supervision.

Candidate paper-level sources:

1. **Action100M manipulation subset** — preferred source for the explicit cheap-Internet-video story.
2. **Ego4D / Ego-Exo4D manipulation clips** — useful as a more controlled human-video source and for analysis.

Do not require either dataset for the first local method prototype.

## 3.2 Robot demonstration data

Robot demonstrations are downstream supervision, not the main source data.

They may contain:

- RGB observations;
- robot actions;
- optional proprioception;
- task metadata.

Ground-truth object poses, full simulator states, segmentation labels, contact flags, or rewards should not be required by the main P2C method. They may be used for evaluation, controlled analysis, or oracle experiments.

---

# 4. Training Paradigm

The main paradigm is:

```text
Stage A: action-free human-video learning / pretraining
        |
        v
Stage B: action-labeled robot imitation learning
        |
        v
Stage C: closed-loop evaluation
```

This is **not** primarily online RL.

It is also **not** classic offline RL because the source Internet videos do not provide `(state, action, reward, next_state)` tuples.

The working description is:

> **Action-free video learning followed by robot imitation learning.**

The exact objective used in Stage A remains part of the P2C method design.

---

# 5. Policy Architecture

The downstream policy should be deliberately conventional so policy novelty does not confound the contribution.

## Main policy

**Visual encoder + Diffusion Policy**

Typical downstream inputs:

```text
RGB visual representation
        +
optional proprioception
        |
        v
Diffusion Policy
        |
        v
action chunk / continuous robot actions
```

## Secondary architecture

**ACT** may be used as an architecture-robustness experiment after the main result is established.

Do not make OpenVLA, GR00T, pi-family models, or a custom VLA architecture part of the critical path before the core P2C effect is established.

---

# 6. Simulation Environments

The simulation stack is separated by purpose.

## 6.1 Current local development environment: RoboCasa

The repository already has a verified working RoboCasa environment:

```text
conda env: robocasa
Python 3.11
RoboCasa 1.0.1
robosuite 1.5.2
MuJoCo 3.3.1
```

This environment is currently the main development environment and **must not be destabilized**.

Existing RoboCasa365 datasets and the current data pipeline may be used for:

- code debugging;
- RGB trajectory loading;
- synthetic partialization;
- controlled experiments;
- small-scale policy sanity checks;
- stage-level analysis where annotations are available.

RoboCasa dataset videos already provide synchronized rendered RGB streams, so many visual-data experiments do not require simulator rendering locally.

## 6.2 Main controlled benchmark: LIBERO

LIBERO is planned as a main controlled manipulation benchmark, especially for transfer/generalization across task factors.

However, **LIBERO must not be installed into the existing RoboCasa environment**.

LIBERO and the current RoboCasa stack depend on incompatible robosuite / Python / dependency versions. Therefore use a separate environment, for example:

```text
p2c-robocasa   -> current development environment
p2c-libero     -> independent LIBERO-compatible environment
```

The P2C implementation should be simulator-agnostic and communicate through a common data interface.

Example logical batch interface:

```python
{
    "images": ...,      # [B, T, C, H, W]
    "proprio": ...,     # optional
    "actions": ...,     # available for robot demonstrations
    "task_id": ...,
}
```

No P2C core module should directly depend on RoboCasa-specific APIs unless isolated in an adapter.

## 6.3 Large realistic benchmark: RoboCasa365

RoboCasa365 is the main large-scale realistic household benchmark.

It is particularly useful for:

- diverse kitchen scenes;
- atomic and composite manipulation tasks;
- long-horizon manipulation;
- stage/subtask analysis;
- realistic visual variation.

## 6.4 Optional cross-simulator benchmark: ManiSkill

ManiSkill is optional and not on the critical path.

Use it only if an additional independent simulator is needed later. The local machine has previously shown Vulkan limitations, so ManiSkill should not block P2C development.

## 6.5 Environments not prioritized

- **DMC:** mainly locomotion/control; weak match to human manipulation video.
- **MetaWorld:** useful manipulation benchmark but lower priority for the current visual Internet-video story.

---

# 7. Local vs Server Execution

## 7.1 Local machine

Purpose:

- verify the full software path;
- inspect data;
- build visual preprocessing;
- prototype partialization;
- train tiny models;
- overfit small subsets;
- test method logic;
- run cheap controlled experiments.

Use the existing RoboCasa data pipeline first. Do not spend time replacing a working stack merely to use a simpler simulator.

The local GTX 1650 and limited RAM are not intended for full paper-scale training.

## 7.2 Remote server

Purpose:

- large video pretraining;
- full Diffusion Policy training;
- multi-seed evaluation;
- LIBERO benchmark runs;
- RoboCasa365 paper-scale runs;
- larger ablations;
- Action100M / Ego4D experiments.

The same P2C core implementation should be reused across local and remote execution.

---

# 8. Real-Robot Evaluation

If a real robot arm is available, use it as a **data-efficiency validation**, not as a way to collect a massive new dataset.

Recommended setting:

- 4-6 tabletop manipulation tasks;
- small demonstration budgets such as 10 / 25 / 50 demos per task;
- visually related human-video source data where possible.

Compare at minimum:

```text
robot demonstrations only
vs
robot demonstrations + raw human-video pretraining
vs
robot demonstrations + P2C
```

Primary question:

> Does P2C reduce the amount of expensive robot demonstration data needed for a given downstream success level?

Report closed-loop task success and data-efficiency curves.

---

# 9. Experimental Roles of Simulator Information

The simulator may expose much more information than the P2C method should consume.

## Main method input

Prefer:

- RGB video;
- optional weak language.

## Downstream robot policy

May use:

- RGB representation;
- robot actions as supervision;
- optional proprioception.

## Analysis / oracle only

May use:

- object poses;
- segmentation;
- contact state;
- full simulator state;
- reward;
- success flag;
- stage labels.

This separation is important for the CVPR framing. A method dependent on privileged simulator state would weaken the claim that it applies to cheap Internet video.

---

# 10. Current Repository State

The repository already contains useful infrastructure from the earlier controlled multi-view study, including:

- RoboCasa365 metadata and video readers;
- memmap frame cache;
- camera-subset logic;
- lightweight visual encoders;
- fusion modules;
- tiny behavior-cloning policy;
- stage and complementarity analysis;
- experiment tracking;
- local configs and tests.

Keep reusable infrastructure, but do not let the old multi-view hypothesis define the final P2C method.

`PROGRESS.md` remains the source of truth for machine-specific facts, verified package versions, downloaded datasets, passing tests, and known environment limitations.

---

# 11. Execution Stages

## Stage 0 — Literature and gap

Goal: establish precisely what is already covered by action-free video learning, latent-action models, video-to-robot transfer, temporal completion, cross-video learning, and partial-observation learning.

Output:

- overlap map;
- strongest baselines;
- explicit research gap;
- falsifiable P2C hypothesis.

## Stage 1 — Method freeze

Deadline: **2026-10-05**.

Output:

- precise definition of partial input;
- precise definition of complementary information;
- training objective;
- model components;
- inference path;
- minimal baselines;
- first falsification experiment.

Do not expand the research question after this point without experimental evidence that the frozen version fails for a fundamental reason.

## Stage 2 — Local proof of pipeline

Use the existing RoboCasa stack.

Requirements:

- create partial visual samples from existing full trajectories;
- run P2C forward/backward;
- overfit a tiny dataset;
- compare against minimal controls;
- verify that the downstream policy can consume the learned representation.

This stage is about correctness, not paper-level numbers.

## Stage 3 — Controlled simulation experiments

Run the first meaningful multi-seed experiments on manipulation tasks.

Primary environments:

1. LIBERO in its own environment;
2. RoboCasa365 in the existing / server-compatible RoboCasa environment.

Evaluate both representation-level metrics where justified and closed-loop robot success.

## Stage 4 — Internet-video scaling

Introduce the real target source distribution:

- Action100M manipulation clips;
- Ego4D / Ego-Exo4D manipulation clips.

Test scaling with source-data quantity and source-data quality / partiality.

## Stage 5 — Real robot

Validate whether P2C improves robot data efficiency and transfers outside simulation.

---

# 12. Baseline Families

The exact list will be frozen after literature review, but experiments should distinguish at least:

1. **Robot-only imitation learning** — no human-video pretraining.
2. **Raw video pretraining** — same source videos without the P2C mechanism.
3. **Standard visual self-supervised / video representation baseline** where appropriate.
4. **Action-free / latent-action video-to-robot baseline** relevant to the final formulation.
5. **Oracle / full-information control** when simulation ground truth allows it.

Do not compare only against weak BC baselines if the final contribution overlaps current action-free video pretraining methods.

---

# 13. Critical Design Rules

1. **P2C core must be simulator-agnostic.**
2. **Do not install LIBERO into the current RoboCasa environment.**
3. **Do not require privileged simulator state in the main method.**
4. **Do not make a new policy architecture the contribution.**
5. **Do not scale before the local hypothesis has a falsifiable signal.**
6. **Do not let the existing multi-view implementation lock the research problem into multi-view selection.**
7. **Simulation results must ultimately support the visual-learning claim, not replace it.**
8. **Real-robot experiments should emphasize data efficiency, not dataset size.**

---

# 14. Immediate Next Step

The next task is **method design**, not further environment setup.

By Monday, the project should have a P2C-v0 specification containing:

```text
problem formulation
partial-data model
complementary-information definition
model architecture
training loss
source / target training flow
inference flow
minimal baselines
first experiment
kill criterion
```

Only after that specification is frozen should implementation expand beyond the already working infrastructure.
