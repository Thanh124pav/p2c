# Harness contract

Rules the experiment harness enforces so that two runs differing in one factor are
actually comparable. They are properties of the apparatus, not of any hypothesis, so they
outlive the research question in [PLAN.md](../PLAN.md) and apply to whatever P2C method is
eventually frozen in Stage 1.

Each rule exists because violating it produces a *clean-looking table of numbers that
means nothing* — the failure mode is silent, which is why these live in code and tests
rather than in a checklist.

## C1 — One factor varies

Every arm of a comparison shares demonstrations, train/validation split, preprocessing,
targets, optimiser settings and model size. Only the condition under study differs.

Enforced: one config file per experiment setting, with the condition passed on the command
line (`configs/README.md` explains why there is no per-arm config file).

## C2 — Capacity matching

A multi-input model must not be larger than a single-input one, or any gain could be extra
capacity rather than extra information.

Enforced: one encoder shared across views plus a view-count-independent fusion (`mean`,
`attn`), so parameter counts are identical for 1, 2 and 3 views. `concat` fusion is
available but emits a warning and is recorded in run metadata as violating this rule.
Asserted by `tests/test_model_forward.py::test_parameter_count_is_identical_across_view_counts`.

## C3 — Fixed sample count and order

A stochastic condition must not change how many samples exist, or in what order.

Enforced: conditions are built from a fixed number of slots; randomness selects *what fills
a slot*, never how many there are. Asserted by
`tests/test_dataset.py::test_sample_count_is_identical_across_conditions`.

## C4 — Targets independent of the condition

Actions and proprioception are read at the same frame index regardless of condition, and
normalised with statistics computed from the **training** episodes only.

Enforced and asserted by
`tests/test_dataset.py::test_action_target_is_unchanged_across_view_subsets`.

## C5 — Shared split

Train/validation is hashed from the cache contents and a split seed — never from the
condition, the model seed, or iteration order. This is what allows per-sample validation
errors to be joined frame by frame across arms, which the paired statistics depend on.

## C6 — Determinism without global RNG

Random draws hash `(seed, episode, frame)` rather than consuming global RNG state, so
dataloader workers, restarts and resumed runs agree.

## C7 — Augmentation is training-only

Augmentation runs in the training loop, not the Dataset. Validation frames therefore stay
byte-identical across conditions, and only training is regularised.

## C8 — Preprocessing happens once

Video is decoded once into a uint8 memmap cache. "Identical preprocessing across
conditions" becomes structural rather than a convention, and camera synchronisation becomes
true by construction because every camera writes to the same global frame index.

## C9 — Controls that separate information from scale

A comparison that only shows "more inputs help" has not shown that *complementary* inputs
help. Three controls separate them:

| Control | Question it answers |
|---|---|
| random extra input | Does it matter *which* extra evidence is added, or just that there is some? |
| duplicate input | Does repeating existing evidence, with no new information, give the same gain? |
| input dropout | Is the gain regularisation rather than information? |

## C10 — Experiment validity precedes falsification

Before any falsification verdict, check that at least one arm beats the predict-the-mean
baseline. If none does, the experiment has no signal, and that is a statement about the
apparatus — underpowered, or a task whose goal is not recoverable from the input — not
evidence against the hypothesis. A broken experiment and a falsified hypothesis are
indistinguishable in a table of numbers.

Enforced: `p2c.analysis.kill_criteria.check_validity`, reported as criterion 0 and
suppressing the ordinary verdict when it fires.

## C11 — Selection statistics need a null

A per-frame minimum over K noisy models is biased downwards even when the models carry no
real per-frame structure. Any "oracle selection headroom" must therefore be compared
against a permutation null that destroys frame-level association while preserving each
model's marginal error distribution.

Enforced: `p2c.analysis.complementarity._selection_bias_floor`. On the first task this
mattered: raw headroom read 32% while the noise floor was 90%, so the apparent headroom was
entirely selection bias.

## C12 — Run metadata is complete enough to re-run

Each run records git commit, resolved config, dataset identity, condition, split, seed,
parameter count, resolution, metric history, checkpoint, and captured stdout/stderr.

Enforced: `p2c.utils.tracking.RunLogger`.
