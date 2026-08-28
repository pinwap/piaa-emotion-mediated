# Methodology

This document explains every decision that affects the numbers reported in
the paper, with the reasoning behind it. Anyone verifying this work should
read this before reading the code.

## 1. Model structure

Every experiment is a two-axis table.

- **Axis one - the mediator**, 7-dimensional, shared across users. It is
  fit only on images seen by the training-group users, then frozen.
- **Axis two - the head**, a single per-user layer. It is fit only on that
  user's own ratings.

Training is **sequential** in both the ridge and MLP cases: the mediator is
fit and frozen first, then the head is fit on the mediator's output. It is
not trained end-to-end.

Mediators compared:

| name | what the mediator is | question it answers |
|---|---|---|
| identity (Direct) | no mediator, raw features | does routing through a mediator help at all |
| emotion (ours) | predicts 7 emotions | does a *meaningful* mediator help |
| pca | unsupervised 7-dim compression | is the gain just dimensionality reduction |
| random | random linear projection to 7 dims | does any 7-dim mediator work |
| shuffled | emotion predictions shuffled across images | does the mediator need to match the image |

## 2. Leak-free data split

The 129 users are partitioned into 10 groups. Each fold uses 7 groups as
train, 1 as validation, and 2 as test, rotating over 5 folds so every user
is a test user exactly once, giving 387 evaluation units (129 users x 3
domains).

This closes three leakage points at once:

1. The mediator is fit only on train-group images, so it never sees a test
   user's images.
2. Every **shared** hyperparameter is selected on the validation group,
   which is disjoint from both train and test users (and whose images are
   disjoint from theirs). See "Where each hyperparameter comes from" below.
3. Each test user's own images are split into support and eval sets, so
   the head never trains on what it's scored on.

### Where each hyperparameter comes from

One rule, applied everywhere:

| component | shared or personal? | hyperparameter selected on |
|---|---|---|
| Stage-1 emotion mediator (ridge) | shared | validation user group |
| Shuffled mediator (ridge) | shared | validation user group |
| Stage-1 emotion mediator (MLP) | shared | validation user group |
| Population / GIAA head (ridge, MLP) | shared | validation user group |
| Pop-zero formula in `efficiency` | shared | validation user group |
| Per-user head (ridge, sparse linear, MLP) | personal | validation users, using the same support/eval protocol as test users |

For a personal head, each candidate is fitted separately to every validation
user's support set and scored on that validation user's 50-image evaluation
set. Scores are averaged across validation-user units, one hyperparameter is
frozen for the fold/domain/configuration, and that value is then applied to
all test users. No test user or test evaluation image participates in model
selection.

PCA (`n_components=7`) and the random projection have no hyperparameter to
select, so nothing is chosen for them from any data.

`uv run main.py verify --splits` checks points 1 and 3 automatically.

### Support/eval split per user

A user's images in each domain are shuffled with `RandomState(42 +
user_id)`, and **the first 50 images are held out as a fixed eval set**;
the rest form the support pool. Holding out eval first makes results
comparable across rating budgets, because a model trained on 10 images and
one trained on 100 are scored on the exact same images.

## 3. Using only first-session ratings

About 5% of the data (4,509 pairs) are images a user rated twice, in
separate sittings. All experiments **use only the first rating** and
reserve the second one exclusively for measuring test-retest reliability.
This leaves 83,327 of 87,836 rows in use.

"first" means first row in the file. Checked that
both ratings of a pair always land in the same split group.

## 4. Metrics

We report **SROCC and PLCC**, following standard practice in image
aesthetic assessment.

Comparisons between models use a **two-sided paired Wilcoxon signed-rank
test** on matched user-domain units. Seeds are first averaged within each
unit. A domain-specific test uses 129 users; the pooled Average concatenates
the three disjoint domains and uses 387 user-domain units.

Tables show mean +/- SEM unless their captions explicitly state otherwise.
Holm--Bonferroni correction is applied within the prespecified families in
`output/statistics/HOLM_FAMILY_MAP.md`; SROCC and PLCC are separate families,
and pooled Average is corrected as its own family.

## 5. Heads

### Ridge

Alpha comes from the same grid everywhere: **17 values from 1e-2 to 1e6**
(`numpy.logspace(-2, 6, 17)`). Features are always standardized first.

How the winner is picked depends on whether the component is shared or
personal (Sec. 2). A **shared** ridge is fit on the train group at each alpha
and scored by MSE on the validation group. A **personal** ridge mirrors the
full support/evaluation protocol across validation users and selects the alpha
with the best mean validation-unit SROCC. The selected value is frozen before
test-user fitting.

Fitting a mediator on shuffled labels and then selecting its alpha honestly
drives it toward the top of the grid (up to 1e6, i.e. maximal shrinkage), because
there is genuinely no signal for the validation group to reward. That is
the control behaving correctly, not a bug.

We also report effective degrees of freedom, defined as
`tr(Z(Z'Z + alpha*I)^-1 Z')` at the selected alpha, computed on standardized
features and averaged across units. This is defined only for linear heads.

### MLP

A single hidden layer of 128 units, ReLU, trained with MSE loss and no weight
decay (`alpha=0`). The learning rate is selected from 11 values from `1e-4`
to `1e1`; shared components use validation-group MSE, while personal heads
mirror the support/evaluation protocol across validation users and maximize
mean validation-unit SROCC. The selected rate is then frozen. Every fit uses
the fixed 500-epoch budget specified in advance. Early stopping is disabled.

## 6. Backbones

| name | feature file | dim |
|---|---|---|
| CLIP frozen | `clip_features.npz` | 512 |
| CLIP-ft (score) | `clip_ftpf_overall_v4_fold{k}.npz` | 768 |
| CLIP-ft (emotion) | `clip_ftpf_emotion_v4_fold{k}.npz` | 768 |
| Qwen3-VL 4B | `vlm4b_LT15.npz` | 2560 |
| Qwen3-VL 8B | `vlm_LT15.npz` | 4096 |

A fine-tuned backbone must always use the **per-fold** version, fine-tuned
only on that fold's train users.

## 7. Two upper bounds

- **GT emotions** uses the user's true emotion ratings instead of the
  mediator's prediction. This is the ceiling of the mediator pathway - how
  well the model would do if the mediator were perfect. It is measured
  within the same session, so it is optimistic.
- **Test-retest reliability** is a user's agreement with their own rating
  across sessions - a more realistic ceiling for deployment.

Neither is included when identifying the best-performing predictive model.

## 8. Faithfulness tests

- **Ablation** holds one emotion constant at its support-set mean (the
  highest-weighted, an average-weighted, or the lowest-weighted concept)
  and re-scores the user, to see how much each concept's presence is
  actually load-bearing for that user's predictions.
- **Formula swap** compares a user's own formula against the population-mean
  formula and against five other users' formulas sampled without replacement
  by a fixed random seed. Sampling is restricted to the same fold and domain,
  where coefficients share a scaler and population model.
- **Image swap** keeps the user's formula fixed but replaces the aligned
  emotion vector by another evaluation image's vector. The anchored analysis
  uses all 49 nonzero cyclic shifts; a no-image control uses the support-set
  mean emotion vector. The unanchored swap is reported as a diagnostic.
- **Weight vs. empirical correlation** compares the 7 weights the ridge
  head learned against the true correlation between each emotion and that
  user's own ratings.

## 9. Reproducibility

Every source of randomness is seeded. Formula swap resets `RandomState(0)`
for the anchored and unanchored analyses; image swap uses exhaustive cyclic
shifts. Both are deterministic arithmetic over stored formulas and refit no
model.

For experiments with stochastic model components, points are repeated under
**3 seeds (0, 1, 2)** and averaged within the user-domain unit before
summarizing, computing SEM, or testing. Seed 0 reproduces the original
single-seed numbers; seeds 1 and 2 offset every base seed below by
`+ run_seed * 1_000_003`.

### What "reproducible" does and does not promise

**Same machine, same environment: byte-identical.** `verify --repro` checks
this and passes on `efficiency`, `stage1_emotion_acc` and
`stage2_emotion_importance`.

**Different platform: identical to about 1e-5, not bit-identical.** A run on
Linux/Python 3.14 against Windows/Python 3.13 agrees to ~1e-5 on SROCC and
~1e-8 on PLCC, because a different BLAS build sums in a different order.
Every number the paper reports to three decimals is unaffected.

The one quantity that was *not* robust to this was the percentage of positive
coefficients, because it is a sign test and ridge shrinks weak emotions to
exactly zero - the sign of a zero is decided by the last bit. That is now
counted with a tolerance (`SIGN_TOL` in `stage2_emotion_importance.py`), so it
agrees across platforms too. Anyone reproducing on other hardware should
expect the third decimal of a correlation to move and nothing else.

| point | base seed source (seed 0) |
|---|---|
| support/eval split per user | `42 + user_id` (never reseeded - fixed across runs) |
| mediator random projection and shuffle | one generator per fold, drawing R first, then the permutation |
| per-user MLP | `user_id` |
| population-level MLP | `100 + fold` |
| MLP emotion mediator | `fold` |
| PCA | 0 (never reseeded - PCA is deterministic given the data) |

The order in which the random and shuffled mediators draw random numbers is
fixed; changing that order changes the numbers even though the method is
unchanged.
