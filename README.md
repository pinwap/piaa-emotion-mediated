# Emotion-Mediated Personalized Image Aesthetic Assessment

Code for all experiments in the paper. We predict a personal beauty score by
routing prediction through a 7-dimensional layer of aesthetic emotions,
instead of predicting from image features directly.

The core idea is a mediator that can be read: a model shared across all
users predicts how an image tends to make people feel (7 emotions), and
then a per-user linear formula with only 7 coefficients turns those emotions
into that user's score, compared with an unrestricted head whose width follows
the visual backbone (512--4,096 features in the reported experiments).


## Code layout

```
main.py                 single entry point (argument parser)
src/
  config.py              every setting that affects the numbers lives here, once
  data/
    data.py               loads ratings + the first-session filter
    splits.py              leak-free (v4) split protocol
    build_group_split.py   builds the v4 10-group split from raw user metadata
    extract_clip_features.py   extract frozen CLIP ViT-B/16 features
    extract_vlm_features.py    extract raw Qwen3-VL features (all layers)
    select_vlm_layer.py        pick one layer out of the raw VLM output
    finetune_clip_perfold.py   fine-tune CLIP per fold (leak-free)
  modeling/
    backbones.py           CLIP / per-fold CLIP-ft / Qwen3-VL
    mediators.py            identity, emotion, pca, random, shuffled
    heads.py                RidgeHead, SparseLinearHead, MLPHead
    pipeline.py             wires the three together and runs the evaluation loop
  utils/
    metrics.py              SROCC, PLCC, Wilcoxon, participant-cluster bootstrap, effective DoF
    plots.py                figures
    verify.py               checks the data split is leak-free
  experiments/              one file per reported table
```

Every experiment is a **mediator x head** grid. Adding a new mediator or
head is one new class in `mediators.py` or `heads.py`; nothing else in
`pipeline.py` needs to change.


## Documentation

- `docs/METHODOLOGY.md` - experimental protocol, metrics, and the reasoning
  behind each design decision
- `docs/DATA.md` - dataset, split, and feature files


## Getting started
# use uv
```bash
uv sync
uv run main.py <experiment name>            
```

# use pip
```bash
python -m venv .venv && .venv/bin/pip install numpy pandas scipy scikit-learn matplotlib
.venv/bin/python main.py <experiment name>
```

Data must be placed as follows first (see `docs/DATA.md` for more details):

```
Dataset/maked/ratings.csv           rating data
Dataset/split_v4_10group/fold{0..4} user split (already in this repo)
features/clip_features.npz          pre-extracted features
```


## Commands

| command | produces |
|---|---|
| `uv run main.py table1` | main table: every mediator x head at 100 ratings/user |
| `uv run main.py backbone --backbones clip,clip_ft,clip_ft_emo,qwen4b,qwen8b` | compares five backbones |
| `uv run main.py efficiency` | results by rating budget (10/25/50/100) |
| `uv run main.py formulas` | fits and stores the final per-user formulas |
| `uv run main.py faithfulness` | formula swap, concept ablation, image swap, and weight alignment |
| `uv run main.py fig_faithfulness` | regenerates the faithfulness figure from stored CSVs |
| `uv run main.py fig_efficiency` | regenerates the efficiency figures with seed-averaged SEM |
| `uv run main.py verify --splits` | checks the data split for leakage |

Every command writes to `output/<experiment name>/`, with `config.json`
recording every setting that could affect the numbers, alongside
`per_unit.csv` (raw, one row per evaluation unit) and `summary.csv`
(mean/sd/best-flag/significance, ready to paste into a spreadsheet).
The reviewer release keeps a compact numerical core rather than every
intermediate run folder: `raw_all_final.csv`, the consolidated formula-unit
and formula-image databases, Stage-1 accuracy, the GT upper-bound units,
emotion-importance units, the compact MLP-architecture units, the two-row
reference-bound summary, within-user test--retest units, the controlled regularizer units, and the
Holm-adjusted audit.
This lets a reviewer regenerate and inspect the reported summaries without
repeating model training. Generated LaTeX fragments and figures are not
required in the release; the CSVs contain every reported numerical value. The
scripts rebuild the quantitative efficiency and faithfulness plots and
reproduce the disclosed ranking used to select the qualitative example.


## Verifying results match the paper

To rebuild the downstream faithfulness results, all 300 audited comparisons,
the paper-facing table fragments, and the data figures from the compact
release CSVs, without retraining or modifying the frozen outputs, run:

```bash
uv run python reproduce.py
```

The isolated results are written under `tmp/release_reproduction/`. The
script also checks the regenerated `holm_corrected.csv` row-for-row against
the frozen audit. This quick audit does not require the licensed raw ratings
or extracted visual features.

To verify the split against the licensed data after placing it locally, run:

```bash
uv run main.py verify --splits
```

Checks that the data split is actually leak-free, at both the user level
and the image level - see `docs/DATA.md` for the exact per-fold numbers
this should print (0 overlapping images in every fold/domain).

### Reference values for a quick sanity check

The following values identify the frozen primary configuration.

| check | expected value |
|---|---|
| rows after the first-session filter | 83,327 of 87,836 |
| evaluation units | 387 (129 users x 3 domains) |
| each user is a test user | exactly once |
| Hybrid ridge, Qwen3-VL 8B, pooled Average | SROCC 0.459, PLCC 0.469 |
| Direct ridge, Qwen3-VL 8B, pooled Average | SROCC 0.478, PLCC 0.488 |
| anchored own vs. five-peer formula swap | SROCC 0.459 vs. 0.396 |
| image-aligned vs. image-swapped emotions | SROCC 0.459 vs. 0.384 |
| weight vs. empirical correlation | pooled Spearman 0.525, 94.6% positive |


## Things to watch out for

- `Config.first_session_only` (`src/data/data.py`) controls which ratings
  get used - it prints the row count on load, should always say 83,327.
- Fine-tuned backbones must load per-fold features (`PerFoldBackbone` in
  `backbones.py`), not a single all-user file, or it leaks.
- `mediators.py`'s Random and Shuffled mediators draw from one RNG per fold
  in a fixed order (projection, then permutation); preserve that order for
  deterministic reproduction.
- Domain-specific model comparisons use `wilcoxon_paired` (paired, keyed on
  fold/domain/user); pooled Average inference uses the participant-cluster
  bootstrap rather than treating 387 domain rows as independent.
- MLP learning rate is selected on the held-out validation-user group from
  the grid in `src/config.py`; the 500-epoch budget and zero weight decay are
  fixed in advance, and early stopping is disabled.

## Statistical outputs

For three-seed experiments, seeds are averaged within each
`(fold, domain, user_id, model configuration)` unit before any test or SEM is
computed. Domain-specific tests therefore use 129 paired units. For the pooled
Average, the three domains are retained as one cluster per participant; SEM,
confidence intervals, and p-values use a 10,000-sample participant-cluster
bootstrap over 129 participants.
`output/statistics/holm_corrected.csv` is the source of truth for raw and
Holm-adjusted p-values. `output/statistics/HOLM_FAMILY_MAP.md` gives the
human-readable family definitions used for every table.

For maintainers who also have the historical intermediate run folders, the
consolidated databases and their provenance manifests are rebuilt with:

```bash
uv run python -m src.experiments.make_raw_all_final \
  --output output/raw_all_final.csv \
  --manifest-output output/raw_all_sources.csv
uv run python -m src.experiments.make_formula_database --output-dir output
uv run python -m src.experiments.make_release_auxiliary --output-dir output
```

Those historical intermediate folders are not part of the compact reviewer
package. Their manifests and SHA-256 hashes are retained for provenance, but
reviewers should use `uv run python reproduce.py` for the supported exact
downstream reproduction path. Full training from the licensed ratings and
extracted features produces fresh unit-level run files.

## Adding a new experiment

New mediator: add a class that subclasses `Mediator` in `mediators.py`,
then register it in `build_shared_mediators`.

New head: add a class that subclasses `Head` in `heads.py`, then add it to
`make_head`.

New backbone: add an entry to `BACKBONE_SPECS` in `backbones.py`.

New table: create a file in `experiments/`, then add its name to `main.py`.

None of this requires touching `pipeline.py`.
