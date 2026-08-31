# Reviewer release manifest

This document identifies the compact, frozen numerical release used to audit
the paper. Raw XPASS-Vis ratings are not redistributed; access and expected
placement are described in `docs/DATA.md`.

## Files to release

| File | Rows | Purpose |
|---|---:|---|
| `output/raw_all_final.csv` | 381,969 | Seed-level results for 329 model configurations; source for the main, backbone, anchor, efficiency, MLP, and distribution summaries |
| `output/formula_units_final.csv` | 774 | One fitted formula per user-domain unit and setting (387 anchored and 387 unanchored) |
| `output/formula_images_final.csv` | 38,700 | The 50 held-out image predictions associated with every formula unit |
| `output/stage1_accuracy.csv` | 105 | Fold-domain-emotion Stage-1 accuracy used by the Stage-1 table |
| `output/upper_bound_units_final.csv` | 387 | Per-user-domain GT-emotion upper-bound results |
| `output/emotion_importance_units_final.csv` | 8,127 | GT, predicted, and difference coefficients for seven emotions over 387 units |
| `output/mlp_architecture_units_final.csv` | 1,548 | Four seed-averaged MLP architecture conditions (387 units each), including the three joint-loss weights |
| `output/reference_bounds_final.csv` | 2 | Validated GT-emotion summary and test--retest reliability computed from licensed ratings |
| `output/regularizer_units_final.csv` | 3,870 | Controlled Ridge, Lasso, and Elastic Net rerun: ten seed-averaged conditions with 387 units each |
| `output/supplementary_v2/test_retest_units.csv` | 387 | Within-user--domain test--retest correlations underlying the supplementary reference quantity |
| `output/statistics/holm_corrected.csv` | 300 | Raw and Holm-adjusted p-values, family identifiers, means, differences, clustered uncertainty, and decisions |

The three small provenance manifests should be released beside these files:

- `output/raw_all_sources.csv` records the 25 physical input CSVs, their
  priority when runs overlap, selected-row counts, and SHA-256 hashes.
- `output/formula_sources.csv` records the six formula inputs and their
  SHA-256 hashes. The anchored and unanchored Stage-1 files are verified to be
  identical before one 105-row table is written.
- `output/auxiliary_sources.csv` records the GT upper-bound,
  emotion-importance, reference-bound, and MLP-architecture inputs and their
  SHA-256 hashes.

Frozen SHA-256 checksums:
| File | SHA-256 |
|---|---|
| `raw_all_final.csv` | `9ed6146cdf1de4a0b278f6d290fdadccfef710587a867c63fd80045e4ed95640` |
| `formula_units_final.csv` | `c91fafca03d891a34a2d879fbfaecb11ebd233183b9b56a208b9442e161243e7` |
| `formula_images_final.csv` | `d3894f8342535eed30a298bcdfec6501f0a490bb2d06be167ba62d50e646793e` |
| `stage1_accuracy.csv` | `ee35ec62361be83fb8425d2a08d72951e2d59834da8b8e57af5d57e0f76f0754` |
| `upper_bound_units_final.csv` | `8499a80207e070303704d14bc06e320b57ab900f766152a7c58d70578a41d6e7` |
| `emotion_importance_units_final.csv` | `6860f2b07ccd91234e7aec40d2b1897dd6f0a054bc14803439286e5c3030fd5b` |
| `mlp_architecture_units_final.csv` | `ead9826457f20b7bce876b671336cff963516eb583e5410ae1f651c513145665` |
| `reference_bounds_final.csv` | `2cdbb63504b8ef06eb1b20757406ab73937e8cdd1639751beb3b2c0ff84bdc6e` |
| `regularizer_units_final.csv` | `a1b8d62666f6fad449e885a8492d74723ac4ac719957a7b23771c33ad4be81e4` |
| `test_retest_units.csv` | `572b6e3794fd333c9d126ff38660bc171506972274f96ae97afbdc4b9e9a9d83` |
| `holm_corrected.csv` | `4ec82c7397154b8d6af21bb90640ad166a4253ca854c3047244617ca0e5b2f11` |

Generated LaTeX fragments and intermediate summary CSVs are not part of the
minimal release. The numerical source is the released CSV set; plots can be
regenerated in an isolated directory using the commands below.

## Consolidation guarantees

`make_raw_all_final.py` declares every source explicitly and applies a stable
priority order when historical run files overlap. It rejects duplicate final
keys, missing seeds, incomplete configurations, and any result that is not
exactly 387 user-domain units by three seeds. The current build reads 476,010
input rows, resolves 94,041 overlaps, and produces 381,969 unique rows. A
reference comparison confirms that all keys, SROCC values, and PLCC values are
identical to the frozen database.

`make_formula_database.py` validates unique `(setting, fold, domain, user_id)`
formula keys, 50 unique held-out images per unit, matching anchored/unanchored
evaluation image keys and targets, and identical Stage-1 inputs.

`make_release_auxiliary.py` validates 387 unique GT upper-bound units (129
participants in each domain), 8,127 unique emotion-importance rows (three
settings by seven emotions by 387 units), and 1,548 MLP-architecture units
(four conditions by 387 units). It also verifies the GT row in the two-row
reference summary against those units, including participant-clustered Average
SEMs. An isolated rebuild is byte-identical to all frozen outputs.

The released regularizer ablation replaces the legacy mixed-run table. Ridge,
Lasso, and Elastic Net were rerun together under the current split,
support-set, validation-selection, and three-seed protocol. The released file
contains exactly 387 seed-averaged units for each of its ten conditions. The
table is descriptive and does not add a multiple-testing family.

## Frozen faithfulness protocol

- Formula swap uses five peer formulas sampled without replacement within the
  same fold and domain with `RandomState(0)`. Population mean uses every
  eligible peer in that fold and domain.
- Image swap uses all 49 non-zero cyclic shifts of each 50-image evaluation
  set. It introduces no image-sampling choice.
- Both analyses are deterministic arithmetic over stored formulas; neither
  trains nor refits a model.

Expected pooled values are:

| Setting | Formula | SROCC | PLCC |
|---|---|---:|---:|
| Anchored | Own | 0.459 | 0.469 |
| Anchored | Population mean | 0.439 | 0.449 |
| Anchored | Five-peer swap | 0.396 | 0.403 |
| Unanchored | Own | 0.446 | 0.454 |
| Unanchored | Population mean | 0.425 | 0.434 |
| Unanchored | Five-peer swap | 0.368 | 0.376 |

The exhaustive all-peer formula-swap result from an earlier diagnostic is not
the reported experiment. Image swap, in contrast, intentionally remains
exhaustive over cyclic image shifts.

## Non-destructive verification

The following commands rebuild every audited derivative under
`tmp/release_verify/`; they do not overwrite the frozen files in `output/` or
the manuscript repository.

The first three consolidation commands are a maintainer audit and require the
historical source-run folders named in the manifests. Those folders are not
redistributed in the compact reviewer package. In that package, the supported
exact downstream check is simply `uv run python reproduce.py`; it starts from
the compact frozen audit-facing CSV files and requires neither the licensed raw
ratings nor extracted image features.

```bash
uv run python -m src.experiments.make_raw_all_final \
  --output tmp/release_verify/raw_all_final.csv \
  --manifest-output tmp/release_verify/raw_all_sources.csv \
  --compare output/raw_all_final.csv

uv run python -m src.experiments.make_formula_database \
  --output-dir tmp/release_verify/formula_database

uv run python -m src.experiments.make_release_auxiliary \
  --output-dir tmp/release_verify/auxiliary

uv run main.py faithfulness \
  --output-dir tmp/release_verify/formula_database

uv run python -m src.experiments.make_holm_statistics \
  --raw tmp/release_verify/raw_all_final.csv \
  --faithfulness-dir tmp/release_verify/formula_database/faithfulness \
  --output-dir tmp/release_verify/statistics

uv run python reproduce.py --output-dir tmp/reviewer_check
```

The current audit regenerated 300 statistical tests with identical raw and
Holm-adjusted p-values and identical significance decisions. The canonical
table fragments were reproduced exactly, and corrected descriptive MLP and
regularizer tables were generated with participant-clustered Average SEMs. The regenerated
efficiency and faithfulness PDFs differ only in PDF metadata; their rendered
pixels are identical. A targeted Qwen3-VL 8B fold-0 model rerun reproduced all
156 saved metric rows exactly.

## Software environment

The released `uv.lock` is the executable reference environment. The final
release audit used Python 3.13.5, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0,
scikit-learn 1.9.0, and Matplotlib 3.11.1.

### Descriptive effective-DoF field

The frozen model database retains the effective-degrees-of-freedom diagnostic
used when the submitted table was generated. During the release audit, the
diagnostic for anchored heads was corrected to use the already standardized
fitted design matrix directly rather than standardizing it a second time.
Targeted primary-backbone reruns leave the displayed Hybrid and Direct values
unchanged at one decimal (3.5 and 13.3); the same-width PCA diagnostic changes
from 4.2 to 4.1. This diagnostic is not used for fitting, hyperparameter
selection, prediction, uncertainty, or hypothesis testing, so SROCC, PLCC,
confidence intervals, p-values, and all conclusions are unaffected. New model
runs use the corrected implementation in `src/utils/metrics.py`.

Historical model runs were distributed across Windows, Linux, and macOS. They
used the same fixed data splits, support-set protocol, and random seeds. We do
not claim historical byte identity across platforms; the consolidated release
provides the frozen per-unit results, and the locked reference environment
regenerates their downstream statistics, tables, and figures deterministically.
