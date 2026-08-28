# Reviewer release manifest

This file identifies the frozen artifacts used by the paper. The repository
contains code, split definitions, per-unit results, fitted formulas, statistical
audit files, and final figures. Raw XPASS-VIS ratings are not redistributed;
see `docs/DATA.md` for access and placement.

## Sources of reported results

| Paper result | Reviewer artifact |
|---|---|
| Main baselines and MLP rows | `output/raw_all_final.csv` |
| Upper bounds | `output/upper_bounds.csv` |
| Rating-budget efficiency | `output/efficiency/qwen8b/raw.csv` |
| Five-backbone robustness | rows selected from `output/raw_all_final.csv` |
| Stage-1 emotion accuracy | `output/stage1_emotion_acc/` |
| Emotion importance | `output/stage2_emotion_importance/` |
| Final anchored formulas | `output/formulas/` |
| Final unanchored formulas | `output/unanchored/formulas/` |
| Faithfulness diagnostics | `output/faithfulness/` |
| Corrected statistical tests | `output/statistics/holm_corrected.csv` |
| Readable correction families | `output/statistics/HOLM_FAMILY_MAP.md` |
| Final plots generated from CSV | `output/paper_figures/` |

The corrected LaTeX fragments in `output/statistics/` are convenience exports;
the CSV files remain the numerical source of truth.

## Frozen faithfulness protocol

- Formula swap uses five peer formulas sampled without replacement within the
  same fold and domain with `RandomState(0)`. Population mean uses every
  eligible peer in that fold and domain.
- Image swap uses all 49 nonzero cyclic shifts of each 50-image evaluation
  set. It introduces no image-sampling choice.
- Both analyses read the stored formulas and do not train or refit a model.

Expected pooled values are:

| Condition | SROCC | PLCC |
|---|---:|---:|
| Anchored own formula | 0.459 | 0.469 |
| Anchored population-mean formula | 0.439 | 0.449 |
| Anchored five-peer formula swap | 0.396 | 0.403 |
| Anchored correct-image emotions | 0.459 | 0.469 |
| Anchored swapped-image emotions | 0.384 | 0.392 |
| Anchored no-image emotions | 0.439 | 0.449 |
| Unanchored own formula | 0.446 | 0.454 |
| Unanchored population-mean formula | 0.425 | 0.434 |
| Unanchored five-peer formula swap | 0.368 | 0.376 |

Running `uv run main.py faithfulness` twice from the frozen formula files
produces identical output bytes. The verified SHA-256 values are:

| File | SHA-256 |
|---|---|
| `formula_swap.csv` | `90765CCF2EA1FBA865542AB55074C6A52C020E637843DB7ADE6069BF3403318C` |
| `formula_swap_unanchored.csv` | `8B94AE31381D40B97C3A6F8AC3E023239D4B49EA789EA9F64F02825E96753CAF` |
| `image_swap.csv` | `73011150A28A8867182FEABBC67B39E56342137073F167FAF7BB3D4CBDBBB227` |
| `image_swap_unanchored.csv` | `C508EABE1DF81CECF2C34D4CA61B433AE53B65162FD1754569F746CF9FE20967` |

## Reproduction commands

```bash
uv sync
uv run main.py verify --splits
uv run main.py faithfulness
uv run main.py fig_faithfulness
uv run main.py fig_efficiency
uv run python -m src.experiments.make_holm_statistics
uv run python -m src.experiments.make_holm_tables
```

The statistics scripts average seeds within user-domain units before computing
SEM or paired tests. They do not retrain any model.
