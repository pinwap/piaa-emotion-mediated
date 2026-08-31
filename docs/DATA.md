# Data and feature files

## Dataset

We use the XPASS-VIS dataset, which records beauty ratings and aesthetic
emotion ratings from 129 participants on 6,526 images across 3 domains,
87,836 interactions in total.

| domain | images | interactions |
|---|---|---|
| art | 2,345 | 31,543 |
| fashion | 2,082 | 28,154 |
| landscape | 2,099 | 28,139 |

Each interaction has an overall beauty rating (1-7 scale) and 9 aesthetic
emotion ratings (1-5 scale). This work uses 7 of the 9 as the mediator,
dropping `like` and `beautiful` because they overlap in meaning with the
dependent variable.

Raw data is not included in this repository and must be requested from the
dataset owner, then placed at `Dataset/maked/ratings.csv`.

### Rating scale conversion

The 9 emotion items are stored 0-4 in the raw file; the loader adds 1 to
match the 1-5 scale. `Aesthetic` (overall) is stored 0-6 and remains on that
coding internally. Adding 1 would not change SROCC or PLCC, the metrics used
throughout the paper.
## Data split

```
Dataset/split_v4_10group/fold{0..4}/
    train_users.txt         7 groups
    val_users.txt           1 group
    test_users.txt          2 groups
    giaa_train_images.txt   images used to fit the shared mediator
```

This split is included in the repository because it is small and required
to reproduce every result. See `docs/METHODOLOGY.md` section 2 for why it
is designed this way.

### Per-fold composition

All 129 users are partitioned into 10 groups; each fold uses 7 groups as
train, 1 as validation, and 2 as test, rotating so that every user is a
test user in exactly one fold (387 = 129 users x 3 domains total
evaluation units). Verified with `uv run main.py verify --splits`.

| fold | train users | val users | test users | GIAA train images | test images (art / fashion / landscape) |
|---|---|---|---|---|---|
| 0 | 97 | 6 | 26 | 4,561 | 470 / 420 / 420 |
| 1 | 95 | 12 | 22 | 4,564 | 470 / 420 / 419 |
| 2 | 92 | 14 | 23 | 4,563 | 470 / 418 / 420 |
| 3 | 83 | 16 | 30 | 4,561 | 470 / 420 / 420 |
| 4 | 84 | 17 | 28 | 4,582 | 465 / 404 / 420 |

Train, validation, and test users are disjoint within every fold, and the
images seen by train users never overlap with the images evaluated for test
users, in every fold and domain. `uv run main.py verify --splits` checks
both of these directly against the loaded data and prints a pass/fail per
fold; it does not just re-check the split files against each other.

## Feature files

Extracted in advance and stored as `.npz`, each with two keys:
`stimulus_ids` and `features`. Place them under `features/`.

| file | backbone | dim | notes |
|---|---|---|---|
| `clip_features.npz` | CLIP ViT-B/16, frozen | 512 | default for every experiment |
| `clip_ftpf_overall_v4_fold{0..4}.npz` | CLIP fine-tuned per fold | 512 | must be loaded per fold |
| `clip_ftpf_emotion_v4_fold{0..4}.npz` | CLIP fine-tuned for emotion per fold | 512 | must be loaded per fold |
| `vlm4b_LT15.npz` | Qwen3-VL 4B, layer 15 | 2560 | |
| `vlm_LT15.npz` | Qwen3-VL 8B, layer 15 | 4096 | |

A fine-tuned backbone must always use the per-fold version (not a single
file fine-tuned on everyone) - otherwise it leaks.
files are fine-tuned only on that fold's train users
