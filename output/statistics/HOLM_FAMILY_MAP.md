# Holm--Bonferroni Family Map

This document states exactly which comparisons were corrected together. It is
intended as a readable companion to `holm_corrected.csv`, which contains the
complete raw and adjusted p-values.

## Statistical procedure used throughout

- For experiments with three-seed outputs, seeds 0, 1, and 2 are averaged
  within each user--domain unit before testing.
- A domain-specific test contains 129 paired user units. A pooled Average
  stores 387 seed-averaged user--domain values, but inference resamples 129
  participants as clusters and retains all three domains for each selected
  participant.
- The Average is corrected as its own family and is never included in a
  domain-specific family.
- Domain-specific model comparisons use two-sided paired Wilcoxon signed-rank
  tests; Average comparisons use the two-sided participant-cluster bootstrap.
  Weight alignment uses its prespecified one-sided alternative against zero.
- Holm--Bonferroni correction is applied separately within every family below.
  SROCC and PLCC are always separate families.
- A dagger is retained only when the Holm-adjusted p-value is below .05.

## Main Table 1: Ridge baselines

**Eight families:** `{Artwork, Fashion, Landscape, Average} x {SROCC, PLCC}`.

Each family contains these four comparisons:

1. Hybrid Ridge versus Population Ridge
2. Hybrid Ridge versus Direct Ridge
3. Hybrid Ridge versus Random Ridge
4. Hybrid Ridge versus PCA Ridge

**Family size: 4.** The MLP rows removed from the main table are not included
in these families.

Manuscript action: remove four former raw-p daggers:

- Population -- Artwork PLCC
- Population -- Fashion SROCC
- Direct -- Fashion SROCC
- Random -- Fashion SROCC

## Anchored versus unanchored table

**Two families:** pooled Average SROCC and pooled Average PLCC.

The anchored and unanchored halves use the same 387 units and are therefore
kept together. Each metric family contains:

1. Anchored Hybrid versus anchored Population
2. Anchored Hybrid versus anchored Direct
3. Anchored Hybrid versus anchored Random
4. Anchored Hybrid versus anchored PCA
5. Unanchored Hybrid versus unanchored Population
6. Unanchored Hybrid versus unanchored Direct
7. Unanchored Hybrid versus unanchored Random
8. Unanchored Hybrid versus unanchored PCA

**Family size: 8.** No significance decision changes after correction; the
existing 12 significant and 4 nonsignificant comparisons remain unchanged.

## Table S2: backbone robustness

**Eight families:** `{Artwork, Fashion, Landscape, Average} x {SROCC, PLCC}`.

Each family contains Hybrid versus Population for the following five
backbones:

1. CLIP frozen
2. CLIP fine-tuned for overall score
3. CLIP fine-tuned for emotion
4. Qwen3-VL 4B
5. Qwen3-VL 8B

**Family size: 5.** Direct is an unrestricted-personalization reference and is
reported descriptively; Direct comparisons are not part of the confirmatory
family and should not carry daggers.

The pooled-Average claim survives for all five backbones in both metrics. The
largest Holm-adjusted p-value is .0106 (Qwen3-VL 4B SROCC), so the defensible
summary is all $p<.05$, not all $p<.005$. This claim should not be generalized
to every individual domain.

## Efficiency table and figure

**Eight families:** `{Artwork, Fashion, Landscape, Average} x {SROCC, PLCC}`.

Each family contains:

1. Hybrid versus Population at N = 10
2. Hybrid versus Population at N = 25
3. Hybrid versus Population at N = 50
4. Hybrid versus Population at N = 100
5. Direct versus Population at N = 10
6. Direct versus Population at N = 25
7. Direct versus Population at N = 50
8. Direct versus Population at N = 100

**Family size: 8.** For the pooled Average, Hybrid significantly exceeds
Population at N = 50 and N = 100 in both metrics, but not at N = 10 or N = 25.

The current detailed table is descriptive and uses the corrected
participant-cluster Average SEM. The inferential decisions attached to the
efficiency figure remain the eight comparisons in each family above.

## Faithfulness figure and detail tables

**Eight families:** `{Artwork, Fashion, Landscape, Average} x {SROCC, PLCC}`.

The three panels use the same units and are kept in one family. Each family
contains these 13 comparisons:

**Formula swap (6)**

1. Anchored Own versus Population-mean formula
2. Anchored Own versus Other-user formula (five fixed-seed peer swaps)
3. Anchored Population-mean versus Other-user formula
4. Unanchored Own versus Population-mean formula
5. Unanchored Own versus Other-user formula (five fixed-seed peer swaps)
6. Unanchored Population-mean versus Other-user formula

**Concept ablation (3)**

7. Full model versus Bottom-weight emotion ablation
8. Full model versus Average-weight emotion ablation
9. Full model versus Top-weight emotion ablation

**Image swap (4)**

10. Anchored Correct-image emotion versus Swapped-image emotion
11. Anchored Correct-image emotion versus No-image emotion
12. Anchored No-image emotion versus Swapped-image emotion
13. Unanchored Correct-image emotion versus Swapped-image emotion

**Family size: 13.** The unanchored no-image condition is undefined and is not
tested. Adding the direct anchored control comparison changes no existing
significance decision. The pooled faithfulness conclusions survive, including
that No-image outperforms Swapped-image input in both metrics. Bottom-weight
ablation is not detected for SROCC but is detected for PLCC; Average- and
Top-weight ablations are significant in both metrics.

## Weight-alignment table

**Eight families:** `{Artwork, Fashion, Landscape, Average} x
{Spearman, Pearson}`.

Each family contains one directional one-sided test asking whether the
weight--empirical-effect alignment is greater than zero.

**Family size: 1.** All tests remain significant. Holm makes no numerical
change when a family contains one test.

## Descriptive MLP architecture table

The current manuscript reports the MLP baseline and joint/sequential variants
descriptively and carries no significance markers. The corresponding tests
remain in `holm_corrected.csv` as an audit trail, but they are not used for a
manuscript claim or displayed dagger.

## Descriptive distribution-valued mediator table

The current manuscript reports this experiment descriptively and carries no
significance markers. Historical exploratory contrasts remain in
`holm_corrected.csv` for auditability but are not used for a manuscript claim.

## Descriptive results with no multiple-testing family

The Stage-1 emotion-accuracy table, emotion-importance table, and regularizer
ablation report descriptive estimates and contain no hypothesis tests. They
therefore receive no Holm correction and should not use inferential daggers.

## How to use the output files

- `holm_corrected.csv`: source of truth for all 300 tests, including raw and
  Holm-adjusted p-values.
- `holm_dagger_changes.csv`: comparisons that had raw p < .05 but no longer
  pass after correction.
- `tableS_mlp_baseline_corrected.tex` and
  `tableS_mlp_architecture_corrected.tex`: current descriptive MLP tabulars.
- `tableS_distribution_corrected.tex`: current descriptive distribution
  tabular with participant-cluster Average SEM.
