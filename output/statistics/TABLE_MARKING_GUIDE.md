# Table marking guide after Holm correction

This guide describes manuscript changes. The exact values remain in
`holm_corrected.csv`.

## Main Table 1

Remove four existing daggers:

- Population, Artwork PLCC
- Population, Fashion SROCC
- Direct, Fashion SROCC
- Random, Fashion SROCC

All other Ridge-row markers follow `holm_corrected.csv`.

## Anchored versus unanchored

No raw significance decision changes after correction. The two halves are
corrected together in one eight-test family per metric.

## Backbone table

Remove all daggers from Direct because it is retained as a descriptive
unrestricted-personalization reference. Hybrid-versus-Population is the only
formal contrast. Six raw Hybrid markers no longer survive Holm correction;
their cells are listed in `holm_dagger_changes.csv`.

## Efficiency

Regenerate rather than editing manually. The existing table pooled seed rows
when computing SEM, so its SEM values are too small. The corrected tests use
eight comparisons per domain-metric family: Hybrid and Direct versus
Population at four rating budgets. Ten raw significance decisions do not
survive correction.

## Faithfulness detail tables

- Formula swap: both replacement formulas differ from Own in both metrics;
  the Population-mean versus Other-user claim also survives correction.
- Concept ablation: Bottom is not significant. Average and Top are significant
  in both SROCC and PLCC. Add the currently missing PLCC daggers for Average
  and Top.
- Image swap: all displayed, defined comparisons remain significant in both
  metrics. The unanchored no-image condition is undefined and is not tested.

The three correlated panels are corrected together as one 12-test family per
domain and metric.

## Weight alignment

All one-sided alignment-greater-than-zero tests remain significant.

## MLP architecture

The sequential MLP is the inferential reference; Ridge is descriptive. Each
domain-metric family contains the three joint-loss weights. Remove two
existing double daggers from joint lambda=1:

- Artwork SROCC
- Fashion SROCC

The corrected LaTeX is in `tableS_mlp_corrected.tex`.

## Distribution-valued mediator

The family contains 12 pooled-Average SROCC tests: three mediator widths at
four rating budgets, each versus Population. No significance decision changes.
`tableS_distribution_corrected.tex` marks the adjusted decisions with daggers;
exact adjusted p-values remain in `tableS_distribution_holm_corrected.csv`.

## Descriptive tables

Stage-1 emotion accuracy, emotion importance, and the regulariser ablation do
not contain hypothesis tests and require no Holm correction.
