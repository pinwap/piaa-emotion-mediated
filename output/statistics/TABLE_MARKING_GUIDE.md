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

The three correlated panels are corrected together as one 13-test family per
domain and metric.

## Weight alignment

All one-sided alignment-greater-than-zero tests remain significant.

## MLP architecture

The final supplementary table is descriptive and carries no daggers or
double-daggers. The three joint-versus-sequential contrasts remain in
`holm_corrected.csv` as an audit trail, but they do not support an inferential
claim in the manuscript. Use `tableS_mlp_architecture_corrected.tex`.

## Distribution-valued mediator

The final supplementary table is descriptive and carries no daggers. The 12
pooled-Average SROCC contrasts remain in `holm_corrected.csv` as an audit
trail, but the manuscript does not use them to claim statistical
significance. Use `tableS_distribution_corrected.tex`.

## Descriptive tables

Stage-1 emotion accuracy, emotion importance, MLP, distribution, and the
regulariser ablation do not carry manuscript hypothesis tests.
