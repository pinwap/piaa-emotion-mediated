# Holm-corrected significance audit

## Scope

This audit covers the current main performance table, the anchored-versus-
unanchored table, the five-backbone table, the efficiency results, the
faithfulness results, the weight-alignment table, the MLP architecture
ablation, and the distribution-valued mediator table. Stage-1 emotion
accuracy, emotion importance, and the regulariser ablation are descriptive and
contain no hypothesis tests.

## Statistical unit

For results read from `output/raw_all_final.csv`, seeds 0, 1, and 2 are first
averaged within each `(fold, domain, user_id, model configuration)` unit. Each
domain-specific test therefore uses 129 paired units. The pooled Average test
uses 387 seed-averaged user-domain units, not 1,161 seed-level rows.

The faithfulness files already contain one row per user-domain unit and
therefore require no additional seed aggregation.

All model comparisons use two-sided paired Wilcoxon signed-rank tests. The
weight-alignment table follows its stated directional hypothesis and uses a
one-sided Wilcoxon test against zero. Holm correction is applied within each
predefined family. SROCC and PLCC are separate families, and the pooled Average
is kept separate from each domain-specific family.

## Families

| Reported result | Family definition | Tests per family |
|---|---|---:|
| Main Table 1 | domain x metric | 4 |
| MLP architecture | domain x metric | 3 |
| Distribution-valued mediator | pooled Average SROCC | 12 |
| Anchored vs unanchored | metric; pooled Average; both halves combined | 8 |
| Five-backbone table | domain x metric | 5 |
| Efficiency | domain x metric | 8 |
| Faithfulness | domain x metric; correlated panels combined | 12 |
| Weight alignment | domain x correlation metric | 1 |

The full definitions, including every contrast and input file, are listed in
`holm_family_definitions.csv`.

## Priority findings

- The five-backbone pooled-Average claim survives. Hybrid exceeds Population
  for all five backbones in both metrics. The largest Holm-adjusted p-value is
  0.002207 for SROCC and 0.000599 for PLCC.
- In Main Table 1, 4 cells that were significant before correction are no
  longer significant: Artwork PLCC for Hybrid versus Population, and Fashion
  SROCC for Hybrid versus Direct, Population, and Random.
- In the anchored-versus-unanchored table, none of the raw significance
  decisions changes after Holm correction. Twelve of the sixteen comparisons
  remain significant.
- For the pooled Average efficiency analysis, Hybrid exceeds Population at
  N=50 and N=100 for both metrics after correction. It does not do so at N=10
  or N=25. Thus the statement "from N >= 50" survives for the pooled Average.
- All pooled-Average faithfulness conclusions survive. Holding the
  bottom-weight emotion does not cause a significant performance change, as
  expected. Formula swap, top/average concept ablation, and image swap claims
  remain significant after correction.
- All weight-alignment tests remain significant.
- In the MLP architecture table, the joint lambda=1 differences from the
  sequential MLP no longer survive correction for Artwork SROCC or Fashion
  SROCC. All lambda=0.1 differences remain significant; lambda=10 remains
  indistinguishable from sequential.
- The distribution-table significance decisions do not change. The replacement
  LaTeX reports daggers from the Holm-adjusted decisions; exact adjusted
  p-values are retained in the audit CSV.

## Required presentation corrections

1. In the backbone table, Direct is a descriptive unrestricted-personalization
   reference. Remove its significance daggers. Only Hybrid-versus-Population
   is included in the five-test families.
2. Update table notes to say "Holm-adjusted p < .05" and state that seeds were
   averaged within user-domain units before testing.
3. The former detailed efficiency table computed SEM by pooling the three seed
   rows. Its means were unaffected, but its SEM values were too small. The
   replacement LaTeX now averages seeds within each user-domain unit before
   computing SEM. This did not require retraining or rerunning the experiment.
4. Use `holm_dagger_changes.csv` to remove significance markers that do not
   survive correction. `holm_corrected.csv` is the complete source of truth for
   all retained markers.

## Output files

- `holm_corrected.csv`: all 292 tests with raw and adjusted p-values.
- `holm_family_definitions.csv`: family definitions and sources.
- `holm_family_summary.csv`: one row per family.
- `holm_dagger_changes.csv`: tests that were significant at raw p < .05 but not
  after Holm correction.
- `tableS2_holm_corrected.csv`: five-backbone tests only.
