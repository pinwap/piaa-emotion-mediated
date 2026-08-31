# Supplementary audit

## Resolved in the release candidate

- Ridge selection grid corrected to 17 values from `1e-2` through `1e6`.
- MLP baseline caption corrected to Qwen3-VL 8B, population-anchored, `N=100`.
- MLP tables regenerated as descriptive tables with seeds averaged within
  user--domain units and Average SEM clustered over 129 participants.
- Distribution and detailed efficiency tables use corrected clustered Average
  SEMs and are presented descriptively.
- Formula Swap is explicitly frozen to five fixed-seed peer swaps (`K=5`).
- Concept ablation retains the submitted anchored total-weight definition and
  Holm family. The supplementary states its interpretation boundary directly;
  no post-submission unanchored ablation is added.
- The no-image condition is defined as the support-set mean emotion vector,
  not a zero vector.
- Formula-swap, concept-ablation, and image-swap tables use the regenerated
  clustered SEMs and current Holm decisions.
- The qualitative values were regenerated from
  `output/formula_images_final.csv` and `output/formula_units_final.csv`.
  The displayed coefficients are
  `w_tot`, and the caption now states that raw emotion values are standardized
  before the head and that the full prediction includes the population branch
  and personal intercept.
- Software versions now describe the locked release-audit environment rather
  than claiming that every historical run used one machine and one environment.

## Regularizer status

The controlled regularizer rerun is complete. Ridge, Lasso, and Elastic Net
were evaluated together under the current split, support-set, validation, and
three-seed protocol. Every displayed condition has 387 seed-averaged unit
records. Three sparse Hybrid fits produce constant predictions, leaving 385
valid correlations for Lasso and 386 for Elastic Net; this is disclosed in the
table note. The final table is descriptive and is generated from
`output/regularizer_units_final.csv`; historical mixed-run files are not used.

## Remaining submission gates

- Compile the supplementary in Overleaf and inspect float placement, widths,
  references, and numbering. All
  three seeds and 387 seed-averaged units per displayed configuration passed
  the audit.
- Compile the candidate once in the manuscript Overleaf project and inspect
  float placement, table width, references, and final table numbering.
- Package the compiled supplementary PDF with the minimal reviewer code release;
  do not include `.git`, caches, historical run directories, generated LaTeX
  fragments, licensed ratings, or large extracted features.

## Interpretation boundary to preserve

Anchor C executes

`y_hat = y_pop(raw image features) + z @ delta_user + b_user`.

Thus `delta_user` is the coefficient vector in the exact computational path.
The reported concept-ablation diagnostic instead uses
`w_tot = w_pop + delta_user` to represent an absolute emotion-space direction.
This follows the submitted total-weight definition, but it is not an algebraic
decomposition of the raw-feature population branch. The revised supplementary
states that boundary explicitly. Changing the ablation to `delta_user` would be
a different analysis: its pooled SROCC is approximately `.442` after top-weight
ablation rather than `.361`, so it must not be changed silently.

## Submitted-main-paper SEM note

The release CSVs and supplementary candidate use the corrected
participant-cluster SEMs. A small SEM-only transcription discrepancy in the
already-submitted anchor table does not change a mean, confidence interval,
Holm decision, or conclusion. No special correction note is needed in the
supplement unless the venue requests one; if asked, disclose it as a table
transcription correction.
