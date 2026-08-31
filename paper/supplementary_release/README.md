# Supplementary release candidate

This directory contains a corrected standalone supplementary source and the
table/figure fragments it reads. It is intentionally separate from the
submitted manuscript so the original files are not overwritten during
verification.

## Files to copy into the manuscript project

- Add `supplementary.tex` as the standalone Overleaf main file.
- Copy the contents of `Table_Figure/` into the manuscript project's
  `Table_Figure/` directory.
- Keep the manuscript project's existing `preamble.tex`, `main.bib`, WACV
  style files, and image assets. Set `supplementary.tex` as the main document
  when compiling the supplementary PDF.

The revised source replaces the legacy regularizer table with a controlled
three-seed rerun in which Ridge, Lasso, and Elastic Net were evaluated together
under the current protocol. It also corrects
the Ridge grid, cluster-bootstrap Average SEMs, MLP captions, faithfulness
condition definitions, and the qualitative-figure explanation. Stage-1
accuracy and emotion-importance tables are included because they were moved
out of the page-limited main manuscript.

## Controlled regularizer rerun

The code now supports an isolated, current-protocol rerun without appending to
the frozen `output/raw_all.csv` database:

```bash
uv run main.py table1 --backbone clip --n-train 100 --stage2 plain \
  --heads ridge,lasso,elastic --mediators identity,emotion \
  --output-dir tmp/regularizer_rerun
```

Expected outputs include:

- `tmp/regularizer_rerun/table1/per_unit_rle_m2_seed{0,1,2}.csv`
- `tmp/regularizer_rerun/table1/per_unit_rle_m2_by_seed.csv`
- `tmp/regularizer_rerun/table1/per_unit_rle_m2.csv`
- `tmp/regularizer_rerun/table1/summary_rle_m2.csv`
- one selection log per seed and an isolated `tmp/regularizer_rerun/raw_all.csv`

The completed rerun contains all three seeds and 387 seed-averaged units for
every displayed mediator--head combination. Historical files under `no_use`
were not used for the final table. The compact audit-facing copy is
`output/regularizer_units_final.csv`; `reproduce.py` validates its unit coverage
and regenerates `tab_regularizer.tex` from it.

## Compilation check

This host does not currently have a LaTeX engine. The source has been checked
for balanced environments, labels, references, and all `Table_Figure` inputs.
The remaining `preamble.tex`, `main.bib`, and WACV style dependencies already
exist in the manuscript project. Compile once in Overleaf after copying the
files and inspect float placement, table width, references, and final numbering
before submission.
