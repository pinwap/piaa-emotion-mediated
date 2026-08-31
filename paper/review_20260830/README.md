# Review package — WACV 2027 #2267

Produced 2026-08-30 from the submitted PDF, `paper/supplementary_release/supplementary.tex`,
and the code in `src/`. All quantitative statements were recomputed from the project's own
frozen release CSVs; no model was refitted.

| File | What it is |
|---|---|
| `REVIEW.md` | Adversarial reviewer report. Findings ranked A (rejection risk) / B (substantive objection) / C (correctness and presentation), each with the recomputed numbers behind it. |
| `TODO_ADDITIONAL_WORK.md` | Only the things that must be **computed, run or produced**. Tiered by effort: free / cheap / requires a grid re-run. |
| `WORDING_CHANGES.md` | Only the **text changes**. Current sentence, replacement, reason, and a consistency table for the numbers that must agree across the manuscript. |
| `supplementary_additions.tex` | Drop-in LaTeX: two patches to existing sections plus new sections S5–S10 with their tables, written to match the supplementary's register. |
| `supp_v2_analyses.py` | Reproduces every number in the additions from `output/*.csv` and `Dataset/maked/ratings.csv`. |

## Reading order

1. `REVIEW.md` §A1–A3 — the three claims that are stated more broadly than the evidence supports.
2. `TODO_ADDITIONAL_WORK.md` Tier 1 — seven analyses, all computable without retraining, that
   close A1–A3 and B1–B2.
3. `WORDING_CHANGES.md` — the sentences to change once those numbers exist.
4. `supplementary_additions.tex` — paste into the supplementary.

## Running the analyses

```bash
python supp_v2_analyses.py \
    --output-dir output \
    --ratings Dataset/maked/ratings.csv \
    --dest output/supplementary_v2
```

Writes `supplementary_v2_values.json` plus two per-unit CSVs. Requires numpy, pandas, scipy —
no project imports, so it runs against the reviewer release as well as the working tree.

Regenerate and diff against the tables in `supplementary_additions.tex` before submitting; the
numbers there were produced from the release CSVs as of 2026-08-30 and are marked in the table
notes where a value still needs to come from a fresh run.
