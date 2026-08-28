# Corrected LaTeX table index

Every file below is generated from seed-averaged user--domain units and uses
the Holm-adjusted decisions in `holm_corrected.csv`. Each file contains a
`tabular` block, not a complete floating `table` environment. Replace the
current `tabular` in Overleaf and retain the manuscript's caption and label.

| Result | LaTeX file | Marker definition |
|---|---|---|
| Main Table 1, Ridge rows | `table1_corrected.tex` | versus Hybrid Ridge |
| Anchored versus unanchored | `table_anchor_corrected.tex` | versus Hybrid in the same setting |
| Table S2, backbones | `tableS2_backbone_corrected.tex` | Hybrid versus Population; Direct is descriptive |
| Efficiency | `table_efficiency_corrected.tex` | Direct/Hybrid versus Population at the same budget |
| Formula swap | `table_faithfulness_formula_swap_corrected.tex` | exact adjusted p-values versus Own |
| Concept ablation | `table_faithfulness_ablation_corrected.tex` | versus Full |
| Image swap | `table_faithfulness_image_swap_corrected.tex` | versus Correct image in the same setting |
| Weight alignment | `table_weight_alignment_corrected.tex` | one-sided alignment greater than zero |
| MLP architecture | `tableS_mlp_corrected.tex` | joint setting versus sequential MLP |
| Distribution-valued mediator | `tableS_distribution_corrected.tex` | Hybrid width versus Population |

`tableS_distribution_full_corrected.tex` is also provided as a complete table
environment with a corrected caption and tablenotes, ready to replace the full
distribution table in the manuscript.

The Stage-1 emotion-accuracy, emotion-importance, and regularizer-ablation
tables are descriptive. They contain no inferential marker and therefore do
not need Holm-corrected replacement tables.

## Efficiency SEM correction

The former detailed efficiency table treated the three seed rows as if they
were independent observations. For example, Hybrid at N = 100, pooled SROCC,
was summarized over 1,161 rows, giving SEM = 0.00527. The corrected procedure
first averages the three seeds within each unit and then summarizes the 387
units, giving SD = 0.17966 and SEM = 0.00913. Means are unchanged; no model
rerun is required.
