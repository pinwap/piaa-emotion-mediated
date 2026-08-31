# Additional work required

Separated from the text changes in `WORDING_CHANGES.md`. This file lists only things that have
to be **computed, run or produced**, ordered by how much they reduce rejection risk per unit of
effort.

Effort key: **free** = already computable from the frozen release CSVs, no model refitting;
**cheap** = one script run over existing CSVs; **run** = requires re-running a training grid.

---

## Tier 1 — do these before resubmission

### T1.1 Mediator content across all five backbones (free)

**Why.** The Conclusion's "an unsupervised basis of the same width does no better under the
anchor" is false on four of five backbones and significantly false on three. This is the single
largest exposure in the paper.

**What.** A supplementary table giving Population / Direct / Hybrid / PCA / Random mean SROCC
for each of the five backbones at *N* = 100, in both the anchored and unanchored settings, with
the participant-cluster bootstrap difference of Hybrid against PCA, Random and Direct.

**How.** `supp_v2_analyses.py --output-dir output` → `S7_backbone_mediators`. All rows exist in
`raw_all_final.csv`. Rendered as Table S13 in `supplementary_additions.tex`.

**Also needed:** the corresponding wording changes (`WORDING_CHANGES.md` §1, §6).

---

### T1.2 Peer-shifted null for the weight-alignment test (free)

**Why.** The reported ρ = .525 is 86 % reproducible with a stranger's weights. Without a null,
Table 3 does not support the claim attached to it.

**What.** For each user–domain unit, recompute the alignment using (a) the user's own empirical
associations and (b) those of a peer in the same fold and domain, under two weight definitions
(w_tot and δ_u) and two association sources (predicted emotions and the user's own reported
emotions). Report own, null, and the clustered difference with a 95 % CI.

**How.** `supp_v2_analyses.py` → `S10_alignment_null`. Rendered as Table S15.

**Note.** The `*_rated` columns needed for the reported-emotion variant are already in
`formula_images_final.csv`; no new extraction is required.

---

### T1.3 Matched-variance control for the image swap (free)

**Why.** The claim that wrong emotions "mislead" is indistinguishable from the effect of adding
any emotion-shaped noise, and a reviewer can show this in twenty lines.

**What.** Add two null conditions to the image-swap table: emotion vectors drawn from a Gaussian
matched to the unit's own cov(ê), and per-column independent shuffles. Report alongside correct,
no-image and swapped.

**How.** `supp_v2_analyses.py` → `S10_image_swap_controls`. Rendered as Table S14.

**Also record** that the anchored no-image predictor is ŷ_pop plus a constant, so its SROCC
equals the Population row by construction (.4391 in both) — this explains the coincidence a
reader will otherwise notice.

---

### T1.4 Test–retest reliability on the unit the models are scored on (cheap)

**Why.** The reported .684 is a pooled across-participant correlation; every model number is a
within-participant correlation. They are not comparable, and the paper explicitly asks the
reader to compare them.

**What.** Compute test–retest per user–domain over that unit's repeat items, average over the
387 units, and report with a participant-cluster CI. Report the pooled figure alongside it and
say which one Table 1 should be read against.

**How.** `supp_v2_analyses.py --ratings Dataset/maked/ratings.csv` → `S6_test_retest`.
Result: within-unit .585 [.550, .617] versus pooled .684; median 12 repeat items per unit.
Rendered as Table S11.

**Downstream:** Table 1's Test–retest row and the Discussion sentence about the GT bound both
need the revised number (`WORDING_CHANGES.md` §2, §3).

---

### T1.5 Equivalence tests with a prespecified margin (cheap)

**Why.** Two conclusions in the paper are equivalence claims drawn from non-significant
differences. `tost_equivalence` is already implemented and never called.

**What.** Fix δ = .02 SROCC on substantive grounds *before running* (roughly two bootstrap SEMs;
below the Hybrid–Direct gap the paper calls small), report δ = .01 as a stricter secondary, and
apply TOST to: Hybrid vs PCA (Average), Hybrid vs Random (Average), Hybrid vs Direct (Average),
Hybrid vs Direct (Landscape). State the margin and its justification in the text before the
results.

**How.** `supp_v2_analyses.py` → `S8_equivalence_and_folds.tost`. Rendered as Table S12.

**Caution.** Record the margin in the supplementary as fixed in advance and do not adjust it
after seeing the table; the function's own docstring is the standard to hold to here.

---

### T1.6 Fold-level robustness (cheap)

**Why.** All test users in a fold share a Stage-1 model and a frozen hyperparameter, so
fold-level variance is absent from every reported SEM.

**What.** Mean SROCC per fold for Population / Direct / Hybrid / PCA, and a paired *t* test over
the five folds for Hybrid−Direct, Hybrid−Population and Hybrid−PCA. Present as a descriptive
robustness check, not as a new inferential family.

**How.** `supp_v2_analyses.py` → `S8_equivalence_and_folds.fold`. Rendered as Table S12b.

---

### T1.7 What the personal term contributes (free)

**Why.** Answers the standing objection that the interpretable part explains only a residual of
a black box, with numbers instead of assertion.

**What.** Report mean sd(ŷ_pop) and mean sd(personal term), the median share of prediction
variance carried by the personal term, and ‖w_pop‖ against ‖δ_u‖. Add the γ-scaling curve
(scaling δ_u by γ ∈ [0, 2]) as evidence that the formula is a well-behaved control surface.

**How.** `supp_v2_analyses.py` → `S9_decomposition`. Rendered as Table S16 and §S9.

**Disclose** in the same paragraph that the curve peaks at γ ≈ 0.75 (+.0036 over γ = 1), and
that the protocol-selected value is reported rather than the post-hoc optimum.

---

## Tier 2 — strongly recommended, one grid re-run each

### T2.1 Support-set resampling (run)

**Why.** Every result is conditional on one deterministic support draw per user. This is the
most standard robustness question in PIAA and the paper currently has no answer.

**What.** Re-run the primary configuration (Qwen3-VL 8B, anchor C, ridge, Population / Direct /
PCA / Random / Hybrid, *N* = 100) with three additional support-set draws — split seeds
`142+user_id`, `242+user_id`, `342+user_id` — holding the 50-image evaluation set fixed so the
target does not move. Report mean SROCC and the across-draw spread per condition, and confirm
the sign of Hybrid−Population and Hybrid−Direct in every draw.

**Cost.** One `table1` grid at one backbone and one budget, times three. Keep it out of the
frozen `raw_all.csv`: use `--output-dir tmp/support_resample` as the README already documents
for the regularizer rerun.

**Fallback if not run.** State in Limitations that all results are conditional on a single fixed
support draw. This is honest but noticeably weaker.

---

### T2.2 Width-matched controls for the distribution-valued targets (run)

**Why.** The supplementary reports that the 35-dimensional histogram target is the strongest
Stage-1 target from *N* = 25 onward. On Qwen3-VL 8B the width-matched controls are already
available and show the gain is width, not disagreement information:
`emotion_hist` .4644, `pca35` .4639, `random35` .4643 — all within .0005 of each other, and all
above `emotion` .4592. The CLIP table in the supplementary has no width-35 controls, so as it
stands the histogram row reads as an information result.

**What.** Run `pca35`, `random35` and `shuffled35` on frozen CLIP at all four budgets so the
distribution table carries width-matched controls in the backbone it is reported for.

**Cost.** One `table1`-style grid, CLIP only, four budgets.

**If not run**, add the Qwen3-VL 8B width-matched row to the distribution table's notes and say
explicitly that the histogram advantage is consistent with width rather than with the
disagreement information the target was designed to carry.

---

### T2.3 Regularizer table, or its removal (run — already scoped)

`AUDIT.md` already records that the Ridge row and the Lasso/Elastic-net rows come from different
historical run blocks and that the table must be omitted until all three heads are rerun
together under the current protocol. The command is in `paper/supplementary_release/README.md`.
Either complete that rerun or leave the table out; do not ship the mixed-protocol version.

---

## Tier 3 — code fixes (no paper number changes, but the code is released)

### T3.1 `participant_cluster_bootstrap` must not impute or mutate

`src/utils/metrics.py:98–104`. Replace

```python
v = np.asarray(values, float)
...
v[~np.isfinite(v)] = 0.0
```

with a copy, a drop rather than an imputation, and a reported count:

```python
v = np.array(values, dtype=float, copy=True)
ids = np.asarray(participant_ids)
finite = np.isfinite(v)
n_dropped = int((~finite).sum())
v, ids = v[finite], ids[finite]
```

and return `n_dropped` in the result dict. Under pandas 3.x the current line raises
`ValueError: assignment destination is read-only` for a Series argument and silently rewrites
the caller's buffer for a float ndarray. Re-run `reproduce.py` afterwards to confirm
`holm_corrected.csv` is unchanged row-for-row (it should be: the only non-finite rows in the
reported conditions are the unanchored no-image rows, which are already excluded upstream).

### T3.2 `effective_dof` should use the design the ridge actually solved

`src/utils/metrics.py:138–148` re-standardizes `X` with a fresh `StandardScaler`, but the
anchored head was fitted on features standardized by the *training-group* scaler. Either pass
the already-standardized matrix and drop the internal scaler, or document in the table notes
that the reported effective degrees of freedom are computed on re-standardized features and
exclude the intercept.

### T3.3 Record per-test *n* and the bootstrap resolution floor

`wilcoxon_paired` drops non-finite pairs without recording how many remained. Add the surviving
*n* to every row of `output/statistics/holm_corrected.csv`, and record `n_boot` so that
"*p* < .001" is traceable to the 10⁻⁴ resolution limit rather than read as an exact zero.

### T3.4 Report the ablation under both weight conventions

`experiments/faithfulness.py:165–203` supports `*_w` as well as `*_wtot`. Emit both and print
both in `summary.csv`, so the supplementary can show them side by side rather than describing
the difference in prose (`AUDIT.md` records .098 versus approximately .042 for the top-weight
drop).

### T3.5 Assert the invariants the README already states

`main.py` and `reproduce.py` should fail loudly, not silently, if: the first-occurrence filter
does not leave 83,327 rows; there are not 387 units; a participant is a test user more than
once; or any reported condition contains a non-finite per-unit metric. Three assertions, and
they make the reviewer package self-checking.

---

## Not recommended

**A user study before this submission.** It is the right next paper, not a patch. Attempting a
small one now would produce an underpowered result that weakens rather than strengthens the
claim. Scope the contribution to computational faithfulness (see `WORDING_CHANGES.md` §10) and
state the protocol as declared future work.

**An editing-improves-accuracy experiment.** Tested on the released formulas and it does not
work: blending each user's weights toward the association measured on half their evaluation
images lowers SROCC on the other half (.4494 → .4433 using the user's own reported emotions,
against .4418 for a random edit of the same norm). Twenty-five images are too few to estimate
the correction. Report controllability (T1.7) instead of correction gain.
