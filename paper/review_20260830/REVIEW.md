# Adversarial review — WACV 2027 submission #2267

*Can Aesthetic Emotions Explain Individual Taste?* — paper + code audit.

Reviewed material: the submitted PDF (10 pages), `paper/supplementary_release/supplementary.tex`,
and the code under `src/` (`utils/metrics.py`, `modeling/pipeline.py`, `modeling/heads.py`,
`experiments/faithfulness.py`, `experiments/make_upper_bounds.py`, `data/data.py`).

Every number quoted below was recomputed from the project's own frozen release CSVs
(`output/raw_all_final.csv`, `output/formula_units_final.csv`, `output/formula_images_final.csv`,
`Dataset/maked/ratings.csv`). The script that reproduces them is
`supp_v2_analyses.py`, included with this review.

---

## Summary judgement

The study is well posed, the protocol is leak-free, the statistical machinery is unusually
careful for this area, and the internal audit (`tmp/release_verify/PAPER_DIFF_REPORT.md`)
already caught most transcription-level problems. The paper's weaknesses are not sloppiness;
they are **three claims that are stated more broadly than the evidence supports**, and each of
the three is discoverable by a reviewer who runs the released code. Disclosing and scoping them
in the supplementary turns all three from rejection risks into evidence of rigour.

Severity key: **A** = would justify rejection if a reviewer finds it and the paper did not
disclose it; **B** = will draw a substantive objection; **C** = correctness or presentation.

---

## A1. The central RQ2 claim does not hold on four of five backbones

The Conclusion states:

> "That margin is the price of the bottleneck's width, not of insisting its dimensions be
> nameable: an unsupervised basis of the same width does no better under the population anchor."

This is tested only on Qwen3-VL 8B. Under the anchor at *N* = 100, with the participant-cluster
bootstrap over 129 participants:

| Backbone | Population | Direct | Hybrid | PCA | Hybrid − PCA | 95 % CI | *p* |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B/16 | .416 | .456 | .435 | **.445** | −.0102 | [−.018, −.003] | .007 |
| CLIP-ft (aesthetic) | .418 | .459 | .434 | **.447** | −.0129 | [−.021, −.005] | <.001 |
| CLIP-ft (emotion) | .406 | .449 | .423 | **.441** | −.0179 | [−.025, −.011] | <.001 |
| Qwen3-VL 4B | .436 | .470 | .448 | .454 | −.0055 | [−.014, +.002] | .156 |
| Qwen3-VL 8B | .439 | .478 | **.459** | .458 | +.0011 | [−.006, +.008] | .761 |

On three backbones PCA is **significantly better** than the emotion bottleneck under the anchor,
by up to .018 SROCC — comparable to the Hybrid–Direct gap the paper treats as the headline cost.
"Naming costs nothing" is a property of the primary backbone, not of the design.

The unanchored comparison, which is where the paper's real argument lives, does survive:

| Backbone | Hybrid − PCA (unanchored) | 95 % CI | *p* |
|---|---|---|---|
| CLIP | +.0413 | [+.029, +.053] | <.001 |
| CLIP-ft (aesthetic) | +.0020 | [−.007, +.011] | .670 |
| CLIP-ft (emotion) | −.0024 | [−.011, +.006] | .574 |
| Qwen3-VL 4B | +.0732 | [+.061, +.086] | <.001 |
| Qwen3-VL 8B | +.0633 | [+.051, +.075] | <.001 |

Without the anchor the named basis is better on three backbones and indistinguishable on the
other two; it is never worse. That is a defensible claim. The anchored claim is not.

**Fix.** Report the full mediator × backbone table (anchored and unanchored) in the
supplementary, and rewrite the Conclusion and Discussion sentences to say what is actually true:
under the anchor, naming is free on the primary backbone and costs .006–.018 SROCC on the
CLIP family; without the anchor, the named basis is never worse than PCA and is better on three
of five backbones. See `WORDING_CHANGES.md` §1 and §6.

---

## A2. The weight-alignment result is mostly a population effect, not a personal one

Table 3 reports mean Spearman ρ = .525 between the fitted weight vector and the user's empirical
emotion–rating associations, and the paper reads it as evidence that the formula "recovers
associations visible in the person's own ratings".

The test has no null. Because the anchored model reports the **total** weight
w_tot = w_pop + δ_u, and w_pop is shared by every user in a fold–domain, most of the alignment
is available to any user's weight vector. Applying each user's weights to a **different** user's
empirical associations (cyclic peer shift within fold and domain) gives:

| Weight vector | Association computed from | Own ρ | Peer-shifted ρ | Personal excess | 95 % CI | *p* |
|---|---|---|---|---|---|---|
| w_tot | predicted emotions | .525 | .453 | **+.072** | [+.041, +.102] | <.001 |
| δ_u (residual) | predicted emotions | .122 | .025 | **+.097** | [+.060, +.135] | <.001 |
| w_tot | the user's own reported emotions | .478 | .458 | +.020 | [−.012, +.052] | .112 |
| δ_u | the user's own reported emotions | .042 | .000 | **+.041** | [+.005, +.077] | .013 |

Three things follow. (i) 86 % of the headline .525 is shared, not personal. (ii) The personal
excess is real and significant, but an order of magnitude smaller than the reported number
suggests. (iii) Against the user's own *reported* emotions — the only genuinely independent
check, since the model never saw them — only the residual coefficient vector shows a personal
excess, and the total vector does not.

A reviewer who asks "what would a stranger's weights score?" gets .453 and the claim collapses
into a much weaker one. The paper should ask that question itself.

**Fix.** Add the peer-shifted null and the reported-emotion variant as a supplementary table
(§S10.2 in the additions), and restate the Results and Conclusion sentences so the claim is
"positive and specific to the owner, at a magnitude the null makes explicit" rather than
"ρ = .525". See `WORDING_CHANGES.md` §5 and §8.

---

## A3. "Wrong emotions mislead" is not supported — any noise of the same scale does this

Results, lines 445–449, and Discussion, line 513:

> "This falls below the 0.439 obtained when the emotion inputs are zeroed out entirely: wrong
> emotional cues do not merely fail to inform, they appear to mislead it."

Two problems.

**(a) The stated control is wrong.** The no-image condition supplies the user's support-set mean
emotion vector, not zeros — the supplementary says so explicitly (§S4, "This is a baseline
emotion vector rather than setting the seven raw emotion ratings to zero"). The main paper and
its own supplementary contradict each other on this point, and the internal audit already
flagged it (`PAPER_DIFF_REPORT.md`, `sec/5_result.tex` lines 102–104; `sec/6_discussion` line 46).
The correction was not carried into the submitted PDF.

**(b) The inference does not hold.** Because the anchored no-image predictor is
ŷ_pop(x) + constant, its SROCC is *identical to the Population row by construction* (.4391 in
both). Anything added to it that is uncorrelated with the target must lower the correlation. A
control confirms that the drop is a variance effect, not a "misleading" one:

| Condition (anchored, *N* = 100) | Mean SROCC | Bootstrap SEM |
|---|---|---|
| Correct image | .4592 | .0114 |
| No image (support-set mean) | .4391 | .0110 |
| Swapped image (49 cyclic shifts) | .3838 | .0098 |
| **Gaussian noise matched to cov(ê)** | **.3852** | .0098 |
| Independently shuffled emotion columns | .3688 | .0100 |

Swapped emotions (.3838) and covariance-matched Gaussian noise (.3852) are indistinguishable.
The image-swap test does establish image specificity — correct .459 vs swapped .384 — but it
says nothing about "misleading", and the comparison against the no-image condition carries no
information beyond arithmetic.

**Fix.** Add the matched-noise control (a strength: it shows the authors knew the objection) and
delete the "mislead" inference from Results, Discussion and Conclusion. See
`WORDING_CHANGES.md` §4, §7, §9.

---

## B1. Test–retest reliability is computed on a different unit from every model number

`make_upper_bounds.py` computes test–retest as **one pooled correlation over all 4,509 repeat
pairs**, across users and images. Every model number in Table 1 is a **within-user** correlation
over that user's 50 evaluation images, averaged over units. The paper then writes that
test–retest "sets the scale on which differences in Tab. 1 should be read" — but the two
quantities are not on the same scale. The pooled figure absorbs between-participant variance in
rating level that the within-user metric removes.

Recomputed on the unit the models are actually scored on (median 12 repeat items per
user–domain, 387 units, no undefined units):

| | Artwork | Fashion | Landscape | Average |
|---|---|---|---|---|
| Pooled over pairs (as reported) | .672 | .688 | .695 | .684 |
| **Within user–domain** | **.577** | **.593** | **.584** | **.585** [.550, .617] |

The corrected reference is .585, not .684. This helps the paper — Hybrid's .459 sits much closer
to the target's own reliability — but the current figure is not comparable to the table it is
placed in, and a statistically minded reviewer will say so.

**Fix.** Report both, label the within-unit value as the one comparable to Table 1, and note
that with ~12 repeat items per unit the within-unit estimate is itself noisy and
downward-biased. See §S6 of the additions and `WORDING_CHANGES.md` §3.

---

## B2. Two null results are read as equivalence; TOST is implemented but never called

`src/utils/metrics.py` contains `tost_equivalence` with a docstring that states the problem
exactly — "A non-significant difference test does NOT mean two methods are the same" — and the
function is called nowhere in the repository. Meanwhile the paper draws two equivalence
conclusions from non-significant differences: Hybrid ≈ PCA (*p* = .761) and Landscape Hybrid ≈
Direct (*p* = .51).

Running the implemented test on the paired participant-level units closes this cleanly:

| Contrast | Mean diff | 90 % CI | Equivalent at δ = .01 | at δ = .02 |
|---|---|---|---|---|
| Hybrid − PCA (Average) | +.0011 | [−.0050, +.0073] | **yes** (*p* = .009) | yes (*p* < .001) |
| Hybrid − Random (Average) | +.0118 | [+.0049, +.0187] | no (*p* = .668) | yes (*p* = .025) |
| Hybrid − Direct (Average) | −.0183 | [−.0242, −.0125] | no (*p* = .991) | no (*p* = .319) |
| Hybrid − Direct (Landscape) | −.0044 | [−.0123, +.0035] | no (*p* = .122) | **yes** (*p* < .001) |

This is strictly better evidence than the *p*-values now in the paper, and it costs one script
run. The margin δ must be fixed on substantive grounds before the test — the function's own
docstring says so; propose δ = .02 SROCC (roughly two bootstrap SEMs, and below the
Hybrid–Direct gap the paper calls small) with δ = .01 reported as a stricter secondary.

---

## B3. No human-subject evidence for a human-centred claim, and "correct" is untested

The abstract promises "aesthetic systems a person can inspect and correct". Inspection is
supported by the faithfulness checks. **Correction is never tested**, and no person ever reads a
formula anywhere in the study. For the Applications Track this is the most predictable reviewer
question.

An editing experiment run on the released formulas does not rescue the claim: blending each
user's weights toward the empirical association measured on half their evaluation images and
scoring on the other half *lowers* SROCC (.4494 → .4433 using the user's own reported emotions;
.4418 for a random edit). With ~25 images the correction signal is too noisy to help.

What the data does support is **controllability**: uniformly scaling the personal correction by
γ traces a smooth unimodal curve.

| γ | 0 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|---|---|
| SROCC | .4391 | .4516 | .4600 | **.4628** | .4592 | .4512 | .4377 | .4075 |

Two things worth saying out loud. The response to a weight edit is smooth and single-peaked, so
the formula behaves as a control surface rather than a brittle fit. And the protocol-selected
head is mildly under-regularized: a post-hoc uniform shrinkage to γ = 0.75 would add .0036
SROCC, which the paper does not take because γ cannot be chosen without touching test data.
Disclosing that pre-empts a reviewer who computes it.

**Fix.** Scope the abstract and conclusion to inspection, present editability as an
architectural property with the γ curve as evidence of well-behaved control, and state the
user-study protocol as declared future work rather than implied contribution.

---

## B4. The ICI / MIR comparison is uninformative and should be demoted

The first contribution bullet and the Conclusion both say the seven emotions "numerically exceed
the ICI and MIR reference values on average". So does the **content-free random projection**:
Random reaches .447 against ICI .433 and MIR .438. The comparison is confounded by backbone,
features and protocol, and the paper already disclaims significance testing against it. Keeping
it in the contribution list invites a reviewer to make exactly the observation above.

**Fix.** Remove the claim from the contributions and the Conclusion; keep the rows in Table 1 as
context with the existing disclaimer, and add one supplementary sentence noting that Random also
exceeds them, which is why the comparison is not evidence about the mediator.

---

## B5. Units within a fold are not independent, and the paper's tests assume they are

All test users in a fold share one Stage-1 mediator, one population head and one frozen
hyperparameter. The paired Wilcoxon over 129 participants and the participant-cluster bootstrap
both treat the fold as fixed, so fold-level variance never enters the standard errors.

A fold-level check (five paired observations, one per fold) reaches the same conclusions:

| Fold | Population | Direct | Hybrid | PCA |
|---|---|---|---|---|
| 0 | .4155 | .4533 | .4273 | .4164 |
| 1 | .4331 | .4731 | .4468 | .4603 |
| 2 | .4532 | .4865 | .4736 | .4716 |
| 3 | .4688 | .4889 | .4799 | .4788 |
| 4 | .4225 | .4841 | .4647 | .4618 |

Hybrid − Direct is negative in all five folds (mean −.0187, paired *t* *p* = .006);
Hybrid − Population is positive in all five (mean +.0199, *p* = .027); Hybrid − PCA is +.0007
(*p* = .870). Adding this as a robustness paragraph costs nothing and removes the objection.

---

## B6. One support-set draw per user; support-set variance is unquantified

`per_user_split` uses a single deterministic draw (seed 42 + user_id) for every user. Which 100
images a person happened to rate is a source of variance the paper never measures, and PIAA
results are known to be sensitive to it. Every reported SEM is conditional on that one draw.

The "three seeds" that are averaged do not help here: for the emotion mediator with a ridge head
the pipeline is deterministic, and all three seeds give bit-identical results (.459236 for each).
Seeds vary only the Random/Shuffled mediators and the MLP. The paper's "we average the per-seed
metrics over three seeds" is true but conveys more robustness than it delivers.

**Fix.** Either run three additional support draws for the primary configuration and report the
spread, or state plainly that results are conditional on a single fixed support draw and that
seed averaging affects only the stochastic conditions. The first is much stronger and is a
single re-run of one grid.

---

## B7. The interpretable part explains the residual, not the prediction

Under Anchor C the executed model is ŷ = ŷ_pop(x) + Σ w_i ê_i, where ŷ_pop is a ridge over
4,096 raw features. The seven readable weights are therefore an explanation of the *deviation
from a black box*, not of the prediction. The abstract's "Everything personal is then seven named
weights" is literally true, but a reader can easily take it as "everything is seven weights".

The paper handles this correctly in Discussion ("the two components divide the labour"), but
never quantifies it. From the released per-image records:

* mean sd(ŷ_pop) = .579, mean sd(personal term) = .311 — the personal term is about half the
  scale of the population term, not a rounding correction;
* median share of prediction variance carried by the personal term = .193 (mean .530, the mean
  inflated by units with near-constant ŷ_pop);
* mean ‖w_pop‖ = .387 versus mean ‖δ_u‖ = .385 — the personal correction is the same magnitude
  as the shared emotion-space direction.

Reporting these numbers converts a rhetorical vulnerability into a quantitative answer.

---

## C1. Code correctness

**`participant_cluster_bootstrap` imputes zero for undefined correlations, in place.**
`metrics.py:102` does `v[~np.isfinite(v)] = 0.0`. Two consequences.

* `np.asarray(values, float)` returns the caller's buffer for a float64 array, so passing a
  NumPy array silently rewrites the caller's data (verified: `[0.1, nan, 0.3, 0.4]` becomes
  `[0.1, 0.0, 0.3, 0.4]`). Under the locked environment (pandas 3.x) passing a Series instead
  raises `ValueError: assignment destination is read-only`, so the function is either
  destructive or crashes depending on the argument type.
* Substantively, replacing an undefined SROCC with 0 is a downward-biased imputation, not a
  neutral one. `summarize_image_swap` pre-filters non-finite rows; `summarize_ablation`,
  `summarize_swap` and `summarize_align` do not.

Fix: copy the array, drop non-finite rows rather than imputing, and report the number dropped.
In the reported conditions this changes nothing — the only undefined rows are the unanchored
no-image row already excluded — but the code as written cannot be defended if a reviewer reads it.

**`effective_dof` re-standardizes features the model never saw in that form.**
`metrics.py:145` applies a fresh `StandardScaler` to `X`, but the anchored ridge was solved on
`Z` standardized by the *training-group* scaler. The reported effective degrees of freedom are
therefore computed on a different design matrix from the one the penalty acted on, and the
intercept is not counted. The values in Table 1 are indicative rather than exact.

**The GT-emotion reference uses a different hyperparameter rule from every other row.**
`make_upper_bounds.py:46` calls `make_head("ridge", cfg).fit(...)` with `val=None`, which routes
to `RidgeCV` on the user's own support set (`heads.py:127`). Every predictive row uses a penalty
frozen on the validation group. The code comments this deliberately ("it is meant to be
generous"), and the supplementary describes the GT block as "diagnostic", but the paper never
tells the reader that the 0.749 figure is additionally favoured by per-user tuning. One clause
fixes it.

**Two faithfulness tests attribute the emotion channel differently.** `ablation` uses
`w_tot = w_pop + δ_u`; `image_swap` uses `δ_u` and reconstructs the prediction exactly.
`AUDIT.md` records that switching the ablation to δ_u changes the top-weight drop from .098 to
approximately .042. The supplementary explains the boundary, but the two conventions sit in
adjacent tables without a shared statement. Report both ablation variants side by side.

**Minor.** `wilcoxon_paired` uses the normal approximation without continuity correction and
drops non-finite pairs without recording *n*; the per-test *n* should appear in the released
statistics CSV. Domain rows in `summarize_align` report mean ± 1.96 SEM as a "95 % CI", which is
a normal approximation, while the Average row uses a bootstrap percentile interval — the table
should say so.

---

## C2. Presentation and internal consistency

1. **The audit's own corrections did not all reach the PDF.** `PAPER_DIFF_REPORT.md` lists
   `sec/5_result.tex` lines 102–104 ("the no-image control does not zero the emotion inputs")
   and `sec/6_discussion` line 46; both sentences are still in the submitted text.
2. **Table 1 is easy to misread.** PCA (.510) exceeds Hybrid (.496) in Artwork, and Random
   (.495) is within .001 of it, but neither carries an arrow because neither reaches
   significance. A reader scanning the column will notice the ordering before the footnote. A
   sentence in Results naming the Artwork PCA result is safer than leaving it to be found.
3. **The bootstrap *p*-value floor is not stated.** With 10,000 resamples the smallest attainable
   *p* is 10⁻⁴, and the percentile-inversion rule can return exactly 0. "*p* < .001" is the
   correct way to report it; the supplementary should say that is a resolution limit.
4. **Prespecification.** The supplementary calls the Holm families "prespecified". Without a
   registration, "fixed before the audit was run" is the accurate and equally sufficient phrasing.
5. **"First occurrence" assumes file order is session order.** `data.py` keeps the first row per
   (user, image) in file order. If the released file is not ordered by session this is a nominal
   rather than temporal first session. One sentence disposing of this is cheap.
6. **Figure 1 caption** says "one personal head per user g_u, 387 in total" — 387 is
   users × domains, not users. Rephrase as "one personal head per user and domain".
7. **`\dagger` in Table S1** is defined in the notes but the caption of `tab_backbone` gives the
   family as "across five backbones"; make explicit that Direct is excluded from that family.

---

## What the paper gets right, and should keep

These are worth protecting in revision, because they are the reasons to accept:

* The split protocol is genuinely leak-free at both user and image level, and the validation
  group mirrors the test protocol rather than using an internal CV shortcut. This is stricter
  than most PIAA work.
* Every mediator, including the content-free controls, receives the population anchor. The
  config comment explaining why the earlier design was wrong is exactly the right decision.
* The anchored/unanchored contrast (Table 2) is the paper's best idea and its cleanest result.
  The +.192 versus +.014 asymmetry is a real finding.
* The formula-swap result — another person's formula is worse than no personalization
  (.396 vs .439) — is non-trivial and survives every control I applied.
* The GT-emotion reference at .749 versus Direct at .478 is a strong and well-caveated statement
  about where the bottleneck's loss comes from.
* The internal numerical audit and the released per-unit CSVs are better provenance than most
  submissions provide.

---

## Recommended disposition

With the wording corrections in `WORDING_CHANGES.md` and the supplementary additions in
`supplementary_additions.tex` — in particular the backbone-wide mediator table (A1), the
alignment null (A2), the matched-noise control (A3) and the corrected reliability reference
(B1) — the paper's claims match its evidence, and the remaining objections are ordinary
limitations. Without them, A1 and A2 are each sufficient grounds for a reviewer to argue that
the central conclusions are not supported.
