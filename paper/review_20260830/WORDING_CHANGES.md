# Wording changes

Separated from the computational work in `TODO_ADDITIONAL_WORK.md`. Each item gives the current
sentence, the replacement, and the reason. Line numbers refer to the numbered lines in the
submitted PDF. Replacements are written to match the surrounding register: plain, declarative,
no new vocabulary.

Items marked **[blocked]** require a number from `TODO_ADDITIONAL_WORK.md` before the
replacement can be inserted.

---

## Main paper

### §1. Conclusion — the naming claim (lines 562–566) **[blocked: T1.1]**

Severity A. As written the sentence is contradicted on four of five backbones.

**Current**

> That margin is the price of the bottleneck's width, not of insisting its dimensions be
> nameable: an unsupervised basis of the same width does no better under the population anchor,
> and worse without it.

**Replace with**

> Most of that margin is the price of the bottleneck's width rather than of insisting its
> dimensions be nameable. On the primary backbone an unsupervised basis of the same width does
> no better under the population anchor (mean difference .001, 90 % CI [−.005, +.007],
> equivalent within .01 SROCC); on the CLIP backbones it is .010 to .018 better, so under the
> anchor naming is free on some representations and carries a small cost on others. Without the
> anchor the named basis is never worse than the unsupervised one and is better on three of the
> five backbones.

---

### §2. Table 1 — Test–retest row **[blocked: T1.4]**

Severity B. The row is currently a pooled across-participant correlation placed in a table of
within-participant correlations.

**Current row values** .672 / .686 / .688 / .691 / .695 / .703 / .684 / .693

**Replace with** the within-user–domain values .577 / .589 / .593 / .595 / .584 / .607 / .585 /
.597, and add to the table notes:

> Test–retest reliability is computed within each user–domain unit over that unit's repeated
> items (median 12 items per unit), which is the unit every model row is scored on. The pooled
> correlation over all 4,509 repeat pairs is higher (.684 SROCC) because it also contains
> between-participant variance in rating level; it is reported in the supplementary material.

---

### §3. §3.4, Reference points (lines 299–303)

Severity B/C. The comparability claim, and the undisclosed selection rule for the GT bound.

**Current**

> GT emotions replaces the Stage-1 predictions with the user's own emotion ratings; collected in
> the same session as the aesthetic score, it is an optimistic bound on what perfect extraction
> would buy rather than a reachable target. Test–retest reliability, the correlation between
> repeat evaluations by the same participant, is not a ceiling on prediction but a measure of how
> stable the target itself is, and sets the scale on which differences in Tab. 1 should be read.

**Replace with**

> GT emotions replaces the Stage-1 predictions with the user's own emotion ratings; collected in
> the same session as the aesthetic score, it is an optimistic bound on what perfect extraction
> would buy rather than a reachable target. Because it is a reference and not a compared
> condition, its penalty is selected per user on that user's own support set rather than frozen
> on the validation group, which makes it deliberately generous. Test–retest reliability, the
> correlation between a participant's repeat evaluations, is not a ceiling on prediction but a
> measure of how stable the target itself is. We compute it within each user–domain unit, the
> unit the models are scored on, so that it sets the scale on which differences in Tab. 1 should
> be read.

---

### §4. §4.3, Specific to the image (lines 439–449) **[blocked: T1.3]**

Severity A. The stated control is factually wrong and the inference does not follow.

**Current**

> ... drops SROCC from 0.459 to 0.384 (∆ = −0.075, Holm-adjusted p < 0.001; Fig. 3c). This falls
> below the 0.439 obtained when the emotion inputs are zeroed out entirely: wrong emotional cues
> do not merely fail to inform, they appear to mislead it. The formula is therefore reading the
> picture in front of it, not a stable prior about the user.

**Replace with**

> ... drops SROCC from 0.459 to 0.384 (∆ = −0.075, Holm-adjusted p < 0.001; Fig. 3c). The
> control condition supplies the user's support-set mean emotion vector rather than an image's
> own, which under the anchor makes the prediction the population score plus a constant and
> returns 0.439. Mismatched emotions fall below that, but so does a random vector drawn to match
> the covariance of the predicted emotions (0.385), so the additional drop reflects the variance
> the emotion term injects rather than any misleading content; the supplementary reports both
> controls. What the comparison establishes is that the formula reads the picture in front of
> it: correctly paired emotions are worth 0.075 SROCC over mismatched ones.

---

### §5. §4.3, Agreeing with the person's own data (lines 450–458) **[blocked: T1.2]**

Severity A. Reported without a null; most of the effect is shared across users.

**Current**

> The two agree closely: a mean Spearman correlation of 0.525 (95 % CI [0.494, 0.554]) and a
> mean Pearson correlation of 0.549 [0.522, 0.574]; the alignment is positive in 94.6 % of units
> (89.9 % Artwork, 96.1 % Fashion, 97.7 % Landscape). The formulas are faithful to the model and
> recover associations visible in the person's own ratings.

**Replace with**

> The two agree: a mean Spearman correlation of 0.525 (95 % CI [0.494, 0.554]) and a mean Pearson
> correlation of 0.549 [0.522, 0.574], positive in 94.6 % of units. Because the total weight
> vector contains the shared population direction, much of this agreement is available to any
> user's formula: applying a peer's weights to the same person's associations still reaches
> 0.453. The personal excess over that null is +0.072 (95 % CI [0.041, 0.102], Holm-adjusted
> p < .001), and it is larger for the personal coefficients alone (+0.097 [0.060, 0.135]). The
> formulas therefore recover associations visible in the person's own ratings, and a measurable
> part of that recovery belongs to the individual rather than to the population.

---

### §6. §5, The anchor and the naming do different work (lines 482–496) **[blocked: T1.1]**

Severity A. Same over-generalization as §1.

**Current final sentence**

> Naming the dimensions costs nothing measurable under the anchor, and without it the named
> basis carries more.

**Replace with**

> On the primary backbone, naming the dimensions costs nothing measurable under the anchor: the
> Hybrid–PCA difference is equivalent to zero within .01 SROCC. That result does not generalize
> across representations. On the three CLIP backbones an unsupervised basis of the same width is
> .010 to .018 SROCC better under the anchor, so where the anchor already supplies the consensus,
> naming can carry a small cost. Without the anchor the ordering reverses and holds across
> backbones: the named basis is better than PCA on three of five and indistinguishable on the
> other two, and better than a content-free projection on all five.

---

### §7. §5, Where seven dimensions suffice (lines 510–515)

Severity A, same error as §4.

**Current**

> ... and swapping in another image's emotions is worse than supplying no emotions at all
> (0.384 vs 0.439), so the formula reads the picture in front of it rather than a stable prior
> about the user.

**Replace with**

> ... and swapping in another image's emotions costs 0.075 SROCC against the correctly paired
> ones, so the formula reads the picture in front of it rather than a stable prior about the
> user.

---

### §8. §7 Conclusion, the faithfulness summary (lines 566–572) **[blocked: T1.2]**

**Current**

> ... it reads the image in front of it rather than a stable prior about its owner, and it
> recovers associations visible in that person's own ratings.

**Replace with**

> ... it reads the image in front of it rather than a stable prior about its owner, and it
> recovers associations visible in that person's own ratings, in part beyond what a peer's
> formula would recover on the same data.

---

### §9. Abstract (lines 017–024)

Severity B. Two adjustments: scope "correct", and drop the ICI/MIR comparison from the headline.

**Current**

> The resulting formulas pass tests of owner specificity, weight sensitivity, image specificity,
> and alignment with each person's own ratings. ... These are the properties an editable personal
> model needs, and a first step toward aesthetic systems a person can inspect and correct in the
> vocabulary they already use for their own experience.

**Replace with**

> The resulting formulas pass tests of owner specificity, weight sensitivity, image specificity,
> and alignment with each person's own ratings, the last of which we report against a
> peer-formula null. ... These are the properties an editable personal model needs. We test what
> such a model can be inspected for, not whether people can use it, and offer the design as a
> first step toward aesthetic systems stated in the vocabulary people already use for their own
> experience.

---

### §10. Contributions, first bullet (lines 088–093)

Severity B. Remove the ICI/MIR claim; a content-free random projection also exceeds them.

**Current**

> A cost accounting for emotion-mediated PIAA. Seven emotions numerically exceed the ICI and MIR
> reference values on average, although these comparisons are descriptive rather than
> significance-tested, while trailing unrestricted direct regression by 0.019–0.026 SROCC across
> five backbones using only seven personal coefficients.

**Replace with**

> A cost accounting for emotion-mediated PIAA. Seven personal coefficients trail unrestricted
> direct regression by 0.019–0.026 SROCC across five backbones, and the gap narrows at small
> rating budgets. We report ICI and MIR as descriptive context only; they use different features
> and a different protocol, and our content-free control also exceeds them, so their ordering is
> not evidence about the mediator.

The Conclusion sentence at lines 558–560 carries the same claim and should be cut to match.

---

### §11. Figure 1 caption

**Current** "one personal head per user g_u, 387 in total"

**Replace with** "one personal head per user and domain, 387 in total"

---

### §12. §3.5, Metrics (lines 338–340)

Severity C. The seed statement conveys more than it delivers.

**Current**

> We compute PLCC and SROCC per user–domain unit and average the per-seed metrics over three
> seeds before inference.

**Replace with**

> We compute PLCC and SROCC per user–domain unit and average the per-seed metrics over three
> seeds before inference. The ridge pipeline over the emotion, PCA and identity mediators is
> deterministic, so seed averaging affects only the stochastic conditions (Random, Shuffled and
> the MLP variants).

---

### §13. §4.1, Results (after line 369) — optional but advised

Severity C. Pre-empt the Artwork ordering a reader will notice in Table 1.

**Add**

> In Artwork the unsupervised PCA mediator is numerically ahead of Hybrid (0.510 against 0.496);
> the domain-wise contrast does not reach significance, and the supplementary reports the same
> comparison across all five backbones.

---

## Supplementary

### §14. §S1.3, prespecification

**Current** "Holm–Bonferroni correction is applied separately within each family in Table S2."
and "prespecified alternative".

**Change** every occurrence of "prespecified" to "fixed before the audit was run". The families
were not registered; the weaker phrasing is accurate and carries the same weight.

### §15. §S1.3, add the bootstrap resolution limit

**Add after the cluster-bootstrap sentence**

> Bootstrap p-values are obtained by inverting the percentile interval and are bounded below by
> 1/10{,}000; values at that bound are reported as p < .001 rather than as zero.

### §16. §S1.1, the first-occurrence rule

**Add after "we retain the first occurrence in the released file"**

> This is the first occurrence in file order. We treat file order as session order, which is how
> the released file is documented; nothing in the analysis depends on which member of a repeated
> pair is retained except the direction of the test–retest reference.

### §17. §S4, the two weight conventions **[blocked: T3.4]**

The ablation uses w_tot and the image swap uses δ_u. The current text explains each separately.
**Add one sentence at the head of §S4**

> Two conventions appear in this section. The ablation reports the total emotion-space weight
> w_tot = w_pop + δ_u, which is the quantity the paper interprets; the image swap reconstructs
> the executed prediction and therefore uses δ_u. Table S9 reports the ablation under both, so
> the difference between the two readings is visible rather than described.

### §18. §S3.3, distribution-valued targets **[blocked: T2.2]**

**Current note** "The histogram target is highest from N = 25 onward, but uses five times the
width of the named bottleneck. We therefore retain the seven-dimensional mean target for the
primary study."

**Replace with**

> The histogram target is highest from N = 25 onward, but it uses five times the width of the
> named bottleneck. On Qwen3-VL 8B, where width-matched controls are available, a
> 35-dimensional PCA basis (.464) and a 35-dimensional random projection (.464) match the
> histogram target (.464) to within .001, so the advantage is consistent with width rather than
> with the disagreement information the target was designed to carry. We therefore retain the
> seven-dimensional mean target for the primary study.

### §19. Table notes on interval type

**Add to the weight-alignment table notes**

> Domain intervals are normal-approximation intervals, mean ± 1.96 SEM; the Average interval is
> a bootstrap percentile interval. The two are not interchangeable and are labelled accordingly.

### §20. §S1.4, software claim

**Current** "Historical training runs were produced on more than one platform; all reported
values are retained as per-unit outputs..."

This is already honest. **Add one clause** so a reviewer does not have to infer it:

> No reported value is recomputed from a rerun on a different platform; the locked environment
> regenerates only the downstream statistics and presentation artifacts from the frozen per-unit
> outputs, and `reproduce.py` checks the regenerated Holm table row-for-row against the frozen
> audit.

---

## Consistency check before submission

Once §1–§13 are applied, these numbers must agree everywhere they appear:

| Quantity | Value | Appears in |
|---|---|---|
| Test–retest, within unit | .585 [.550, .617] | Table 1, §3.4, §5, Table S11 |
| Test–retest, pooled | .684 | Table S11 notes only |
| Hybrid − PCA, primary backbone | +.001, 90 % CI [−.005, +.007] | §4.2, §5, §7, Table S12 |
| Hybrid − PCA, CLIP family | −.010 to −.018 | §5, §7, Table S13 |
| Image swap, matched noise | .385 | §4.3, Table S14 |
| Alignment peer null | .453; excess +.072 [.041, .102] | §4.3, §7, Table S15 |
| Personal-term scale | sd .311 vs sd(ŷ_pop) .579 | §5, Table S16 |

The phrase "zeroed out entirely" must not survive anywhere in the manuscript.
