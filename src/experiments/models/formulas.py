"""Fit and store the per-user formulas used by downstream diagnostics.

The module fits one canonical formula database for the requested Stage-2
variant. Faithfulness and emotion-importance analyses read these stored fits
without refitting a separate head.

That works because the personal head is linear. Under the population anchor,

    y_hat(image) = y_pop(image) + z(image) . delta + b            (population anchor)
    y_hat(image) = z(image) . w + b                               (plain unanchored)

so a prediction, an ablated prediction, or one user's formula applied to
another user's images is arithmetic over saved numbers, not a refit.

Stored weight vectors
---------------------
w_pop     the population's own mapping from the seven emotions to a score,
          fitted on the pooled training group in the training group's units.
delta     what population anchor fits: this user's departure from that mapping. It is
          a residual fit that ridge shrinks toward zero.
w_total   w_pop + delta, the absolute personal mapping. This is literally the
          formula variant B fits and predicts with. Under population anchor the
          prediction uses y_pop on raw features instead, which carries more
          information than z . w_pop can, so w_total is the emotion-space
          reading of the same person rather than a term in C's prediction.
          It is used for the reported emotion-space interpretation and the
          comparison with the ground-truth formula.

Outputs, all in output/formulas/:

    user_formula.csv         per unit - 387 rows:
                                delta, w_pop, w_total, bias, alpha, accuracy, and the support-set mean of z,
                                which is what an ablation holds an emotion at
    user_formula_gt.csv     per unit - 387 rows: the same head fitted on the emotions the
                             user actually reported, which is the ceiling the
                             predicted-emotion formula is measured against
    per_image.csv           per (unit, evaluation image) ~19350 data rows:
                                the true score, the population prediction, the personal prediction,
                                and each emotion predicted, standardised, and as its contribution
    stage1_accuracy.csv     per (fold, domain, emotion) - 105 rows:
                                how well Stage-1 predicts the population-mean emotion

Usage:
    uv run main.py formulas --backbone qwen8b --stage2 C --n-train 100
    uv run main.py formulas --subsets
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data import CORE7, DOMAINS
from src.modeling.heads import make_head
from src.modeling.pipeline import _PopAnchoredRidge, _ResidualHead
from src.utils.metrics import evaluate, plcc, srocc

OUT = "formulas"
N_SUBSET = 50
SUBSETS = ["A", "B", "C"]
N_PARTITIONS = 10

def _fit(head, M_tr, y_tr, X_tr, alpha):
    """One calling convention for the anchored and unanchored heads."""
    if isinstance(head, (_ResidualHead, _PopAnchoredRidge)):
        return head.fit(M_tr, y_tr, X_raw=X_tr, frozen_alpha=alpha)
    return head.fit(M_tr, y_tr, frozen_alpha=alpha)

def _predict(head, M_ev, X_ev):
    """Predict with a fitted personal head (anchored heads also need the raw features for y_pop)."""
    if isinstance(head, (_ResidualHead, _PopAnchoredRidge)):
        return head.predict(M_ev, X_raw=X_ev)
    return head.predict(M_ev)

def _intercept(head) -> float:
    """Return the fitted intercept across the supported head wrappers."""
    if hasattr(head, "_b"):
        return float(head._b)
    pipe = getattr(head, "_pipe", None)
    return float(pipe[-1].intercept_) if pipe is not None else float("nan")

def population_context(cfg, pipeline, fold, dom: str, feats, n_train: int,
                       mediator: str = "emotion", variant: str = "C", seed: int = 0) -> dict:
    """Everything the test users of one (fold, domain) share:
    GIAA (pop_ridge), Stage-1 (mediator_model), the Stage-2 anchor (scaler, w_pop, b_pop)
    and the personal alpha for n_train images, selected on the validation users."""
    
    want = list(dict.fromkeys([mediator, "identity", "emotion", "pca",
                                           "random", "shuffled"]))
    Xg, Eg, yg, meds = pipeline.shared_context(fold, dom, feats, seed=seed, want=want)
    med = meds[mediator]
    Xv, _, yv = pipeline.val_data(fold, dom, feats) #tuning set, for alpha selection
    pop_ridge = make_head("ridge", cfg).fit(Xg, yg, val=(Xv, yv))

    anchor = None
    if variant == "C" and mediator in set(cfg.stage2_variant_mediators):
        anchor = pipeline.pop_anchor(med, Xg, yg, Xv, yv,
                                        mediator_name=mediator)
    
    # --- select the personal alpha for this fold/domain/variant ------
    alpha = pipeline.select_personal_hyperparam(
        fold, dom, feats, med, "ridge", n_train, variant, anchor, pop_ridge,
        mediator_name=mediator)
    return {"mediator_model": med, "pop_ridge": pop_ridge, "anchor": anchor, "alpha": alpha}

def run(cfg, pipeline, variant: str = "C", n_train: int | None = None,
        mediator: str = "emotion", seed: int = 0,
        domains: list[str] | None = None):
    out_dir = cfg.run_dir(OUT)
    domains = domains or DOMAINS
    n = int(n_train or cfg.n_train)
    variant = variant or "plain"

    unit_rows, gt_rows, image_rows, s1_rows = [], [], [], []

    for fold in pipeline.split.folds():
        feats = pipeline.backbone.features_for_fold(fold.index)
        for dom in domains:
            context = population_context(cfg, pipeline, fold, dom, feats, n, mediator, variant, seed)
            med, pop_ridge, anchor, alpha = context["mediator_model"], context["pop_ridge"], context["anchor"], context["alpha"]
            
            # ---- Stage-1 accuracy, on the images the test users rated ------
            test_df = pipeline.ds.subset(domain=dom, users=fold.test_users)
            test_df = test_df[test_df["stimulus_id"].astype(str).isin(feats)]
            gt_img = pipeline.ds.per_stimulus(test_df)
            if len(gt_img) >= 10:
                P = np.asarray(med.transform(
                    pipeline.backbone.matrix(feats, gt_img.index)), float)
                for j, e in enumerate(CORE7):
                    truth = gt_img[e].to_numpy(float)
                    s1_rows.append({"fold": fold.index, "domain": dom,
                                    "emotion": e, "n_images": len(gt_img),
                                    "srocc": srocc(truth, P[:, j]),
                                    "plcc": plcc(truth, P[:, j])})

            # ---- the population's own formula, in emotion space -------------
            scaler = anchor[0] if anchor is not None else None
            w_pop = (np.asarray(anchor[1], float).ravel() if anchor is not None
                     else np.full(len(CORE7), np.nan))
            b_pop = float(anchor[2]) if anchor is not None else float("nan")

            # --- the personal formulas, one row per unit --------------------
            for unit in pipeline.iter_units(fold, dom, feats, n_train=n):
                # extract the features for this unit's support set and evaluation set, and transform them into the mediator space
                M_tr = med.transform(unit.X_train)
                M_ev = med.transform(unit.X_eval)
                
                head = pipeline.make_personal("ridge", unit.user_id, variant, anchor, pop_ridge)
                _fit(head, M_tr, unit.y_train, unit.X_train, alpha)
                p = _predict(head, M_ev, unit.X_eval)
                p_pop = np.asarray(pop_ridge.predict(unit.X_eval), float).ravel()

                w = np.asarray(head.weights(), float).ravel()
                if len(w) != len(CORE7):
                    raise ValueError(
                        f"expected {len(CORE7)} coefficients, got {len(w)}; "
                        f"mediator '{mediator}' is not seven-dimensional so "
                        f"the emotion labels would be wrong")

                if scaler is None:
                    # A plain RidgeHead owns its StandardScaler -> Z
                    personal_scaler = head._pipe[0]
                    Z_tr = personal_scaler.transform(M_tr)
                    Z_ev = personal_scaler.transform(M_ev)
                    formula_scaler = personal_scaler
                else:
                    Z_tr = scaler.transform(M_tr)
                    Z_ev = scaler.transform(M_ev)
                    formula_scaler = scaler
                Z_tr, Z_ev = np.asarray(Z_tr, float), np.asarray(Z_ev, float)
                zbar = Z_tr.mean(axis=0)

                m = evaluate(unit.y_eval, p)
                base = {"fold": unit.fold, "domain": unit.domain,
                        "user_id": unit.user_id}
                unit_rows.append({
                    **base, "n_train": n, "variant": variant,
                    "mediator": mediator, "backbone": cfg.backbone,
                    "seed": int(seed), "alpha": float(alpha),
                    "bias": _intercept(head), "b_pop": b_pop,
                    **{f"{e}_w": float(v) for e, v in zip(CORE7, w)},
                    **{f"{e}_wpop": float(v) for e, v in zip(CORE7, w_pop)},
                    **{f"{e}_wtot": float(b if anchor is None else a + b)
                       for e, a, b in zip(CORE7, w_pop, w)},
                    # what an ablation holds an emotion at: its mean over this
                    # user's own support set, in the same standardised units
                    **{f"{e}_zbar": float(v) for e, v in zip(CORE7, zbar)},
                    **{f"{e}_zmean": float(v) for e, v in
                       zip(CORE7, formula_scaler.mean_)},
                    **{f"{e}_zscale": float(v) for e, v in
                       zip(CORE7, formula_scaler.scale_)},
                    **m,
                    "pop_srocc": float(srocc(unit.y_eval, p_pop)),
                    "pop_plcc": float(plcc(unit.y_eval, p_pop)),
                    "gain_srocc": float(m["srocc"] - srocc(unit.y_eval, p_pop)),
                    "n_eval": int(len(unit.y_eval)),
                    "y_sd": float(np.std(unit.y_eval)),
                })

                # ---- the ceiling row: the same head on the emotions the user reported --------
                gt_head = make_head("ridge", cfg).fit(unit.E_train, unit.y_train)
                p_gt = gt_head.predict(unit.E_eval)
                gt_rows.append({
                    **base, "n_train": n, "backbone": cfg.backbone,
                    "seed": int(seed), "alpha": float(gt_head.alpha_),
                    "bias": _intercept(gt_head),
                    **{f"{e}_w": float(v) for e, v in
                       zip(CORE7, np.asarray(gt_head.weights(), float).ravel())},
                    **evaluate(unit.y_eval, p_gt),
                })

                contrib = Z_ev * w[None, :]
                ids = (unit.eval_ids if getattr(unit, "eval_ids", None)
                       else [""] * len(unit.y_eval))
                for i, sid in enumerate(ids):
                    image_rows.append({
                        **base, "stimulus_id": sid,
                        "y_true": float(unit.y_eval[i]),
                        "y_pred": float(p[i]), "y_pop": float(p_pop[i]),
                        **{f"{e}_pred": float(M_ev[i, j]) for j, e in enumerate(CORE7)},
                        **{f"{e}_z": float(Z_ev[i, j]) for j, e in enumerate(CORE7)},
                        **{f"{e}_c": float(contrib[i, j]) for j, e in enumerate(CORE7)},
                        **{f"{e}_rated": float(unit.E_eval[i, j]) for j, e in enumerate(CORE7)},
                    })
        print(f"  fold {fold.index} done ({len(unit_rows)} units)", flush=True)

    # ----------------------------------------------------------
    units = pd.DataFrame(unit_rows)
    frames = {"user_formula.csv": units,
              "user_formula_gt.csv": pd.DataFrame(gt_rows),
              "per_image.csv": pd.DataFrame(image_rows),
              "stage1_accuracy.csv": pd.DataFrame(s1_rows)}
    for name, df in frames.items():
        df.to_csv(out_dir / name, index=False)

    counts = ", ".join(f"{name}={len(df):,}" for name, df in frames.items())
    print(f"[formulas] wrote {counts} to {out_dir}")
    return frames

# ----------------------------------------------------------------------------------------------
def run_subset(cfg, pipeline):
    """All images each user rated in a domain are shuffled 10 times; each shuffle is cut into
    3 disjoint subsets of 50 and one formula is fit per subset.
    output/formulas/subset_formulas.csv: 387 units x 10 partitions x 3 subsets = 11,610 rows."""
    
    out_dir = cfg.run_dir(OUT)
    rows = []
    for fold in pipeline.split.folds():
        feats = pipeline.backbone.features_for_fold(fold.index)
        for dom in DOMAINS:
            context = population_context(cfg, pipeline, fold, dom, feats, N_SUBSET)

            domain_df = pipeline.ds.subset(domain=dom)
            domain_df = domain_df[domain_df["stimulus_id"].astype(str).isin(feats)]
            
            for user_id in sorted(fold.test_users):
                user_df = domain_df[domain_df["user_id"] == user_id]
                per_image = pipeline.ds.per_stimulus(user_df)
                stimuli = user_df["stimulus_id"].astype(str).unique()
                
                # all images this user rated in this domain
                user_images = sorted(image for image in stimuli if image in per_image.index)
                assert len(user_images) >= N_SUBSET * len(SUBSETS), f"user {user_id} {dom}: too few images"
                
                # reshuffle and partition to 3 disjoint subsets of 50
                for partition in range(N_PARTITIONS):
                    rng = np.random.RandomState(1000*partition + int(user_id))
                    shuffled_images = list(rng.permutation(user_images))
                    for index, subset in enumerate(SUBSETS):
                        train_ids = shuffled_images[index * N_SUBSET:(index + 1) * N_SUBSET]
                        
                        X_train = pipeline.backbone.matrix(feats, train_ids)
                        y_train = per_image.loc[train_ids, "overall"].to_numpy(float)
                        head = pipeline.make_personal("ridge", user_id, "C", context["anchor"], context["pop_ridge"])
                        
                        _fit(head, context["mediator_model"].transform(X_train), y_train, X_train, context["alpha"])
                        
                        rows.append({"fold": fold.index, "domain": dom, "user_id": int(user_id),
                                     "partition": partition, "subset": subset,
                                     "n_images": len(shuffled_images), "alpha": float(context["alpha"]),
                                     "bias": _intercept(head),
                                     **{f"{e}_w": float(v) for e, v in zip(CORE7, np.asarray(head.weights(), float).ravel())}})
            print(f"  fold {fold.index} {dom} done", flush=True)
    subset_df = pd.DataFrame(rows)
    subset_df.to_csv(out_dir / "subset_formulas.csv", index=False)
    print(f"[formulas] wrote {len(subset_df):,} rows to {out_dir / 'subset_formulas.csv'}")
    return subset_df