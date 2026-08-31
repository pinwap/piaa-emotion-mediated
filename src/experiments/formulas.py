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

    population_formula.csv   per (fold, domain) - 15 rows: w_pop, b_pop, alpha
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
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data import CORE7, DOMAINS
from src.modeling.heads import make_head
from src.modeling.pipeline import _PopAnchoredRidge, _ResidualHead
from src.utils.metrics import evaluate, plcc, srocc

OUT = "formulas"


def _fit_predict(head, M_tr, y_tr, M_ev, X_tr, X_ev, alpha):
    """One calling convention for the anchored and unanchored heads."""
    if isinstance(head, (_ResidualHead, _PopAnchoredRidge)):
        head.fit(M_tr, y_tr, X_raw=X_tr, frozen_alpha=alpha)
        return head.predict(M_ev, X_raw=X_ev)
    head.fit(M_tr, y_tr, frozen_alpha=alpha)
    return head.predict(M_ev)

def _intercept(head) -> float:
    """Return the fitted intercept across the supported head wrappers."""
    if hasattr(head, "_b"):
        return float(head._b)
    pipe = getattr(head, "_pipe", None)
    return float(pipe[-1].intercept_) if pipe is not None else float("nan")


def run(cfg, pipeline, variant: str = "C", n_train: int | None = None,
        mediator: str = "emotion", seed: int = 0,
        domains: list[str] | None = None):
    out_dir = cfg.run_dir(OUT)
    domains = domains or DOMAINS
    n = int(n_train or cfg.n_train)
    variant = variant or "plain"

    pop_rows, unit_rows, gt_rows, image_rows, s1_rows = [], [], [], [], []

    for fold in pipeline.split.folds():
        feats = pipeline.backbone.features_for_fold(fold.index)
        for dom in domains:
            want = list(dict.fromkeys([mediator, "identity", "emotion", "pca",
                                       "random", "shuffled"]))
            Xg, Eg, yg, meds = pipeline.shared_context(fold, dom, feats,
                                                       seed=seed, want=want)
            med = meds[mediator]
            Xv, _, yv = pipeline.val_data(fold, dom, feats) #tuning set, for alpha selection
            pop_ridge = make_head("ridge", cfg).fit(Xg, yg, val=(Xv, yv))

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
            anchor = None
            if variant == "C" and mediator in set(cfg.stage2_variant_mediators):
                anchor = pipeline.pop_anchor(med, Xg, yg, Xv, yv,
                                             mediator_name=mediator)
            scaler = anchor[0] if anchor is not None else None
            w_pop = (np.asarray(anchor[1], float).ravel() if anchor is not None
                     else np.full(len(CORE7), np.nan))
            b_pop = float(anchor[2]) if anchor is not None else float("nan")
            pop_rows.append({
                "fold": fold.index, "domain": dom, "backbone": cfg.backbone,
                "variant": variant, "n_train": n, "seed": int(seed),
                "b_pop": b_pop,
                **{f"{e}_wpop": float(v) for e, v in zip(CORE7, w_pop)},
            })

            alpha = pipeline.select_personal_hyperparam(
                fold, dom, feats, med, "ridge", n, variant, anchor, pop_ridge,
                mediator_name=mediator)

            for unit in pipeline.iter_units(fold, dom, feats, n_train=n):
                # extract the features for this unit's support set and evaluation set, and transform them into the mediator space
                M_tr = med.transform(unit.X_train)
                M_ev = med.transform(unit.X_eval)
                
                head = pipeline.make_personal("ridge", unit.user_id, variant, anchor, pop_ridge)
                p = _fit_predict(head, M_tr, unit.y_train, M_ev,
                                 unit.X_train, unit.X_eval, alpha)
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
    frames = {"population_formula.csv": pd.DataFrame(pop_rows),
              "user_formula.csv": units,
              "user_formula_gt.csv": pd.DataFrame(gt_rows),
              "per_image.csv": pd.DataFrame(image_rows),
              "stage1_accuracy.csv": pd.DataFrame(s1_rows)}
    for name, df in frames.items():
        df.to_csv(out_dir / name, index=False)

    counts = ", ".join(f"{name}={len(df):,}" for name, df in frames.items())
    print(f"[formulas] wrote {counts} to {out_dir}")
    return frames
