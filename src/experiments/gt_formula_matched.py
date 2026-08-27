"""Parallel ground-truth formula fit under the same Variant-C protocol.

This is a diagnostic companion to ``user_formula_gt.csv``.  It keeps true
per-user emotion ratings as the Stage-2 input, but otherwise mirrors the
deployable predicted-emotion formula: a training-group scaler, a population
emotion anchor, and one ridge penalty selected on validation-user units per
(fold, domain).  It writes a new file and never overwrites the paper's
existing formula outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data.data import CORE7, DOMAINS
from src.modeling.heads import ALPHA_TIE_RTOL, head_grid, select_alpha_on_val
from src.modeling.pipeline import _PopAnchoredRidge
from src.utils.metrics import evaluate, srocc


OUT = "formulas"
FILENAME = "user_formula_gt_matched.csv"


def _anchor(E_train_group, y_train_group, E_val_group, y_val_group, cfg):
    """Fit the GT population emotion formula in training-group units."""
    scaler = StandardScaler().fit(E_train_group)
    Zg = scaler.transform(E_train_group)
    Zv = scaler.transform(E_val_group)
    alpha = select_alpha_on_val(
        Zg, y_train_group, (Zv, y_val_group), np.asarray(cfg.ridge_alphas, float),
        stage="gt_matched_anchor", component="population_emotion", mediator="gt_emotion")
    ridge = Ridge(alpha=alpha).fit(Zg, y_train_group)
    return scaler, ridge.coef_.ravel(), float(ridge.intercept_), float(alpha)


def _select_personal_alpha(pipeline, fold, domain, feats, n_train, anchor, pop_head):
    """Select one GT personal penalty on validation users, matching Variant C."""
    cfg = pipeline.cfg
    candidates = [float(a) for a in head_grid("ridge", cfg)]
    scores = {a: [] for a in candidates}

    for unit in pipeline.iter_units(fold, domain, feats, n_train=n_train,
                                    users=fold.val_users):
        for alpha in candidates:
            head = _PopAnchoredRidge(cfg, "C", *anchor, pop_head, kind="ridge")
            head.fit(unit.E_train, unit.y_train, X_raw=unit.X_train,
                     frozen_alpha=alpha)
            pred = head.predict(unit.E_eval, X_raw=unit.X_eval)
            scores[alpha].append(srocc(unit.y_eval, pred))

    means = {a: (float(np.mean(v)) if v else -np.inf) for a, v in scores.items()}
    best = max(means.values())
    tied = [a for a, score in means.items()
            if score >= best - abs(best) * ALPHA_TIE_RTOL]
    return max(tied), means


def run(cfg, pipeline, n_train: int | None = None,
        domains: list[str] | None = None):
    """Write matched GT formula rows to output/formulas without touching others."""
    out_dir = cfg.run_dir(OUT)
    n = int(n_train or cfg.n_train)
    domains = domains or DOMAINS
    rows = []

    for fold in pipeline.split.folds():
        feats = pipeline.backbone.features_for_fold(fold.index)
        for domain in domains:
            Xg, Eg, yg, _ = pipeline.shared_context(
                fold, domain, feats, want=["emotion"])
            Xv, Ev, yv = pipeline.val_data(fold, domain, feats)
            # The image-based population score is the same y_pop used by
            # Variant C in the predicted-emotion formula.
            from src.modeling.heads import make_head
            score_pop = make_head("ridge", cfg).fit(Xg, yg, val=(Xv, yv))
            scaler, w_pop, b_pop, anchor_alpha = _anchor(Eg, yg, Ev, yv, cfg)
            anchor = (scaler, w_pop, b_pop)
            alpha, _ = _select_personal_alpha(
                pipeline, fold, domain, feats, n, anchor, score_pop)

            for unit in pipeline.iter_units(fold, domain, feats, n_train=n):
                head = _PopAnchoredRidge(cfg, "C", *anchor, score_pop, kind="ridge")
                head.fit(unit.E_train, unit.y_train, X_raw=unit.X_train,
                         frozen_alpha=alpha)
                pred = head.predict(unit.E_eval, X_raw=unit.X_eval)
                delta = np.asarray(head.weights(), float).ravel()
                total = w_pop + delta
                base = {"fold": unit.fold, "domain": unit.domain,
                        "user_id": unit.user_id}
                rows.append({
                    **base, "n_train": n, "variant": "C",
                    "mediator": "gt_emotion_matched", "backbone": cfg.backbone,
                    "alpha": float(alpha), "anchor_alpha": anchor_alpha,
                    "bias": float(head._b), "b_pop": b_pop,
                    **{f"{e}_w": float(v) for e, v in zip(CORE7, delta)},
                    **{f"{e}_wpop": float(v) for e, v in zip(CORE7, w_pop)},
                    **{f"{e}_wtot": float(v) for e, v in zip(CORE7, total)},
                    **evaluate(unit.y_eval, pred),
                    "n_eval": int(len(unit.y_eval)),
                })
        print(f"  fold {fold.index} done ({len(rows)} units)", flush=True)

    result = pd.DataFrame(rows)
    path = out_dir / FILENAME
    result.to_csv(path, index=False)

    # A separate summary makes the parallel comparison inspectable without
    # changing Table 5 or its existing GT/pred source files.
    from src.experiments.stage2_emotion_importance import summarize
    pred = pd.read_csv(out_dir / "user_formula.csv")
    importance = []
    for setting, frame in (("gt_matched", result), ("pred", pred)):
        suffix = "_wtot"
        for record in frame.itertuples():
            for emotion in CORE7:
                importance.append({"setting": setting, "fold": record.fold,
                                   "domain": record.domain, "user_id": record.user_id,
                                   "emotion": emotion,
                                   "w": float(getattr(record, f"{emotion}{suffix}"))})
    comparison_dir = cfg.run_dir("stage2_emotion_importance_matched")
    per_unit = pd.DataFrame(importance)
    per_unit.to_csv(comparison_dir / "per_unit.csv", index=False)
    summarize(per_unit).to_csv(comparison_dir / "summary.csv", index=False)

    print(f"[gt matched] {len(result)} units -> {path}")
    return result
