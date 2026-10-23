"""How well does the shared Stage-1 model predict each of the 7 emotions?

Fit the shared mediator on train users, then predict emotions for images
rated by test users (image-level, population mean per image), compare to
the true population mean per emotion.

Output: output/stage1_emotion_acc/per_fold_domain.csv, summary.csv
"""
from __future__ import annotations

import pandas as pd

from src.data.data import CORE7, DOMAINS
from src.utils.metrics import plcc, sem, srocc


def run(cfg, pipeline):
    out_dir = cfg.run_dir("stage1_emotion_acc")
    rows = []
    for fold in pipeline.split.folds():
        feats = pipeline.backbone.features_for_fold(fold.index)
        for dom in DOMAINS:
            _, _, _, meds = pipeline.shared_context(fold, dom, feats)
            emotion = meds["emotion"]

            test_df = pipeline.ds.subset(domain=dom, users=fold.test_users)
            test_df = test_df[test_df["stimulus_id"].astype(str).isin(feats)]
            gt = pipeline.ds.per_stimulus(test_df)
            if len(gt) < 10:
                continue
            X = pipeline.backbone.matrix(feats, gt.index)
            P = emotion.transform(X)
            G = gt[CORE7].to_numpy(float)
            for j, e in enumerate(CORE7):
                rows.append(dict(fold=fold.index, domain=dom, emotion=e,
                                 srocc=srocc(G[:, j], P[:, j]),
                                 plcc=plcc(G[:, j], P[:, j])))
        print(f"  fold {fold.index} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_fold_domain.csv", index=False)

    lines = []
    for e in CORE7:
        d = df[df.emotion == e]
        row = dict(emotion=e)
        for dom in DOMAINS:
            dd = d[d.domain == dom]
            row[f"{dom}_srocc_mean"] = dd.srocc.mean()
            row[f"{dom}_srocc_sd"] = dd.srocc.std()
            row[f"{dom}_srocc_sem"] = sem(dd.srocc)
            row[f"{dom}_plcc_mean"] = dd.plcc.mean()
            row[f"{dom}_plcc_sd"] = dd.plcc.std()
            row[f"{dom}_plcc_sem"] = sem(dd.plcc)
        row["avg_srocc_mean"] = d.srocc.mean()
        row["avg_srocc_sd"] = d.srocc.std()
        row["avg_srocc_sem"] = sem(d.srocc)
        row["avg_plcc_mean"] = d.plcc.mean()
        row["avg_plcc_sd"] = d.plcc.std()
        row["avg_plcc_sem"] = sem(d.plcc)
        lines.append(row)

    summary = pd.DataFrame(lines)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summary
