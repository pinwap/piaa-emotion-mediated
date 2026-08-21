r"""The two reference ceilings at the bottom of Table 1.

Both are excluded from every fairness comparison in the paper -- they are
references, not conditions -- so neither goes through the validation-group
selection protocol that the compared rows do.

  GT emotions   replaces Stage 1's predictions with the user's own emotion
                ratings and fits the usual personal ridge on top, which
                isolates what is left to gain if Stage 1 were perfect. Its
                penalty is chosen per user on that user's own support set,
                the way a ceiling should be: it is meant to be generous.
  Test-retest   the correlation between a user's two scores for the same
                image in separate sessions -- the noise floor of the target
                itself, not of any model. Pooled over pairs, so there is no
                across-unit spread to report.

Writes output/upper_bounds.csv, which make_tab_main.py reads.

Usage:  python -m src.experiments.make_upper_bounds
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.data import DOMAINS                        # noqa: E402
from src.modeling.heads import make_head                 # noqa: E402
from src.utils.metrics import plcc, sem, srocc           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BACKBONE, N_TRAIN = "qwen8b", 100


def run(cfg, pipe, ds) -> pd.DataFrame:
    rows = []
    for fold in pipe.split.folds():
        feats = pipe.backbone.features_for_fold(fold.index)
        for dom in DOMAINS:
            for unit in pipe.iter_units(fold, dom, feats, n_train=N_TRAIN):
                h = make_head("ridge", cfg).fit(unit.E_train, unit.y_train)
                p = h.predict(unit.E_eval)
                rows.append(dict(domain=dom, srocc=srocc(unit.y_eval, p),
                                 plcc=plcc(unit.y_eval, p),
                                 eff_dof=h.effective_dof()))
        print(f"  fold {fold.index} done ({len(rows)} units)", flush=True)
    gt = pd.DataFrame(rows)
    print(f"[gt_emotion] {len(gt)} units")

    pairs = ds.retest_pairs(cfg.data_dir)

    out = []
    r = {"label": "GT emotions", "stage2": "Ridge",
         "eff_dof": f"{gt.eff_dof.mean():.1f}"}
    for key, d in [(dom, gt[gt.domain == dom]) for dom in DOMAINS] + [("avg", gt)]:
        for m in ("srocc", "plcc"):
            r[f"{key}_{m}_mean"] = float(d[m].mean())
            r[f"{key}_{m}_sem"] = float(sem(d[m]))
    out.append(r)

    r = {"label": "Test--retest reliability", "stage2": "---", "eff_dof": "---"}
    for key, d in ([(dom, pairs[pairs["domain"] == dom]) for dom in DOMAINS]
                   + [("avg", pairs)]):
        y1 = d["overall_r1"].to_numpy(float)
        y2 = d["overall_r2"].to_numpy(float)
        r[f"{key}_srocc_mean"], r[f"{key}_plcc_mean"] = srocc(y1, y2), plcc(y1, y2)
        # one correlation over pooled pairs, not an average of per-unit ones,
        # so there is no across-unit spread to put a +- on
        r[f"{key}_srocc_sem"] = r[f"{key}_plcc_sem"] = np.nan
    out.append(r)

    df = pd.DataFrame(out)
    dest = ROOT / "output" / "upper_bounds.csv"
    df.to_csv(dest, index=False)
    print(df[["label", "avg_srocc_mean", "avg_plcc_mean"]].to_string(index=False))
    print(f"wrote {dest}")
    return df


if __name__ == "__main__":
    from main import build
    from src.config import Config
    cfg = Config()
    ds, bb, sp, pipe = build(cfg, BACKBONE)
    raise SystemExit(0 if run(cfg, pipe, ds) is not None else 1)
