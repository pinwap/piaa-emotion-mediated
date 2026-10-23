"""Summarize emotion-weight magnitude and sign across fitted formulas.

``gt`` uses heads fitted to reported emotions. ``pred`` uses the anchored
model's total emotion-space weight ``w_pop + delta``. The residual coefficient
``delta`` is retained as a separate diagnostic and is not used for the
reported absolute-formula interpretation.

Output: output/stage2_emotion_importance/per_unit.csv, summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data import CORE7, DOMAINS
from src.utils.metrics import participant_cluster_bootstrap, sem

# Treat numerical values below this tolerance as zero when reporting sign.
# No released coefficient magnitude lies between 1e-9 and 1e-3.
SIGN_TOL = 1e-9

SETTINGS = [
    ("gt", "user_formula_gt.csv", "_w",
     "fitted on the emotions the user reported"),
    ("pred", "user_formula.csv", "_wtot",
     "fitted on Stage-1's predicted emotions, absolute formula w_pop + delta"),
    ("delta", "user_formula.csv", "_w",
     "personal residual coefficient under the population anchor"),
]


def run(cfg, pipeline=None):
    out_dir = cfg.run_dir("stage2_emotion_importance")
    d = Path(cfg.output_dir) / "formulas"
    if not (d / "user_formula.csv").exists():
        raise FileNotFoundError(
            f"{d} has no user_formula.csv. Run `main.py formulas` first; "
            "this analysis reads the stored fits without refitting.")

    cache = {}
    rows = []
    for name, fname, suffix, _note in SETTINGS:
        if fname not in cache:
            cache[fname] = pd.read_csv(d / fname)
        f = cache[fname]
        for r in f.itertuples():
            for e in CORE7:
                rows.append({"setting": name, "fold": r.fold,
                             "domain": r.domain, "user_id": r.user_id,
                             "emotion": e,
                             "w": float(getattr(r, f"{e}{suffix}"))})
    per_unit = pd.DataFrame(rows)
    per_unit.to_csv(out_dir / "per_unit.csv", index=False)

    summary = summarize(per_unit)
    summary.to_csv(out_dir / "summary.csv", index=False)

    n_units = len(per_unit) // (len(CORE7) * len(SETTINGS))
    print(f"[emotion importance] wrote {len(per_unit):,} rows "
          f"({n_units} units per setting) to {out_dir}")
    return per_unit, summary


def summarize(per_unit: pd.DataFrame) -> pd.DataFrame:
    """Summarize magnitude with ``abs(w)`` and sign with the original ``w``."""
    rows = []
    for (setting, emotion), g in per_unit.groupby(["setting", "emotion"]):
        r = {"setting": setting, "emotion": emotion}
        for dom in DOMAINS:
            gd = g[g.domain == dom]["w"]
            absd = gd.abs()
            r[f"{dom}_mean"] = float(absd.mean())
            r[f"{dom}_sd"] = float(absd.std(ddof=1))
            r[f"{dom}_sem"] = float(sem(absd))
            r[f"{dom}_pct_pos"] = 100 * float((gd > SIGN_TOL).mean())
        allw = g["w"]
        absw = allw.abs()
        r["avg_mean"] = float(absw.mean())
        r["avg_sd"] = float(absw.std(ddof=1))
        r["avg_sem"] = participant_cluster_bootstrap(
            absw, g["user_id"])["sem"]
        # Report the unweighted mean of the three equal-sized domain shares.
        r["avg_pct_pos"] = float(np.mean(
            [r[f"{dom}_pct_pos"] for dom in DOMAINS]))
        r["pooled_pct_pos"] = 100 * float((allw > SIGN_TOL).mean())
        r["n"] = int(len(allw))
        rows.append(r)
    out = pd.DataFrame(rows)
    out["rank"] = (out.groupby("setting")["avg_mean"]
                      .rank(ascending=False, method="min").astype(int))
    return out.sort_values(["setting", "rank"]).reset_index(drop=True)
