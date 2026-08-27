"""Do the coefficients reflect real emotion importance, or just how easy each
emotion is to predict?

Reads the formulas `main.py formulas` wrote and compares two of them per unit:

  gt        the head fitted on the emotions the user actually reported
  pred      the head fitted on the emotions Stage-1 predicted

If an emotion carries a low weight in the predicted setting only because it is
hard to predict, rather than because it does not matter to that person, the
ground-truth weight should be higher.

Which weight vector to read, and why it matters here more than anywhere
----------------------------------------------------------------------
Under the population anchor the personal head returns `delta`, a departure
from the population formula, fitted on a residual and therefore shrunk toward
zero and near-symmetric about it. Its sign is close to a coin flip for every
emotion, so `pct_positive` computed on delta says nothing about people. This
table reports `w_total = w_pop + delta`, the absolute personal formula, which
is the vector whose sign means "this person weights this emotion up", and the
one that is comparable with the ground-truth formula.

Both are written out, so the difference can be seen rather than taken on
trust: the `delta` block is there precisely to show that its split is an
artefact of the anchor.

Output: output/stage2_emotion_importance/per_unit.csv, summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data import CORE7, DOMAINS
from src.utils.metrics import sem

#: A coefficient this small is zero for our purposes, and the sign of a zero is
#: not a fact about anyone's taste. Ridge shrinks weak emotions all the way
#: down, and whether such a weight lands at +5e-31, -3e-31 or 0.0 depends on
#: the platform's last bit, which moved the reported percentage by a unit
#: between machines. Nothing sits between 1e-9 and 1e-3, so this threshold
#: touches no coefficient that means anything.
SIGN_TOL = 1e-9

SETTINGS = [
    ("gt", "user_formula_gt.csv", "_w",
     "fitted on the emotions the user reported"),
    ("pred", "user_formula.csv", "_wtot",
     "fitted on Stage-1's predicted emotions, absolute formula w_pop + delta"),
    ("delta", "user_formula.csv", "_w",
     "the anchor's residual fit; its sign split is an artefact, kept for "
     "comparison only"),
]


def run(cfg, pipeline=None):
    out_dir = cfg.run_dir("stage2_emotion_importance")
    d = Path(cfg.output_dir) / "formulas"
    if not (d / "user_formula.csv").exists():
        raise FileNotFoundError(
            f"{d} has no user_formula.csv. Run `main.py formulas` first: this "
            f"table reads the fitted formulas rather than refitting, so that "
            f"what it reports is what the tables report.")

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

    print(f"[emotion importance] {per_unit.user_id.nunique()} users, "
          f"{len(per_unit) // (len(CORE7) * len(SETTINGS))} units per setting\n")
    for name, _f, _s, note in SETTINGS:
        print(f"  {name:6s} {note}")
    print()
    show = summary.pivot(index="emotion", columns="setting",
                         values="avg_pct_pos").round(1)
    show = show.reindex([e for e in CORE7])
    print("share of units with a positive coefficient (%)")
    print(show.to_string())
    print("\nthe delta column sits near 50 for every emotion because a residual "
          "fit is centred on zero.\ngt and pred are the columns that say "
          "whether people agree about an emotion.")
    return per_unit, summary


def summarize(per_unit: pd.DataFrame) -> pd.DataFrame:
    """mean/sd/sem are computed on |w| (magnitude), never on the signed w:
    a concept most people weight strongly in ONE direction (e.g. Distasteful,
    negative for ~84% of users) must still come out as high-importance, not
    cancel toward zero. pct_pos is the separate, signed diagnostic that says
    which direction people lean; it is computed on the raw signed w, not on
    the magnitude, so it still means "share of units with a positive
    coefficient" rather than always reading ~100%.
    """
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
        r["avg_sem"] = float(sem(absw))
        # the average of the three domain shares, not the pooled share. The
        # domains are 129 units each so the two agree to a rounding place, but
        # which one is reported has to be stated rather than inferred.
        r["avg_pct_pos"] = float(np.mean(
            [r[f"{dom}_pct_pos"] for dom in DOMAINS]))
        r["pooled_pct_pos"] = 100 * float((allw > SIGN_TOL).mean())
        r["n"] = int(len(allw))
        rows.append(r)
    out = pd.DataFrame(rows)
    out["rank"] = (out.groupby("setting")["avg_mean"]
                      .rank(ascending=False, method="min").astype(int))
    return out.sort_values(["setting", "rank"]).reset_index(drop=True)