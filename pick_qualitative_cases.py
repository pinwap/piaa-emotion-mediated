"""Choose the units the qualitative figure shows, from the formula_weights export.

Two things come out of this:

  1. A contrast pair. Two users in the *same fold and the same domain* whose
     sad coefficient is furthest apart. Same fold and same domain is not a
     convenience: under Anchor C the seven coefficients live in the training
     group's scaler and sit on top of that fold's GIAA model, so only inside
     one (fold, domain) do two users' weights mean the same thing. Comparing
     across folds would be comparing numbers in different units.

     Both users are additionally required to be predicted at least as well as
     the median unit, and to see real variation in predicted sadness across
     their images. A formula that does not work, or a sad axis that never
     moves, illustrates nothing.

  2. Three failure cases, deliberately of three different kinds, so the panel
     says something more useful than "sometimes it is wrong":

       stage1_misread   Stage-1's predicted emotions do not track the
                        emotions the user actually reported for those images.
                        The bottleneck is carrying the wrong signal, so
                        whatever the personal head does with it cannot help.

       personal_hurt    The population model did fine on this unit and the
                        personal correction made it worse. This is the honest
                        cost of personalizing on 100 ratings.

       seven_not_enough Stage-1 read the images correctly and the user's
                        scores do vary, and accuracy is still poor. Nothing
                        is broken; seven emotions simply do not span what
                        this person responds to.

Usage, from the repo root after running the export:
    uv run python pick_qualitative_cases.py
    uv run python pick_qualitative_cases.py --emotion amused --top-images 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CORE7 = ["impressed", "intellectual", "motivated", "amused", "nostalgic",
         "sad", "distasteful"]
KEY = ["fold", "domain", "user_id"]


def emotion_fidelity(images: pd.DataFrame) -> pd.DataFrame:
    """Per unit: how well Stage-1's predicted emotions track the ones the user
    actually reported on the same images, averaged over the seven."""
    rows = []
    for key, g in images.groupby(KEY, sort=False):
        cors = []
        for e in CORE7:
            a, b = g[f"{e}_pred"].to_numpy(float), g[f"{e}_rated"].to_numpy(float)
            if a.std() > 0 and b.std() > 0:
                cors.append(float(np.corrcoef(a, b)[0, 1]))
        rows.append({**dict(zip(KEY, key)),
                     "emotion_fidelity": float(np.mean(cors)) if cors else np.nan,
                     "n_emotions_used": len(cors),
                     "pred_spread": float(g[[f"{e}_pred" for e in CORE7]]
                                          .std().mean()),
                     "personal_swing": float((g["y_pred"] - g["y_pop"]).std())})
    return pd.DataFrame(rows)


def pick_pair(units: pd.DataFrame, images: pd.DataFrame, emotion: str):
    col = f"{emotion}_wtot"
    med_srocc = units["srocc"].median()
    ok = units[units["srocc"] >= med_srocc].copy()

    # the emotion has to actually vary across the images, or its coefficient
    # cannot change any prediction no matter how large it is
    spread = (images.groupby(KEY)[f"{emotion}_z"].std()
              .rename("z_spread").reset_index())
    ok = ok.merge(spread, on=KEY, how="left")
    ok = ok[ok["z_spread"] >= ok["z_spread"].median()]

    best = None
    for (fold, dom), g in ok.groupby(["fold", "domain"]):
        if len(g) < 2:
            continue
        hi = g.loc[g[col].idxmax()]
        lo = g.loc[g[col].idxmin()]
        gap = float(hi[col] - lo[col])
        if best is None or gap > best[0]:
            best = (gap, fold, dom, hi, lo)
    if best is None:
        raise SystemExit("no (fold, domain) had two units passing the filters")
    return best


def shared_panel(images: pd.DataFrame, hi, lo, emotion: str, top_n: int):
    """Both formulas applied to the same images.

    Inside one (fold, domain) every unit's `_z` columns come from the same
    training-group scaler and every `y_pop` from the same GIAA model, so one
    user's formula can be evaluated on the other user's images by arithmetic
    alone. No refitting, and nothing is being extrapolated.
    """
    fold, dom = int(hi["fold"]), hi["domain"]
    pool = images[(images.fold == fold) & (images.domain == dom)].copy()
    pool = pool.drop_duplicates(subset="stimulus_id")

    Z = pool[[f"{e}_z" for e in CORE7]].to_numpy(float)
    out = pool[["stimulus_id", "y_pop"]].copy()
    for tag, u in (("hi", hi), ("lo", lo)):
        w = np.array([u[f"{e}_w"] for e in CORE7], float)
        out[f"y_{tag}"] = pool["y_pop"].to_numpy(float) + Z @ w + float(u["bias"])
        out[f"{emotion}_c_{tag}"] = pool[f"{emotion}_z"].to_numpy(float) * u[f"{emotion}_w"]
    out[f"{emotion}_z"] = pool[f"{emotion}_z"].to_numpy(float)
    out["disagreement"] = out["y_hi"] - out["y_lo"]

    # the user's own score, where that user happened to be shown the image
    for tag, u in (("hi", hi), ("lo", lo)):
        own = images[(images.fold == fold) & (images.domain == dom)
                     & (images.user_id == int(u["user_id"]))]
        out = out.merge(own[["stimulus_id", "y_true"]]
                        .rename(columns={"y_true": f"y_true_{tag}"}),
                        on="stimulus_id", how="left")

    out = out.reindex(out["disagreement"].abs().sort_values(ascending=False).index)
    return out.head(top_n).reset_index(drop=True), out


def pick_failures(units: pd.DataFrame, fid: pd.DataFrame) -> pd.DataFrame:
    u = units.merge(fid, on=KEY, how="left")
    med_pop = u["pop_srocc"].median()
    med_fid = u["emotion_fidelity"].median()
    med_ysd = u["y_sd"].median()

    picks = []

    a = u.dropna(subset=["emotion_fidelity"]).nsmallest(1, "emotion_fidelity")
    if len(a):
        picks.append(("stage1_misread", a.iloc[0]))

    b = u[u["pop_srocc"] >= med_pop].nsmallest(1, "gain_srocc")
    if len(b):
        picks.append(("personal_hurt", b.iloc[0]))

    c = u[(u["emotion_fidelity"] >= med_fid) & (u["y_sd"] >= med_ysd)]
    c = c[~c.index.isin([r.name for _, r in picks])].nsmallest(1, "srocc")
    if len(c):
        picks.append(("seven_not_enough", c.iloc[0]))

    keep = KEY + ["srocc", "pop_srocc", "gain_srocc", "emotion_fidelity",
                  "personal_swing", "y_sd", "alpha"] + [f"{e}_wtot" for e in CORE7]
    out = pd.DataFrame([{"case": name, **{k: r[k] for k in keep}}
                        for name, r in picks])

    # each pick is the most extreme unit of its kind, which is not the same
    # as being a failure. Say so rather than letting the label imply it.
    notes = []
    for _, r in out.iterrows():
        if r["case"] == "personal_hurt" and r["gain_srocc"] >= 0:
            notes.append("NOT a failure: personalization still helped this "
                         "unit, it is only the smallest gain in the sample")
        elif r["case"] == "stage1_misread" and r["emotion_fidelity"] >= 0.5:
            notes.append("weak case: Stage-1 still tracked the reported "
                         "emotions reasonably well here")
        elif r["case"] == "seven_not_enough" and r["srocc"] >= 0.3:
            notes.append("weak case: accuracy on this unit is not actually poor")
        else:
            notes.append("")
    out["caveat"] = notes
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/formulas")
    ap.add_argument("--emotion", default="sad", choices=CORE7)
    ap.add_argument("--top-images", type=int, default=8)
    args = ap.parse_args()

    d = Path(args.dir)
    units = pd.read_csv(d / "user_formula.csv")
    images = pd.read_csv(d / "per_image.csv")
    print(f"{len(units)} units, {len(images)} image rows\n")

    fid = emotion_fidelity(images)
    gap, fold, dom, hi, lo = pick_pair(units, images, args.emotion)

    print(f"=== contrast pair on '{args.emotion}' "
          f"(fold {fold}, {dom}, gap {gap:+.3f}) ===")
    pair = pd.DataFrame([hi, lo], index=["higher", "lower"])
    cols = ["user_id", "srocc", "pop_srocc", "gain_srocc", "bias", "alpha"] \
        + [f"{e}_wtot" for e in CORE7]
    print(pair[cols].round(4).to_string())

    panel, full = shared_panel(images, hi, lo, args.emotion, args.top_images)
    print(f"\n=== the {args.top_images} images the two formulas disagree on most ===")
    show = ["stimulus_id", f"{args.emotion}_z", "y_pop", "y_hi", "y_lo",
            "disagreement", "y_true_hi", "y_true_lo"]
    print(panel[show].round(3).to_string(index=False))

    fails = pick_failures(units, fid)
    print("\n=== failure cases ===")
    print(fails.round(4).to_string(index=False))

    pair.assign(role=["higher", "lower"], emotion=args.emotion,
                gap=gap).to_csv(d / "qualitative_pair.csv", index=False)
    full.to_csv(d / "qualitative_pair_images.csv", index=False)
    fails.to_csv(d / "qualitative_failures.csv", index=False)
    print(f"\nwritten to {d}: qualitative_pair.csv, "
          f"qualitative_pair_images.csv, qualitative_failures.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
