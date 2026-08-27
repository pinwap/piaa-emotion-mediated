"""Shared helpers for reading the per-user formula export.

One place for everything the notebook and the command-line picker both need,
so the two cannot drift apart and neither has to re-implement a criterion.

The export itself comes from `main.py formula_weights`, which re-runs the
reported pipeline and writes two files:

    output/formula_weights/user_formula_weights.csv   one row per unit
    output/formula_weights/per_image_predictions.csv  one row per eval image

Under the population anchor a prediction splits into two named parts:

    y_hat(image) = y_pop(image) + sum_e z_e(image) * w_e + b

y_pop is the GIAA model on raw image features, shared by everyone in that
fold and domain. The seven w_e are the person, and because z is standardised
with the training group's scaler, two people's coefficients are in the same
units and can be put side by side.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

CORE7 = ["impressed", "intellectual", "motivated", "amused", "nostalgic",
         "sad", "distasteful"]
NICE = {e: e.capitalize() for e in CORE7}
KEY = ["fold", "domain", "user_id"]

#: the repo calls it landscape, ratings.csv calls it scenery, and the image
#: folder is scenery_image. One mapping, written down once.
DOMAIN_TO_GENRE = {"art": "art", "fashion": "fashion", "landscape": "scenery"}
DOMAIN_TO_FOLDER = {"art": "art", "fashion": "fashion",
                    "landscape": "scenery_image"}


# ---------------------------------------------------------------- loading

def load(d="output/formula_weights") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(units, images). Both indexed by fold / domain / user_id."""
    d = Path(d)
    units = pd.read_csv(d / "user_formula_weights.csv")
    images = pd.read_csv(d / "per_image_predictions.csv", low_memory=False)
    images["stimulus_id"] = images["stimulus_id"].astype(str)
    return units, images


def stimulus_index(ratings_csv="Dataset/maked/ratings.csv") -> pd.DataFrame:
    """stimulus_id -> (genre, file name), from the ratings file itself.

    The export carries stimulus ids, not file names, because the pipeline
    never needs the pixels. This is the only place the two are joined.
    """
    r = pd.read_csv(ratings_csv, usecols=["sample_id", "genre", "sample_file"])
    r = r.drop_duplicates("sample_id")
    r["stimulus_id"] = r["sample_id"].astype(str)
    return r.set_index("stimulus_id")[["genre", "sample_file"]]


def image_path(sample_dir, domain: str, stimulus_id, index: pd.DataFrame) -> Path:
    """Where the picture for one stimulus lives.

    ratings.csv records the landscape stimuli under the file name of the clip
    they were cut from, which ends in .mp4, while what was rated and what the
    features were extracted from is the still frame sitting next to it under
    the same stem. So the exact name is tried first and the .jpg stem second.
    The three domains account for 2,345 + 2,084 + 2,097 = 6,526 stimuli, which
    is the count the dataset paper reports, so the landscape domain here is
    the image set and not the video one.
    """
    row = index.loc[str(stimulus_id)]
    folder = Path(sample_dir) / DOMAIN_TO_FOLDER[domain]
    exact = folder / str(row["sample_file"])
    if exact.exists():
        return exact
    return folder / (Path(str(row["sample_file"])).stem + ".jpg")


# ------------------------------------------------------------ diagnostics

def emotion_fidelity(images: pd.DataFrame) -> pd.DataFrame:
    """Per unit, how well Stage-1's predicted emotions track the ones the user
    reported on the same images, averaged over the seven.

    A correlation rather than a mean difference: it asks whether Stage-1
    ordered the images the way the person did, which is what the personal
    head downstream actually consumes.
    """
    rows = []
    for (f, d, u), g in images.groupby(KEY, sort=False):
        cs, per = [], {}
        for e in CORE7:
            a = g[f"{e}_pred"].to_numpy(float)
            b = g[f"{e}_rated"].to_numpy(float)
            if a.std() > 0 and b.std() > 0:
                per[f"fid_{e}"] = float(np.corrcoef(a, b)[0, 1])
                cs.append(per[f"fid_{e}"])
        rows.append({"fold": f, "domain": d, "user_id": u,
                     "fidelity": float(np.mean(cs)) if cs else np.nan,
                     "n_emotions_used": len(cs), **per})
    return pd.DataFrame(rows)


def sign_split(units: pd.DataFrame, emotion: str = "sad") -> pd.DataFrame:
    """Share of people whose coefficient on one emotion is positive.

    Under the anchor, positive means the person weights that emotion more
    than the population model already does, so the split is a statement
    about disagreement, not about whether people like the emotion.
    """
    rows = []
    for dom in sorted(units.domain.unique()) + ["ALL"]:
        g = units if dom == "ALL" else units[units.domain == dom]
        c = g[f"{emotion}_w"]
        rows.append({"domain": dom, "n": len(g),
                     "pct_positive": 100 * float((c > 0).mean()),
                     "mean": float(c.mean()), "median": float(c.median()),
                     "sd": float(c.std()),
                     "min": float(c.min()), "max": float(c.max())})
    return pd.DataFrame(rows).set_index("domain")


def shared_images(images: pd.DataFrame, fold, domain, a, b) -> list[str]:
    """Evaluation images both users were actually shown.

    A figure that claims two people judged the same picture differently has
    to use a picture both of them rated. Support and evaluation sets are
    drawn per user, so this is usually a handful of images, not fifty.
    """
    sel = images[(images.fold == fold) & (images.domain == domain)]
    sa = set(sel[sel.user_id == a].stimulus_id)
    sb = set(sel[sel.user_id == b].stimulus_id)
    return sorted(sa & sb)


# -------------------------------------------------------------- selection

def contrast_pairs(units: pd.DataFrame, images: pd.DataFrame,
                   emotion: str = "sad", min_shared: int = 8,
                   require_gain: bool = True,
                   min_srocc: float | None = None) -> pd.DataFrame:
    """Every pair of users that could carry the contrast panel, ranked.

    The filters, and why each one is there:

      same fold and domain   only there do two people's coefficients share a
                             scaler and a population model, so only there is
                             putting them side by side a comparison at all
      opposite signs         the panel is about disagreement; two people who
                             both weight the emotion up illustrate nothing
      min_shared images      the panel shows one picture judged two ways, so
                             both people have to have been shown it
      require_gain           both units are ones where personalization helped.
                             An example user whose personal model was worse
                             than the population model argues against us
      min_srocc              defaults to the median unit, so the formulas on
                             display are ones that work
    """
    if min_srocc is None:
        min_srocc = float(units.srocc.median())
    ok = units[units.srocc >= min_srocc]
    if require_gain:
        ok = ok[ok.gain_srocc > 0]

    spread = images.groupby(KEY)[f"{emotion}_z"].std()
    col = f"{emotion}_w"
    rows = []
    for (f, d), g in ok.groupby(["fold", "domain"]):
        for x, y in itertools.combinations(g.itertuples(), 2):
            hi, lo = (x, y) if getattr(x, col) > getattr(y, col) else (y, x)
            if getattr(hi, col) <= 0 or getattr(lo, col) >= 0:
                continue
            sh = shared_images(images, f, d, hi.user_id, lo.user_id)
            if len(sh) < min_shared:
                continue
            # what actually happens on those shared images. A coefficient gap
            # is a promise; this is whether the promise is kept, in the
            # model's predictions and in the two people's own ratings.
            s = images[(images.fold == f) & (images.domain == d)
                       & images.stimulus_id.isin(sh)]
            pv = s.pivot_table(index="stimulus_id", columns="user_id",
                               values="y_pred")
            tv = s.pivot_table(index="stimulus_id", columns="user_id",
                               values="y_true")
            dp = (pv[hi.user_id] - pv[lo.user_id]).dropna()
            dt = (tv[hi.user_id] - tv[lo.user_id]).dropna()
            # on the shared image the two predictions differ on most, how much
            # of that difference does the emotion being illustrated account
            # for? A pair whose gap is really carried by a different emotion
            # makes a panel that says the wrong thing.
            share = np.nan
            if len(dp):
                sid = dp.idxmax()
                cs = s[s.stimulus_id == sid].set_index("user_id")
                if hi.user_id in cs.index and lo.user_id in cs.index:
                    tot = sum(abs(float(cs.loc[hi.user_id, f"{e}_c"])
                                  - float(cs.loc[lo.user_id, f"{e}_c"]))
                              for e in CORE7)
                    if tot > 0:
                        share = abs(float(cs.loc[hi.user_id, f"{emotion}_c"])
                                    - float(cs.loc[lo.user_id, f"{emotion}_c"])) / tot
            rows.append({
                "fold": f, "domain": d,
                "user_hi": hi.user_id, "user_lo": lo.user_id,
                f"{emotion}_hi": getattr(hi, col),
                f"{emotion}_lo": getattr(lo, col),
                "gap": getattr(hi, col) - getattr(lo, col),
                "shared_images": len(sh),
                "pred_gap_max": float(dp.max()) if len(dp) else np.nan,
                "true_gap_max": float(dt.max()) if len(dt) else np.nan,
                f"{emotion}_share": share,
                # one sortable number: the panel needs the two predictions to
                # separate AND the emotion on show to be what separated them.
                # Either alone picks a pair that illustrates the wrong thing.
                "panel_score": (share * float(dp.max())
                                if len(dp) and share == share else np.nan),
                "srocc_hi": hi.srocc, "srocc_lo": lo.srocc,
                "gain_hi": hi.gain_srocc, "gain_lo": lo.gain_srocc,
                "z_spread_hi": float(spread.get((f, d, hi.user_id), np.nan)),
                "z_spread_lo": float(spread.get((f, d, lo.user_id), np.nan)),
            })
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    # Ranked on what the panel has to be able to show, in order: the two
    # people really did rate a shared picture differently (true_gap_max), and
    # the emotion being illustrated is what drove it (share). The coefficient
    # gap alone is a promise; these two are whether it is kept.
    return out.sort_values("panel_score", ascending=False).reset_index(drop=True)


def misread_units(units: pd.DataFrame, fid: pd.DataFrame,
                  require_gain: bool = True) -> pd.DataFrame:
    """Units ranked by how badly Stage-1 read the images, worst first.

    require_gain keeps only units where the personal head still helped, so
    the case shown is "the bottleneck carried the wrong signal" and not
    "personalization made this person worse". The first is a limit of the
    emotion predictor, which the ground-truth-emotion row already measures.
    The second is an argument against the method.
    """
    m = units.merge(fid, on=KEY, how="left")
    if require_gain:
        m = m[m.gain_srocc > 0]
    return m.sort_values("fidelity").reset_index(drop=True)


# --------------------------------------------------- one image, in numbers

def explain_image(units: pd.DataFrame, images: pd.DataFrame,
                  fold, domain, stimulus_id, user_ids) -> pd.DataFrame:
    """The arithmetic behind one picture, for one or more people.

    Every row is a term of y_pop + z.w + b, in a fixed order and under a
    fixed label, so two people's columns line up and the table adds up in
    front of the reader. The difference between two columns is then
    attributable to named lines rather than asserted.
    """
    sid = str(stimulus_id)
    order = (["population prediction"]
             + [NICE[e] for e in CORE7]
             + ["personal terms, total", "bias", "= personal prediction",
                "what the user actually gave"])
    out = {}
    for uid in user_ids:
        r = images[(images.fold == fold) & (images.domain == domain)
                   & (images.user_id == uid) & (images.stimulus_id == sid)]
        w = units[(units.fold == fold) & (units.domain == domain)
                  & (units.user_id == uid)]
        if r.empty or w.empty:
            continue
        r, w = r.iloc[0], w.iloc[0]
        col = {"population prediction": float(r.y_pop)}
        for e in CORE7:
            col[NICE[e]] = float(r[f"{e}_c"])
        col["personal terms, total"] = float(sum(r[f"{e}_c"] for e in CORE7))
        col["bias"] = float(w.bias)
        col["= personal prediction"] = float(r.y_pred)
        col["what the user actually gave"] = float(r.y_true)
        out[f"user {uid}"] = pd.Series(col)
    t = pd.DataFrame(out).reindex(order)

    r0 = images[(images.fold == fold) & (images.domain == domain)
                & (images.stimulus_id == sid)]
    if not r0.empty:
        r0 = r0.iloc[0]
        t.attrs["predicted_emotions"] = {NICE[e]: round(float(r0[f"{e}_pred"]), 2)
                                         for e in CORE7}
    return t


def formula_table(units: pd.DataFrame, images: pd.DataFrame,
                  fold, domain, stimulus_id, user_ids) -> pd.DataFrame:
    """The seven coefficients next to the image's seven standardised
    emotions, which is where every contribution in explain_image comes from.

    z is a property of the image and is shared; w is the person.
    """
    sid = str(stimulus_id)
    r0 = images[(images.fold == fold) & (images.domain == domain)
                & (images.stimulus_id == sid)]
    if r0.empty:
        return pd.DataFrame()
    r0 = r0.iloc[0]
    cols = {"predicted (1-7)": [float(r0[f"{e}_pred"]) for e in CORE7],
            "z (standardised)": [float(r0[f"{e}_z"]) for e in CORE7]}
    for uid in user_ids:
        w = units[(units.fold == fold) & (units.domain == domain)
                  & (units.user_id == uid)]
        if not w.empty:
            cols[f"w, user {uid}"] = [float(w.iloc[0][f"{e}_w"]) for e in CORE7]
    return pd.DataFrame(cols, index=[NICE[e] for e in CORE7])


def rated_emotions(images: pd.DataFrame, fold, domain, stimulus_id,
                   user_ids) -> pd.DataFrame:
    """What each of these people reported feeling about that image, next to
    what Stage-1 predicted for it."""
    sid = str(stimulus_id)
    sel = images[(images.fold == fold) & (images.domain == domain)
                 & (images.stimulus_id == sid)]
    if sel.empty:
        return pd.DataFrame()
    out = {"Stage-1 predicted": {NICE[e]: float(sel.iloc[0][f"{e}_pred"])
                                 for e in CORE7}}
    for uid in user_ids:
        r = sel[sel.user_id == uid]
        if not r.empty:
            out[f"user {uid} reported"] = {NICE[e]: float(r.iloc[0][f"{e}_rated"])
                                           for e in CORE7}
    return pd.DataFrame(out)