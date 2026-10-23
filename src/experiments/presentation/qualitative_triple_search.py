"""Rank candidate triples for the qualitative personalization figure: two
users who rate the same image, fold and domain but sit at opposite ends of
the Sad-weight (sad_wtot) distribution, plus a third user (same image,
fold, domain) whose Stage-1 prediction diverged sharply from their own
rating on that image.

The script returns a deterministic ranked table; the reported example is its
top-ranked row.

Ranking rule:
  1. Keep sad_wtot-opposite-sign pairs (one positive, one negative) that
     rated at least one image in common, restricted to pairs whose
     Sad-weight gap is at least SAD_GAP_MIN -- "clearly apart" rather than
     a near-zero contrast.
  2. For each shared image, score the pair by how well BOTH of them are
     predicted on that specific image: pair_err = the larger of the two
     |y_true - y_pred| errors. Small pair_err means the figure's headline
     claim -- "same ê, different formula, both still land close to the
     truth" -- is visibly true on the one image being shown, not just true
     on average across a user's whole history (which is what SROCC would
     say). Keep only the single best (lowest pair_err) shared image per
     pair.
  3. On that image, look at every OTHER user who also rated it and pick
     the one with the largest overall rating error |y_true - y_pred| --
     the sharpest failure case available on this exact image -- breaking
     ties by the largest single-emotion Stage-1-predicted-vs-self-reported
     gap (which is what drives the causal explanation in the caption).
  4. Sort all pairs by (pair_err ascending, misread's rating error
     descending): the top of the table is a pair that is both sharply
     contrasting AND both well-predicted, paired with the worst-available
     failure case on the same image.

Run: uv run python -m src.experiments.qualitative_triple_search
"""
from pathlib import Path
from src.config import PROJECT_ROOT
import pandas as pd

CORE7 = ["impressed", "intellectual", "motivated", "amused",
         "nostalgic", "sad", "distasteful"]

PER_IMAGE = PROJECT_ROOT / "output" / "formulas" / "per_image.csv"
USER_FORMULA = PROJECT_ROOT / "output" / "formulas" / "user_formula.csv"
FINAL_IMAGES = PROJECT_ROOT / "output" / "formula_images_final.csv"
FINAL_UNITS = PROJECT_ROOT / "output" / "formula_units_final.csv"

SAD_GAP_MIN = 0.30   # "clearly apart" cutoff on sad_wtot
TOP_N = 10

PRED_COLS = [f"{e}_pred" for e in CORE7]
RATED_COLS = [f"{e}_rated" for e in CORE7]


def load():
    """Load compact formula databases, with a run-directory fallback."""
    if FINAL_IMAGES.exists() and FINAL_UNITS.exists():
        pi = pd.read_csv(FINAL_IMAGES, low_memory=False)
        uf = pd.read_csv(FINAL_UNITS, low_memory=False)
        if "setting" not in pi or "setting" not in uf:
            raise ValueError("consolidated formula databases require a setting column")
        pi = pi[pi.setting == "anchored"].copy()
        uf = uf[uf.setting == "anchored"].copy()
    else:
        pi = pd.read_csv(PER_IMAGE)
        uf = pd.read_csv(USER_FORMULA)
    if len(pi) != 19_350 or len(uf) != 387:
        raise ValueError(
            "qualitative search expects 387 anchored formula units and "
            f"19,350 evaluation-image rows, got {len(uf)} and {len(pi)}"
        )
    return pi, uf


def _user_weights(uf, fold, domain, user_id):
    """7-dim w_total vector for one user, as a {emotion: value} dict."""
    row = uf[(uf.fold == fold) & (uf.domain == domain)
             & (uf.user_id == user_id)].iloc[0]
    return {e: float(getattr(row, f"{e}_wtot")) for e in CORE7}


def best_misread_on_image(grp, fold, domain, stimulus_id, exclude_ids):
    """Among users other than the pair who rated this exact image, the one
    with the largest overall rating error |y_true - y_pred|, tie-broken by
    the largest single-emotion predicted-vs-reported gap."""
    cand = grp[(grp.stimulus_id == stimulus_id)
               & (~grp.user_id.isin(exclude_ids))].copy()
    if cand.empty:
        return None
    cand["rating_err"] = (cand["y_true"] - cand["y_pred"]).abs()
    gap = cand[PRED_COLS].values - cand[RATED_COLS].values
    cand["max_gap"] = gap.max(axis=1)
    cand = cand.sort_values(["rating_err", "max_gap"], ascending=[False, False])
    return cand.iloc[0]


def build_table(pi, uf, top_n=TOP_N):
    pos = uf[uf.sad_wtot > 0][["fold", "domain", "user_id", "sad_wtot"]]
    neg = uf[uf.sad_wtot < 0][["fold", "domain", "user_id", "sad_wtot"]]

    rows = []
    for (fold, domain), grp in pi.groupby(["fold", "domain"]):
        p = pos[(pos.fold == fold) & (pos.domain == domain)]
        n = neg[(neg.fold == fold) & (neg.domain == domain)]
        if p.empty or n.empty:
            continue
        imgs = grp.groupby("user_id")["stimulus_id"].apply(set)
        y_lookup = grp.set_index(["user_id", "stimulus_id"])[["y_true", "y_pred"]]

        for _, h in p.iterrows():
            for _, l in n.iterrows():
                gap = h.sad_wtot - l.sad_wtot
                if gap < SAD_GAP_MIN:
                    continue
                shared = imgs.get(h.user_id, set()) & imgs.get(l.user_id, set())
                if not shared:
                    continue

                best_row = None
                for img in shared:
                    hi_y, hi_yp = y_lookup.loc[(h.user_id, img)]
                    lo_y, lo_yp = y_lookup.loc[(l.user_id, img)]
                    pair_err = max(abs(hi_y - hi_yp), abs(lo_y - lo_yp))

                    m = best_misread_on_image(grp, fold, domain, img,
                                               exclude_ids=[h.user_id, l.user_id])
                    if m is None:
                        continue
                    cand = dict(
                        fold=fold, domain=domain,
                        hi=int(h.user_id), hi_sad=round(float(h.sad_wtot), 3),
                        lo=int(l.user_id), lo_sad=round(float(l.sad_wtot), 3),
                        gap=round(float(gap), 3),
                        image=int(img),
                        hi_y=float(hi_y), hi_ypred=round(float(hi_yp), 2),
                        lo_y=float(lo_y), lo_ypred=round(float(lo_yp), 2),
                        pair_err=round(float(pair_err), 3),
                        mis=int(m.user_id),
                        mis_y=float(m.y_true), mis_ypred=round(float(m.y_pred), 2),
                        mis_err=round(float(m.rating_err), 3),
                        mis_gap=round(float(m.max_gap), 3),
                    )
                    # keep only the single best (lowest pair_err, then
                    # largest mis_err) shared image per pair
                    if (best_row is None
                            or (cand["pair_err"], -cand["mis_err"])
                               < (best_row["pair_err"], -best_row["mis_err"])):
                        best_row = cand
                if best_row is not None:
                    rows.append(best_row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["pair_err", "mis_err"], ascending=[True, False]).head(top_n)
    df = df.reset_index(drop=True)

    for prefix, id_col in [("hi", "hi"), ("lo", "lo"), ("mis", "mis")]:
        for e in CORE7:
            df[f"{prefix}_w_{e}"] = [
                round(_user_weights(uf, r.fold, r.domain, int(getattr(r, id_col)))[e], 3)
                for r in df.itertuples()
            ]
    return df

def main() -> int:
    pi, uf = load()
    table = build_table(pi, uf)
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 100)
    print(f"{len(table)} ranked qualitative candidates "
          f"(minimum Sad-weight gap {SAD_GAP_MIN})")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
