"""Analyses added to the supplementary material (S6-S10).

Every number is recomputed from the frozen release CSVs; nothing is refitted.

Inputs (relative to --output-dir, default `output/`):
    raw_all_final.csv          seed-level model results
    formula_units_final.csv    per-unit fitted formulas
    formula_images_final.csv   per-unit, per-image records
    Dataset/maked/ratings.csv  licensed ratings (only for S6, test-retest)

Usage:
    python supp_v2_analyses.py --output-dir output --ratings Dataset/maked/ratings.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, spearmanr

CORE7 = ["impressed", "intellectual", "motivated", "amused",
         "nostalgic", "sad", "distasteful"]
RATED = {"impressed": "Impressed", "intellectual": "Intellectually",
         "motivated": "Motivated", "amused": "Amused",
         "nostalgic": "Nostalgic", "sad": "Sad", "distasteful": "Distasteful"}
DOMAINS = ["art", "fashion", "landscape"]
KEY = ["fold", "domain", "user_id"]
BACKBONES = ["clip", "clip_ft", "clip_ft_emo", "qwen4b", "qwen8b"]


# ----------------------------------------------------------------- helpers

def srocc(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def cluster_boot(values, ids, n_boot=10_000, seed=0, null=0.0,
                 alternative="two-sided"):
    """Participant-cluster bootstrap. Non-finite rows are dropped, not imputed."""
    v = np.asarray(values, float).copy()
    ids = np.asarray(ids)
    ok = np.isfinite(v)
    v, ids = v[ok], ids[ok]
    uq, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    cm = np.bincount(inv, weights=v) / cnt
    rng = np.random.default_rng(seed)
    boot = cm[rng.integers(0, len(cm), size=(n_boot, len(cm)))].mean(axis=1)
    if alternative == "two-sided":
        p = 2.0 * min(np.mean(boot <= null), np.mean(boot >= null))
    elif alternative == "greater":
        p = float(np.mean(boot <= null))
    else:
        p = float(np.mean(boot >= null))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(cm.mean()), sem=float(boot.std(ddof=1)),
                ci_lo=float(lo), ci_hi=float(hi), p=float(min(1.0, p)),
                n_participants=int(len(cm)), n_dropped=int((~ok).sum()))


def tost(a, b, delta, alpha=0.05):
    """Paired two one-sided tests for equivalence within +/- delta."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    df = n - 1
    p_lo = stats.t.sf((d.mean() + delta) / se, df)
    p_hi = stats.t.cdf((d.mean() - delta) / se, df)
    t_crit = stats.t.ppf(1 - alpha, df)
    return dict(n=n, mean=float(d.mean()), p_tost=float(max(p_lo, p_hi)),
                ci_lo=float(d.mean() - t_crit * se),
                ci_hi=float(d.mean() + t_crit * se),
                equivalent=bool(max(p_lo, p_hi) < alpha))


def load(output_dir: Path):
    raw = pd.read_csv(output_dir / "raw_all_final.csv", low_memory=False)
    units = pd.read_csv(output_dir / "formula_units_final.csv", low_memory=False)
    images = pd.read_csv(output_dir / "formula_images_final.csv", low_memory=False)
    return raw, units, images


# ------------------------------------------------- S6  target reliability

def s6_test_retest(ratings_path: Path):
    cols = list(RATED.values())
    df = pd.read_csv(ratings_path).dropna(subset=cols + ["Aesthetic"])
    df = df.reset_index(drop=True)
    df["domain"] = df["genre"].map({"art": "art", "fashion": "fashion",
                                    "landscape": "landscape",
                                    "scenery": "landscape"})
    df["occ"] = df.groupby(["user_id", "sample_id"]).cumcount()
    m = (df[df.occ == 0]
         .merge(df[df.occ == 1], on=["user_id", "sample_id", "domain"],
                suffixes=("_r1", "_r2")))
    out = {"n_pairs": int(len(m)), "pooled": {}, "within_unit": {}}
    for dom in DOMAINS + ["all"]:
        d = m if dom == "all" else m[m.domain == dom]
        out["pooled"][dom] = dict(
            n=int(len(d)),
            srocc=srocc(d.Aesthetic_r1, d.Aesthetic_r2),
            plcc=float(np.corrcoef(d.Aesthetic_r1, d.Aesthetic_r2)[0, 1]))
    rows = []
    for (u, dom), d in m.groupby(["user_id", "domain"]):
        a = d.Aesthetic_r1.to_numpy(float)
        b = d.Aesthetic_r2.to_numpy(float)
        rows.append(dict(user_id=u, domain=dom, n=len(d),
                         srocc=srocc(a, b),
                         plcc=(np.nan if a.std() == 0 or b.std() == 0
                               else float(np.corrcoef(a, b)[0, 1]))))
    r = pd.DataFrame(rows)
    out["items_per_unit"] = dict(median=int(r.n.median()),
                                 min=int(r.n.min()), max=int(r.n.max()))
    out["n_units"] = int(len(r))
    for dom in DOMAINS:
        d = r[r.domain == dom]
        out["within_unit"][dom] = dict(srocc=float(d.srocc.mean()),
                                       plcc=float(d.plcc.mean()),
                                       sd=float(d.srocc.std()))
    boot = cluster_boot(r.srocc, r.user_id)
    out["within_unit"]["all"] = dict(srocc=boot["mean"], ci=[boot["ci_lo"],
                                                             boot["ci_hi"]],
                                     plcc=float(r.plcc.mean()))
    return out, r


# --------------------------------- S7  mediator content across backbones

def s7_backbones(raw: pd.DataFrame):
    sel = raw[(raw["head"] == "ridge") & (raw["n_train"] == 100)]
    out = {}
    for variant, label in [("C", "anchored"), ("plain", "unanchored")]:
        block = {}
        s = sel[sel.variant == variant]
        wide = (s.groupby(KEY + ["mediator"])["srocc"].mean()
                 .unstack().reset_index())
        for bb in BACKBONES:
            sb = s[s.backbone == bb]
            w = (sb.groupby(KEY + ["mediator"])["srocc"].mean()
                   .unstack().reset_index())
            if "pca" not in w or "emotion" not in w:
                continue
            row = {"means": {m: float(w[m].mean()) for m in
                             ["population", "identity", "emotion", "pca", "random"]
                             if m in w}}
            for other in ["pca", "random", "identity"]:
                if other not in w:
                    continue
                row[f"emotion_vs_{other}"] = cluster_boot(
                    w["emotion"] - w[other], w["user_id"])
            block[bb] = row
        out[label] = block
    return out


# ----------------------------------------- S8  equivalence and fold-level

def s8_equivalence(raw: pd.DataFrame):
    s = raw[(raw.backbone == "qwen8b") & (raw["head"] == "ridge")
            & (raw.variant == "C") & (raw["n_train"] == 100)]
    w = s.groupby(KEY + ["mediator"])["srocc"].mean().unstack()
    per_participant = w.groupby(level="user_id").mean()
    out = {"tost": {}, "fold": {}}
    for name, other in [("pca", "pca"), ("random", "random"),
                        ("identity", "identity")]:
        out["tost"][name] = {f"{d:.2f}": tost(per_participant["emotion"],
                                              per_participant[other], d)
                             for d in (0.01, 0.02, 0.03)}
    land = w.xs("landscape", level="domain")
    out["tost"]["identity_landscape"] = {
        f"{d:.2f}": tost(land["emotion"], land["identity"], d)
        for d in (0.01, 0.02, 0.03)}
    fold = w.groupby(level="fold").mean()
    out["fold"]["means"] = fold[["population", "identity", "emotion",
                                 "pca", "random"]].round(4).to_dict()
    for other in ["identity", "population", "pca"]:
        d = fold["emotion"] - fold[other]
        out["fold"][f"emotion_minus_{other}"] = dict(
            per_fold=[round(float(x), 4) for x in d],
            mean=float(d.mean()),
            p_paired_t=float(stats.ttest_rel(fold["emotion"],
                                             fold[other]).pvalue))
    return out


# ------------------------------- S9  what the personal term contributes

def s9_decomposition(units: pd.DataFrame, images: pd.DataFrame):
    ua = units[units.setting == "anchored"].set_index(KEY)
    ia = images[images.setting == "anchored"]
    sd_pop, sd_per, share, rows = [], [], [], []
    for key, g in ia.groupby(KEY, sort=False):
        yp = g.y_pred.to_numpy(float)
        pop = g.y_pop.to_numpy(float)
        per = yp - pop
        sd_pop.append(pop.std())
        sd_per.append(per.std())
        share.append(per.var() / yp.var() if yp.var() > 0 else np.nan)
        rows.append(dict(zip(KEY, key)))
    W = ua[[f"{e}_w" for e in CORE7]].to_numpy(float)
    WP = ua[[f"{e}_wpop" for e in CORE7]].to_numpy(float)
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    curve = {g: [] for g in gammas}
    for key, g in ia.groupby(KEY, sort=False):
        y = g.y_true.to_numpy(float)
        pop = g.y_pop.to_numpy(float)
        Z = g[[f"{e}_z" for e in CORE7]].to_numpy(float)
        r = ua.loc[key]
        w = np.array([float(r[f"{e}_w"]) for e in CORE7])
        b = float(r["bias"])
        for gm in gammas:
            curve[gm].append(srocc(y, pop + Z @ (gm * w) + b))
    return dict(
        sd_pop=float(np.mean(sd_pop)), sd_personal=float(np.mean(sd_per)),
        share_mean=float(np.nanmean(share)),
        share_median=float(np.nanmedian(share)),
        norm_w_pop=float(np.linalg.norm(WP, axis=1).mean()),
        norm_delta=float(np.linalg.norm(W, axis=1).mean()),
        gamma_curve={str(g): float(np.nanmean(v)) for g, v in curve.items()})


# ------------------------------- S10  faithfulness controls (null models)

def s10_controls(units: pd.DataFrame, images: pd.DataFrame, seed=0):
    ua = units[units.setting == "anchored"].set_index(KEY)
    ia = images[images.setting == "anchored"]
    rng = np.random.default_rng(seed)

    # (a) image-swap against a matched-covariance noise control
    res = {k: [] for k in ["correct", "swapped", "no_image",
                           "gaussian_matched", "column_shuffled"]}
    keys = []
    for key, g in ia.groupby(KEY, sort=False):
        g = g.sort_values("stimulus_id")
        y = g.y_true.to_numpy(float)
        pop = g.y_pop.to_numpy(float)
        Z = g[[f"{e}_z" for e in CORE7]].to_numpy(float)
        r = ua.loc[key]
        w = np.array([float(r[f"{e}_w"]) for e in CORE7])
        b = float(r["bias"])
        zbar = np.array([float(r[f"{e}_zbar"]) for e in CORE7])
        keys.append(key)
        res["correct"].append(srocc(y, pop + Z @ w + b))
        res["swapped"].append(np.nanmean(
            [srocc(y, pop + np.roll(Z, s, axis=0) @ w + b)
             for s in range(1, len(g))]))
        res["no_image"].append(srocc(y, pop + float(zbar @ w + b) + 0 * y))
        cov = np.cov(Z, rowvar=False)
        mu = Z.mean(0)
        res["gaussian_matched"].append(np.nanmean(
            [srocc(y, pop + rng.multivariate_normal(mu, cov, len(g)) @ w + b)
             for _ in range(20)]))
        res["column_shuffled"].append(np.nanmean(
            [srocc(y, pop + np.column_stack(
                [rng.permutation(Z[:, j]) for j in range(7)]) @ w + b)
             for _ in range(20)]))
    swap = pd.DataFrame(res)
    swap[KEY] = pd.DataFrame(keys, columns=KEY)

    # (b) weight alignment against a peer-shifted null
    emp_pred, emp_rated = {}, {}
    for key, g in ia.groupby(KEY, sort=False):
        y = g.y_true.to_numpy(float)
        vp, vr = [], []
        for e in CORE7:
            z = g[f"{e}_z"].to_numpy(float)
            rr = g[f"{e}_rated"].to_numpy(float)
            vp.append(np.corrcoef(z, y)[0, 1] if z.std() > 0 else np.nan)
            vr.append(np.corrcoef(rr, y)[0, 1] if rr.std() > 0 else np.nan)
        emp_pred[key] = np.array(vp)
        emp_rated[key] = np.array(vr)

    def alignment(weight_suffix, table, shifted):
        rows = []
        for (f, dom), dd in ua.reset_index().groupby(["fold", "domain"]):
            ks = [(f, dom, int(x)) for x in dd.user_id]
            partners = ([ks[j] for j in np.roll(np.arange(len(ks)), 1)]
                        if shifted else ks)
            for k, pk in zip(ks, partners):
                w = np.array([float(ua.loc[k, f"{e}_{weight_suffix}"])
                              for e in CORE7])
                e_ = table.get(pk)
                if e_ is None:
                    continue
                ok = np.isfinite(e_)
                if ok.sum() < 3:
                    continue
                rows.append(dict(zip(KEY, k)) | dict(
                    rho=float(spearmanr(w[ok], e_[ok]).statistic)))
        return pd.DataFrame(rows)

    align = {}
    for suffix in ["wtot", "w"]:
        for tname, table in [("predicted", emp_pred), ("reported", emp_rated)]:
            own = alignment(suffix, table, shifted=False)
            null = alignment(suffix, table, shifted=True)
            merged = own.merge(null, on=KEY, suffixes=("_own", "_null"))
            gap = cluster_boot(merged.rho_own - merged.rho_null,
                               merged.user_id, alternative="greater")
            align[f"{suffix}_vs_{tname}"] = dict(
                own=float(own.rho.mean()), null=float(null.rho.mean()),
                gap=gap["mean"], gap_ci=[gap["ci_lo"], gap["ci_hi"]],
                gap_p=gap["p"],
                pct_positive_own=float(100 * (own.rho > 0).mean()))
    return swap, align


# --------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="output", type=Path)
    ap.add_argument("--ratings", default=None, type=Path)
    ap.add_argument("--dest", default="output/supplementary_v2", type=Path)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)

    raw, units, images = load(a.output_dir)
    report = {}

    if a.ratings and a.ratings.exists():
        tr, tr_units = s6_test_retest(a.ratings)
        report["S6_test_retest"] = tr
        tr_units.to_csv(a.dest / "test_retest_units.csv", index=False)

    report["S7_backbone_mediators"] = s7_backbones(raw)
    report["S8_equivalence_and_folds"] = s8_equivalence(raw)
    report["S9_decomposition"] = s9_decomposition(units, images)

    swap, align = s10_controls(units, images)
    swap.to_csv(a.dest / "image_swap_controls_units.csv", index=False)
    report["S10_alignment_null"] = align
    report["S10_image_swap_controls"] = {
        c: dict(mean=float(np.nanmean(swap[c])),
                **{k: v for k, v in cluster_boot(swap[c], swap.user_id).items()
                   if k in ("sem", "ci_lo", "ci_hi")})
        for c in ["correct", "swapped", "no_image",
                  "gaussian_matched", "column_shuffled"]}

    (a.dest / "supplementary_v2_values.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
