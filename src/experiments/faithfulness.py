"""Is the per-user formula a faithful explanation, or numbers that happen to fit?

Three checks, all computed from the formulas `main.py formulas` wrote. Nothing
here refits anything, which is the point: the earlier version built its own
heads with `make_head("ridge", cfg).fit(M, y)`, so it interpreted an
unanchored head with a per-user penalty while the tables reported an anchored
head with a penalty frozen on the validation group. The two were never the
same model and the output did not say so.

Because the head is linear, every check is arithmetic over saved numbers:

  1. Ablation. Hold one emotion at its support-set mean and see how much
     accuracy drops. Under the anchor the prediction is
     y_pop + z . delta + b, so holding emotion j at zbar_j gives

         y_ablate = y_pred - delta_j * (z_j - zbar_j)

     exactly, with no refit. Dropping the top-weighted emotion should hurt
     more than dropping the bottom-weighted one, or the weights are decoration.

  2. Formula swap. Score a user with their own formula, with the mean of the
     other users' formulas, and with individual other users' formulas. "Own"
     should win if formulas are really personal. A swap is only meaningful
     inside one fold and one domain, because that is the only place two
     users' coefficients share a scaler and a population model. The earlier
     version pooled by domain across folds, which mixed units.

  3. Weight against empirical correlation. Do the seven coefficients line up
     with how much each predicted emotion actually correlates with that user's
     scores? Under the anchor the target is the residual the head was fitted
     on, y - y_pop, not the raw score.

Output: output/faithfulness/{ablation,formula_swap,weight_vs_empirical}.csv
        and summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.data import CORE7, DOMAINS
from src.utils.metrics import mean_sd, plcc, sem, srocc, wilcoxon_paired

N_SWAP = 5          # other users' formulas sampled per unit
SWAP_SEED = 0
KEY = ["fold", "domain", "user_id"]


def _load(cfg):
    d = Path(cfg.output_dir) / "formulas"
    need = ["user_formula.csv", "per_image.csv"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{d} is missing {missing}. Run `main.py formulas` first: this "
            f"experiment reads the fitted formulas rather than refitting, so "
            f"that what it interprets is what the tables report.")
    units = pd.read_csv(d / "user_formula.csv")
    images = pd.read_csv(d / "per_image.csv", low_memory=False)
    images["stimulus_id"] = images["stimulus_id"].astype(str)
    return units, images


def run(cfg, pipeline=None):
    out_dir = cfg.run_dir("faithfulness")
    units, images = _load(cfg)
    variant = str(units["variant"].iloc[0])
    print(f"[faithfulness] reading formulas fitted under variant {variant}, "
          f"{len(units)} units")

    abl = ablation(units, images)
    abl.to_csv(out_dir / "ablation.csv", index=False)

    swap = formula_swap(units, images)
    swap.to_csv(out_dir / "formula_swap.csv", index=False)

    align = weight_vs_empirical(units, images)
    align.to_csv(out_dir / "weight_vs_empirical.csv", index=False)

    summary = pd.concat([summarize_ablation(abl), summarize_swap(swap),
                         summarize_align(align)],
                        keys=["ablation", "formula_swap", "weight_vs_empirical"],
                        names=["test"])
    summary.to_csv(out_dir / "summary.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    return abl, swap, align


# ------------------------------------------------------------------ 1. ablation

def ablation(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    u_idx = units.set_index(KEY)
    for key, g in images.groupby(KEY, sort=False):
        if key not in u_idx.index:
            continue
        u = u_idx.loc[key]
        y = g["y_true"].to_numpy(float)
        p = g["y_pred"].to_numpy(float)
        w = np.array([float(u[f"{e}_w"]) for e in CORE7])
        Z = g[[f"{e}_z" for e in CORE7]].to_numpy(float)
        zbar = np.array([float(u[f"{e}_zbar"]) for e in CORE7])

        # holding emotion j at zbar_j removes exactly one term
        held = p[:, None] - (Z - zbar[None, :]) * w[None, :]

        aj = int(np.abs(w).argmax())
        bj = int(np.abs(w).argmin())
        s = [srocc(y, held[:, j]) for j in range(len(CORE7))]
        pl = [plcc(y, held[:, j]) for j in range(len(CORE7))]
        rows.append(dict(
            zip(KEY, key)) | dict(
            full_srocc=srocc(y, p), full_plcc=plcc(y, p),
            top1_srocc=s[aj], top1_plcc=pl[aj],
            bottom1_srocc=s[bj], bottom1_plcc=pl[bj],
            avg1_srocc=float(np.mean(s)), avg1_plcc=float(np.mean(pl)),
            top_emotion=CORE7[aj], bottom_emotion=CORE7[bj]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 2. formula swap

def formula_swap(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Apply other people's formulas to this person's images.

    Only within one (fold, domain): elsewhere the coefficients are in a
    different scaler and sit on a different population model, so applying them
    would not be a swap, it would be a unit error.
    """
    rng = np.random.RandomState(SWAP_SEED)
    rows = []
    u_by_fd = {k: g for k, g in units.groupby(["fold", "domain"], sort=False)}

    for key, g in images.groupby(KEY, sort=False):
        f, d, uid = key
        peers = u_by_fd.get((f, d))
        if peers is None:
            continue
        others = peers[peers.user_id != uid]
        if len(others) < 2:
            continue

        y = g["y_true"].to_numpy(float)
        y_pop = g["y_pop"].to_numpy(float)
        Z = g[[f"{e}_z" for e in CORE7]].to_numpy(float)
        own = g["y_pred"].to_numpy(float)

        def score_with(row):
            w = np.array([float(row[f"{e}_w"]) for e in CORE7])
            return y_pop + Z @ w + float(row["bias"])

        pick = rng.choice(len(others), size=min(N_SWAP, len(others)),
                          replace=False)
        oth = [score_with(others.iloc[i]) for i in pick]
        pop = np.mean([score_with(others.iloc[i]) for i in range(len(others))],
                      axis=0)
        rows.append(dict(zip(KEY, key)) | dict(
            n_others=len(others),
            own_srocc=srocc(y, own), own_plcc=plcc(y, own),
            other_srocc=float(np.mean([srocc(y, o) for o in oth])),
            other_plcc=float(np.mean([plcc(y, o) for o in oth])),
            pop_srocc=srocc(y, pop), pop_plcc=plcc(y, pop)))
    return pd.DataFrame(rows)


# ------------------------------------------------------ 3. weight vs empirical

def weight_vs_empirical(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Do the coefficients agree with what each emotion actually tracks?

    The head was fitted on the residual against the population model, so the
    thing a coefficient claims to explain is that residual. Correlating the
    weights against the raw score instead would ask a question the head was
    never fitted to answer, and would credit the coefficients with whatever
    y_pop already got right.
    """
    rows = []
    u_idx = units.set_index(KEY)
    for key, g in images.groupby(KEY, sort=False):
        if key not in u_idx.index:
            continue
        u = u_idx.loc[key]
        w = np.array([float(u[f"{e}_w"]) for e in CORE7])
        y = g["y_true"].to_numpy(float)
        resid = y - g["y_pop"].to_numpy(float)

        def emp_against(target):
            out = []
            for e in CORE7:
                z = g[f"{e}_z"].to_numpy(float)
                out.append(np.corrcoef(z, target)[0, 1]
                           if z.std() > 0 and target.std() > 0 else np.nan)
            return np.asarray(out, float)

        emp = emp_against(resid)
        ok = np.isfinite(emp)
        if ok.sum() < 3:
            continue
        row = dict(zip(KEY, key)) | dict(
            n_emotions=int(ok.sum()),
            spearman=float(spearmanr(w[ok], emp[ok]).statistic),
            pearson=float(np.corrcoef(w[ok], emp[ok])[0, 1]))
        # the same question asked against the raw score rather than the
        # residual. It is the wrong target for an anchored head -- it credits
        # the coefficients with whatever y_pop already got right -- but the
        # earlier version reported it, so both are written out and the choice
        # is visible instead of silent.
        emp_raw = emp_against(y)
        ok2 = np.isfinite(emp_raw)
        if ok2.sum() >= 3:
            row["spearman_vs_raw_y"] = float(spearmanr(w[ok2], emp_raw[ok2]).statistic)
            row["pearson_vs_raw_y"] = float(np.corrcoef(w[ok2], emp_raw[ok2])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ summaries

def summarize_ablation(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, k in [("full", "full"), ("top1_held", "top1"),
                     ("avg1_held", "avg1"), ("bottom1_held", "bottom1")]:
        r = dict(condition=label)
        for m in ("srocc", "plcc"):
            mean, sd = mean_sd(b[f"{k}_{m}"])
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            r[f"{m}_sem"] = sem(b[f"{k}_{m}"])
            r[f"{m}_drop"] = float(b[f"full_{m}"].mean() - b[f"{k}_{m}"].mean())
            p = (wilcoxon_paired(b[f"full_{m}"], b[f"{k}_{m}"])
                 if k != "full" else np.nan)
            r[f"{m}_p_vs_full"] = p
        rows.append(r)
    return pd.DataFrame(rows).set_index("condition")


def summarize_swap(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, k in [("own", "own"), ("population_mean", "pop"),
                     ("other_user", "other")]:
        r = dict(formula=label)
        for m in ("srocc", "plcc"):
            mean, sd = mean_sd(b[f"{k}_{m}"])
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            r[f"{m}_sem"] = sem(b[f"{k}_{m}"])
            p = (wilcoxon_paired(b[f"own_{m}"], b[f"{k}_{m}"])
                 if k != "own" else np.nan)
            r[f"{m}_p_vs_own"] = p
            r[f"{m}_sig_vs_own"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows).set_index("formula")


def summarize_align(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dom in DOMAINS + ["ALL"]:
        d = b if dom == "ALL" else b[b.domain == dom]
        r = dict(domain=dom, n=len(d),
                 pct_positive=100 * float((d["spearman"] > 0).mean()))
        for m in ("spearman", "pearson"):
            mean, sd = mean_sd(d[m])
            p = wilcoxon_paired(d[m], np.zeros(len(d)))
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            r[f"{m}_sem"] = sem(d[m])
            r[f"{m}_p"] = p
            r[f"{m}_sig"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows).set_index("domain")