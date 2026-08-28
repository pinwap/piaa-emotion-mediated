r"""Is the per-user formula a faithful explanation, or numbers that happen to fit?

Four checks, all computed from the formulas `main.py formulas` wrote. Nothing
here refits anything, which is the point: the earlier version built its own
heads with `make_head("ridge", cfg).fit(M, y)`, so it interpreted an
unanchored head with a per-user penalty while the tables reported an anchored
head with a penalty frozen on the validation group. The two were never the
same model and the output did not say so.

Because the head is linear, every check is arithmetic over saved numbers:

  1. Ablation. Hold one emotion at its support-set mean and see how much
     accuracy drops. Under the anchor the prediction is
     \sum_j (w_tot_j * z_j) + bias_tot, so holding emotion j at zbar_j gives

         y_ablate = y_pred - w_tot_j * (z_j - zbar_j)

     exactly, with no refit. Dropping the top-weighted emotion (max |w_tot|) should hurt
     more than dropping the bottom-weighted one (min |w_tot|), 
     proving that the total emotion weights faithfully explain the score.

  2. Formula swap. Score a user with their own formula, with the mean of all
     eligible peer formulas, and with five fixed-seed sampled peer formulas.
     "Own" should win if formulas are really personal. A swap is only meaningful
     inside one fold and one domain, because that is the only place two
     users' coefficients share a scaler and a population model. The earlier
     version pooled by domain across folds, which mixed units.

  3. Weight against empirical correlation. Do the seven total-formula weights
     line up with how much each predicted emotion correlates with that user's
     raw aesthetic scores? This is the reported total-formula interpretation;
     it is not a residual-only diagnostic.

  4. Image swap. Keep the user's fitted formula and the target image's
     population prediction fixed, but replace its emotion vector with the
     vector of another evaluation image. A no-image control replaces the
     vector with the user's support-set mean. Correctly aligned emotions
     should outperform both controls if the channel is image-specific.

Output: output/faithfulness/{ablation,formula_swap,weight_vs_empirical,image_swap}.csv
        and summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.data.data import CORE7, DOMAINS
from src.utils.metrics import mean_sd, plcc, sem, srocc, wilcoxon_paired

N_SWAP = 5
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


def _load_formula_dir(d: Path):
    """Load a separately fitted formula set (for example, variant plain)."""
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

    img_swap = image_swap(units, images)
    img_swap.to_csv(out_dir / "image_swap.csv", index=False)

    parts = [summarize_ablation(abl), summarize_swap(swap),
             summarize_align(align), summarize_image_swap(img_swap)]
    names = ["ablation", "formula_swap", "weight_vs_empirical", "image_swap"]
    
    # plain: swap
    plain_dir = Path(cfg.output_dir) / "unanchored" / "formulas"
    if (plain_dir / "user_formula.csv").exists():
        plain_units, plain_images = _load_formula_dir(plain_dir)
        swap_plain = formula_swap(plain_units, plain_images)
        swap_plain.to_csv(out_dir / "formula_swap_unanchored.csv", index=False)
        img_swap_plain = image_swap(plain_units, plain_images)
        img_swap_plain.to_csv(out_dir / "image_swap_unanchored.csv", index=False)
        parts.append(summarize_swap(swap_plain))
        names.append("formula_swap_unanchored")
        parts.append(summarize_image_swap(img_swap_plain))
        names.append("image_swap_unanchored")

    summary = pd.concat(parts, keys=names,
                        names=["test"])
    summary.to_csv(out_dir / "summary.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    return abl, swap, align, img_swap, summary


# ------------------------------------------------------------------ 1. concept ablation

def ablation(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    u_idx = units.set_index(KEY)
    for key, g in images.groupby(KEY, sort=False):
        if key not in u_idx.index:
            continue
        u = u_idx.loc[key]
        y = g["y_true"].to_numpy(float)
        p = g["y_pred"].to_numpy(float)
        w = np.array([float(u[f"{e}_wtot"]) for e in CORE7])
        Z = g[[f"{e}_z" for e in CORE7]].to_numpy(float)
        zbar = np.array([float(u[f"{e}_zbar"]) for e in CORE7])

        # holding emotion j at zbar_j removes exactly one term
        # add dimension -> [50, 7] and broadcast -> matrix [50, 7] for 50 test images x when cut emotion j
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
            # average of srocc when cut every each emotion
            avg1_srocc=float(np.mean(s)), avg1_plcc=float(np.mean(pl)),
            top_emotion=CORE7[aj], bottom_emotion=CORE7[bj]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 2. formula swap

def formula_swap(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Apply five reproducibly sampled peer formulas to this person's images.

    Only within one (fold, domain): elsewhere the coefficients are in a
    different scaler and sit on a different population model, so applying them
    would not be a swap, it would be a unit error. ``RandomState(0)`` is reset
    at the start of each anchored/unanchored call, making the selected peers
    and every reported value deterministic across reruns.
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
        M = g[[f"{e}_pred" for e in CORE7]].to_numpy(float)
        own = g["y_pred"].to_numpy(float)

        def score_with(row):
            w = np.array([float(row[f"{e}_w"]) for e in CORE7])
            if str(row["variant"]) == "plain":
                mean = np.array([float(row[f"{e}_zmean"]) for e in CORE7])
                scale = np.array([float(row[f"{e}_zscale"]) for e in CORE7])
                z = (M - mean) / scale
                return z @ w + float(row["bias"])
            return y_pop + Z @ w + float(row["bias"])

        pick = rng.choice(
            len(others), size=min(N_SWAP, len(others)), replace=False
        )
        oth = [score_with(others.iloc[i]) for i in pick]
        pop = np.mean(
            [score_with(others.iloc[i]) for i in range(len(others))], axis=0
        )
        rows.append(dict(zip(KEY, key)) | dict(
            n_others=len(others),
            n_swaps=len(pick),
            own_srocc=srocc(y, own), own_plcc=plcc(y, own),
            other_srocc=float(np.mean([srocc(y, o) for o in oth])),
            other_plcc=float(np.mean([plcc(y, o) for o in oth])),
            pop_srocc=srocc(y, pop), pop_plcc=plcc(y, pop)))
    return pd.DataFrame(rows)


# ------------------------------------------------------ 3. weight vs empirical

def weight_vs_empirical(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Do the coefficients agree with what each emotion actually tracks?

    The paper interprets the total personal formula, so this check uses the
    stored total weights and the user's raw aesthetic score. The component
    residual weights are not the estimand reported in the alignment table.
    """
    rows = []
    u_idx = units.set_index(KEY)
    for key, g in images.groupby(KEY, sort=False):
        if key not in u_idx.index:
            continue
        u = u_idx.loc[key]
        w = np.array([float(u[f"{e}_wtot"]) for e in CORE7])
        y = g["y_true"].to_numpy(float)
        def emp_against(target):
            out = []
            for e in CORE7:
                z = g[f"{e}_z"].to_numpy(float)
                out.append(np.corrcoef(z, target)[0, 1]
                           if z.std() > 0 and target.std() > 0 else np.nan)
            return np.asarray(out, float)

        emp = emp_against(y)
        ok = np.isfinite(emp)
        if ok.sum() < 3:
            continue
        row = dict(zip(KEY, key)) | dict(
            n_emotions=int(ok.sum()),
            spearman=float(spearmanr(w[ok], emp[ok]).statistic),
            pearson=float(np.corrcoef(w[ok], emp[ok])[0, 1]))
        # Retain explicit aliases for compatibility with earlier output readers.
        # They deliberately repeat the reported raw-score association.
        emp_raw = emp_against(y)
        ok2 = np.isfinite(emp_raw)
        if ok2.sum() >= 3:
            row["spearman_vs_raw_y"] = float(spearmanr(w[ok2], emp_raw[ok2]).statistic)
            row["pearson_vs_raw_y"] = float(np.corrcoef(w[ok2], emp_raw[ok2])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)

# --------------------------------------------------------------- 4. image swap

def image_swap(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Destroy image--emotion alignment without changing the fitted model.

    Under anchor C, the prediction is ``y_pop(x_i) + z_i @ delta_u + b_u``;
    under plain it is ``z_i @ w_u + b_u``. We retain every term except
    ``z_i`` and change only the image--emotion pairing.
    All non-zero cyclic shifts are evaluated, so every target image receives
    every other evaluation image's emotion vector exactly once, with no fixed
    points and no Monte-Carlo variation.
    """
    rows = []
    u_idx = units.set_index(KEY)
    
    for key, g in images.groupby(KEY, sort=False):
        if key not in u_idx.index:
            continue
        
        u = u_idx.loc[key]
        g = g.sort_values("stimulus_id")
        y = g["y_true"].to_numpy(float)
        p = g["y_pred"].to_numpy(float)
        y_pop = g["y_pop"].to_numpy(float)
        w = np.array([float(u[f"{e}_w"]) for e in CORE7])
        bias = float(u["bias"])
        zbar = np.array([float(u[f"{e}_zbar"]) for e in CORE7])
        Z = g[[f"{e}_z" for e in CORE7]].to_numpy(float)

        variant = str(u["variant"])
        base = y_pop if variant == "C" else np.zeros_like(y_pop)
        reconstructed = base + Z @ w + bias
        
        # sanity check: predict p - reconstructed p
        if not np.allclose(reconstructed, p, rtol=1e-6, atol=1e-5):
            raise ValueError(f"saved prediction does not match formula for {key}")

        no_image = base + float(zbar @ w + bias)
        swapped = np.stack([
            base + np.roll(Z, shift=shift, axis=0) @ w + bias
            for shift in range(1, len(g))
        ])

        def row_corr(predictions, target, ranked=False):
            """
            corr(a,b) = (a - a.mean) @ (b - b.mean) / sqrt(sum(a-a.mean)^2 * sum(b-b.mean)^2)
            spearman <= ranked == True:  else: pearson
            """
            a = (rankdata(predictions, axis=1) if ranked
                 else np.asarray(predictions, float))
            b = rankdata(target) if ranked else np.asarray(target, float)
            a = a - a.mean(axis=1, keepdims=True)
            b = b - b.mean()
            denom = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b))
            return np.divide(a @ b, denom,
                             out=np.full(len(a), np.nan), where=denom > 0)

        swap_s = row_corr(swapped, y, ranked=True)
        swap_p = row_corr(swapped, y, ranked=False)
        no_image_constant = float(np.std(no_image)) <= 1e-12

        rows.append(dict(zip(KEY, key)) | dict(
            n_shifts=len(g) - 1,
            correct_srocc=srocc(y, p), correct_plcc=plcc(y, p),
            swapped_srocc=float(np.mean(swap_s)),
            swapped_plcc=float(np.mean(swap_p)),
            no_image_srocc=(np.nan if no_image_constant
                            else srocc(y, no_image)),
            no_image_plcc=(np.nan if no_image_constant
                           else plcc(y, no_image))))
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
            # paired test to full 
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
            s_sem = sem(d[m])
            p = wilcoxon_paired(d[m], np.zeros(len(d)))
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            r[f"{m}_sem"] = s_sem
            r[f"{m}_ci95_lower"] = float(mean - 1.96 * s_sem)
            r[f"{m}_ci95_upper"] = float(mean + 1.96 * s_sem)
            r[f"{m}_p"] = p
            r[f"{m}_sig"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows).set_index("domain")


def summarize_image_swap(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    conditions = [
        ("correct_image", "correct"),
        ("swapped_image", "swapped"),
        ("no_image_emotion", "no_image"),
    ]
    for label, k in conditions:
        r = dict(condition=label)
        for m in ("srocc", "plcc"):
            mean, sd = mean_sd(b[f"{k}_{m}"])
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            r[f"{m}_sem"] = sem(b[f"{k}_{m}"])
            r[f"{m}_drop"] = float(
                b[f"correct_{m}"].mean() - b[f"{k}_{m}"].mean())
            p = (wilcoxon_paired(b[f"correct_{m}"], b[f"{k}_{m}"])
                 if k != "correct" else np.nan)
            r[f"{m}_p_vs_correct"] = p
            r[f"{m}_sig_vs_correct"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows).set_index("condition")
