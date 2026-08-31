r"""Faithfulness diagnostics computed from stored per-user formulas.

All four checks are deterministic arithmetic over the stored formula and
evaluation-image records; no model is refitted.

Because the head is linear, every check is arithmetic over saved numbers:

  1. Ablation. Hold one emotion at its support-set mean and see how much
     accuracy drops. The reported diagnostic assigns the emotion-space total
     weight w_tot = w_pop + delta to that concept, so holding emotion j at
     zbar_j gives

         y_ablate = y_pred - w_tot_j * (z_j - zbar_j)

     with no refit. This is a total emotion-space attribution diagnostic, not
     an algebraic decomposition of Anchor C's raw-feature population branch:
     the executable predictor is y_pop(raw image features) + z . delta + b.
     Dropping the top-weighted emotion (max |w_tot|) should hurt more than
     dropping the bottom-weighted one (min |w_tot|).

  2. Formula swap. Score a user with their own formula, with the mean of all
     eligible peer formulas, and with five fixed-seed sampled peer formulas.
     Swaps are restricted to one fold and domain so coefficients share a
     scaler and population model.

  3. Weight against empirical correlation. Do the seven total-formula weights
     line up with how much each predicted emotion correlates with that user's
     raw aesthetic scores? This uses the reported total-formula weights.

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
from src.utils.metrics import (mean_sd, participant_cluster_bootstrap, plcc,
                               sem, srocc, wilcoxon_paired)

N_SWAP = 5
SWAP_SEED = 0
KEY = ["fold", "domain", "user_id"]


def _load(cfg):
    d = Path(cfg.output_dir) / "formulas"
    need = ["user_formula.csv", "per_image.csv"]
    missing = [f for f in need if not (d / f).exists()]
    if not missing:
        units = pd.read_csv(d / "user_formula.csv")
        images = pd.read_csv(d / "per_image.csv", low_memory=False)
    else:
        units, images = _load_consolidated(Path(cfg.output_dir), "anchored")
    images["stimulus_id"] = images["stimulus_id"].astype(str)
    return units, images


def _load_formula_dir(d: Path):
    """Load a separately fitted formula set (for example, variant plain)."""
    units = pd.read_csv(d / "user_formula.csv")
    images = pd.read_csv(d / "per_image.csv", low_memory=False)
    images["stimulus_id"] = images["stimulus_id"].astype(str)
    return units, images


def _load_consolidated(output_dir: Path, setting: str):
    """Load one setting from the two-table release formula database."""
    unit_path = output_dir / "formula_units_final.csv"
    image_path = output_dir / "formula_images_final.csv"
    missing = [p.name for p in [unit_path, image_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{output_dir} is missing {missing}. Run `main.py formulas` and "
            "`python -m src.experiments.make_formula_database` first."
        )
    units = pd.read_csv(unit_path, low_memory=False)
    images = pd.read_csv(image_path, low_memory=False)
    if "setting" not in units or "setting" not in images:
        raise ValueError("consolidated formula tables require a setting column")
    units = units[units.setting == setting].copy()
    images = images[images.setting == setting].copy()
    if len(units) != 387 or len(images) != 19_350:
        raise ValueError(
            f"{setting}: expected 387 formula units and 19,350 image rows, "
            f"got {len(units)} and {len(images)}"
        )
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

    align_peer = weight_alignment_peer_null(units, images)
    align_peer.to_csv(out_dir / "weight_alignment_peer_null.csv", index=False)

    img_swap = image_swap(units, images)
    img_swap.to_csv(out_dir / "image_swap.csv", index=False)

    image_control = image_swap_covariance_control(units, images)
    image_control.to_csv(
        out_dir / "image_swap_covariance_control.csv", index=False
    )

    parts = [summarize_ablation(abl), summarize_swap(swap),
             summarize_align(align), summarize_image_swap(img_swap)]
    names = ["ablation", "formula_swap", "weight_vs_empirical", "image_swap"]
    
    # plain: swap
    plain_dir = Path(cfg.output_dir) / "unanchored" / "formulas"
    if (plain_dir / "user_formula.csv").exists():
        plain_units, plain_images = _load_formula_dir(plain_dir)
    elif (Path(cfg.output_dir) / "formula_units_final.csv").exists():
        plain_units, plain_images = _load_consolidated(
            Path(cfg.output_dir), "unanchored"
        )
    else:
        plain_units = plain_images = None
    if plain_units is not None:
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
    print(f"[faithfulness] wrote {len(summary)} summary rows to {out_dir}")
    return abl, swap, align, img_swap, summary


# ------------------------------------------------------------------ 1. concept ablation

def ablation(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Total emotion-space attribution diagnostic.

    ``w_tot`` combines the separately fitted population emotion-space mapping
    with the personal residual coefficient. Anchor C itself predicts with the
    raw-feature population model plus ``delta``; therefore this intervention
    does not literally remove a node from that raw-feature branch. Use
    ``*_w`` instead of ``*_wtot`` only for a distinct, computational-path
    ablation of the personal residual term.
    """
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

        # Subtract the assigned total emotion-space contribution for concept j.
        # Broadcasting produces one counterfactual prediction column per concept.
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
            # Mean score across the seven one-concept interventions.
            avg1_srocc=float(np.mean(s)), avg1_plcc=float(np.mean(pl)),
            top_emotion=CORE7[aj], bottom_emotion=CORE7[bj]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 2. formula swap

def formula_swap(units: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Apply five reproducibly sampled peer formulas to this person's images.

    Swaps stay within one (fold, domain), where coefficients share a scaler and
    population model. ``RandomState(0)`` is reset
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
    """Compare total-formula weights with empirical emotion associations."""
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
        # Explicit aliases retained for existing output readers.
        emp_raw = emp_against(y)
        ok2 = np.isfinite(emp_raw)
        if ok2.sum() >= 3:
            row["spearman_vs_raw_y"] = float(spearmanr(w[ok2], emp_raw[ok2]).statistic)
            row["pearson_vs_raw_y"] = float(np.corrcoef(w[ok2], emp_raw[ok2])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


def weight_alignment_peer_null(
    units: pd.DataFrame, images: pd.DataFrame
) -> pd.DataFrame:
    """Compare each total weight vector with own and fixed-peer associations.

    Peers are assigned by a cyclic shift of sorted user order within each fold
    and domain.  The pairing is deterministic, one-to-one, and has no fixed
    points.  This is a scope check for the shared population direction in
    ``w_tot`` rather than a replacement for the primary alignment analysis.
    """
    associations = {}
    for key, g in images.groupby(KEY, sort=False):
        y = g["y_true"].to_numpy(float)
        values = []
        for emotion in CORE7:
            z = g[f"{emotion}_z"].to_numpy(float)
            values.append(
                np.corrcoef(z, y)[0, 1]
                if z.std() > 0 and y.std() > 0 else np.nan
            )
        associations[key] = np.asarray(values, float)

    indexed = units.set_index(KEY)
    rows = []
    for (fold, domain), group in units.groupby(["fold", "domain"], sort=True):
        user_ids = sorted(int(value) for value in group["user_id"].unique())
        keys = [(fold, domain, user_id) for user_id in user_ids]
        peer_keys = [keys[-1], *keys[:-1]]
        for key, peer_key in zip(keys, peer_keys):
            weights = np.array([
                float(indexed.loc[key, f"{emotion}_wtot"])
                for emotion in CORE7
            ])

            def correlation(values):
                valid = np.isfinite(values)
                return (
                    float(spearmanr(weights[valid], values[valid]).statistic)
                    if valid.sum() >= 3 else np.nan
                )

            own = correlation(associations[key])
            peer = correlation(associations[peer_key])
            rows.append(dict(zip(KEY, key)) | {
                "peer_user_id": peer_key[2],
                "own_spearman": own,
                "peer_spearman": peer,
                "excess_spearman": own - peer,
            })
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
        
        # Validate the stored prediction against the reconstructed formula.
        if not np.allclose(reconstructed, p, rtol=1e-6, atol=1e-5):
            raise ValueError(f"saved prediction does not match formula for {key}")

        no_image = base + float(zbar @ w + bias)
        swapped = np.stack([
            base + np.roll(Z, shift=shift, axis=0) @ w + bias
            for shift in range(1, len(g))
        ])

        def row_corr(predictions, target, ranked=False):
            """Vectorized Pearson correlation; rank inputs for Spearman."""
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


def image_swap_covariance_control(
    units: pd.DataFrame,
    images: pd.DataFrame,
    *,
    seed: int = 0,
    n_draws: int = 20,
) -> pd.DataFrame:
    """Matched-scale control for the anchored image-swap diagnostic.

    For each unit, draw emotion vectors from a Gaussian with the observed
    seven-dimensional mean and covariance.  The fixed seed and draw count make
    this post-hoc scale check deterministic.
    """
    rng = np.random.default_rng(seed)
    indexed = units.set_index(KEY)
    rows = []
    for key, group in images.groupby(KEY, sort=False):
        # The released IDs are numeric but consolidated CSV loading normalizes
        # them to strings.  Sort by their numeric value so this control is
        # invariant to that storage dtype.
        g = group.assign(
            _stimulus_order=pd.to_numeric(group["stimulus_id"], errors="raise")
        ).sort_values("_stimulus_order")
        y = g["y_true"].to_numpy(float)
        y_pop = g["y_pop"].to_numpy(float)
        z = g[[f"{emotion}_z" for emotion in CORE7]].to_numpy(float)
        row = indexed.loc[key]
        weights = np.array([float(row[f"{emotion}_w"]) for emotion in CORE7])
        bias = float(row["bias"])
        mean = z.mean(axis=0)
        covariance = np.cov(z, rowvar=False)
        gaussian_scores = [
            srocc(
                y,
                y_pop
                + rng.multivariate_normal(mean, covariance, len(g)) @ weights
                + bias,
            )
            for _ in range(n_draws)
        ]
        shuffled_scores = [
            srocc(
                y,
                y_pop
                + np.column_stack([
                    rng.permutation(z[:, column])
                    for column in range(z.shape[1])
                ]) @ weights
                + bias,
            )
            for _ in range(n_draws)
        ]
        rows.append(dict(zip(KEY, key)) | {
            "n_draws": n_draws,
            "gaussian_matched_srocc": float(np.nanmean(gaussian_scores)),
            "column_shuffled_srocc": float(np.nanmean(shuffled_scores)),
        })
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
            r[f"{m}_sem"] = participant_cluster_bootstrap(
                b[f"{k}_{m}"], b["user_id"])["sem"]
            r[f"{m}_drop"] = float(b[f"full_{m}"].mean() - b[f"{k}_{m}"].mean())
            p = (participant_cluster_bootstrap(
                    b[f"full_{m}"] - b[f"{k}_{m}"], b["user_id"])["p_value"]
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
            r[f"{m}_sem"] = participant_cluster_bootstrap(
                b[f"{k}_{m}"], b["user_id"])["sem"]
            p = (participant_cluster_bootstrap(
                    b[f"own_{m}"] - b[f"{k}_{m}"], b["user_id"])["p_value"]
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
            if dom == "ALL":
                boot = participant_cluster_bootstrap(
                    d[m], d["user_id"], alternative="greater")
                s_sem = boot["sem"]
                ci_lower = boot["ci95_lower"]
                ci_upper = boot["ci95_upper"]
                p = boot["p_value"]
            else:
                s_sem = sem(d[m])
                ci_lower = float(mean - 1.96 * s_sem)
                ci_upper = float(mean + 1.96 * s_sem)
                p = wilcoxon_paired(
                    d[m], np.zeros(len(d)), alternative="greater")
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            r[f"{m}_sem"] = s_sem
            r[f"{m}_ci95_lower"] = ci_lower
            r[f"{m}_ci95_upper"] = ci_upper
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
            finite = np.isfinite(b[f"{k}_{m}"].to_numpy(float))
            r[f"{m}_sem"] = (participant_cluster_bootstrap(
                b.loc[finite, f"{k}_{m}"], b.loc[finite, "user_id"])["sem"]
                if finite.any() else np.nan)
            r[f"{m}_drop"] = float(
                b[f"correct_{m}"].mean() - b[f"{k}_{m}"].mean())
            differences = b[f"correct_{m}"] - b[f"{k}_{m}"]
            difference_finite = np.isfinite(differences.to_numpy(float))
            p = (participant_cluster_bootstrap(
                    differences[difference_finite],
                    b.loc[difference_finite, "user_id"])["p_value"]
                 if k != "correct" and difference_finite.any() else np.nan)
            r[f"{m}_p_vs_correct"] = p
            r[f"{m}_sig_vs_correct"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows).set_index("condition")
