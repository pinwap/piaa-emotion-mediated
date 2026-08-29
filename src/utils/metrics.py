# Every metric used here -- CCC / PLCC / SROCC / Wilcoxon / effective DoF.
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon


def ccc(y_true, y_pred) -> float:
    # Concordance correlation coefficient
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mt, mp = y_true.mean(), y_pred.mean()
    cov = np.mean((y_true - mt) * (y_pred - mp))
    denom = y_true.var() + y_pred.var() + (mt - mp) ** 2
    return float(2 * cov / denom) if denom > 0 else 0.0


def srocc(y_true, y_pred) -> float:
    # Spearman rank-order correlation
    s = spearmanr(y_true, y_pred).statistic
    return float(s) if np.isfinite(s) else np.nan


def plcc(y_true, y_pred) -> float:
    # Pearson linear correlation
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.std() == 0 or y_pred.std() == 0:
        return np.nan
    r = pearsonr(y_true, y_pred)[0]
    return float(r) if np.isfinite(r) else np.nan


METRICS = {"ccc": ccc, "srocc": srocc, "plcc": plcc}

def evaluate(y_true, y_pred) -> dict[str, float]:
    return {k: f(y_true, y_pred) for k, f in METRICS.items()}


def wilcoxon_paired(a, b) -> float:
    """Paired Wilcoxon signed-rank -> p-value.

    Two models compared on the same units (user x domain), so this is
    always paired. Returns nan if all differences are zero or too few
    units remain.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.sum(np.abs(a - b)) == 0:
        return np.nan
    return float(wilcoxon(a, b, method="approx").pvalue)


def mean_sd(values) -> tuple[float, float]:
    """Mean and sd (ddof=1) across units, ignoring nan."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), np.nan
    return float(v.mean()), float(v.std(ddof=1))


def sem(values) -> float:
    """Standard error of the mean (sd/sqrt(n)), ignoring nan.

    n is counted from `values` itself, never assumed -- tables differ in what
    one unit is (387 user-domain units for the model tables, 5 fold-domain
    estimates for Stage-1 accuracy)
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    return float(v.std(ddof=1) / np.sqrt(len(v)))


def participant_cluster_bootstrap(
    values,
    participant_ids,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    null: float = 0.0,
    alternative: str = "two-sided",
) -> dict[str, float | int]:
    """Bootstrap a pooled mean while keeping each participant's rows together.

    ``values`` may contain several correlated observations per participant
    (three domains in this project).  They are first averaged within
    participant; bootstrap samples then resample participants, never individual
    user-domain rows.  The returned p-value follows the empirical-tail rule
    used in the manuscript audit.
    """
    v = np.asarray(values, float)
    ids = np.asarray(participant_ids)
    if v.ndim != 1 or ids.ndim != 1 or len(v) != len(ids):
        raise ValueError("values and participant_ids must be aligned 1-D arrays")
    ok = np.isfinite(v)
    v, ids = v[ok], ids[ok]
    if len(v) == 0:
        raise ValueError("cluster bootstrap received no finite values")

    unique_ids, inverse, counts = np.unique(ids, return_inverse=True,
                                             return_counts=True)
    if len(unique_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two participants")
    if not np.all(counts == counts[0]):
        raise ValueError("each participant must contribute the same number of rows")
    cluster_sums = np.bincount(inverse, weights=v)
    cluster_means = cluster_sums / counts

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(cluster_means), size=(n_boot, len(cluster_means)))
    boot = cluster_means[sampled].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    if alternative == "two-sided":
        p = 2.0 * min(np.mean(boot <= null), np.mean(boot >= null))
    elif alternative == "greater":
        p = np.mean(boot <= null)
    elif alternative == "less":
        p = np.mean(boot >= null)
    else:
        raise ValueError(f"unknown alternative: {alternative}")

    return {
        "mean": float(cluster_means.mean()),
        "sem": float(boot.std(ddof=1)),
        "ci95_lower": float(lo),
        "ci95_upper": float(hi),
        "p_value": float(min(1.0, p)),
        "n_participants": int(len(unique_ids)),
        "rows_per_participant": int(counts[0]),
    }

def effective_dof(X, alpha: float) -> float:
    """Ridge effective degrees of freedom: tr(Z(Z'Z + alpha*I)^-1 Z').

    Computed on standardized features (the pipeline always standardizes
    before RidgeCV). Only defined for linear heads.
    """
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.asarray(X, float))
    G = Z.T @ Z
    p = G.shape[0]
    return float(np.trace(Z @ np.linalg.solve(G + alpha * np.eye(p), Z.T)))


def tost_equivalence(a, b, delta: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests: is `a` equivalent to `b` to within +/- delta?

    A non-significant difference test does NOT mean two methods are the same
    -- it means the data could not tell them apart, which is also what a
    badly underpowered study looks like. The claim this paper actually wants
    is stronger and directional: a 7-parameter head is *not meaningfully
    worse* than a 512-parameter one. That is an equivalence claim, and TOST
    is the test for it.

    TOST rejects "the difference is at least delta in some direction" from
    both sides. Both one-sided tests must pass at `alpha`, so the reported
    p is the larger of the two. Rejecting both means the true mean
    difference sits inside (-delta, +delta) at the chosen confidence.

    delta has to be fixed on substantive grounds *before* looking at the
    result -- the smallest SROCC gap that would matter -- not tuned until
    the test passes. Choosing it afterwards turns TOST into a way of
    proving whatever the data happens to show.

    Paired: a and b are per-unit scores in the same order.
    """
    import numpy as np
    from scipy import stats

    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return dict(n=n, mean_diff=float("nan"), p_tost=float("nan"),
                    equivalent=False, ci_lo=float("nan"), ci_hi=float("nan"))

    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        equiv = abs(d.mean()) < delta
        return dict(n=n, mean_diff=float(d.mean()), p_tost=0.0 if equiv else 1.0,
                    equivalent=equiv, ci_lo=float(d.mean()), ci_hi=float(d.mean()))

    df = n - 1
    # H0_lower: mean <= -delta   H0_upper: mean >= +delta
    p_lower = stats.t.sf((d.mean() + delta) / se, df)
    p_upper = stats.t.cdf((d.mean() - delta) / se, df)
    p = max(p_lower, p_upper)

    # the (1-2*alpha) CI is the interval TOST actually compares to +/-delta
    t_crit = stats.t.ppf(1 - alpha, df)
    lo, hi = d.mean() - t_crit * se, d.mean() + t_crit * se
    return dict(n=n, mean_diff=float(d.mean()), p_tost=float(p),
                equivalent=bool(p < alpha), ci_lo=float(lo), ci_hi=float(hi))
