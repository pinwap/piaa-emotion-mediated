"""Main results table: every mediator x head combo at 100 ratings/user.

Rows: Population (no personal ratings used), Direct (no mediator, 512
params/user), Random/Shuffled (content-free mediators), PCA (unsupervised),
Hybrid (our 7-dim emotion mediator), plus two upper bounds -- GT emotions
(uses true ratings instead of predicted ones) and test-retest reliability.

Everything is scored on the same users/images so rows compare directly with
a paired test.

Every row is run under 3 seeds (0,1,2) and averaged per unit before
summarizing, since the random/shuffled mediators are stochastic.
seed=0 alone reproduces the original single-seed run bit-for-bit.

Writes per_unit.csv (seed-averaged), per_unit_by_seed.csv (raw, one row per
seed), and summary.csv (mean/sd per domain, plus a best-row flag and
Wilcoxon significance vs. Hybrid+Ridge) to output/table1/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data import DOMAINS
from src.utils.metrics import mean_sd, plcc, sem, srocc, wilcoxon_paired
from src.utils.results_db import record

#: row order (mediator, head)
ROWS = [
    ("population", "ridge"),
    ("identity", "ridge"),
    ("random", "ridge"),
    ("shuffled", "ridge"),
    ("pca", "ridge"),
    ("emotion", "ridge"),
]

REFERENCE = ("emotion", "ridge")   # row that Wilcoxon significance is computed against

#: rows involving random/shuffled mediators are stochastic
#: (random projection, label permutation);
#: we repeat the whole grid under these seeds and average per unit so a
#: single unlucky draw doesn't set the reported number. seed=0 reproduces
#: the original single-seed run bit-for-bit (see pipeline.run_grid).
SEEDS = (0, 1, 2)


def _tag(variant: str) -> str:
    """Results for a non-default Stage-2 variant go to their own files, so
    variants can be compared instead of overwriting each other."""
    return "" if variant in (None, "plain") else f"_{variant}"


HEADS = ["ridge"]


def _htag(heads) -> str:
    """A partial-head run goes to its own file so it cannot overwrite a full
    run's results (which hold the rows it is not recomputing)."""
    return "" if list(heads) == HEADS else "_" + "".join(h[0] for h in heads)


MEDIATORS = ["identity", "random", "shuffled", "pca", "emotion"]


def _mtag(mediators) -> str:
    return "" if list(mediators) == MEDIATORS else "_m" + str(len(mediators))


def run_one_seed(cfg, pipeline, seed: int, variant: str | None = None,
                 heads=None, mediators=None) -> pd.DataFrame:
    """One seed's full grid, written to its own file so seeds can run as
    separate processes in parallel (they are completely independent)."""
    out_dir = cfg.run_dir("table1")
    variant = variant or cfg.stage2_variant
    heads = list(heads or HEADS)
    mediators = list(mediators or MEDIATORS)
    print(f"[table1] seed {seed} variant {variant} heads {heads} meds {mediators}")
    d = pipeline.run_grid(
        mediators=mediators,
        heads=heads,
        include_population=True,
        # the ceiling is a ridge-only row; a partial run would not produce
        # it, and asking for it anyway would write a duplicate of a row the
        # full run already has
        include_gt_upper_bound=("ridge" in heads),
        seed=seed,
        stage2_variant=variant,
    )
    d["seed"] = seed
    d["stage2_variant"] = variant
    f = (out_dir /
         f"per_unit{_tag(variant)}{_htag(heads)}{_mtag(mediators)}_seed{seed}.csv")
    d.to_csv(f, index=False)
    print(f"[table1] seed {seed} variant {variant} written ({len(d)} rows) -> {f.name}")
    record(d, "table1", backbone=cfg.backbone, variant=variant,
           n_train=cfg.n_train)
    return d


def run(cfg, pipeline, dataset, seeds=None, variant: str | None = None) -> pd.DataFrame:
    """Run the seeds that aren't on disk yet, then merge and summarize."""
    out_dir = cfg.run_dir("table1")
    seeds = list(SEEDS if seeds is None else seeds)
    variant = variant or cfg.stage2_variant
    tag = _tag(variant)

    raw = []
    for s in seeds:
        f = out_dir / f"per_unit{tag}_seed{s}.csv"
        if f.exists():
            print(f"[table1] seed {s} already on disk, reusing")
            raw.append(pd.read_csv(f))
        else:
            raw.append(run_one_seed(cfg, pipeline, s, variant))

    raw_df = pd.concat(raw, ignore_index=True)
    raw_df.to_csv(out_dir / f"per_unit{tag}_by_seed.csv", index=False)

    key = ["mediator", "head", "fold", "domain", "user_id"]
    value_cols = ["ccc", "srocc", "plcc", "eff_dof"]
    df = raw_df.groupby(key, as_index=False)[value_cols].mean()
    df.to_csv(out_dir / f"per_unit{tag}.csv", index=False)

    retest = test_retest(cfg, dataset)
    summary = summarize(df, retest)
    summary.to_csv(out_dir / f"summary{tag}.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def test_retest(cfg, dataset) -> dict:
    """User's self-agreement across sessions (a second upper bound)

    Uses the second-occurrence ratings that were held out by the
    first-session filter.
    """
    pairs = dataset.retest_pairs(cfg.data_dir)
    out = {}
    for dom in DOMAINS:
        d = pairs[pairs["domain"] == dom]
        y1 = d["overall_r1"].to_numpy(float)
        y2 = d["overall_r2"].to_numpy(float)
        out[dom] = dict(srocc=srocc(y1, y2), plcc=plcc(y1, y2), n=len(d))
    y1 = pairs["overall_r1"].to_numpy(float)
    y2 = pairs["overall_r2"].to_numpy(float)
    out["avg"] = dict(srocc=srocc(y1, y2), plcc=plcc(y1, y2), n=len(pairs))
    return out


def _key(df, med, head):
    # bracket notation for "head" -- df.head is DataFrame.head(), not the column
    return df[(df.mediator == med) & (df["head"] == head)].set_index(
        ["fold", "domain", "user_id"])


def summarize(df, retest) -> pd.DataFrame:
    ref = _key(df, *REFERENCE)
    rows = []
    for med, head in ROWS + [("gt_emotion", "ridge")]:
        s = _key(df, med, head)
        if len(s) == 0:
            continue
        r = dict(mediator=med, head=head, eff_dof=s["eff_dof"].mean())
        for dom in DOMAINS:
            d = s[s.index.get_level_values("domain") == dom]
            for m in ("srocc", "plcc"):
                mean, sd = mean_sd(d[m])
                r[f"{dom}_{m}_mean"], r[f"{dom}_{m}_sd"] = mean, sd
                r[f"{dom}_{m}_sem"] = sem(d[m])
        for m in ("srocc", "plcc"):
            mean, sd = mean_sd(s[m])
            r[f"avg_{m}_mean"], r[f"avg_{m}_sd"] = mean, sd
            r[f"avg_{m}_sem"] = sem(s[m])
            j = s[[m]].merge(ref[[m]], left_index=True, right_index=True,
                             suffixes=("", "_ref")).dropna()
            p = (np.nan if (med, head) == REFERENCE
                 else wilcoxon_paired(j[m], j[f"{m}_ref"]))
            r[f"avg_{m}_sig"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)

    # test-retest is one correlation over all pairs pooled, not an average of per-unit correlations, so there is no sd/sem across units to report
    rt = dict(mediator="test_retest", head="---", eff_dof=np.nan)
    for dom in DOMAINS:
        rt[f"{dom}_srocc_mean"] = retest[dom]["srocc"]
        rt[f"{dom}_plcc_mean"] = retest[dom]["plcc"]
        rt[f"{dom}_srocc_sd"] = rt[f"{dom}_plcc_sd"] = np.nan
        rt[f"{dom}_srocc_sem"] = rt[f"{dom}_plcc_sem"] = np.nan
    for m in ("srocc", "plcc"):
        rt[f"avg_{m}_mean"] = retest["avg"][m]
        rt[f"avg_{m}_sd"] = np.nan
        rt[f"avg_{m}_sem"] = np.nan
        rt[f"avg_{m}_sig"] = False
    rows.append(rt)

    out = pd.DataFrame(rows)
    upper_bound = out.mediator.isin(["gt_emotion", "test_retest"])
    for m in ("srocc", "plcc"):
        top = out.loc[~upper_bound, f"avg_{m}_mean"].max()
        out[f"avg_{m}_best"] = np.isclose(out[f"avg_{m}_mean"], top) & ~upper_bound
    return out
