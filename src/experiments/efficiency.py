"""How much does the mediator help when a user has few ratings?

Sweeps support size n (10/25/50/100) against a fixed eval set, so results are
comparable across budgets, and reports every mediator against the population
(GIAA) baseline.

The sweep runs through Pipeline.run_grid, the same code path as Table 1, so
one n is exactly a Table 1 run at a different support size: the personal
hyperparameter is frozen per (fold, domain, mediator, head, n) on the
validation user group, and the population row is the real GIAA head on raw
features rather than a population formula read off the mediator. An earlier
version of this file fitted heads with per-user RidgeCV and called the
population formula on the emotion mediator "pop_zero"; neither matched Table
1, so the two tables could not be read against each other.

`--stage2 B/C` anchors the personal head on GIAA, which is the interesting
case here: it asks at what support size a personalized model stops losing to
the population model.

Output: output/efficiency/<backbone>/raw{tag}.csv, summary{tag}.csv
(two files per run: the raw per-seed rows, and the reported summary)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.metrics import mean_sd, sem, wilcoxon_paired
from src.utils.results_db import record

#: mediators the sweep reports. Population is added by run_grid.
MEDIATORS = ["identity", "pca", "emotion"]
REFERENCE = "population"     # what every row is tested against


def _tag(variant, heads, mediators, backbone) -> str:
    """Distinct filename per configuration. Everything that changes the
    numbers is in the name, so parallel runs cannot overwrite one another and
    a file can be traced back to the command that made it."""
    v = "" if variant in (None, "plain") else f"_{variant}"
    h = "" if list(heads) == ["ridge"] else "_" + "".join(x[0] for x in heads)
    b = "" if backbone in (None, "clip") else f"_{backbone}"
    # The mediator part has to name the mediators, not count them: two
    # different 3-mediator runs (e.g. emotion/emotion_sd/emotion_hist and
    # emotion/emotion_sd/emotion_hist) both used to render as "_m3" and the
    # second one silently overwrote the first.
    if list(mediators) == MEDIATORS:
        m = ""
    else:
        m = "_m" + "-".join(x.replace("emotion", "emo") for x in mediators)
        if len(m) > 60:                      # keep paths sane, stay unique
            import hashlib
            m = "_m%d_%s" % (len(mediators),
                             hashlib.md5("|".join(mediators).encode()).hexdigest()[:8])
    return v + h + b + m


def run(cfg, pipeline, n_list: list[int], variant: str | None = None,
        heads=None, mediators=None, seeds=(0,), backbone=None) -> pd.DataFrame:
    # One folder per backbone, so a finished set (4 variants x 4 support
    # sizes x 3 seeds) sits together instead of 50-odd files sharing a flat
    # directory with every other backbone.
    bb = backbone or cfg.backbone
    out_dir = cfg.run_dir("efficiency") / bb
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = variant or cfg.stage2_variant
    heads = list(heads or ["ridge"])
    mediators = list(mediators or MEDIATORS)
    tag = _tag(variant, heads, mediators, backbone or cfg.backbone)

    frames = []
    for n in n_list:
        for seed in seeds:
            print(f"[efficiency] n={n} seed={seed} variant={variant} heads={heads}",
                  flush=True)
            d = pipeline.run_grid(
                mediators=mediators, heads=heads, n_train=n,
                include_population=True, include_gt_upper_bound=False,
                seed=seed, stage2_variant=variant)
            d["n_train"], d["seed"], d["stage2_variant"] = n, seed, variant
            frames.append(d)
            record(d, "efficiency", backbone=backbone or cfg.backbone,
                   variant=variant, n_train=n)

    # Two files per run and no more: the raw per-unit-per-seed rows, which
    # every later table and figure can be rebuilt from, and the summary that
    # is actually reported. The seed-averaged per-unit file that used to sit
    # between them was never read by anything -- it is one groupby away from
    # the raw file, and having it on disk mostly created doubt about which of
    # the two was the real one.
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out_dir / f"raw{tag}.csv", index=False)

    key = ["n_train", "mediator", "head", "fold", "domain", "user_id"]
    df = raw.groupby(key, as_index=False)[["ccc", "srocc", "plcc", "eff_dof"]].mean()

    summary = summarize(df, n_list)
    summary.to_csv(out_dir / f"summary{tag}.csv", index=False)
    cols = [c for c in summary.columns
            if c in ("n_train", "mediator", "head", "n")
            or c.endswith(("srocc_mean", "srocc_sig_vs_pop", "srocc_beats_pop"))]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def summarize(df: pd.DataFrame, n_list: list[int]) -> pd.DataFrame:
    """One row per (n, mediator, head), each tested against the population
    baseline at that same n on the units both scored."""
    rows = []
    for n in n_list:
        d = df[df.n_train == n]
        if d.empty:
            continue
        for head in sorted(d["head"].unique()):
            dh = d[d["head"] == head]
            ref = dh[dh.mediator == REFERENCE].set_index(
                ["fold", "domain", "user_id"])
            for med in sorted(dh.mediator.unique()):
                s = dh[dh.mediator == med].set_index(["fold", "domain", "user_id"])
                r = dict(n_train=n, mediator=med, head=head, n=len(s),
                         eff_dof=s["eff_dof"].mean())
                for m in ("srocc", "plcc"):
                    mean, sd = mean_sd(s[m])
                    r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
                    r[f"{m}_sem"] = sem(s[m])
                    j = s[[m]].merge(ref[[m]], left_index=True, right_index=True,
                                     suffixes=("", "_pop")).dropna()
                    p = (np.nan if med == REFERENCE
                         else wilcoxon_paired(j[m], j[f"{m}_pop"]))
                    # the question the reviewer asks: does personalization beat the
                    # population model at this support size, and is it real?
                    r[f"{m}_beats_pop"] = bool(len(j) and
                                               j[m].mean() > j[f"{m}_pop"].mean())
                    r[f"{m}_sig_vs_pop"] = bool(np.isfinite(p) and p < 0.05)
                rows.append(r)
    return pd.DataFrame(rows)
