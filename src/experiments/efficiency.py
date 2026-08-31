"""How much does the mediator help when a user has few ratings?

Sweeps support size n (10/25/50/100) against a fixed eval set, so results are
comparable across budgets, and reports every mediator against the population
(GIAA) baseline.

The sweep uses the same ``Pipeline.run_grid`` path as the primary comparison.
The personal hyperparameter is frozen per (fold, domain, mediator, head, n)
on validation users, and the population row is the shared raw-feature model.

`--stage2 B/C` anchors the personal head on GIAA, which is the interesting
case here: it asks at what support size a personalized model stops losing to
the population model.

Output: output/efficiency/<backbone>/raw{tag}.csv, summary{tag}.csv
(two files per run: the raw per-seed rows, and the reported summary)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import selection_log
from src.utils.metrics import mean_sd, participant_cluster_bootstrap
from src.utils.results_db import record

#: mediators the sweep reports. Population is added by run_grid.
MEDIATORS = ["identity", "pca", "emotion"]
REFERENCE = "population"     # what every row is tested against


def _tag(variant, heads, mediators, backbone, folds=None) -> str:
    """Distinct filename per configuration. Everything that changes the
    numbers is in the name, so parallel runs cannot overwrite one another and
    a file can be traced back to the command that made it."""
    v = "" if variant in (None, "plain") else f"_{variant}"
    h = "" if list(heads) == ["ridge"] else "_" + "".join(x[0] for x in heads)
    b = "" if backbone in (None, "clip") else f"_{backbone}"
    # Encode mediator names rather than only their count so distinct subsets
    # cannot share an output filename.
    if list(mediators) == MEDIATORS:
        m = ""
    else:
        m = "_m" + "-".join(x.replace("emotion", "emo") for x in mediators)
        if len(m) > 60:                      # keep paths sane, stay unique
            import hashlib
            m = "_m%d_%s" % (len(mediators),
                             hashlib.md5("|".join(mediators).encode()).hexdigest()[:8])
    # a partial-fold run gets its own file, so folds running as separate
    # processes cannot overwrite one another; ingest_runs.py merges them
    f = "" if folds is None else "_f" + "".join(str(int(x)) for x in sorted(folds))
    return v + h + b + m + f


def run(cfg, pipeline, n_list: list[int], variant: str | None = None,
        heads=None, mediators=None, seeds=(0,), backbone=None,
        folds=None) -> pd.DataFrame:
    # One folder per backbone, so a finished set (4 variants x 4 support
    # sizes x 3 seeds) sits together instead of 50-odd files sharing a flat
    # directory with every other backbone.
    bb = backbone or cfg.backbone
    out_dir = cfg.run_dir("efficiency") / bb
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = variant or cfg.stage2_variant
    heads = list(heads or ["ridge"])
    mediators = list(mediators or MEDIATORS)
    tag = _tag(variant, heads, mediators, backbone or cfg.backbone, folds)

    frames, selections = [], []
    for n in n_list:
        for seed in seeds:
            print(f"[efficiency] n={n} seed={seed} variant={variant} "
                  f"heads={heads} folds={folds or 'all'}", flush=True)
            d = pipeline.run_grid(
                mediators=mediators, heads=heads, n_train=n,
                include_population=True, include_gt_upper_bound=False,
                seed=seed, stage2_variant=variant, folds=folds)
            d["n_train"], d["seed"], d["stage2_variant"] = n, seed, variant
            frames.append(d)
            selections.extend(pipeline.selection_records)
            record(d, "efficiency", db_path=cfg.output_dir / "raw_all.csv",
                   backbone=backbone or cfg.backbone,
                   variant=variant, n_train=n)

    # Store seed-level rows and the derived summary; seed-averaged units are
    # reconstructed from the raw file when needed.
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out_dir / f"raw{tag}.csv", index=False)
    # Retain every evaluated hyperparameter candidate. ``edge`` identifies
    # selections on a search-grid boundary.
    if selections:
        selection_log.write(selections, out_dir / f"selection{tag}.csv")
        print(f"[efficiency] selection log -> selection{tag}.csv "
              f"({len(selections)} rows)")

    key = ["n_train", "mediator", "head", "fold", "domain", "user_id"]
    df = raw.groupby(key, as_index=False)[["ccc", "srocc", "plcc", "eff_dof"]].mean()

    if folds is not None:
        # Partial fold runs are merged from their raw rows before summarizing.
        print(f"[efficiency] folds {folds} written -> raw{tag}.csv ({len(raw)} rows)")
        return df

    summary = summarize(df, n_list)
    summary.to_csv(out_dir / f"summary{tag}.csv", index=False)
    print(f"[efficiency] wrote {len(summary)} summary rows to "
          f"{out_dir / f'summary{tag}.csv'}")
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
                    s_users = s.index.get_level_values("user_id")
                    r[f"{m}_sem"] = participant_cluster_bootstrap(
                        s[m], s_users)["sem"]
                    j = s[[m]].merge(ref[[m]], left_index=True, right_index=True,
                                     suffixes=("", "_pop")).dropna()
                    p = (np.nan if med == REFERENCE
                         else participant_cluster_bootstrap(
                             j[m] - j[f"{m}_pop"],
                             j.index.get_level_values("user_id"))["p_value"])
                    # Direction and uncorrected paired test versus Population.
                    r[f"{m}_beats_pop"] = bool(len(j) and
                                               j[m].mean() > j[f"{m}_pop"].mean())
                    r[f"{m}_sig_vs_pop"] = bool(np.isfinite(p) and p < 0.05)
                rows.append(r)
    return pd.DataFrame(rows)
