"""Does the emotion mediator still help as the image backbone gets stronger?

Direct vs. Hybrid on 4 backbones, 100 ratings/user, same users/images so we
can use a paired test. Fine-tuned backbones must load per-fold features
(see modeling/backbones.py) or this leaks.

*** common image set ***
The backbones do not all cover the same images: the CLIP-ft and Qwen-4B
extractions are missing 36 landscape images that frozen CLIP and Qwen-8B
have. Left alone that is 387 units for two backbones and 386 for the other
two, with 16 landscape units drawing different support/eval images -- so a
difference between two rows of this table would be partly a difference in
which images were scored. Every backbone is therefore restricted to the
images all of them share, which is what makes the comparison paired.

Output: output/backbone/per_unit.csv, summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.backbones import backbone_label
from src.utils.metrics import mean_sd, sem, wilcoxon_paired
from src.utils.results_db import record


def common_stimuli(cfg, backbone_names: list[str]) -> set[str]:
    """Stimulus ids covered by every backbone, in every fold."""
    from src.modeling.backbones import get_backbone

    shared: set[str] | None = None
    for name in backbone_names:
        bb = get_backbone(name, cfg.features_dir)
        for k in range(cfg.n_folds):
            got = set(bb.features_for_fold(k))
            shared = got if shared is None else (shared & got)
    return shared or set()


def run(cfg, backbone_names: list[str], variant: str | None = None,
        heads=None) -> pd.DataFrame:
    from main import build

    out_dir = cfg.run_dir("backbone")
    variant = variant or cfg.stage2_variant
    heads = list(heads or ["ridge"])
    tag = ("" if variant in (None, "plain") else f"_{variant}") + \
          ("" if heads == ["ridge"] else "_" + "".join(h[0] for h in heads))

    shared = common_stimuli(cfg, backbone_names)
    print(f"[backbone] common image set across {backbone_names}: {len(shared)}")

    frames = []
    for name in backbone_names:
        print(f"[backbone] {name}")
        ds, _, _, pipe = build(cfg, name)
        ds.restrict_to_features(shared)
        # the population row is what the personalized rows have to beat, so it
        # is computed per backbone rather than assumed constant
        df = pipe.run_grid(mediators=["identity", "emotion"], heads=heads,
                           include_population=True, include_gt_upper_bound=False,
                           stage2_variant=variant)
        df["backbone"] = name
        # run_grid does not stamp the run-level seed (efficiency.py adds it
        # afterwards); this table is a single-seed run, and the results
        # database requires the column
        df["seed"] = 0
        frames.append(df)
        record(df, "backbone", backbone=name, variant=variant,
               n_train=cfg.n_train)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(out_dir / f"per_unit{tag}.csv", index=False)

    summary = summarize(allf, backbone_names)
    summary.to_csv(out_dir / f"summary{tag}.csv", index=False)
    # the full frame is wide; print what answers "does personalization beat
    # the population model on this backbone", read the csv for the rest
    cols = ["label", "head", "n_srocc", "pop_srocc_mean",
            "direct_srocc_mean", "hybrid_srocc_mean",
            "direct_beats_pop_srocc", "direct_sig_vs_pop_srocc",
            "hybrid_beats_pop_srocc", "hybrid_sig_vs_pop_srocc",
            "hybrid_srocc_best", "hybrid_srocc_sig"]
    print(summary[[c for c in cols if c in summary.columns]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return allf


def summarize(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """One row per backbone, with the per-domain columns Table 2 prints plus
    the average over all units. Direct and Hybrid are paired on
    (fold, domain, user_id) before anything is averaged, so the Wilcoxon test
    and the two means are always over the same set of units."""
    rows = []
    for name, head in [(n, h) for n in names
                       for h in sorted(df["head"].unique())]:
        d = df[(df.backbone == name) & (df["head"] == head)]
        if d.empty:
            continue
        idx = ["fold", "domain", "user_id"]
        direct = d[d.mediator == "identity"].set_index(idx)
        hybrid = d[d.mediator == "emotion"].set_index(idx)
        pop = d[d.mediator == "population"].set_index(idx)
        r = dict(backbone=name, label=backbone_label(name), head=head)

        # the comparison the reviewer asks for first: is either personalized model
        # actually ahead of the no-personalization baseline on this backbone?
        for m in ("srocc", "plcc"):
            if pop.empty:
                continue
            r[f"pop_{m}_mean"], _ = mean_sd(pop[m])
            r[f"pop_{m}_sem"] = sem(pop[m])
            for lab, cand in (("direct", direct), ("hybrid", hybrid)):
                j = cand[[m]].merge(pop[[m]], left_index=True, right_index=True,
                                    suffixes=("", "_pop")).dropna()
                p = wilcoxon_paired(j[m], j[f"{m}_pop"])
                r[f"{lab}_beats_pop_{m}"] = bool(len(j) and
                                                 j[m].mean() > j[f"{m}_pop"].mean())
                r[f"{lab}_sig_vs_pop_{m}"] = bool(np.isfinite(p) and p < 0.05)

        for m in ("srocc", "plcc"):
            j = direct[[m]].merge(hybrid[[m]], left_index=True, right_index=True,
                                  suffixes=("_direct", "_hybrid")).dropna()
            # "avg" = every unit pooled; each domain name = that domain only
            groups = [("avg", j)] + [(dom, jd) for dom, jd in
                                     j.groupby(level="domain", sort=True)]
            for tag, jj in groups:
                pre = "" if tag == "avg" else f"{tag}_"
                dm, dsd = mean_sd(jj[f"{m}_direct"])
                hm, hsd = mean_sd(jj[f"{m}_hybrid"])
                p = wilcoxon_paired(jj[f"{m}_hybrid"], jj[f"{m}_direct"])
                r[f"{pre}n_{m}"] = int(len(jj))   # equal across backbones = paired
                r[f"{pre}direct_{m}_mean"], r[f"{pre}direct_{m}_sd"] = dm, dsd
                r[f"{pre}direct_{m}_sem"] = sem(jj[f"{m}_direct"])
                r[f"{pre}hybrid_{m}_mean"], r[f"{pre}hybrid_{m}_sd"] = hm, hsd
                r[f"{pre}hybrid_{m}_sem"] = sem(jj[f"{m}_hybrid"])
                r[f"{pre}hybrid_{m}_best"] = bool(hm >= dm)
                r[f"{pre}hybrid_{m}_sig"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows)
