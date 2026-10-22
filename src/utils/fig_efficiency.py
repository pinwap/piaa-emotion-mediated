"""Efficiency curves 

from output/efficiency/<backbone>/raw.csv 

err="sem" -> band is +/- 1 standard error of the mean (sd / sqrt(n))

Output: figures_dir/fig_efficiency_{sd,sem}.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.data import DOMAIN_LABEL, DOMAINS
from src.utils.plots import FULL_WIDTH, COLOR, FONT, grid, save, setup
from src.utils.metrics import participant_cluster_bootstrap

STYLE = {
    "population": dict(color=COLOR["amber_mid"], ls=":", marker=None, label="Population (0 user params)", band=False),
    "identity":   dict(color=COLOR["grey"], ls="--", marker="^", label="Direct (4096 params)", band=True),
    "emotion":   dict(color=COLOR["blue"], ls="-", marker="o", label="Hybrid (7 params)", band=TRUE),
}
LEGEND_DEFAULT = dict(fontsize=6, frameon=False, handlelength=0.8, columnspacing=0.6,
                      loc="lower center", bbox_to_anchor=(0.5, 1.0))

def stats(df: pd.DataFrame, method: str, metric: str, n_list: list[int], 
          err: str, domain: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = df[df.mediator == method]
    if domain is not None:
        d = d[d.domain == domain]
    means, errs = [], []
    for n in n_list:
        dn = d[d.n_train == n]
        v = dn[metric].to_numpy(float)
        ids = dn["user_id"].to_numpy()
        keep = np.isfinite(v)
        v, ids = v[keep], ids[keep]
        means.append(v.mean())
        if err == "sem" and domain is None:
            #Average 3 domain per participant
            result = participant_cluster_bootstrap(v, ids)
            errs.append(result["sem"])
        else:
            sd = v.std(ddof=1)
            errs.append(sd / np.sqrt(len(v)) if err == "sem" else sd)
    return np.array(means), np.array(errs)


def plot_panel(ax, df, metric, n_list, err, domain: str | None = None, annotate_crossover: bool = False):
    final_points = []
    for method, s in STYLE.items():
        means, errs = stats(df, method, metric, n_list, err, domain)
        
        if method == "population":
            ax.plot(n_list, means, color=s["color"], ls=s["ls"], lw=1.3, label=s["label"])
        else:
            ax.plot(n_list, means, color=s["color"], ls=s["ls"], marker=s["marker"],ms=3.5, lw=1.2, label=s["label"])
            # error band
            ax.fill_between(n_list, means - errs, means + errs, color=s["color"], alpha=0.12, lw=0)
            
        final_points.append((means[-1], s["color"]))
    
    if annotate_crossover:
        # Annotate last value by rank
        final_points.sort(key=lambda item: item[0], reverse=True)
        offsets = [5, -1, -5]
        for rank, (value, color) in enumerate(final_points):
            ax.annotate(f"{value:.3f}", xy=(n_list[-1], value), xytext=(6, offsets[rank]),
                       textcoords="offset points", fontsize=FONT["annot"], color=color)

    ax.set_xscale("log")
    ax.set_xticks(n_list)
    ax.set_xticklabels([str(n) for n in n_list])
    ax.tick_params(labelsize=FONT["tick"], pad=2)
    grid(ax)
    ax.set_axisbelow(True)

def run(cfg, err: str = "sem", domain_split: bool = False):
    setup()
    # raw{tag}.csv holds one row per (unit, seed); average seeds before plotting.
    # The fallback supports run directories created with the earlier layout.
    eff = cfg.output_dir / "efficiency"
    src = eff / cfg.backbone / "raw.csv"
    if not src.exists():
        src = eff / "per_unit.csv"
    df = pd.read_csv(src)
    df = df[df["mediator"].isin(STYLE)]
    if "seed" in df.columns:
        df = df.groupby(["n_train", "mediator", "head", "fold", "domain",
                         "user_id"], as_index=False)[
            ["ccc", "srocc", "plcc", "eff_dof"]].mean()

    n_list = sorted(df.n_train.unique().tolist())

    if domain_split:
        domain_keys = DOMAINS + [None]
        titles = [DOMAIN_LABEL[d] for d in DOMAINS] + ["Average"]
        
        # sharey too: with free y-limits fashion looks much closer to the
        # other domains than it is, which is the opposite of the point
        fig, axes = plt.subplots(1, len(domain_keys), figsize=(FULL_WIDTH, 1.35), sharex=True, sharey=True)
        
        for ax, dom, title in zip(axes, domain_keys, titles):
            plot_panel(ax, df, "srocc", n_list, err, domain=dom, annotate_crossover=True)
            
            ax.set_title(title, fontsize=FONT["title"], pad=3)
            ax.set_xlabel("ratings / user", fontsize=FONT["xlabel"], labelpad=2)
        
        axes[0].set_ylabel("SROCC", fontsize=FONT["ylabel"], labelpad=2)
        axes[0].legend(fontsize=FONT["legend"], loc="lower left", framealpha=0.85, handlelength=1.2)
        fig.tight_layout(pad=0.3)
        
        save(fig, cfg.figures_dir / f"fig_efficiency_domains_{err}")
    
    else: #only average 
        fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))
        plot_panel(axes[0], df, "srocc", n_list, err, annotate_crossover=True)
        axes[0].set_ylabel("SROCC")
        axes[0].set_xlabel("ratings / user")
        
        plot_panel(axes[1], df, "plcc", n_list, err, annotate_crossover=True)
        axes[1].set_ylabel("PLCC")
        axes[1].set_xlabel("ratings / user")
        
        axes[1].legend(fontsize=FONT["legend"], loc="lower right")
        fig.tight_layout()
        save(fig, cfg.figures_dir / f"fig_efficiency_{err}")

if __name__ == "__main__":
    from src.config import Config
    run(Config(), "sem", domain_split=True)