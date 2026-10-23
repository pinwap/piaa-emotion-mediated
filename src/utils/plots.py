"""Figures -- font/size settings matched to what's actually used in the paper.

Every figure is 7in wide (full two-column page width), 8pt serif font,
saved as both .pdf (for the paper) and .png (for a quick look).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from src.utils.metrics import participant_cluster_bootstrap

FULL_WIDTH = 7.0
COL_WIDTH = 3.3
HEIGHT_2PANEL = 1.9
HEIGHT_3PANEL = 2.2

COLOR = {
    "blue": "#185FA5",     # our model
    "blue_mid": "#5B9BD5", 
    "blue_light": "#AFC9E8",  
    "grey": "#8B8B8B",     # baseline
    "grey_mid": "#B0B0B0",
    "grey_light": "#D0D0D0",
    "amber":  "#8C4A0F",    # content-free control
    "amber_mid": "#BA7517",
    "amber_light": "#E8B87A",
    "teal": "#1D9E75",
    "ink": "#222222",     #text
    "ink_mid": "#555555",
    "ink_light": "#999999",    
}
FONT = dict(title=8, legend=6, tick=6, xlabel=8, ylabel=8,
            annot=6, star=5.5, star_gap=0.6)

def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.3,
        "legend.frameon": False,
        "pdf.fonttype": 42,          
        "text.color": COLOR["ink"],
        "axes.labelcolor": COLOR["ink"],
        "xtick.color":  COLOR["ink"],
        "ytick.color": COLOR["ink"],
        
        # box plot
        "boxplot.meanprops.marker": "x",
        "boxplot.meanprops.markersize": 2.5,
        "boxplot.meanprops.markeredgecolor": COLOR["ink"],
        "boxplot.medianprops.color": COLOR["ink"],
        "boxplot.medianprops.linewidth": 0.5,
        "boxplot.boxprops.linewidth": 0.5,
        "boxplot.whiskerprops.color": COLOR["ink_mid"],
        "boxplot.whiskerprops.linewidth": 0.5,
        "boxplot.capprops.color": COLOR["ink_mid"],
        "boxplot.capprops.linewidth": 0.5,
        "boxplot.flierprops.marker": "o",
        "boxplot.flierprops.markersize": 1.2,
        "boxplot.flierprops.markerfacecolor": COLOR["ink_light"],
        "boxplot.flierprops.markeredgecolor": "none",
        # error bar
        "errorbar.capsize": 1.5,
        "errorbar.elinewidth": 0.6,        
    })


def grid(ax, axis:str = "both") -> None:
    ax.grid(axis=axis, alpha=0.25, linewidth=0.4)


def save(fig, path: str | Path) -> Path:
    """Save both .pdf and .png."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.with_suffix('.pdf').name} and .png")
    return p


# --------holm significance
def get_holm_sig(holm_df: pd.DataFrame | None, table: str, block: str, domain: str, contrast: str, metric: str="srocc")-> str: 
    if holm_df is None:
        return ""
    row = holm_df[(holm_df.table == table) &(holm_df.block == block) & (holm_df.domain == domain) 
                  & (holm_df.contrast == contrast) & (holm_df.metric == metric)]
    if len(row) != 1:
        print(f"Warning: {len(row)} Holm-corrected p-value found for {block}, {domain}, {contrast}, {metric}")
        return ""
    pval = float(row["p_holm"].iloc[0])
    return "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."


# --- draw
def bar_err(ax, x:float, values, color:str, width:float, err="sem",
            ids=None, label:str | None = None, hatch:str | None = None) -> float: #return top of error bar to 
    v = np.asanyarray(values, dtype=float)
    
    """calculate cluster bootstrap SEM for Average domain (ids = 3)"""
    if err=='sem' and ids is not None:
        res = participant_cluster_bootstrap(v, np.asarray(ids))
        mean, e = res["mean"], res["sem"]
    else: #sem in 3 domain or sd
        v=v[np.isfinite(v)] 
        mean, sd = v.mean(), v.std(ddof=1)
        e = sd/np.sqrt(len(v)) if err == "sem" else sd 
          
    ax.bar(x, mean, width, yerr=e, label=label,
           color=color, edgecolor="white", hatch = hatch,
           capsize=1.5, linewidth=0.4, zorder=3,
           error_kw = dict(linewidth=0.6))
    
    return float(mean+e)

def half_violin(ax, x:float, values, color:str, width: float,
                label: str |None= None, hatch: str| None = None,
                alpha: float = 0.85) -> tuple[float, float]:
    """Gausian KDE, Silverman bandwidth at left of x
    return (bottom, top) for size reference
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v,1), np.percentile(v,99)
    pad = (hi - lo)*0.08
    y = np.linspace(lo - pad, hi+pad,200)
    
    dens = gaussian_kde(v, bw_method="silverman")(y)
    dens = dens / dens.max() * width
    
    ax.fill_betweenx(y, x-dens, x, facecolor=color, alpha=alpha, edgecolor="white", 
                     linewidth=0, hatch=hatch, label=label, zorder=2)
    ax.plot(x-dens, y, color=color, linewidth=0.5, zorder=3)
    
    return float(min(y[0], v.min())), float(max(y[-1], v.max()))

def box(ax, x: float, values, color: str, width:float) -> None:
    """Box plot at x; line-median, x=mean, dot=outliner"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    ax.boxplot(
        [v], positions=[x], widths=[width], orientation="vertical",
        patch_artist=True, showmeans=True, manage_ticks=False,
        
        boxprops = dict(facecolor = "white", edgecolor=COLOR["ink"]),
        meanprops = dict(markeredgewidth=0.5),
        
        zorder = 4,
    )

def sig_bracket(ax, x1: float, x2: float, y: float, text: str, h: float, star_gap: float, fontsize: float =5.5) -> None:
    # line coordinate with h; height of the bracket, and significant sign on top
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color=COLOR["ink"], linewidth=0.6, clip_on=False, zorder=7)
    va = "baseline" if '*' in text else "bottom"
    ax.annotate(text, xy=((x1+x2)/2, y+h), xytext=(0, star_gap), textcoords="offset points", 
                ha="center", va=va, fontsize=fontsize, zorder=8)