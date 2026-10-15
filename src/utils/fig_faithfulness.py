"""Faithfulness figure -- formula swap, concept ablation, and image swap.
    mode = {"bar", "raincloud"->half violin KDE+boxplot}
    show_sig = True/False
    
    Use:
    uv run python -m src.utils.fig_faithfulness              # bars + sig
    uv run python -m src.utils.fig_faithfulness raincloud    # raincloud + sig
    uv run python -m src.utils.fig_faithfulness bars nosig   # bars ไม่มี sig เก่า
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import PROJECT_ROOT
from src.data.data import DOMAINS
from src.utils.plots import COLOR, grid, save, setup, FULL_WIDTH, COL_WIDTH
from src.utils.metrics import participant_cluster_bootstrap

GROUPS = DOMAINS + [None]
XLABELS = ["Artwork", "Fashion", "Landscape", "Average"]

def _get_holm_sig(holm_df: pd.DataFrame):
    pass

def stat(series: pd.Series, err: str = "sem") -> tuple[float, float]:
    v = series.to_numpy(float)
    v = v[np.isfinite(v)]
    m, sd = v.mean(), v.std(ddof=1)
    return m, (sd / np.sqrt(len(v)) if err == "sem" else sd)


def grouped(df: pd.DataFrame, col: str, err: str) -> tuple[np.ndarray, np.ndarray]:
    means, errs = [], []
    for dom in GROUPS:
        d = df if dom is None else df[df.domain == dom]
        if dom is None and err == "sem":
            result = participant_cluster_bootstrap(d[col], d["user_id"])
            m, e = result["mean"], result["sem"]
        else:
            m, e = stat(d[col], err)
        means.append(m)
        errs.append(e)
    return np.array(means), np.array(errs)


def panel_swap(ax, formula_anc: pd.DataFrame, formula_plain: pd.DataFrame, err: str):
    x = np.arange(len(XLABELS))
    width = 0.125

    # Pop-Anchored (Primary: Solid BLUE tones)
    m_own_a, e_own_a = grouped(formula_anc, "own_srocc", err)
    m_pop_a, e_pop_a = grouped(formula_anc, "pop_srocc", err)
    m_oth_a, e_oth_a = grouped(formula_anc, "other_srocc", err)

    # Plain (Comparison: Soft Grey with academic Hatch)
    m_own_p, e_own_p = grouped(formula_plain, "own_srocc", err)
    m_pop_p, e_pop_p = grouped(formula_plain, "pop_srocc", err)
    m_oth_p, e_oth_p = grouped(formula_plain, "other_srocc", err)

    # Draw 6 bars per x-location:
    ax.bar(x - 2.5 * width, m_own_a, width, yerr=e_own_a, capsize=2, label="Own (Anchor)",
           color=COLOR["blue"], edgecolor="white", linewidth=0.4)
    ax.bar(x - 1.5 * width, m_pop_a, width, yerr=e_pop_a, capsize=2, label="Pop (Anchor)",
               color=COLOR["blue_mid"], edgecolor="white", linewidth=0.4)
    ax.bar(x - 0.5 * width, m_oth_a, width, yerr=e_oth_a, capsize=2, label="Other (Anchor)",
               color=COLOR["blue_light"], edgecolor="white", linewidth=0.4)

    ax.bar(x + 0.5 * width, m_own_p, width, yerr=e_own_p, capsize=2, label="Own (Plain)",
           color=COLOR["grey"], hatch="///", edgecolor="white", linewidth=0.4)
    ax.bar(x + 1.5 * width, m_pop_p, width, yerr=e_pop_p, capsize=2, label="Pop (Plain)",
           color=COLOR["grey_mid"], hatch="///", edgecolor="white", linewidth=0.4)
    ax.bar(x + 2.5 * width, m_oth_p, width, yerr=e_oth_p, capsize=2, label="Other (Plain)",
           color=COLOR["grey_light"], hatch="///", edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=7)
    # ax.set_ylabel("Spearman SROCC")
    ax.set_title("(a) Formula Swap (Personalization)", fontsize=8.5)
    ax.set_ylim(0.28, 0.59)
    grid(ax)
    handles, labels = ax.get_legend_handles_labels()
    order = 0, 3, 1, 4, 2, 5
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=6, loc="upper right", ncol=3, frameon=False, handlelength=0.8, columnspacing=0.6, bbox_to_anchor=(1.0, 1.16))


def panel_ablation(ax, abl: pd.DataFrame, err: str):
    full_col = "full_srocc" if "full_srocc" in abl.columns else "base_srocc"
    series = {
        "Full Model": grouped(abl, full_col, err),
        "w/o Bottom": grouped(abl, "bottom1_srocc", err),
        "w/o Avg": grouped(abl, "avg1_srocc", err),
        "w/o Top": grouped(abl, "top1_srocc", err),
    }
    colors = {
        "Full Model": COLOR["blue"],
        "w/o Bottom": COLOR["amber_light"],
        "w/o Avg": COLOR["amber_mid"],
        "w/o Top": COLOR["amber"]
    }

    n = len(series)
    width = 0.125 #0.8 / n
    x = np.arange(len(XLABELS))
    for i, (label, (means, errs)) in enumerate(series.items()):
        off = (i - (n - 1) / 2) * width
        ax.bar(x + off, means, width, yerr=errs, capsize=2, label=label,
               color=colors[label], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=7)
    ax.set_title("(b) Concept Ablation (Importance)", fontsize=8.5, y=0.88)
    ax.set_ylim(0.28, 0.58)
    grid(ax)
    ax.legend(fontsize=6, loc="upper right", ncol=4, frameon=False, handlelength=0.8, columnspacing=0.6, bbox_to_anchor=(1.0, 1.05))


def panel_image_swap(ax, anchored: pd.DataFrame, err: str):
    x = np.arange(len(XLABELS))
    width = 0.125
    series = [
        ("Correct Image", anchored, "correct_srocc", COLOR["blue"]),
        ("No-Image", anchored, "no_image_srocc", COLOR["blue_mid"]),
        ("Swapped Image", anchored, "swapped_srocc", COLOR["blue_light"]),
    ]
    offsets = [-1.0, 0.0, 1.0]
    for off, (label, frame, col, color) in zip(offsets, series):
        means, errs = grouped(frame, col, err)
        ax.bar(x + off * width, means, width, yerr=errs, capsize=2,
               label=label, color=color, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=7)
    # ax.set_ylabel("Spearman SROCC")
    ax.set_title("(c) Image Swap (Stimulus Specificity)", fontsize=8.5, y=0.88)
    ax.set_ylim(0.28, 0.58)
    grid(ax)
    ax.legend(fontsize=6, loc="upper right", ncol=3, frameon=False,
              handlelength=0.8, columnspacing=0.6,
              bbox_to_anchor=(1.0, 1.05))


def run(cfg, err: str = "sem", mode: str = "raincloud", show_sig: bool = True):
    setup()
    out_dir = cfg.output_dir / "faithfulness"
    holm_df = pd.read_csv(PROJECT_ROOT / "output" / "statistics" / "holm_corrected.csv")
    
    abl = pd.read_csv(out_dir / "ablation.csv")
    formula_anc = pd.read_csv(out_dir / "formula_swap.csv")
    formula_plain = pd.read_csv(out_dir / "formula_swap_unanchored.csv")
    image_anc= pd.read_csv(out_dir / "image_swap.csv")
    image_plain = pd.read_csv(out_dir / "image_swap_unanchored.csv")

    fig, axes = plt.subplots(3, 1, figsize=(COL_WIDTH, 3.4), sharex=True)
    panel_swap(axes[0], formula_anc, formula_plain, err)
    panel_ablation(axes[1], abl, err)
    panel_image_swap(axes[2], image_anc, err)
    
    fig.supylabel("Spearman SROCC", fontsize=8, x=-0.054)
    fig.subplots_adjust(hspace=0.28, left=0.07, right=0.98, top=0.90, bottom=0.22)
    
    save(fig, cfg.figures_dir / f"fig_faithfulness_{mode}_{err}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.config import Config

    cfg = Config()
    run(cfg, "sem")
