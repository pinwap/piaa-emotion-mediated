"""Faithfulness figure -- formula swap, concept ablation, and image swap.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.data import DOMAINS
from src.utils.plots import BLUE, COL_WIDTH, GREY, AMBER, grid, save, setup, FULL_WIDTH

DOM_LABEL = {"art": "Artwork", "fashion": "Fashion", "landscape": "Landscape"}
GROUPS = DOMAINS + [None]
XLABELS = [DOM_LABEL[d] for d in DOMAINS] + ["Average"]


def stat(series: pd.Series, err: str = "sem") -> tuple[float, float]:
    v = series.to_numpy(float)
    v = v[np.isfinite(v)]
    m, sd = v.mean(), v.std(ddof=1)
    return m, (sd / np.sqrt(len(v)) if err == "sem" else sd)


def grouped(df: pd.DataFrame, col: str, err: str) -> tuple[np.ndarray, np.ndarray]:
    means, errs = [], []
    for dom in GROUPS:
        d = df if dom is None else df[df.domain == dom]
        m, e = stat(d[col], err)
        means.append(m)
        errs.append(e)
    return np.array(means), np.array(errs)


def panel_swap(ax, swap_anc: pd.DataFrame, swap_pln: pd.DataFrame, err: str):
    x = np.arange(len(XLABELS))
    width = 0.125

    # Pop-Anchored (Primary: Solid BLUE tones)
    m_own_a, e_own_a = grouped(swap_anc, "own_srocc", err)
    m_pop_a, e_pop_a = grouped(swap_anc, "pop_srocc", err)
    m_oth_a, e_oth_a = grouped(swap_anc, "other_srocc", err)

    # Plain (Comparison: Soft Grey with academic Hatch)
    m_own_p, e_own_p = grouped(swap_pln, "own_srocc", err)
    m_pop_p, e_pop_p = grouped(swap_pln, "pop_srocc", err)
    m_oth_p, e_oth_p = grouped(swap_pln, "other_srocc", err)

    # Draw 6 bars per x-location:
    ax.bar(x - 2.5 * width, m_own_a, width, yerr=e_own_a, capsize=2, label="Own (Anchor)",
           color=BLUE, edgecolor="white", linewidth=0.4)
    ax.bar(x - 1.5 * width, m_pop_a, width, yerr=e_pop_a, capsize=2, label="Pop (Anchor)",
               color="#5B9BD5", edgecolor="white", linewidth=0.4)
    ax.bar(x - 0.5 * width, m_oth_a, width, yerr=e_oth_a, capsize=2, label="Other (Anchor)",
               color="#AFC9E8", edgecolor="white", linewidth=0.4)

    ax.bar(x + 0.5 * width, m_own_p, width, yerr=e_own_p, capsize=2, label="Own (Plain)",
           color=GREY, hatch="///", edgecolor="white", linewidth=0.4)
    ax.bar(x + 1.5 * width, m_pop_p, width, yerr=e_pop_p, capsize=2, label="Pop (Plain)",
           color="#B0B0B0", hatch="///", edgecolor="white", linewidth=0.4)
    ax.bar(x + 2.5 * width, m_oth_p, width, yerr=e_oth_p, capsize=2, label="Other (Plain)",
           color="#D0D0D0", hatch="///", edgecolor="white", linewidth=0.4)

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
        "Full Model": BLUE,
        "w/o Bottom": "#E8B87A",
        "w/o Avg": AMBER,
        "w/o Top": "#8C4A0F"
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
        ("Correct Image", anchored, "correct_srocc", BLUE),
        ("No-Image", anchored, "no_image_srocc", "#5B9BD5"),
        ("Swapped Image", anchored, "swapped_srocc", "#AFC9E8"),
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


def run(cfg, err: str = "sem"):
    setup()
    out_dir = cfg.output_dir / "faithfulness"
    abl = pd.read_csv(out_dir / "ablation.csv")
    swap_anc = pd.read_csv(out_dir / "formula_swap.csv")
    swap_pln = pd.read_csv(out_dir / "formula_swap_unanchored.csv")
    image = pd.read_csv(out_dir / "image_swap.csv")
    image_plain = pd.read_csv(out_dir / "image_swap_unanchored.csv")

    fig, axes = plt.subplots(3, 1, figsize=(COL_WIDTH, 3.4), sharex=True)
    panel_swap(axes[0], swap_anc, swap_pln, err)
    panel_ablation(axes[1], abl, err)
    panel_image_swap(axes[2], image, err)
    
    # Tight margins, minimal horizontal gap between panels
    # fig.tight_layout()
    fig.supylabel("Spearman SROCC", fontsize=8, x=-0.054)
    fig.subplots_adjust(hspace=0.28, left=0.07, right=0.98, top=0.90, bottom=0.22)
    
    save(fig, cfg.figures_dir / f"fig_faithfulness_{err}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.config import Config

    cfg = Config()
    run(cfg, "sem")
