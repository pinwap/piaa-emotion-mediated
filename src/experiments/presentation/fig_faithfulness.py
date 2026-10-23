"""Faithfulness figure -- formula swap, concept ablation, and image swap.
    mode = {"bar", "raincloud"->half violin KDE+boxplot}
    show_sig = draw Holm-corrected significant brackets
    
    Use:
    uv run python -m src.utils.fig_faithfulness                     # raincloud + sig
    
    domain (conditions)
    legend_label = for color (Own, Pop, Other)/(Full, Average, Bottom, Top)
    
    EDIT:
    add/delete graph          -> add/ delete Condition() and related Comparison()
    add/delete signification  -> add/delete Comparison()
    swap position             -> swap Condtion order
    change width, size        -> LAYOUT 
        step : x distance between conditions (None = share 0.84 in group)
        violin_frac, box_frac : width as a fraction of step
        bracket_h : y height of bracket's leg
        bracket_gap : y distance between stacked bracket level
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_PATH, Config
from src.data.data import DOMAINS, DOMAIN_LABEL
from src.utils.plots import *

log = logging.getLogger(__name__)

DOMAIN_KEYS = DOMAINS + ["average"]
X_LABELS = [DOMAIN_LABEL[d] for d in DOMAINS] + ["Average"]
GROUP_CENTERS = np.arange(len(DOMAIN_KEYS), dtype=float) # = 0 1 2 3
BAR_YMIN = 0.28

LAYOUT = {
    "bars" : dict(step=0.125, box_frac=0.0, bracket_h=0.008, bracket_gap=0.028),
    "raincloud" : dict(step=None, violin_frac=0.65, box_frac=0.35, bracket_h=0.02, bracket_gap=0.14),
}

LEGEND_DEFAULT = dict(fontsize=6, frameon=False, handlelength=0.8, columnspacing=0.6,
                      loc="lower center", bbox_to_anchor=(0.5, 1.0))

def draw_panel(ax, panel: dict, holm: pd.DataFrame|None, 
               mode: str, show_sig: bool, err: str)->None:
    layout = LAYOUT[mode]
    n = len(panel["conditions"])
    step = layout["step"] or 0.84/n
    offsets = (np.arange(n)-(n-1)/2)*step # -> [-step, 0, +step]
    
    for group_index, (domain, group_center) in enumerate(zip(DOMAIN_KEYS,GROUP_CENTERS)):
        group_tops = [] #for significant height reference 
        for (table, col, label, color, hatch), offset in zip(panel["conditions"], offsets):
            domain_rows = table if domain == "average" else table[table["domain"] == domain]
            x = group_center+offset
            legend_label = label if group_index == 0 else None #average
            
            if mode == "bars":
                user_ids = domain_rows["user_id"] if domain == "average" else None 
                top = bar_err(ax, x=x, values=domain_rows[col], color=color, width=step, 
                              err=err, ids=user_ids, label=legend_label, hatch=hatch)
            else:
                lo, top = half_violin(ax, x=x, values=domain_rows[col], color=color, 
                                      width=step*layout["violin_frac"], label=legend_label, hatch=hatch)
                box(ax, x=x, values=domain_rows[col], color=color, width=step*layout["box_frac"])
            group_tops.append(top)
            
        if show_sig:
            base=max(group_tops)+layout["bracket_h"]
            for i, j, contrast, level in panel["comparisons"]:
                star = get_holm_sig(holm_df=holm, table="figure_faithfulness", block=panel["block"], domain=domain, contrast=contrast)
                if star:
                    sig_bracket(ax, x1=group_center+offsets[i], x2=group_center+offsets[j],
                                y=base+level*layout["bracket_gap"], h=layout["bracket_h"],
                                text=star, star_gap=FONT["star_gap"], fontsize=FONT["star"])
        
    handles, labels = ax.get_legend_handles_labels()
    order =  panel.get("legend_order", range(len(handles)))
    ax.set_title(panel["title"], fontsize= FONT["title"],
                 **panel.get("title_detail",{}))
    ax.legend([handles[k] for k in order], [labels[k] for k in order], 
              **{**LEGEND_DEFAULT, **panel.get("ax_legend_detail",{})},
              )

def run(cfg, err: str = "sem", mode: str = "raincloud", show_sig: bool = True,
        width: float = COL_WIDTH, height:float = 4.6):
    setup()
    out_dir = cfg.output_dir / "faithfulness"
    holm_path = DATA_PATH["holm_path"]
    
    holm_df = pd.read_csv(holm_path) if holm_path.exists() else None
    if show_sig and holm_df is None:
        log.warning("%s not found -> figure has no significance brackets", holm_path)
    
    # ----tables-----
    ablation = pd.read_csv(out_dir / "ablation.csv")
    formula_swap_anchored = pd.read_csv(out_dir / "formula_swap.csv")
    formula_swap_plain = pd.read_csv(out_dir / "formula_swap_unanchored.csv")
    image_swap_anchored= pd.read_csv(out_dir / "image_swap.csv")
    image_swap_plain = pd.read_csv(out_dir / "image_swap_unanchored.csv")

    # ----panel detail-----
    panel_swap = {
        "title" : "(a) Formula Swap",
        "block" : "formula_swap",
        "conditions" : [# data, column, label, color, hatch
            # Pop-Anchored (Primary: Solid BLUE tones)
            (formula_swap_anchored, "own_srocc", "Own (Anchor)", COLOR["blue"], None),
            (formula_swap_anchored, "pop_srocc", "Pop (Anchor)", COLOR["blue_mid"], None),
            (formula_swap_anchored, "other_srocc", "Other (Anchor)", COLOR["blue_light"], None),
            # Plain (Comparison: Soft Grey with /// Hatch)
            (formula_swap_plain, "own_srocc", "Own (Plain)", COLOR["grey"], "///"),
            (formula_swap_plain, "pop_srocc", "Pop (Plain)", COLOR["grey_mid"], "///"),
            (formula_swap_plain, "other_srocc", "Other (Plain)", COLOR["grey_light"], "///"),
        ],
            
        "comparisons" : [# condition i, condition j, Holm contrast, bracket level
            (0, 1, "Own vs Population mean [Anchored]", 0),
            (0, 2, "Own vs Other user [Anchored]", 1),
            (3, 4, "Own vs Population mean [Unanchored]", 0),
            (3, 5, "Own vs Other user [Unanchored]", 1),
        ],
        "ax_legend_detail" : dict(ncol=3),
        "legend_order": [0, 3, 1, 4, 2, 5],
        "title_detail" : dict(pad=23),
    }
    panel_ablation = {
        "title" : "(b) Concept Ablation",
        "block" : "concept_ablation",
        "conditions" : [
            (ablation, "full_srocc", "Full Model", COLOR["blue"], None),
            (ablation, "bottom1_srocc", "w/o Bottom", COLOR["amber_light"], None),
            (ablation, "avg1_srocc", "w/o Avg", COLOR["amber_mid"], None),
            (ablation, "top1_srocc", "w/o Top", COLOR["amber"], None),
        ],
            
        "comparisons" : [
            (0, 1, "Full model vs Bottom-weight emotion held", 0),
            (0, 2, "Full model vs Average emotion held", 1),
            (0, 3, "Full model vs Top-weight emotion held", 2),
        ],
        
        "ax_legend_detail" : dict(ncol=4),
        "title_detail" : dict(pad=14),
    }
    panel_image_swap = {
        "title" : "(c) Image Swap",
        "block" : "image_swap",
        "conditions" : [
            (image_swap_anchored, "correct_srocc", "Correct Image", COLOR["blue"], None),
            (image_swap_anchored, "no_image_srocc", "No-Image", COLOR["blue_mid"], None),
            (image_swap_anchored, "swapped_srocc", "Swapped Image", COLOR["blue_light"], None)
        ],
            
        "comparisons" : [
            (0, 1, "Correct image vs No-image emotion", 0),
            (0, 2, "Correct image vs Swapped image", 1),
        ],

        "ax_legend_detail" : dict(ncol=3),
        "title_detail" : dict(pad=14),
    }

    # ----draw-------------------------------------------------
    panels = [panel_swap, panel_ablation, panel_image_swap]
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, height), sharex=True)
    
    for ax, panel in zip(axes, panels):
        draw_panel(ax, panel, holm_df, mode, show_sig, err)
        if mode == "bars":
            ax.set_ylim(bottom=BAR_YMIN)
        grid(ax, axis="y")
        ax.tick_params(labelsize=FONT["tick"])
    axes[-1].set_xticks(GROUP_CENTERS, X_LABELS)
    axes[-1].set_xlim(GROUP_CENTERS[0]-0.5, GROUP_CENTERS[-1]+0.5)       
    
    fig.supylabel("Spearman SROCC", fontsize=FONT["ylabel"])    
    fig.tight_layout(pad=0.3, h_pad=0.6)

    return save(fig, cfg.figures_dir / (f"fig_faithfulness_{mode}_{err}"+("_sig" if show_sig else "")))

if __name__ == "__main__":
    run(Config(), "sem", mode="raincloud", show_sig=True, height=4.6)