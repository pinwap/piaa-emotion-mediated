"""
    1 person have 1 Big Five personal trait - 1 formula(*3 domain) 
    -> Spearman Rank correlation between personal trait and formula weights in each domain
    
    Use:  
    uv run main.py trait_correlation
    
    Output: Heatmap and a DataFrame with the correlation coefficient, p-values, p_holm for each combination of domain, trait, and emotion.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.experiments.analysis.make_holm_statistics import holm_adjust
from src.config import Config
from src.data.data import DOMAINS, CORE7
from src.utils.plots import PANEL_LABEL, COLOR, FONT, setup, save, FULL_WIDTH, heatmap

TRAIT_LABEL = {"extraversion": "Extraversion", "agreeableness": "Agreeableness",         
               "conscientiousness": "Conscientiousness",
               "emotional_stability": "Emotional stability", "openness": "Openness"}
TRAITS = list(TRAIT_LABEL.keys())
EMOTION_LABEL = {"impressed": "Impressed", "intellectual": "Intellectual", "motivated": "Motivated",
                 "amused": "Amused", "nostalgic": "Nostalgic", "sad": "Sad", "distasteful": "Distasteful"}


def big_five_trait(user_df: pd.DataFrame) -> pd.DataFrame:
    # 10 personality questions (0-6 score) based on the Big Five -> 5 traits
    reverse = lambda score: 6-score
    return pd.DataFrame({
        "user_id": user_df["user_id"],
        "extraversion":        (user_df["Q1"] + reverse(user_df["Q6"])) / 2,
        "agreeableness":       (reverse(user_df["Q2"]) + user_df["Q7"]) / 2,
        "conscientiousness":   (user_df["Q3"] + reverse(user_df["Q8"])) / 2,
        "emotional_stability": (reverse(user_df["Q4"]) + user_df["Q9"]) / 2,
        "openness":            (user_df["Q5"] + reverse(user_df["Q10"])) / 2,
    })

def trait_emotion_correlation(trait_df: pd.DataFrame, formula_df: pd.DataFrame) -> pd.DataFrame: 
    weight_cols = [f"{emotion}_w" for emotion in CORE7]
    
    # 1 row -> 1 user 1 domain {5trait} {7emotion_w}
    merged_df = pd.merge(formula_df[["user_id", "domain"] + weight_cols], trait_df, on="user_id")
    
    # Average = average emotion_w across 3 domains -> 1 row -> 1 user  {5trait} {7emotion_w}
    mean_df = merged_df.groupby("user_id")[TRAITS + weight_cols].mean().reset_index()
    mean_df["domain"] = "average"
    
    panels_df = pd.concat([merged_df, mean_df], ignore_index=True)
    
    domain_correlation = []
    for domain in PANEL_LABEL.keys(): # 4 domains: art, fashion, landscape, average
        domain_df = panels_df[panels_df["domain"] == domain] #129 rows: 1 user 1 domain 5trait + 7emotion_w
        for trait in TRAITS:
            for emotion in CORE7:
                corr, p_value = spearmanr(domain_df[trait], domain_df[f"{emotion}_w"])
                domain_correlation.append({
                    "domain": domain,
                    "trait": trait,
                    "emotion": emotion,
                    "correlation": corr,
                    "p_value": p_value
                })
    return pd.DataFrame(domain_correlation) # 4 panel * 5 trait * 7 emotion = 140 rows
    
def draw_panel(panels_df: pd.DataFrame, figure_dir: Path) -> None:
    setup()
    fig, axes = plt.subplots(1, len(panels_df["domain"].unique()), figsize=(FULL_WIDTH, 1.8), sharey=True)
    # Create a heatmap for each domain
    # choose limit from the maximum absolute correlation value across all domains
    limit = np.ceil((panels_df["correlation"].abs().max())*10)/10
    ticks = [-limit, -limit/2, 0, limit/2, limit]
    
    for ax, domain in zip(axes, panels_df["domain"].unique()):
        # data -> matrix[emotion, trait] = correlation
        panel_df = panels_df[panels_df["domain"] == domain]
        correlation_matrix = panel_df.pivot(index="emotion", columns="trait", values="correlation").loc[CORE7, TRAITS].rename(index=EMOTION_LABEL, columns=TRAIT_LABEL)
        p_holm = panel_df.pivot(index="emotion", columns="trait", values="p_holm").loc[CORE7, TRAITS].rename(index=EMOTION_LABEL, columns=TRAIT_LABEL)
        
        image = heatmap(ax=ax, limit=limit,
                        data=correlation_matrix, p_holm=p_holm)
        ax.set_title(PANEL_LABEL[domain], fontsize=FONT["title"], pad=3)
    # Legend
    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01, ticks=ticks)
    colorbar.ax.tick_params(labelsize=FONT["tick"], length=2)
    colorbar.set_label("Spearman correlation", fontsize=FONT["ylabel"])
    colorbar.outline.set_linewidth(0.4)
    
    save(fig, figure_dir / "fig_trait_emotion_correlation")
    
def run(cfg: Config | None = None) -> None:
    users_df = pd.read_csv(cfg.data_dir / "users.csv")
    formulas_df = pd.read_csv(cfg.output_dir / "formula_units_final.csv")
    output_dir = cfg.output_dir / "analysis"
    
    trait_df = big_five_trait(users_df[["user_id"]+[f"Q{i}" for i in range(1, 11)]])
    formulas_df = formulas_df[formulas_df["setting"] == "anchored"]
    
    trait_emo_df = trait_emotion_correlation(trait_df, formulas_df)
    
    # calculate Holm significance 
    trait_holm_df = trait_emo_df.copy()
    trait_holm_df["family"] = trait_holm_df["domain"].where(trait_holm_df["domain"] == "average", "by_domain")
    trait_holm_df["family_size"] = trait_holm_df.groupby("family")["p_value"].transform("size")
    trait_holm_df["p_holm"] = trait_holm_df.groupby("family")["p_value"].transform(holm_adjust)
    
    trait_holm_df.to_csv(output_dir / "trait_emotion_correlation.csv", index=False)
    
    draw_panel(trait_holm_df, cfg.figures_dir)
    
if __name__ == "__main__":
    run(Config())