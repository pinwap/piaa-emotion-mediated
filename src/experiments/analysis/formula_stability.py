"""Formula stability analysis: Cross-Domain consistency and Cross-Sample set

Cross-Domain: 
    own = my formula in domain 1 vs my formula in domain 2
    peer = my formula in domain 1 vs other formula in domain 2 (from the same fold, same y_pop, Stage-1, scaler)

Cross-Sample:
    own = my formula from subset 1 vs my formula from subset 2 (same partition, same fold, same domain)
    peer = my formula from subset 1 vs other formula from subset 2 

Significant test:
    Wilcoxon p-value of the Own-Peer different
    Bootstrap 

Use:
    uv run main.py formula_stability
    uv run main.py fig_formula_stability   (redraw the figure only)

Output: 
    cross_domain_per_user.csv : (129,11) -> user_id, fold, own_{domain1}_{domain2}, peer_{domain1}_{domain2}, own_avg_corr, peer_avg_corr
    cross_sample_per_user.csv : (129,70) -> user_id, fold, own_{domain}_{partition}, peer_{domain}_{partition}, own_avg_{domain}, peer_avg_{domain}, own_avg, peer_avg
    formula_stability_summary.csv : (8,17) -> 1 row per own-vs-peer test: own/peer/excess mean + bootstrap CI, Wilcoxon p, n_own_greater, family_size, p_holm
    
    paper/figures/fig_formula_stability.pdf/.png
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import spearmanr, wilcoxon
import itertools

from src.config import Config
from src.data.data import DOMAINS, CORE7
from src.experiments.analysis.make_holm_statistics import holm_adjust
from src.experiments.models.formulas import SUBSETS, N_PARTITIONS
from src.utils.plots import setup, save, COL_WIDTH, COLOR, FONT, get_holm_sig

DOMAIN_PAIRS = list(itertools.combinations(DOMAINS, 2))
SUBSET_PAIRS = list(itertools.combinations(SUBSETS,2))
COMPARISON_LABEL = {"average": "Avg.", 
                    "art_fashion": "Art–\nFash.", "art_landscape": "Art–\nLand.", "fashion_landscape": "Fash.–\nLand.",
                    "art": "Art", "fashion": "Fash.", "landscape": "Land."}

def cross_domain_stability(formula_df: pd.DataFrame) -> pd.DataFrame:
    # 1 row = 1 fold, 1 user, 1 domain, formula = list of 7 delta in CORE7 order
    formula = formula_df[[f"{i}_w" for i in CORE7]].to_numpy().tolist()
    formula_df = formula_df.loc[:,["fold","user_id","domain"]]
    formula_df["formula"] = formula

    rows = []
    for uid in formula_df["user_id"].unique():
        my_df = formula_df.loc[formula_df["user_id"]==uid]
        my_fold = my_df["fold"].iloc[0]
        other_df = formula_df[(formula_df["fold"]==my_fold) & (formula_df["user_id"]!=uid)]
        row = {"user_id": uid, "fold": my_fold}
        
        for d1, d2 in DOMAIN_PAIRS:
            my_form1 = my_df.loc[my_df["domain"]==d1, "formula"].iloc[0]
            my_form2 = my_df.loc[my_df["domain"]==d2, "formula"].iloc[0]
            
            # Own
            row[f"own_{d1}_{d2}"] = spearmanr(my_form1, my_form2)[0]
            
            # Peer
            corrs = []
            for other_id in other_df["user_id"].unique():
                other = other_df[other_df["user_id"]==other_id] #each other: 3domain formulas
                
                other_form1 = other.loc[other["domain"]==d1, "formula"].iloc[0]
                other_form2 = other.loc[other["domain"]==d2, "formula"].iloc[0]
                
                corrs.append(spearmanr(my_form1, other_form2)[0])
                corrs.append(spearmanr(my_form2, other_form1)[0])
            row[f"peer_{d1}_{d2}"] = np.mean(corrs)
            
        rows.append(row)
    result_df = pd.DataFrame(rows)
    result_df["own_avg_corr"] = result_df[[f"own_{d1}_{d2}" for d1, d2 in DOMAIN_PAIRS]].mean(axis=1)
    result_df["peer_avg_corr"] = result_df[[f"peer_{d1}_{d2}" for d1, d2 in DOMAIN_PAIRS]].mean(axis=1)
    result_df["excess"] = result_df["own_avg_corr"] - result_df["peer_avg_corr"]
    return result_df
      
def cross_sample_stability(sample_df: pd.DataFrame) -> pd.DataFrame:
    # 1 row = (1 paritition, 1 fold, 1 user, 1 domain, 1 subset), formula               
    formula = sample_df[[f"{i}_w" for i in CORE7]].to_numpy().tolist()
    sample_df = sample_df.loc[:, ["fold", "user_id", "domain", "partition", "subset"]]
    sample_df["formula"] = formula
    
    formula_of = {}
    for row, formula in zip(sample_df.itertuples(), sample_df["formula"]):
        formula_of[(row.user_id, row.domain, row.partition, row.subset)]= formula
    fold_of = sample_df.groupby("user_id")["fold"].first().to_dict()
    
    rows = []
    for user_id, my_fold in fold_of.items():
        # make all peer for this user
        others = []
        for other_id, fold in fold_of.items():
            if fold == my_fold and other_id != user_id:
                others.append(other_id)
        
        row = {"user_id": user_id, "fold": my_fold}
        
        for dom in DOMAINS:
            for partition in range(N_PARTITIONS):
                own, peer = [], []
                for s1, s2 in SUBSET_PAIRS:
                    # Own
                    my_form1 = formula_of[(user_id, dom, partition,s1)]
                    my_form2 = formula_of[(user_id, dom, partition,s2)]
                    own.append(spearmanr(my_form1, my_form2)[0])
                    
                    # Peer
                    # every other user in the same fold same domain same partition: mean all subset pairs
                    for other_id in others:
                        other_form1 = formula_of[(other_id, dom, partition,s1)]
                        other_form2 = formula_of[(other_id, dom, partition,s2)]
                        
                        peer.append(spearmanr(my_form1, other_form2)[0])
                        peer.append(spearmanr(my_form2, other_form1)[0])
                row[f"own_{dom}_{partition}"] = np.mean(own)
                row[f"peer_{dom}_{partition}"] = np.mean(peer)
        rows.append(row) 
    result_df = pd.DataFrame(rows)
    
    # average 10 partition
    for dom in DOMAINS:
        result_df[f"own_avg_{dom}"] = result_df[[f"own_{dom}_{partition}" for partition in range(N_PARTITIONS)]].mean(axis=1)
        result_df[f"peer_avg_{dom}"] = result_df[[f"peer_{dom}_{partition}" for partition in range(N_PARTITIONS)]].mean(axis=1)
    # average all domain
    result_df["own_avg"] = result_df[[f"own_avg_{dom}" for dom in DOMAINS]].mean(axis=1)
    result_df["peer_avg"] = result_df[[f"peer_avg_{dom}" for dom in DOMAINS]].mean(axis=1)
    return result_df

def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """95% CI of the mean: resample the users with replacement 10,000 times (seed 0)."""
    rng = np.random.default_rng(0)
    bootstrap_means = [rng.choice(values, len(values)).mean() for _ in range(10_000)]
    return np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5)

def significant_cal(own: pd.Series, peer: pd.Series, task:str, comparison: str, family: str) -> dict:
    #own>peer? : p_holm, bootstrap 95% CI of the mean excess (10,000 resamples, seed 0)  
    excess = (own-peer).to_numpy()
    own_ci_low, own_ci_high = bootstrap_ci(own.to_numpy())
    peer_ci_low, peer_ci_high = bootstrap_ci(peer.to_numpy())
    excess_ci_low, excess_ci_high = bootstrap_ci(excess)
    return {
        "task": task, "comparison": comparison, "family": family,
        "n_users": len(excess),
        "own_mean": own.mean(), "own_ci_low": own_ci_low, "own_ci_high": own_ci_high,
        "peer_mean": peer.mean(), "peer_ci_low": peer_ci_low, "peer_ci_high": peer_ci_high,
        "excess_mean": excess.mean(), "excess_ci_low": excess_ci_low, "excess_ci_high": excess_ci_high,
        "p_value": wilcoxon(own, peer, method="approx").pvalue,
        "n_own_greater": int((excess > 0).sum()),
    }

def draw_figure(summary_df: pd.DataFrame, figure_dir)-> None:
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH, 1.75), sharey=True)
    task_title = {"cross_domain": "Cross-domain", "cross_sample": "Cross-sample"}   
    for ax, (task, panels_df) in zip(axes, summary_df.groupby("task", sort=False)):   
        ax.set_title(task_title[task], fontsize=FONT["title"], pad=2)
        for x, row in enumerate(panels_df.itertuples()):
            ax.plot([x, x], [row.peer_mean, row.own_mean], color=COLOR["grey_light"], linewidth=2.2,
                    solid_capstyle="round", zorder=1)
            
            ax.errorbar(x, row.own_mean, yerr=[[row.own_mean - row.own_ci_low], [row.own_ci_high - row.own_mean]],
                    fmt="o", markersize=2.8, color=COLOR["blue"], zorder=3, label="Own" if x == 0 else None)
            ax.errorbar(x, row.peer_mean, yerr=[[row.peer_mean - row.peer_ci_low], [row.peer_ci_high - row.peer_mean]],
                    fmt="o", markersize=2.8, color=COLOR["grey"], zorder=3, label="Peer" if x == 0 else None)
            # holm star and n_own_greater
            ax.annotate(get_holm_sig(row.p_holm), xy=(x, row.own_ci_high), xytext=(0, 1),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FONT["star"])
            ax.annotate(f"{row.n_own_greater}/{row.n_users}", xy=(x, row.own_ci_high), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FONT["annot"] - 1.5,
                        color=COLOR["ink_mid"])
        ax.axhline(0, color=COLOR["ink_light"], linewidth=0.5, linestyle="--", zorder=0)
        ax.set_xticks(range(len(panels_df)), [COMPARISON_LABEL[comparison] for comparison in panels_df["comparison"]],
                    fontsize=FONT["tick"] - 0.5)
        ax.set_xlim(-0.5, len(panels_df) - 0.5)
        ax.tick_params(axis="both", labelsize=FONT["tick"], length=2, pad=1.5)
    
    axes[0].set_ylim(-0.05, 0.58)
    axes[0].set_ylabel("Formula agreement (ρ)", fontsize=FONT["ylabel"])
    axes[0].legend(fontsize=FONT["legend"], loc="upper left", handletextpad=0.2, borderaxespad=0.1)
    fig.subplots_adjust(wspace=0.08)
    save(fig, figure_dir / "fig_formula_stability")
    
def run(cfg: Config) -> None:
    output_dir = cfg.output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cross-Domain stability
    formula_df = pd.read_csv(cfg.output_dir / "formula_units_final.csv")
    formula_df = formula_df.loc[formula_df["setting"] == "anchored"]
        
    cross_domain_df = cross_domain_stability(formula_df)
    cross_domain_df.to_csv(output_dir / "cross_domain_per_user.csv", index=False)
    
    # Cross-Sample stability 
    subset_df = pd.read_csv(cfg.output_dir / "formulas" / "subset_formulas.csv")
    cross_sample_df = cross_sample_stability(subset_df)
    cross_sample_df.to_csv(output_dir / "cross_sample_per_user.csv", index=False)

    # Holm for average and per domain
    summary = [significant_cal(cross_domain_df["own_avg_corr"], cross_domain_df["peer_avg_corr"],
                         "cross_domain", "average", "cross_domain:average")]
    for domain1, domain2 in DOMAIN_PAIRS:
        summary.append(significant_cal(cross_domain_df[f"own_{domain1}_{domain2}"], cross_domain_df[f"peer_{domain1}_{domain2}"],
                                 "cross_domain", f"{domain1}_{domain2}", "cross_domain:by_part"))
    summary.append(significant_cal(cross_sample_df["own_avg"], cross_sample_df["peer_avg"],
                             "cross_sample", "average", "cross_sample:average"))
    for domain in DOMAINS:
        summary.append(significant_cal(cross_sample_df[f"own_avg_{domain}"], cross_sample_df[f"peer_avg_{domain}"],
                                 "cross_sample", domain, "cross_sample:by_part"))

    summary_df = pd.DataFrame(summary)
    summary_df["family_size"] = summary_df.groupby("family")["p_value"].transform("size")
    summary_df["p_holm"] = summary_df.groupby("family")["p_value"].transform(holm_adjust)
    summary_df.to_csv(output_dir / "formula_stability_summary.csv", index=False)

    draw_figure(summary_df, cfg.figures_dir)
    
def plot_only(cfg: Config) -> None:
    summary_df = pd.read_csv(cfg.output_dir / "analysis" / "formula_stability_summary.csv")
    draw_figure(summary_df, cfg.figures_dir)

if __name__ == "__main__":   
    run(Config() )
    