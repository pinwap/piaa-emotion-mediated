"""Formula stability analysis: Cross-Domain consistency and Cross-Support set

own = my formula in domain 1 vs my formula in domain 2
peer = my formula in domain 1 vs other formula in domain 2 (from the same fold, same y_pop, Stage-1, scaler)

Use:
uv run main.py formula_stability

Output: 
    cross_domain_per_user.csv : (129,11) -> user_id, fold, own_{domain1}_{domain2}, peer_{domain1}_{domain2}, own_avg_corr, peer_avg_corr

"""
from __future__ import annotations

from pandas import pd
from numpy import np
from matplotlib import pyplot as plt

from scipy.stats import spearmanr, wilcoxon
import itertools

from src.config import Config, DATA_PATH
from src.data.data import DOMAINS, CORE7
from src.experiments.analysis.make_holm_statistics import holm_adjust

DOMAIN_PAIRS = list(itertools.combinations(DOMAINS, 2))

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
        row = {"user_id": uid, "fold": my_fold, "own_avg_corr": 0, "peer_avg_corr": 0}
        
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
      
def cross_support_stability(formula_df: pd.DataFrame) -> pd.DataFrame:
    
  
def run(cfg: Config | None = None) -> None:
    output_dir = cfg.output_dir / "analysis"
    formula_df = pd.read_csv(cfg.output_dir / "formula_units_final.csv")
    formula_df = formula_df.loc[formula_df["setting"] == "anchored"]
        
    cross_domain_df = cross_domain_stability(formula_df)
    cross_domain_df.to_csv(output_dir / "cross_domain_per_user.csv", index=False)

    cross_support_df = cross_support_stability(formula_df)
    cross_support_df.to_csv(output_dir / "cross_support_per_user.csv", index=False)

if __name__ == "__main__":
    run(Config())