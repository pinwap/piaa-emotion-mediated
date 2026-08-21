"""Central config for every experiment.

Anything that can change the reported numbers lives here, once, and gets
dumped as config.json next to every output so a run can be traced back to
its settings later.

ridge_alphas: logspace(-2, 3, 11), 11 values from 1e-2 to 1e3.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # paths
    data_dir: Path = PROJECT_ROOT / "Dataset" / "maked"
    split_dir: Path = PROJECT_ROOT / "Dataset" / "split_v4_10group"
    features_dir: Path = PROJECT_ROOT / "features"
    output_dir: Path = PROJECT_ROOT / "output"
    figures_dir: Path = PROJECT_ROOT / "paper" / "figures"

    # evaluation protocol
    first_session_only: bool = True #
    n_folds: int = 5
    n_eval: int = 50        # eval images per user, fixed across n_train
    n_train: int = 100      # ratings per user used to fit the head
    split_seed: int = 42    # per-user split seed (used as seed + user_id)
    min_test: int = 20      # skip a user if fewer eval images remain

    backbone: str = "qwen8b"  #"qwen3-vl-4b", "qwen3-vl-8b"

    # ridge head. The grid runs to 1e6, well past the point where a 7- or
    # 512-feature head on <=100 standardized samples is fully shrunk, so the
    # top of the grid is a safety floor the selector can actually reach rather
    # than a cliff it is cut off before. Matters most for stage2_variant B/C,
    # where full shrinkage lands on the population formula instead of on a
    # constant.
    ridge_alphas: tuple = field(default_factory=lambda: tuple(np.logspace(-2, 6, 17)))

    # Lasso / ElasticNet personal heads. Their penalty is on a different scale
    # from ridge's: with standardized features the smallest alpha that zeroes
    # every coefficient is order 1, so the grid runs from far below any useful
    # penalty up past full sparsity. Same tie-break rule as ridge (strongest
    # penalty wins a tie), which here means the sparsest model.
    sparse_alphas: tuple = field(default_factory=lambda: tuple(np.logspace(-4, 1, 17)))
    elastic_l1_ratio: float = 0.5
    sparse_max_iter: int = 5000

    # Stage-2 variant (how the personal head relates to the population model):
    #   plain  ordinary ridge on the mediator, shrinks toward 0
    #   A      append the GIAA prediction as an extra feature, so w_pop=1 with
    #          all other weights 0 reproduces the population model exactly
    #   B      shrink the weights toward w_pop (the pooled training-group
    #          Stage-2 coefficients) instead of toward 0
    #   C      fit the head on the residual y - y_pop
    stage2_variant: str = "C"

    # Which mediators the variant applies to. The reviewer instruction is to
    # apply it to every mediator, Random and Shuffled included: if only Hybrid
    # (or only Hybrid/Direct/PCA) carried the population prior, any edge those
    # rows show would be a fact about who got the prior, not about the
    # mediator's own content. An earlier version of this list excluded Random
    # and Shuffled on the reasoning that a content-free control "shouldn't
    # benefit" from GIAA -- that reasoning was wrong: withholding the anchor
    # from them is itself the confound the paper is trying to avoid, so every
    # mediator gets the same treatment and the comparison stays about content.
    # Every mediator that can be a row in a variant table has to be listed,
    # including the distribution-valued Stage-1s and the Stage-1 capacity
    # variants. Leaving one out does not turn its anchor off cleanly -- it
    # silently runs that row *unanchored* while the rows next to it are
    # anchored, so a column headed "anchor C" would be comparing two
    # different methods.
    stage2_variant_mediators: tuple = ("identity", "pca", "emotion",
                                       "random", "shuffled",
                                       "emotion_sd", "emotion_hist",
                                       "pca35", "random35", "shuffled35",
                                       "emotion_mlp", "emotion_joint")

    # --- the MLP series --------------------------------------------------
    # One hidden layer, the same width for the extractor (Stage-1, features
    # -> 7 concepts) and the predictor (Stage-2, 7 concepts -> 1 score).
    # 128 and 256 were both offered; 128 is what we run, and the paper says so.
    mlp_hidden: int = 128

    # Fixed epoch budget, fixed in advance rather than tuned. early_stopping
    # is off, and tol=0 with n_iter_no_change=mlp_max_iter is set at
    # construction so the budget is actually spent: max_iter on its own does
    # not guarantee it, because sklearn also halts on a training-loss plateau,
    # and a row that quietly stopped at epoch 80 while its neighbour ran 500
    # is not the controlled comparison this table is for.
    mlp_max_iter: int = 500

    # Step size and weight decay, selected *together* on the validation user
    # group, by the same procedure and the same criterion as the ridge
    # penalty. Selecting the step size but not the penalty would give ridge a
    # 17-point regularization search and the MLP none, on a network with far
    # more parameters than samples -- the table would then be reporting that
    # handicap rather than the model family. Small grids because a Stage-1
    # fit on 4096-d features costs ~25 s; the values are stated in the paper.
    mlp_lr_grid: tuple = (1e-3, 3e-3, 1e-2)
    mlp_alpha_grid: tuple = (1e-3, 1e-1, 1e1)

    # Joint Stage-1: loss = MSE(concepts) + w * MSE(score). Fixed at 1 in
    # advance, not tuned. The two targets sit on comparable scales (emotion
    # sd ~= 0.55, score sd ~= 0.70), so w = 1 is close to equal weighting.
    joint_score_weight: float = 1.0

    mediator_width: int = 7

    def dump(self, path: Path) -> None:
        # make config.json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                # numeric grids dump as floats; name lists (e.g.
                # stage2_variant_mediators) dump as-is
                d[k] = [float(x) if isinstance(x, (int, float)) else x
                        for x in v]
            else:
                d[k] = v
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_dir(self, name: str) -> Path:
        d = Path(self.output_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        self.dump(d / "config.json")
        return d
