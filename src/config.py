"""Central config for every experiment.

Anything that can change the reported numbers lives here, once, and gets
dumped as config.json next to every output so a run can be traced back to
its settings later.

ridge_alphas: logspace(-2, 6, 17), 17 values from 1e-2 to 1e6.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = {
    "holm_path": PROJECT_ROOT/ "output" / "statistics" / "holm_corrected.csv"
}

@dataclass
class Config:
    # paths
    data_dir: Path = PROJECT_ROOT / "Dataset" / "maked"
    split_dir: Path = PROJECT_ROOT / "Dataset" / "split_v4_10group"
    features_dir: Path = PROJECT_ROOT / "features" #backbone extracted features
    output_dir: Path = PROJECT_ROOT / "output"
    figures_dir: Path = PROJECT_ROOT / "paper" / "figures"

    # evaluation protocol
    first_session_only: bool = True
    n_folds: int = 5
    n_eval: int = 50        # eval images per user, fixed across n_train
    n_train: int = 100      # ratings per user used to fit the head
    split_seed: int = 42    # per-user split seed (used as seed + user_id)
    min_test: int = 20      # skip a user if fewer eval images remain

    backbone: str = "qwen8b"
    n_jobs: int = -1        # CPU workers (-1 = all available cores, 1 = serial)

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

    # Apply the selected Stage-2 variant to every reported mediator, including
    # the content-free and distribution-valued controls. Omitting a mediator
    # here silently falls back to an unanchored fit for that row.
    stage2_variant_mediators: tuple = ("identity", "pca", "emotion",
                                       "random", "shuffled",
                                       "emotion_sd", "emotion_hist",
                                       "pca35", "random35", "shuffled35",
                                       "emotion_mlp", "emotion_joint")

    # --- the MLP series --------------------------------------------------
    # One hidden layer, the same width for the extractor (Stage-1, features
    # -> 7 concepts) and the predictor (Stage-2, 7 concepts -> 1 score).
    # The reported experiments use a width of 128.
    mlp_hidden: int = 128

    # Fixed epoch budget. Early stopping is disabled at construction so every
    # MLP condition receives the same number of optimization steps.
    mlp_max_iter: int = 500

    # Weight decay is fixed at zero rather than selected.
    mlp_alpha: float = 0.0

    # Learning rate is selected on the validation group, by the same
    # procedure every other hyperparameter uses: mirror the test protocol
    # inside the validation users, average SROCC over their units, freeze the
    # winner. Ascending, so the tie-break toward the smallest step has a
    # well-defined direction.
    #
    # Exact learning-rate grid used by the frozen MLP results. The selected
    # value is chosen on the validation-user group and recorded in each run's
    # selection log.
    mlp_lr_grid: tuple = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
                          3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1)
    # A single value keeps the epoch budget fixed while the learning rate is
    # selected on validation users.
    mlp_epoch_grid: tuple = (500,)

    # Joint Stage-1: L = MSE(score) + lam * MSE(concepts), as in Koh et al.
    # lam = 1 is their equal weighting and the default here.
    joint_concept_weight: float = 1.0

    mediator_width: int = 7

    def dump(self, path: Path) -> None:
        # make config.json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, Path):
                # Keep released run snapshots portable and anonymous. Paths
                # inside this repository are recorded relative to its root;
                # genuinely external paths remain absolute.
                try:
                    d[k] = str(v.resolve().relative_to(PROJECT_ROOT.resolve()))
                except ValueError:
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
