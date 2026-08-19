"""What the distributional Stage-1 is actually fitted on.

Stage-1 is a shared component: one model per fold, fitted on the training
group's images, then frozen and used for every test user. So its targets are
population quantities, one row per image -- not per-user ratings. This script
draws them, for fold 0, so the two designs can be judged on the data rather
than on the description:

  emotion_hist  the rating histogram per emotion (7 x 5 = 35 numbers/image)
  emotion_sd    the across-rater spread per emotion (7 numbers/image)

Output: paper/figures/fig_stage1_targets.{png,pdf}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config import Config                       # noqa: E402
from src.data.data import CORE7, XpassDataset       # noqa: E402
from src.data.splits import V4Split                 # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
NICE = {"impressed": "Impressed", "intellectual": "Intellectual",
        "motivated": "Motivated", "amused": "Amused",
        "nostalgic": "Nostalgic", "sad": "Sad", "distasteful": "Distasteful"}
BLUE, GREY = "#1565c0", "#37474f"


def main() -> None:
    cfg = Config()
    ds = XpassDataset(cfg.data_dir, first_session_only=cfg.first_session_only)
    split = V4Split(cfg.split_dir)
    fold = split.load_fold(0)

    rows = ds.df[ds.df["user_id"].isin(fold.train_users)]
    giaa = set(map(str, fold.giaa_images))
    rows = rows[rows["stimulus_id"].astype(str).isin(giaa)]

    hist = ds.per_stimulus_hist(rows)          # (n_img, 35)
    sd = ds.per_stimulus_spread(rows)          # (n_img, 7)
    n_img = len(hist)
    n_raters = rows.groupby(rows["stimulus_id"].astype(str)).size()

    print(f"fold 0 Stage-1 training set: {n_img} images, "
          f"{rows['user_id'].nunique()} training users")
    print(f"raters per image: min {n_raters.min()}, median "
          f"{int(n_raters.median())}, max {n_raters.max()}")
    print()
    print("across-rater sd per emotion (what emotion_sd predicts):")
    print(f"  {'emotion':14s} {'mean':>6s} {'sd':>6s} {'min':>6s} {'max':>6s}")
    for c in CORE7:
        v = sd[c].to_numpy(float)
        print(f"  {NICE[c]:14s} {v.mean():6.3f} {v.std():6.3f} "
              f"{v.min():6.3f} {v.max():6.3f}")

    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.32)

    # --- top: the mean histogram Stage-1 is asked to predict ---------------
    for j, c in enumerate(CORE7):
        ax = fig.add_subplot(gs[0, j] if j < 4 else gs[1, j - 4])
        cols = [f"{c}_b{b}" for b in range(1, 6)]
        share = hist[cols].to_numpy(float)
        ax.bar(range(1, 6), share.mean(0), color=BLUE, alpha=.85, width=.72)
        for b in range(5):
            lo, hi = np.percentile(share[:, b], [10, 90])
            ax.plot([b + 1, b + 1], [lo, hi], color=GREY, lw=1.4, alpha=.75)
        ax.set_title(NICE[c], fontsize=10)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, .85)
        ax.grid(axis="y", alpha=.25)
        if j in (0, 4):
            ax.set_ylabel("share of raters")
        ax.set_xlabel("rating", fontsize=8)

    # --- bottom right: the sd distribution --------------------------------
    ax = fig.add_subplot(gs[1, 3])
    ax.boxplot([sd[c].to_numpy(float) for c in CORE7],
               tick_labels=[NICE[c][:4] for c in CORE7], vert=True,
               patch_artist=True,
               boxprops=dict(facecolor="#cfd8dc", edgecolor=GREY),
               medianprops=dict(color=BLUE, lw=1.8),
               flierprops=dict(marker=".", markersize=2, alpha=.3))
    ax.set_title("across-rater sd", fontsize=10)
    ax.set_ylabel("sd")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.grid(axis="y", alpha=.25)

    fig.suptitle(
        f"What Stage-1 is fitted on (fold 0, {n_img} images, population-level)\n"
        "bars = mean share of raters per rating; vertical line = 10th-90th "
        "percentile across images", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_stage1_targets.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nwrote {(OUT / 'fig_stage1_targets.png')}")


if __name__ == "__main__":
    main()
