"""Training diagnostics: did each head actually get trained well?

Three panels, all built from files the pipeline already writes, so nothing here
is a separate experiment that could drift from the reported numbers.

  (a) MLP.   Validation-group SROCC against the epoch budget, one line per
             learning rate, with the selected point ringed. A curve that rises
             and then falls is overfitting, and the ring should sit at the
             peak. A curve still rising at the right edge means the budget
             grid stops too early.

  (b) Ridge. Validation-group score against the penalty, with the selected
             point ringed. If the ring sits at either end of the x-axis the
             grid ran out before the search did.

  (c) Both.  Effective degrees of freedom of the fitted personal ridge head,
             per evaluation unit. For a seven-dimensional mediator this runs
             from 0 (the formula has collapsed onto the population anchor) to
             7 (no shrinkage at all). Values in between are the penalty doing
             its job.

Panel (a) and (b) read selection*.csv. Panel (c) reads the results file, so it
works before any re-run.

Usage, from the repo root:
    uv run python -m src.utils.fig_training_diagnostics
    uv run python -m src.utils.fig_training_diagnostics --mlp-log "output/mlp_new/selection*.csv"
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.utils.plots import BLUE, COL_WIDTH, GREY, grid, save, setup

# lines here are ordered by magnitude, not identity, so the ramp is one hue
# light to dark rather than a categorical set. Two readers with different
# colour vision see the same ordering.
RAMP = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"]


def _read(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                     ignore_index=True)


def _shade(i: int, n: int) -> str:
    if n <= 1:
        return BLUE
    return RAMP[round(i * (len(RAMP) - 1) / (n - 1))]


def panel_mlp(ax, log: pd.DataFrame) -> str | None:
    d = log[(log.get("kind") == "selection") & (log.component == "personal_mlp")]
    if d.empty or "epochs" not in d or d["epochs"].isna().all():
        return ("no MLP selection rows with an epoch axis; run the MLP series "
                "after applying patch_mlp_epoch_selection.py")

    # one (fold, domain) would be a single noisy curve, so average the
    # validation score of each candidate over every slice in the log
    m = (d.groupby(["lr", "epochs"], as_index=False)
          .agg(score=("score", "mean"), picked=("selected", "sum")))
    # the budgets are a short ordered list, so they get evenly spaced ticks;
    # a log axis would crowd the last two labels into each other
    budgets = sorted(m["epochs"].unique())
    pos = {e: i for i, e in enumerate(budgets)}
    lrs = sorted(m["lr"].unique())
    for i, lr in enumerate(lrs):
        g = m[m.lr == lr].sort_values("epochs")
        x = [pos[e] for e in g["epochs"]]
        ax.plot(x, g["score"], marker="o", markersize=3,
                color=_shade(i, len(lrs)), label=f"{lr:g}")
        if g["picked"].max() > 0:
            best = g.loc[g["picked"].idxmax()]
            ax.plot(pos[best["epochs"]], best["score"], marker="o", markersize=7,
                    markerfacecolor="none", markeredgewidth=1.2,
                    color=_shade(i, len(lrs)), zorder=5)

    ax.set_xlabel("Epoch budget")
    ax.set_ylabel("Validation-group SROCC")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([f"{int(e)}" for e in budgets])
    ax.set_xlim(-0.35, len(budgets) - 0.65)
    ax.set_title("(a) Personal MLP", loc="left", fontsize=8)
    leg = ax.legend(title="Learning rate", fontsize=6, title_fontsize=6,
                    loc="lower right", ncol=2, handlelength=1.4,
                    borderpad=0.3, labelspacing=0.25, columnspacing=1.0)
    leg._legend_box.align = "left"
    grid(ax)
    return None


def panel_ridge(ax, log: pd.DataFrame) -> str | None:
    d = log[(log.get("kind") == "selection")
            & (log.component.isin(["personal_ridge", "giaa_ridge"]))]
    d = d[d["alpha"].notna()]
    if d.empty:
        return ("no ridge selection rows; run efficiency once after applying "
                "patch_efficiency_selection_log.py")

    comp = ("personal_ridge" if (d.component == "personal_ridge").any()
            else "giaa_ridge")
    d = d[d.component == comp]
    m = (d.groupby("alpha", as_index=False)
          .agg(score=("score", "mean"), picked=("selected", "sum")))
    m = m.sort_values("alpha")
    lower_is_better = bool(d["lower_is_better"].iloc[0])

    ax.plot(m["alpha"], m["score"], marker="o", markersize=3, color=BLUE)
    if m["picked"].max() > 0:
        best = m.loc[m["picked"].idxmax()]
        ax.plot(best["alpha"], best["score"], marker="o", markersize=7,
                markerfacecolor="none", markeredgewidth=1.2, color=BLUE, zorder=5)
        ax.annotate(f"selected $\\alpha$ = {best['alpha']:g}",
                    (best["alpha"], best["score"]), textcoords="offset points",
                    xytext=(0, -16), fontsize=6, color="#222222", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel(r"Ridge penalty $\alpha$")
    ax.set_ylabel("Validation MSE" if lower_is_better
                  else "Validation-group SROCC")
    label = "Personal ridge" if comp == "personal_ridge" else "GIAA ridge"
    ax.set_title(f"(b) {label}", loc="left", fontsize=8)
    grid(ax)
    return None


def panel_dof(ax, raw: pd.DataFrame, mediator: str, width: int) -> str | None:
    d = raw[(raw.mediator == mediator) & (raw["head"] == "ridge")]
    d = d[d["eff_dof"].notna()]
    if d.empty:
        return f"no eff_dof rows for mediator '{mediator}'"

    ns = sorted(d["n_train"].unique())
    data = [d[d.n_train == n]["eff_dof"].to_numpy(float) for n in ns]
    bp = ax.boxplot(data, positions=range(len(ns)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.1),
                    whiskerprops=dict(linewidth=0.6, color="#555555"),
                    capprops=dict(linewidth=0.6, color="#555555"),
                    boxprops=dict(linewidth=0.4, edgecolor="white"))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_shade(i, len(ns)))

    # x in axes fraction, y in data units, so the labels sit on their lines
    # and cannot be pushed outside the panel by the box positions
    tr = ax.get_yaxis_transform()
    ax.axhline(width, color=GREY, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(0.98, width, f"no shrinkage ({width})", transform=tr, fontsize=6,
            ha="right", va="bottom", color="#555555")
    ax.axhline(0, color=GREY, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(0.98, 0, "collapsed onto the anchor (0)", transform=tr, fontsize=6,
            ha="right", va="top", color="#555555")

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"{int(n)}" for n in ns])
    ax.set_xlabel("Support size $N$")
    ax.set_ylabel("Effective degrees of freedom")
    ax.set_ylim(-1.15, width + 0.9)
    ax.set_title("(c) Fitted personal ridge", loc="left", fontsize=8)
    grid(ax)
    return None


def run(cfg=None, *, mlp_log="output/**/selection*.csv",
        ridge_log="output/**/selection*.csv",
        raw="output/raw_all_final.csv", backbone="qwen8b", variant="C",
        mediator="emotion", width=7, out="paper/figures/training_diagnostics"):
    import matplotlib.pyplot as plt
    setup()

    log_m = _read(mlp_log)
    log_r = _read(ridge_log) if ridge_log != mlp_log else log_m
    df = pd.read_csv(raw, low_memory=False)
    df = df[(df.backbone == backbone) & (df.stage2_variant == variant)]

    # draw each panel on a throwaway axis first so a panel with no data yet
    # is left out of the layout entirely rather than printed as an empty box
    wanted = [("(a)", lambda ax: panel_mlp(ax, log_m) if not log_m.empty
               else "no MLP selection log found yet"),
              ("(b)", lambda ax: panel_ridge(ax, log_r) if not log_r.empty
               else "no ridge selection log found yet"),
              ("(c)", lambda ax: panel_dof(ax, df, mediator, width))]

    ready, skipped = [], []
    for tag, draw in wanted:
        probe = plt.figure()
        note = draw(probe.add_subplot(111))
        plt.close(probe)
        (skipped if note else ready).append((tag, draw, note))

    if not ready:
        print("nothing to plot yet:")
        for tag, _, note in skipped:
            print(f"  {tag} {note}")
        return None

    fig, axes = plt.subplots(1, len(ready), figsize=(7.0 * len(ready) / 3, 2.1),
                             squeeze=False)
    for ax, (_tag, draw, _n) in zip(axes[0], ready):
        draw(ax)

    fig.tight_layout(pad=0.4, w_pad=1.4)
    p = save(fig, out)
    for tag, _, note in skipped:
        print(f"  [panel {tag} not drawn] {note}")
    return p


def _wrap(text: str, n: int) -> list[str]:
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-log", default="output/**/selection*.csv")
    ap.add_argument("--ridge-log", default=None)
    ap.add_argument("--raw", default="output/raw_all_final.csv")
    ap.add_argument("--backbone", default="qwen8b")
    ap.add_argument("--variant", default="C")
    ap.add_argument("--mediator", default="emotion")
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--out", default="paper/figures/training_diagnostics")
    a = ap.parse_args()
    run(mlp_log=a.mlp_log, ridge_log=a.ridge_log or a.mlp_log, raw=a.raw,
        backbone=a.backbone, variant=a.variant, mediator=a.mediator,
        width=a.width, out=a.out)