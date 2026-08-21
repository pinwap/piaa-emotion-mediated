r"""Rebuild every data figure in the paper from the merged results.

The figure modules predate the current output layout: fig_efficiency wants
one `raw.csv` per backbone, while runs now land as `raw{tag}.csv` and are
merged into output/raw_all_final.csv. Rather than teach every figure the new
layout on a deadline, this writes the slice each one expects, from the same
merged file the tables are built from, so a figure and a table can never
disagree about what was run.

Figures are written into the paper repo, not this one.

Usage:  python -m src.experiments.make_paper_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import Config                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
from src.experiments.paper_paths import FIGURES as PAPER_FIGS  # noqa: E402

BACKBONE, VARIANT, HEAD = "qwen8b", "C", "ridge"


def stage_efficiency_slice() -> Path:
    """Write the (backbone, anchor) slice fig_efficiency reads.

    Only the three mediators the figure plots, so a stray row cannot change a
    curve, and only the reported anchor.
    """
    raw = pd.read_csv(ROOT / "output" / "raw_all_final.csv", low_memory=False)
    d = raw[(raw.backbone == BACKBONE) & (raw.variant == VARIANT)
            & (raw["head"] == HEAD)
            & raw.mediator.isin(["population", "identity", "emotion"])]
    if d.empty:
        raise SystemExit(f"no {BACKBONE}/{VARIANT} rows to plot")

    n = sorted(d.n_train.unique())
    seeds = sorted(d.seed.unique())
    dest = ROOT / "output" / "efficiency" / BACKBONE / "raw.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest, index=False)
    print(f"[efficiency] {len(d)} rows, n_train={n}, seeds={seeds} -> {dest.name}")
    return dest


def main() -> int:
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.backbone = BACKBONE
    cfg.figures_dir = PAPER_FIGS

    stage_efficiency_slice()

    # SEM only: the paper reports the standard error of the mean, and a
    # second set of SD figures beside it only invites reading the wrong band
    from src.utils import fig_efficiency
    fig_efficiency.run(cfg, "sem")
    fig_efficiency.run(cfg, "sem", domain_split=True)
    print("[efficiency] figures written")

    from src.utils import fig_faithfulness
    fig_faithfulness.run(cfg, "sem")
    print("[faithfulness] figures written")

    for p in sorted(PAPER_FIGS.glob("*.pdf")):
        print(f"  {p.name:44s} {p.stat().st_size / 1024:6.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
