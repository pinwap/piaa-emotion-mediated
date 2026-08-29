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

import argparse
import shutil
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


def stage_efficiency_slice(raw_path: Path, output_dir: Path) -> Path:
    """Write the (backbone, anchor) slice fig_efficiency reads.

    Only the three mediators the figure plots, so a stray row cannot change a
    curve, and only the reported anchor.
    """
    raw = pd.read_csv(raw_path, low_memory=False)
    d = raw[(raw.backbone == BACKBONE) & (raw.variant == VARIANT)
            & (raw["head"] == HEAD)
            & raw.mediator.isin(["population", "identity", "emotion"])]
    if d.empty:
        raise SystemExit(f"no {BACKBONE}/{VARIANT} rows to plot")

    n = sorted(d.n_train.unique())
    seeds = sorted(d.seed.unique())
    dest = output_dir / "efficiency" / BACKBONE / "raw.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest, index=False)
    print(f"[efficiency] {len(d)} rows, n_train={n}, seeds={seeds} -> {dest.name}")
    return dest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=ROOT / "output" / "raw_all_final.csv")
    parser.add_argument("--faithfulness-dir", type=Path,
                        default=ROOT / "output" / "faithfulness")
    parser.add_argument("--work-output-dir", type=Path,
                        default=ROOT / "output")
    parser.add_argument("--figures-dir", type=Path, default=PAPER_FIGS)
    args = parser.parse_args(argv)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.backbone = BACKBONE
    cfg.output_dir = args.work_output_dir
    cfg.figures_dir = args.figures_dir

    stage_efficiency_slice(args.raw, cfg.output_dir)

    # Faithfulness figure readers expect this conventional layout.  Copy only
    # the seven derived CSVs into the isolated work tree; never stage over the
    # canonical output directory during release verification.
    staged_faithfulness = cfg.output_dir / "faithfulness"
    staged_faithfulness.mkdir(parents=True, exist_ok=True)
    for source in sorted(args.faithfulness_dir.glob("*.csv")):
        destination = staged_faithfulness / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    # SEM only: the paper reports the standard error of the mean, and a
    # second set of SD figures beside it only invites reading the wrong band
    from src.utils import fig_efficiency
    fig_efficiency.run(cfg, "sem")
    fig_efficiency.run(cfg, "sem", domain_split=True)
    print("[efficiency] figures written")

    from src.utils import fig_faithfulness
    fig_faithfulness.run(cfg, "sem")
    print("[faithfulness] figures written")

    for p in sorted(args.figures_dir.glob("*.pdf")):
        print(f"  {p.name:44s} {p.stat().st_size / 1024:6.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
