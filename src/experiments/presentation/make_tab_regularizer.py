"""Render the descriptive regularizer table from seed-averaged unit results."""
from __future__ import annotations

import argparse
from pathlib import Path
from src.config import PROJECT_ROOT

import pandas as pd

from src.utils.metrics import participant_cluster_bootstrap

UNIT = ["fold", "domain", "user_id"]
ROWS = [("ridge", "Ridge"), ("lasso", "Lasso"),
        ("elastic", "Elastic net")]
BLOCKS = [("population", "Population"), ("identity", "Direct"),
          ("emotion", "Hybrid")]


def _cell(data: pd.DataFrame, metric: str) -> str:
    mean = float(data[metric].mean())
    sem = participant_cluster_bootstrap(
        data[metric], data["user_id"]
    )["sem"]
    text = rf"${mean:.3f}_{{\pm {sem:.3f}}}$"
    return text.replace("0.", ".")


def build(source: Path, destination: Path) -> None:
    data = pd.read_csv(source, low_memory=False)
    required = {*UNIT, "mediator", "head", "srocc", "plcc"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"regularizer units missing columns: {sorted(missing)}")
    if data.duplicated(["mediator", "head", *UNIT], keep=False).any():
        raise ValueError("regularizer units contain duplicate condition-unit keys")
    for head, _ in ROWS:
        for mediator, _ in BLOCKS:
            selected = data[(data["head"] == head) &
                            (data["mediator"] == mediator)]
            if len(selected) != 387:
                raise ValueError(
                    f"{head}/{mediator}: expected 387 units, got {len(selected)}"
                )

    lines = [
        r"\begin{tabular}{@{}l cc cc cc@{}}",
        r"\toprule",
        (r"\multirow{2}{*}{Head} & \multicolumn{2}{c}{Population} & "
         r"\multicolumn{2}{c}{Direct} & \multicolumn{2}{c}{Hybrid} \\"),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"& SROCC & PLCC & SROCC & PLCC & SROCC & PLCC " + r"\\",
        r"\midrule",
    ]
    for head, label in ROWS:
        cells = []
        for mediator, _ in BLOCKS:
            selected = data[(data["head"] == head) &
                            (data["mediator"] == mediator)]
            cells.extend([_cell(selected, "srocc"), _cell(selected, "plcc")])
        lines.append(label + " & " + " & ".join(cells) + " " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {destination}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=PROJECT_ROOT / "output" / "regularizer_units_final.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
