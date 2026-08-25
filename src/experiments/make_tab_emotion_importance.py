r"""Table: which of the seven concepts carry the judgment.

Reads output/stage2_emotion_importance/summary.csv, written by `main.py
stage2_emotion_importance --backbone qwen8b`. Nothing is recomputed here.

Two blocks, the same seven concepts in each, ordered by their rank within
that block:

  Ground-Truth (GT)   Stage 2 fitted on the user's own emotion ratings
  Predicted (pred)    Stage 2 fitted on Stage 1's predictions

Reading the blocks against each other is the point. A concept can matter to
people and still not matter to the deployable model, if Stage 1 cannot
recover it, and the change in rank between the two blocks is where that
shows up.

Each cell is the mean absolute Stage-2 weight, its SEM across user-domain
units, and in parentheses the share of users whose weight is positive. The
share separates "everybody agrees about this concept" from "people disagree
about its direction": a large mean at a near-even split is a different
finding from a large mean at 98% agreement.

Usage:  python -m src.experiments.make_tab_emotion_importance
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.paper_paths import TABLES, ensure     # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output" / "stage2_emotion_importance" / "summary.csv"

COLS = [("art", "Artwork"), ("fashion", "Fashion"),
        ("landscape", "Landscape"), ("avg", "Average (All)")]
BLOCKS = [("gt", "Ground-Truth (GT)"), ("pred", "Predicted (pred)")]


def _num(x: float) -> str:
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def _cell(mean: float, se: float, pct: float, bold: bool) -> str:
    body = rf"${_num(mean)}_{{\pm {_num(se)}}}$ ({pct:.1f}\%)"
    return r"\textbf{" + body + "}" if bold else body


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run `main.py "
                         f"stage2_emotion_importance --backbone qwen8b` first")
    d = pd.read_csv(SRC)

    out = [r"\begin{tabular}{l cccc c}", r"\toprule",
           r"\textbf{Emotion} & "
           + " & ".join(rf"\textbf{{{lab}}}" for _, lab in COLS)
           + r" & \textbf{Rank} \\", r"\midrule"]

    for bi, (tag, title) in enumerate(BLOCKS):
        block = d[d.setting == tag].sort_values("rank")
        if block.empty:
            raise SystemExit(f"summary.csv has no rows for setting '{tag}'")
        if bi:
            out.append(r"\addlinespace")
        out.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}}}}\\")
        for _, r in block.iterrows():
            cells = [_cell(r[f"{k}_mean"], r[f"{k}_sem"], r[f"{k}_pct_pos"],
                           bold=(k == "avg"))
                     for k, _lab in COLS]
            out.append(f"{r['emotion'].capitalize():<12} & "
                       + " & ".join(cells)
                       + r" & \textbf{" + str(int(r["rank"])) + r"} \\")

    out += [r"\bottomrule", r"\end{tabular}"]

    ensure()
    dest = TABLES / "tab5_emotion_importance.tex"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")

    print("mean |weight| (share positive), average over domains, by rank:")
    for tag, title in BLOCKS:
        print(f"\n  {title}")
        for _, r in d[d.setting == tag].sort_values("rank").iterrows():
            print(f"    {int(r['rank'])}. {r['emotion']:<13} "
                  f"{r['avg_mean']:.3f} +- {r['avg_sem']:.3f} "
                  f"({r['avg_pct_pos']:.1f}% positive)")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
