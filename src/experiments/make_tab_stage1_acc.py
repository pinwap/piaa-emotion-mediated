r"""Table: how well the shared Stage-1 extractor recovers each emotion.

Reads output/stage1_emotion_acc/summary.csv, which `main.py
stage1_emotion_acc --backbone qwen8b` writes. Nothing is recomputed here.

This table is about Stage 1 alone -- no personal head is involved -- so it
carries no anchor and no support size. Spread is across the five folds per
domain (and across all fifteen fold-domain estimates for the average), not
across user-domain units, because a fold is the unit a shared extractor is
estimated on.

Usage:  python -m src.experiments.make_tab_stage1_acc
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
SRC = ROOT / "output" / "stage1_emotion_acc" / "summary.csv"

DOMAINS = [("art", "Artwork"), ("fashion", "Fashion"),
           ("landscape", "Landscape"), ("avg", "Average")]

#: printed in this order, which is the paper's concept order
ORDER = ["amused", "distasteful", "impressed", "intellectual",
         "motivated", "nostalgic", "sad"]


def _num(x: float) -> str:
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run `main.py stage1_emotion_acc "
                         f"--backbone qwen8b` first")
    d = pd.read_csv(SRC).set_index("emotion")
    missing = [e for e in ORDER if e not in d.index]
    if missing:
        raise SystemExit(f"summary.csv has no rows for {missing}")

    best = {}
    for key, _lab in DOMAINS:
        for m in ("srocc", "plcc"):
            best[(key, m)] = d[f"{key}_{m}_mean"].max()

    out = [r"\begin{tabular}{l cc cc cc cc}", r"\toprule",
           r"\multirow{2}{*}{Concept} & "
           + " ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}} &" for _, lab in DOMAINS).rstrip("&")
           + r"\\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
           " & " + " & ".join(["SROCC", "PLCC"] * len(DOMAINS)) + r" \\",
           r"\midrule"]

    for e in ORDER:
        cells = []
        for key, _lab in DOMAINS:
            for m in ("srocc", "plcc"):
                mean = d.loc[e, f"{key}_{m}_mean"]
                se = d.loc[e, f"{key}_{m}_sem"]
                body = rf"{_num(mean)}_{{\pm {_num(se)}}}"
                if mean == best[(key, m)]:
                    body = r"\bm{" + body + "}"
                cells.append(f"${body}$")
        out.append(f"\textit{{{e.capitalize()}}} & " + " & ".join(cells) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}"]

    ensure()
    dest = TABLES / "tab4_emotion_accuracy.tex"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")

    print("Stage-1 recovery, average over domains (SROCC / PLCC):")
    for e in ORDER:
        print(f"  {e:14s} {d.loc[e, 'avg_srocc_mean']:.4f} / "
              f"{d.loc[e, 'avg_plcc_mean']:.4f}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
