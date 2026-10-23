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

import argparse
import sys
from pathlib import Path
from src.config import PROJECT_ROOT
import pandas as pd

SRC = PROJECT_ROOT / "output" / "stage1_emotion_acc" / "summary.csv"

DOMAINS = [("art", "Artwork"), ("fashion", "Fashion"),
           ("landscape", "Landscape"), ("avg", "Average")]

#: printed in this order, which is the paper's concept order
ORDER = ["amused", "distasteful", "impressed", "intellectual",
         "motivated", "nostalgic", "sad"]


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Build the paper summary from the compact fold-domain release table."""
    required = {"fold", "domain", "emotion", "srocc", "plcc"}
    missing = required.difference(per_fold.columns)
    if missing:
        raise ValueError(f"Stage-1 source is missing columns: {sorted(missing)}")
    if len(per_fold) != 105:
        raise ValueError(
            "Stage-1 source must contain 5 folds x 3 domains x 7 emotions "
            f"(105 rows), got {len(per_fold)}"
        )

    rows = []
    for emotion in ORDER:
        d = per_fold[per_fold.emotion == emotion]
        row = {"emotion": emotion}
        for domain, _label in DOMAINS[:3]:
            values = d[d.domain == domain]
            if len(values) != 5:
                raise ValueError(
                    f"expected five fold rows for {emotion}/{domain}, got {len(values)}"
                )
            for metric in ("srocc", "plcc"):
                x = values[metric].astype(float)
                row[f"{domain}_{metric}_mean"] = x.mean()
                row[f"{domain}_{metric}_sd"] = x.std(ddof=1)
                row[f"{domain}_{metric}_sem"] = x.std(ddof=1) / len(x) ** 0.5
        for metric in ("srocc", "plcc"):
            x = d[metric].astype(float)
            row[f"avg_{metric}_mean"] = x.mean()
            row[f"avg_{metric}_sd"] = x.std(ddof=1)
            row[f"avg_{metric}_sem"] = x.std(ddof=1) / len(x) ** 0.5
        rows.append(row)
    return pd.DataFrame(rows)


def _num(x: float) -> str:
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SRC)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "tmp" / "tables" /
                        "tab4_emotion_accuracy.tex")
    args = parser.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(
            f"missing {args.source}; use output/stage1_accuracy.csv from the "
            "compact release, or run `main.py stage1_emotion_acc "
            "--backbone qwen8b` first"
        )
    source = pd.read_csv(args.source)
    d = (summarize(source) if {"fold", "domain", "srocc", "plcc"}
         .issubset(source.columns) else source).set_index("emotion")
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
        out.append(rf"\textit{{{e.capitalize()}}} & " + " & ".join(cells) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}"]

    dest = args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"[stage1 table] wrote {len(ORDER)} emotion rows to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
