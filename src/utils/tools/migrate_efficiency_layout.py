"""Move output/efficiency to one folder per backbone, two files per run.

Before: one flat directory, three files per run --
    per_unit{tag}_by_seed.csv   raw, one row per (unit, seed)
    per_unit{tag}.csv           the seed average of the file above
    summary{tag}.csv            what actually gets reported

After:  output/efficiency/<backbone>/raw{tag}.csv and summary{tag}.csv.

The seed-averaged per_unit{tag}.csv is deleted rather than moved: it is one
groupby away from the raw file and nothing reads it any more. Run with
--apply to make the changes; without it the script only prints its plan.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EFF = ROOT / "output" / "efficiency"
BACKBONES = ["clip_ft_emo", "clip_ft", "qwen4b", "qwen8b", "clip"]  # longest first


def backbone_of(tag: str) -> str:
    for b in BACKBONES:
        if b != "clip" and re.search(rf"_{re.escape(b)}(_|$)", tag):
            return b
    return "clip"                      # the default backbone leaves no mark


def plan():
    moves, deletes = [], []
    for f in sorted(EFF.glob("*.csv")):
        stem = f.stem
        if stem.endswith("_by_seed"):
            tag = stem.removesuffix("_by_seed").removeprefix("per_unit")
            new = f"raw{tag}.csv"
        elif stem.startswith("summary"):
            tag = stem.removeprefix("summary")
            new = f"summary{tag}.csv"
        elif stem.startswith("per_unit"):
            deletes.append(f)          # the seed-averaged intermediate
            continue
        else:
            continue
        moves.append((f, EFF / backbone_of(tag) / new))
    return moves, deletes


def main() -> int:
    apply = "--apply" in sys.argv
    moves, deletes = plan()
    for src, dst in moves:
        print(f"  move   {src.name:50s} -> {dst.parent.name}/{dst.name}")
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                print(f"         SKIP: {dst.name} already there")
                continue
            shutil.move(str(src), str(dst))
    for f in deletes:
        print(f"  delete {f.name:50s} (seed average, rebuildable)")
        if apply:
            f.unlink()
    print(f"\n{len(moves)} moved, {len(deletes)} deleted" if apply else
          f"\nDRY RUN: {len(moves)} to move, {len(deletes)} to delete -- "
          "re-run with --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
