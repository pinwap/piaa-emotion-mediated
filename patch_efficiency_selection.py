"""Make `main.py efficiency` save its selection log.

Pipeline.run_grid already collects every hyperparameter it selected, with the
grid it was selected from and a flag for whether the winner sat at the edge of
that grid, and leaves it on `pipeline.selection_records`. table1.py writes it
out. efficiency.py does not, which is why there is no ridge selection log
anywhere in output/: every log on disk came from a suite run, and every suite
run used --heads mlp.

That matters twice. It is the evidence for "was the ridge penalty a choice or
a grid boundary", and it is the first thing a reviewer opening the repository
would look for.

Three edits, all additive: import the module, keep the records from each
run_grid call, write one file next to raw{tag}.csv.

Run from the repo root:
    uv run python patch_efficiency_selection_log.py --check
    uv run python patch_efficiency_selection_log.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("src/experiments/efficiency.py")

EDITS = [
    ("from src.utils.metrics import mean_sd, sem, wilcoxon_paired\n"
     "from src.utils.results_db import record",

     "from src.utils import selection_log\n"
     "from src.utils.metrics import mean_sd, sem, wilcoxon_paired\n"
     "from src.utils.results_db import record"),

    ("            d[\"n_train\"], d[\"seed\"], d[\"stage2_variant\"] = n, seed, variant\n"
     "            frames.append(d)",

     "            d[\"n_train\"], d[\"seed\"], d[\"stage2_variant\"] = n, seed, variant\n"
     "            frames.append(d)\n"
     "            selections.extend(pipeline.selection_records)"),

    ("    raw = pd.concat(frames, ignore_index=True)\n"
     "    raw.to_csv(out_dir / f\"raw{tag}.csv\", index=False)",

     "    raw = pd.concat(frames, ignore_index=True)\n"
     "    raw.to_csv(out_dir / f\"raw{tag}.csv\", index=False)\n"
     "    # what every hyperparameter search in this run looked at, not just\n"
     "    # what it picked. `edge` names the axis a winner ran out of grid on,\n"
     "    # which is the difference between a choice and a boundary.\n"
     "    if selections:\n"
     "        selection_log.write(selections, out_dir / f\"selection{tag}.csv\")\n"
     "        print(f\"[efficiency] selection log -> selection{tag}.csv \"\n"
     "              f\"({len(selections)} rows)\")"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not TARGET.exists():
        print(f"cannot find {TARGET} -- run this from the repo root")
        return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    flat = raw.replace("\r\n", "\n")

    bad = [f"  edit {i}: found {flat.count(o)} matches, expected 1"
           for i, (o, _) in enumerate(EDITS, 1) if flat.count(o) != 1]

    # `selections` has to exist before the loop that appends to it
    anchor = "    frames = []"
    if flat.count(anchor) != 1:
        bad.append(f"  could not find the '{anchor.strip()}' line to "
                   f"initialise the accumulator next to")
    if bad:
        print("refusing to write, the file is not in the state this patch expects:")
        print("\n".join(bad))
        return 1

    out = flat.replace(anchor, "    frames, selections = [], []", 1)
    for old, new in EDITS:
        out = out.replace(old, new, 1)

    print(f"all {len(EDITS) + 1} edits matched exactly once in {TARGET}")
    if not args.apply:
        print("\ndry run. re-run with --apply to write.")
        return 0
    TARGET.write_text(out.replace("\n", nl), encoding="utf-8", newline="")
    print(f"written: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())