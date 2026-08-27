"""Register the `formulas` command and route the two analyses through it.

`formulas` fits every formula once under the reported variant and writes them
to output/formulas/. `faithfulness` and `stage2_emotion_importance` then read
those files instead of fitting their own heads, so the model being interpreted
and the model being reported cannot come apart.

The old `formula_weights` command stays registered and keeps working; it is
now a subset of what `formulas` writes.

Run from the repo root:
    uv run python patch_register_formulas.py --check
    uv run python patch_register_formulas.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("main.py")

EDITS = [
    ('                            "formula_weights",\n',
     '                            "formula_weights", "formulas",\n'),

    ('''    elif args.experiment == "formula_weights":''',
     '''    elif args.experiment == "formulas":
        from src.experiments import formulas
        formulas.run(
            cfg, pipe, variant=args.stage2 or "C",
            n_train=int(args.n_train.split(",")[-1]),
            mediator=(args.mediators.split(",")[0]
                      if args.mediators else "emotion"),
            seed=int(str(args.seed).split(",")[0]) if args.seed else 0)
    elif args.experiment == "formula_weights":'''),
]

FALLBACK = [
    ('''                    choices=["table1", "backbone", "efficiency", "faithfulness",\n''',
     '''                    choices=["table1", "backbone", "efficiency", "faithfulness",\n                            "formulas",\n'''),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not TARGET.exists():
        print("run this from the repo root")
        return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    flat = raw.replace("\r\n", "\n")

    edits = list(EDITS)
    if flat.count(EDITS[0][0]) != 1:
        # formula_weights was never registered; put the name in the choices
        # list itself instead
        edits[0] = FALLBACK[0]

    bad = [f"  edit {i}: found {flat.count(o)} matches, expected 1"
           for i, (o, _) in enumerate(edits, 1) if flat.count(o) != 1]
    if bad:
        print("refusing to write, main.py is not in the state this patch expects:")
        print("\n".join(bad))
        return 1

    out = flat
    for old, new in edits:
        out = out.replace(old, new, 1)

    print(f"all {len(edits)} edits matched exactly once in {TARGET}")
    if not args.apply:
        print("\ndry run. re-run with --apply to write.")
        return 0
    TARGET.write_text(out.replace("\n", nl), encoding="utf-8", newline="")
    print(f"written: {TARGET}")
    print("\nnow:  uv run main.py formulas --backbone qwen8b --stage2 C --n-train 100")
    return 0


if __name__ == "__main__":
    sys.exit(main())