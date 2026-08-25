r"""Rebuild every table and figure the paper uses, into paper/final/.

One command, so the finished artefacts can never be a mixture of runs: each
generator reads the merged results and writes into paper/final/, and any
generator whose input is missing says so instead of writing a stale file.

Usage:  python -m src.experiments.make_all_paper_outputs
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.paper_paths import FIGURES, TABLES, ensure   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STEPS = [
    ("Table 1  main results", "src.experiments.make_tab_main"),
    ("Table 2  anchored vs unanchored", "src.experiments.make_tab_anchor"),
    ("Table 3  backbones", "src.experiments.make_tab_backbone_final"),
    ("Table 4  Stage-1 emotion accuracy", "src.experiments.make_tab_stage1_acc"),
    ("Table 5  emotion importance", "src.experiments.make_tab_emotion_importance"),
    ("Figures  efficiency + faithfulness", "src.experiments.make_paper_figures"),
]


def main() -> int:
    ensure()
    failed = []
    for label, module in STEPS:
        print(f"\n=== {label} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", module],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        tail = [ln for ln in (r.stdout or "").splitlines()
                if ln.strip()][-3:]
        print("\n".join(tail) if tail else "(no output)")
        if r.returncode:
            failed.append(label)
            err = [ln for ln in (r.stderr or "").splitlines() if ln.strip()][-3:]
            print("  FAILED: " + " | ".join(err))

    print("\n" + "=" * 60)
    for p in sorted(TABLES.glob("*.tex")):
        print(f"  tables/{p.name:34s} {p.stat().st_size / 1024:6.1f} kB")
    for p in sorted(FIGURES.glob("*.pdf")):
        print(f"  figures/{p.name:33s} {p.stat().st_size / 1024:6.1f} kB")
    if failed:
        print("\nincomplete: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
