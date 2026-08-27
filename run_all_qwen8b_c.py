"""Re-run every experiment that feeds a table, on Qwen3-VL 8B with Anchor C,
in dependency order, with a log per experiment and a check at the end.

Order matters. `efficiency` is first because it is the one the main table
reads and the one whose reproduction is already established, so if it does not
come back with the values below, nothing after it is worth running.

    efficiency               main table, efficiency figure
    stage1_emotion_acc       how well Stage-1 predicts each emotion
    stage2_emotion_importance   weight per emotion, GT vs predicted
    faithfulness             ablation, formula swap, weight alignment
    formula_weights          per-user formulas for the qualitative figure
    backbone                 the backbone comparison table

table1 is deliberately not here: it is the multi-seed run and takes far
longer than the rest put together, and the numbers already on disk reproduce.

Each experiment writes to output/<name>/ as usual, and this script also keeps
output/rerun_logs/<name>.log so a failure can be read afterwards rather than
scrolled back to.

    uv run python run_all_qwen8b_C.py --list
    uv run python run_all_qwen8b_C.py                  # everything
    uv run python run_all_qwen8b_C.py --only efficiency,faithfulness
    uv run python run_all_qwen8b_C.py --from faithfulness
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
import os
os.environ.setdefault("PYTHONUTF8", "1")

BACKBONE = "qwen8b"
VARIANT = "C"

# `formulas` comes before the two analyses that read it, and after the
# efficiency sweep whose numbers it has to agree with. The two analyses fit
# nothing themselves, so they take seconds once formulas has run.
STEPS = [
    ("efficiency", ["efficiency", "--backbone", BACKBONE, "--stage2", VARIANT,
                    "--heads", "ridge",
                    # `population` is not a mediator: run_grid adds the GIAA
                    # baseline row itself via include_population. Naming it
                    # here makes the pipeline look for a mediator called
                    # "population" and fail with KeyError.
                    "--mediators", "identity,pca,random,shuffled,emotion",
                    "--n-train", "10,25,50,100", "--seed", "0"]),
    ("formulas", ["formulas", "--backbone", BACKBONE, "--stage2", VARIANT,
                  "--n-train", "100"]),
    ("stage2_emotion_importance", ["stage2_emotion_importance",
                                   "--backbone", BACKBONE, "--stage2", VARIANT]),
    ("faithfulness", ["faithfulness", "--backbone", BACKBONE, "--stage2", VARIANT]),
    ("stage1_emotion_acc", ["stage1_emotion_acc", "--backbone", BACKBONE,
                            "--stage2", VARIANT]),
    ("backbone", ["backbone", "--backbones", "clip,clip_ft,qwen4b,qwen8b",
                  "--stage2", VARIANT, "--heads", "ridge"]),
]

#: what the run has to come back with if it is the model the paper reports.
#: Only the first is checked automatically; the rest are here so the numbers
#: can be compared by eye against the log.
EXPECTED = {
    "efficiency": [
        ("emotion, N=100", 0.4592),
        ("population, N=100", 0.4391),
        ("emotion, N=10", 0.4333),
        ("emotion, N=50", 0.4505),
    ],
    "formulas": [("mean SROCC", 0.4592), ("population", 0.4391)],
}


def run_one(name: str, argv: list[str], log_dir: Path) -> tuple[bool, float]:
    log = log_dir / f"{name}.log"
    print(f"\n{'=' * 70}\n{name}\n  main.py {' '.join(argv)}\n  log -> {log}\n{'=' * 70}",
          flush=True)
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, "main.py", *argv],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             bufsize=1)
        for line in p.stdout:
            fh.write(line)
            # the ridge solver is chatty about conditioning and it is not a
            # problem here; everything else goes to the screen
            if "LinAlgWarning" in line or "dual_coef" in line:
                continue
            print("  " + line.rstrip(), flush=True)
        code = p.wait()
    dt = time.time() - t0
    print(f"\n  {name}: {'OK' if code == 0 else f'FAILED (exit {code})'}"
          f"  in {dt / 60:.1f} min", flush=True)
    return code == 0, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated subset, e.g. --only efficiency")
    ap.add_argument("--from", dest="start", default=None,
                    help="start at this experiment and continue")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for i, (n, a) in enumerate(STEPS, 1):
            print(f"{i}. {n:26s} main.py {' '.join(a)}")
        return 0

    if not Path("main.py").exists():
        print("run this from the repo root")
        return 1

    steps = STEPS
    if args.start:
        names = [n for n, _ in STEPS]
        if args.start not in names:
            print(f"unknown experiment '{args.start}'. try --list")
            return 1
        steps = STEPS[names.index(args.start):]
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        steps = [s for s in steps if s[0] in want]
        if not steps:
            print("nothing matched --only. try --list")
            return 1

    log_dir = Path("output/rerun_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"about to run {len(steps)} experiment(s) on {BACKBONE}, "
          f"Anchor {VARIANT}:")
    for n, _ in steps:
        print(f"  - {n}")
        if n in EXPECTED:
            for label, v in EXPECTED[n]:
                print(f"        expect {label}: {v}")

    results, total = [], 0.0
    for name, argv in steps:
        ok, dt = run_one(name, argv, log_dir)
        results.append((name, ok, dt))
        total += dt
        if not ok:
            print(f"\nstopping: {name} failed. see {log_dir / (name + '.log')}")
            break

    print(f"\n{'=' * 70}\nsummary   ({total / 60:.1f} min total)")
    for name, ok, dt in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:26s} {dt / 60:6.1f} min")
    done = {n for n, ok, _ in results if ok}
    missing = [n for n, _ in steps if n not in done]
    if missing:
        print("\nnot run:", ", ".join(missing))
    print(f"\nlogs in {log_dir}")
    print("check the efficiency log for '0 rows added ... identical values' "
          "or for the four SROCC values above before trusting anything else.")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())