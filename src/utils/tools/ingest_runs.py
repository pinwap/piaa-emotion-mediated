"""Merge a folder of finished runs into output/raw_all_final.csv.

Results arrive as folders of per-run CSVs from whichever machine produced
them. Two things make a naive concatenation unsafe, and both have bitten us:

  - the files carry no `backbone` column, so it has to come from the filename
  - appending a run computed against superseded data silently corrupts every
    table drawn afterwards

So this checks before it writes. The invariant is that the population row
cannot depend on n_train: per_user_split() carves the eval set out first at a
fixed size with a per-user seed, so the population model scores the same
images whatever the support size. A file where it moves was computed against
a different dataset, and is refused.

    uv run python src/utils/tools/ingest_runs.py output/norm-2
    uv run python src/utils/tools/ingest_runs.py output/norm-2 --apply
"""
from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "output" / "raw_all_final.csv"

UNIT = ["fold", "domain", "user_id"]
KEY = ["backbone", "head", "variant", "n_train", "mediator",
       "fold", "domain", "user_id", "seed"]
BACKBONES = ["clip_ft_emo", "clip_ft", "qwen8b", "qwen4b"]      # longest first


def backbone_of(path: Path) -> str:
    """Read the backbone off the filename; plain CLIP leaves no marker."""
    name = path.name
    for bb in BACKBONES:
        if bb in name:
            return bb
    return "clip"


def population_is_flat(df: pd.DataFrame) -> tuple[bool, str]:
    pop = df[(df["mediator"] == "population") & (df["head"] == "ridge")]
    if pop.empty or pop["n_train"].nunique() < 2:
        return True, "nothing to check"
    worst = 0.0
    for _, g in pop.groupby("seed"):
        sizes = sorted(g["n_train"].unique())
        ref = g[g["n_train"] == sizes[0]].set_index(UNIT)["srocc"]
        for n in sizes[1:]:
            cur = g[g["n_train"] == n].set_index(UNIT)["srocc"]
            idx = ref.index.intersection(cur.index)
            if len(idx):
                worst = max(worst, float((ref.loc[idx] - cur.loc[idx]).abs().max()))
    return worst < 1e-9, f"population moves by {worst:.2e} across n_train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = args.folder if args.folder.is_absolute() else ROOT / args.folder
    files = sorted(f for f in src.rglob("raw*.csv") if f.name != "raw_all.csv")
    if not files:
        raise SystemExit(f"no per-run files under {src}")

    keep, reject = [], []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "mediator" not in df.columns:
            reject.append((f, "not a per-unit results file"))
            continue
        df["backbone"] = backbone_of(f)
        df["variant"] = df.get("stage2_variant", "C")
        df["experiment"] = "efficiency"
        df["source_file"] = f.relative_to(ROOT).as_posix()
        ok, why = population_is_flat(df)
        (keep if ok else reject).append((f, df if ok else why))

    print(f"{'file':62s} {'backbone':12s} rows   verdict")
    for f, df in keep:
        print(f"{f.relative_to(ROOT).as_posix():62s} "
              f"{df['backbone'].iloc[0]:12s} {len(df):6d} KEEP")
    for f, why in reject:
        print(f"{f.relative_to(ROOT).as_posix():62s} {'':12s} {'':6s} REFUSE ({why})")

    if not keep:
        raise SystemExit("nothing passed; refusing to touch raw_all_final.csv")

    new = pd.concat([df for _, df in keep], ignore_index=True)
    print(f"\nmediators: {sorted(new.mediator.unique())}")
    print(f"heads    : {sorted(new['head'].unique())}")

    old = pd.read_csv(FINAL, low_memory=False) if FINAL.exists() else None
    merged = new if old is None else pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=KEY, keep="last")

    print("\npopulation per backbone in the merged file "
          "(one value per backbone, or the merge is wrong):")
    pop = merged[(merged.mediator == "population") & (merged["head"] == "ridge")]
    broken = False
    for bb, g in pop.groupby("backbone"):
        v = g.groupby("n_train")["srocc"].mean().round(9)
        flat = v.nunique() == 1
        broken |= not flat
        print(f"  {bb:12s} {v.iloc[0]:.6f}  {'OK' if flat else 'BROKEN ' + str(v.to_dict())}")
    if broken:
        raise SystemExit("merged file violates the invariant; not written")

    if not args.apply:
        print(f"\n[report only] would write {len(merged)} rows "
              f"(from {0 if old is None else len(old)}). Pass --apply.")
        return 0

    if FINAL.exists():
        bak = FINAL.with_name(f"raw_all_final_before_{date.today():%Y%m%d}.csv")
        shutil.copy2(FINAL, bak)
        print(f"\nbacked up -> {bak.name}")
    merged.to_csv(FINAL, index=False)
    print(f"wrote {FINAL.relative_to(ROOT).as_posix()}: {len(merged)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
