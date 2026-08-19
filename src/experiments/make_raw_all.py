"""Concatenate every raw per-unit result file into one long table.

No averaging, no summarizing -- one row stays one (backbone, head, variant,
n_train, mediator, fold, domain, user_id, seed). Every SROCC/PLCC/CCC value
here is exactly what a run wrote to its per_unit file; this script only adds
the columns a file's own name encodes but its rows don't (backbone, n_train,
seed, stage2_variant depending on which experiment wrote it) and drops exact
duplicate cells (same backbone/head/variant/n_train/mediator/fold/domain/
user_id/seed measured by two different commands).

Excluded on purpose:
  output/table1/optionA|B|C/   -- pre-fix Lightning runs, confirmed
                                   contaminated (Random/Shuffled leaked the
                                   population prior). Do not resurrect these.
  output/table1/per_unit_seed{0,1,2}.csv -- same rows as per_unit_by_seed.csv,
                                   just split by seed; using both double-counts.

Output: output/raw_all.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"

VARIANTS = {"A", "B", "C"}
HEADS = {"l": "lasso", "e": "elastic"}
BACKBONES = ["clip_ft_emo", "clip_ft", "qwen4b", "qwen8b"]

KEY = ["backbone", "head", "variant", "n_train", "mediator",
       "fold", "domain", "user_id", "seed"]
CORE_COLS = ["fold", "domain", "user_id", "mediator", "head",
            "eff_dof", "ccc", "srocc", "plcc"]


def parse_tag(stem: str) -> dict:
    """'_C_l_qwen4b' -> {variant: C, head: lasso, backbone: qwen4b}.

    Same convention efficiency._tag()/backbone.run() write with: variant
    token first, then a single-letter head token, then a backbone name --
    each optional, in that order. Backbone names contain their own
    underscores, so they're matched as a whole suffix, not split by "_".
    """
    cfg = {"variant": "plain", "head": "ridge", "backbone": "clip"}
    for b in BACKBONES:
        if stem == f"_{b}" or stem.endswith(f"_{b}"):
            cfg["backbone"] = b
            stem = stem[: -(len(b) + 1)]
            break
    parts = [p for p in stem.split("_") if p]
    i = 0
    if i < len(parts) and parts[i] in VARIANTS:
        cfg["variant"] = parts[i]; i += 1
    if i < len(parts) and parts[i] in HEADS:
        cfg["head"] = HEADS[parts[i]]; i += 1
    if i != len(parts):
        raise ValueError(f"unrecognized filename tag {stem!r} (left: {parts[i:]})")
    return cfg


def load_efficiency() -> list[pd.DataFrame]:
    """backbone missing from the rows -> take it from the filename.
    head/n_train/seed/stage2_variant are already real per-row columns."""
    frames = []
    eff = ROOT / "output" / "efficiency"
    # Both layouts: the flat per_unit*_by_seed.csv written before the
    # reorganisation, and <backbone>/raw*.csv written after it. The backbone
    # comes from the tag in the old layout and from the folder in the new
    # one, so files that predate the change still load.
    files = ([(f, None) for f in sorted(eff.glob("*_by_seed.csv"))]
             + [(f, f.parent.name) for f in sorted(eff.glob("*/raw*.csv"))])
    for f, folder_bb in files:
        stem = f.stem.removesuffix("_by_seed")
        tag = stem.removeprefix("per_unit").removeprefix("raw")
        cfg = parse_tag(tag)
        d = pd.read_csv(f)
        d["backbone"] = folder_bb or cfg["backbone"]
        d = d.rename(columns={"stage2_variant": "variant"})
        frames.append(d[KEY + ["eff_dof", "ccc", "srocc", "plcc"]])
        print(f"  efficiency: {f.name:45s} {len(d):6d} rows -> backbone={cfg['backbone']}")
    return frames


def load_backbone() -> list[pd.DataFrame]:
    """n_train/seed missing entirely -> 100 and 0. variant from filename."""
    frames = []
    for f in sorted((ROOT / "output" / "backbone").glob("per_unit*.csv")):
        tag = f.stem.removeprefix("per_unit")
        cfg = parse_tag(tag)
        d = pd.read_csv(f)
        d["variant"] = cfg["variant"]
        d["n_train"] = 100
        d["seed"] = 0
        frames.append(d[KEY + ["eff_dof", "ccc", "srocc", "plcc"]])
        print(f"  backbone:   {f.name:45s} {len(d):6d} rows -> variant={cfg['variant']}")
    return frames


def load_table1() -> list[pd.DataFrame]:
    """backbone/n_train missing entirely -> clip and 100. Excludes the
    contaminated optionA/B/C dirs and the per-seed duplicates of by_seed."""
    frames = []
    t1 = ROOT / "output" / "table1"
    patterns = ["per_unit_by_seed.csv", "per_unit_le_seed*.csv",
               "per_unit_A_m_seed*.csv", "per_unit_B_m_seed*.csv",
               "per_unit_C_m_seed*.csv", "per_unit_m_seed*.csv"]
    for pat in patterns:
        for f in sorted(t1.glob(pat)):
            d = pd.read_csv(f)
            d["backbone"] = "clip"
            d["n_train"] = 100
            d = d.rename(columns={"stage2_variant": "variant"})
            frames.append(d[KEY + ["eff_dof", "ccc", "srocc", "plcc"]])
            print(f"  table1:     {f.name:45s} {len(d):6d} rows")
    return frames


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Reading efficiency/*_by_seed.csv ...")
    eff = load_efficiency()
    print("Reading backbone/per_unit*.csv ...")
    bb = load_backbone()
    print("Reading table1/per_unit*.csv (excluding optionA/B/C, per-seed dupes) ...")
    t1 = load_table1()

    eff_df = pd.concat(eff, ignore_index=True)
    eff_df["source"] = "efficiency"
    bb_df = pd.concat(bb, ignore_index=True)
    bb_df["source"] = "backbone"
    t1_df = pd.concat(t1, ignore_index=True)
    t1_df["source"] = "table1"

    # The same cell can be measured by more than one experiment, so keep one
    # copy per KEY by source priority: table1 > backbone > efficiency.
    #
    # table1 first because it is the paper's own canonical n=100 run and
    # covers mediators the others don't (random, shuffled). backbone next
    # because it is the only source of clip's B/C at n=100. efficiency last,
    # filling everything at n<100 and pca.
    #
    # Note the image sets differ slightly: backbone/ restricts every backbone
    # to the 386 images all four share (see common_stimuli), while table1/ and
    # efficiency/ use each backbone's own full set (387 for clip). Both are
    # valid; the n column records which one a row came from, so any table
    # built off this file should print n per cell rather than assume it.
    raw = pd.concat([t1_df, bb_df, eff_df], ignore_index=True)
    before = len(raw)
    raw = raw.drop_duplicates(subset=KEY, keep="first")
    print(f"\ndropped {before - len(raw)} duplicate cells "
         f"(same KEY measured by more than one experiment; "
         f"kept table1 > backbone > efficiency)")
    raw = raw.sort_values(["backbone", "head", "variant", "n_train", "mediator",
                          "fold", "domain", "seed", "user_id"])
    raw.to_csv(OUT / "raw_all.csv", index=False)
    print(f"\nwrote {OUT / 'raw_all.csv'} -- {len(raw)} rows, {raw.memory_usage(deep=True).sum()/1e6:.1f} MB")

    print("\ncoverage (backbone x head x variant x n_train), rows per cell:")
    cells = raw.groupby(["backbone", "head", "variant", "n_train"]).size().reset_index(name="rows")
    print(cells.to_string(index=False))


if __name__ == "__main__":
    main()
