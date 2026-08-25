#!/usr/bin/env python
"""
Remove the stale head=='mlp' rows from output/raw_all_final.csv and replace them
with the thundering_lobster run (the one that follows the validation-group
selection protocol).

Run from the repo root:
    python purge_and_reingest_mlp.py            # dry run, prints what it would do
    python purge_and_reingest_mlp.py --apply    # actually rewrites the file
"""
import argparse, os, shutil, sys
import pandas as pd

ROOT   = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(ROOT, "output", "raw_all_final.csv")
LOBSTER = os.path.join(ROOT, "output", "thundering_lobster_modal_all_runs",
                       "thundering_lobster")
SEED_DIRS = ["1_mlp-mse-s0", "2_mlp-mse-s1", "3_mlp-mse-s2"]

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

if not os.path.exists(TARGET):
    sys.exit(f"not found: {TARGET}")

df = pd.read_csv(TARGET, low_memory=False)
print(f"loaded {TARGET}: {len(df):,} rows")
print("  head values:", df["head"].value_counts().to_dict())

stale = df[df["head"] == "mlp"]
print(f"\nstale mlp rows to drop: {len(stale):,}")
if len(stale):
    print(stale.groupby(["backbone", "mediator"]).size().to_string())

kept = df[df["head"] != "mlp"].copy()

# --- load the replacement -------------------------------------------------
parts = []
for d in SEED_DIRS:
    p = os.path.join(LOBSTER, d, "raw_all.csv")
    if not os.path.exists(p):
        sys.exit(f"not found: {p}")
    parts.append(pd.read_csv(p))
new = pd.concat(parts, ignore_index=True)
print(f"\nthundering_lobster rows: {len(new):,}")
print("  heads:", sorted(new['head'].unique()))
print("  mediators:", sorted(new['mediator'].unique()))
print(new.groupby("mediator").size().to_string())

if set(new["head"].unique()) != {"mlp"}:
    sys.exit("refusing: thundering_lobster contains non-mlp rows, check the source")

# --- align columns --------------------------------------------------------
missing_in_new = [c for c in kept.columns if c not in new.columns]
extra_in_new   = [c for c in new.columns if c not in kept.columns]
print(f"\ncolumns only in target: {missing_in_new}")
print(f"columns only in new:    {extra_in_new}")
for c in missing_in_new:
    new[c] = pd.NA
new = new[kept.columns]          # same order, drop extras

out = pd.concat([kept, new], ignore_index=True)

# --- duplicate guard ------------------------------------------------------
KEY = ["backbone", "head", "variant", "n_train", "mediator",
       "fold", "domain", "user_id", "seed"]
dups = out.duplicated(subset=KEY).sum()
print(f"\nresult: {len(out):,} rows, duplicate keys: {dups}")
if dups:
    sys.exit("refusing: duplicate keys after merge, inspect before writing")

print("  head values after merge:", out["head"].value_counts().to_dict())

# sanity check the numbers we expect
chk = out[(out["head"] == "mlp") & (out["backbone"] == "qwen8b")
          & (out["variant"] == "C") & (out["n_train"] == 100)]
for med in ["emotion_mlp", "emotion_joint"]:
    g = chk[chk["mediator"] == med].groupby(["fold", "domain", "user_id"])["srocc"].mean()
    if len(g):
        print(f"  check {med}: srocc = {g.mean():.4f} over {len(g)} units "
              f"(expect .4035 and .3881)")

if not args.apply:
    print("\nDRY RUN. nothing written. re-run with --apply")
    sys.exit(0)

bak = TARGET.replace(".csv", ".before_mlp_purge2.csv")
shutil.copy2(TARGET, bak)
out.to_csv(TARGET, index=False)
print(f"\nbackup written to {bak}")
print(f"rewrote {TARGET}")