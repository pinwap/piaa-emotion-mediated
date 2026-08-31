"""Append per-unit experiment results to ``<output-dir>/raw_all.csv``.

One row = one evaluation unit under one setting:

    setting   backbone, head, variant, stage1, n_train
    unit      mediator, fold, domain, user_id, seed
    result    srocc, plcc, ccc, eff_dof
    provenance  recorded_at (date the run produced it), experiment

*** duplicate policy ***
Re-running the same setting should give the same numbers, so re-recording it
is normally a no-op. `record()` therefore compares the incoming rows to what
is already stored, on the full key, rounded to 4 decimal places:

  identical  -> the stored row is kept and the new one dropped, so the DB does
                not grow every time an experiment is re-run.
  different  -> BOTH rows are kept, each with its own recorded_at.

Distinct results with the same key are retained with their provenance and can
be inspected with ``conflicts()``.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "output" / "raw_all.csv"

#: what makes a run different from another run
SETTING = ["backbone", "head", "variant", "stage1", "n_train"]
#: what makes a row different within one run
UNIT = ["mediator", "fold", "domain", "user_id", "seed"]
KEY = SETTING + UNIT
METRICS = ["srocc", "plcc", "ccc", "eff_dof"]
PROVENANCE = ["recorded_at", "experiment"]
COLUMNS = KEY + METRICS + PROVENANCE

#: metrics equal to this many decimals count as the same result
NDP = 4

#: defaults for settings an experiment does not vary. A new setting added in
#: future gets a new column here, and rows recorded before it existed read
#: back as this default rather than as missing.
DEFAULTS = {"backbone": "clip", "head": "ridge", "variant": "plain",
            "stage1": "mean", "n_train": 100}


def load(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """The database, or an empty frame with the right columns."""
    p = Path(db_path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    d = pd.read_csv(p)
    for col, val in DEFAULTS.items():          # columns added after some rows
        if col not in d.columns:               # were written
            d[col] = val
    for col in METRICS + PROVENANCE:
        if col not in d.columns:
            d[col] = pd.NA
    return d


def _prepare(df: pd.DataFrame, experiment: str, **setting) -> pd.DataFrame:
    d = df.copy()
    for col, val in DEFAULTS.items():
        d[col] = setting.get(col, d[col] if col in d.columns else val)
    missing = [c for c in UNIT if c not in d.columns]
    if missing:
        raise ValueError(f"record(): missing unit columns {missing}")
    for col in METRICS:
        if col not in d.columns:
            d[col] = pd.NA
    d["recorded_at"] = date.today().isoformat()
    d["experiment"] = experiment
    return d[COLUMNS]


def _fingerprint(d: pd.DataFrame) -> pd.Series:
    """'key||metrics rounded to NDP', one string per row.

    Built column by column and formatted explicitly: a DataFrame-wide
    .astype(str) leaves NaN as a float in some pandas versions, and the join
    then raises instead of comparing.
    """
    parts = [d[c].astype("string").fillna("na") for c in KEY]
    for m in METRICS:
        v = pd.to_numeric(d[m], errors="coerce").round(NDP)
        parts.append(v.map(lambda x: "na" if pd.isna(x) else f"{x:.{NDP}f}")
                     .astype("string"))
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


def record(df: pd.DataFrame, experiment: str, db_path: Path | str = DB_PATH,
           **setting) -> pd.DataFrame:
    """Add one experiment's per-unit results to the database.

    df       per-unit rows; must carry the UNIT columns and at least one metric
    experiment  name of the experiment that produced them (provenance only)
    setting  any of SETTING to override; anything omitted takes its default,
             or the column already present on df

    Returns the rows actually appended (empty when everything was a duplicate).
    """
    new = _prepare(df, experiment, **setting)
    old = load(db_path)

    if old.empty:
        appended = new
    else:
        seen = set(_fingerprint(old))
        appended = new[~_fingerprint(new).isin(seen)]

    if appended.empty:
        print(f"[results_db] {experiment}: 0 rows added "
              f"(all {len(new)} already stored with identical values)")
        return appended

    out = pd.concat([old, appended], ignore_index=True) if not old.empty else appended
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(db_path, index=False)

    dupes = len(new) - len(appended)
    note = f" ({dupes} unchanged rows skipped)" if dupes else ""
    print(f"[results_db] {experiment}: +{len(appended)} rows{note} -> {db_path}")

    clash = conflicts(out)
    if not clash.empty:
        print(f"[results_db] WARNING: {len(clash)} keys now hold more than one "
              f"result. Same setting, different numbers -- inspect with "
              f"results_db.conflicts().")
    return appended


def conflicts(db: pd.DataFrame | None = None,
              db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Keys stored with more than one distinct result.

    Empty is the expected state. A non-empty result means one setting produced
    two different numbers on two dates, which needs explaining before either
    is quoted.
    """
    d = load(db_path) if db is None else db
    if d.empty:
        return d
    r = d.copy()
    for m in METRICS:
        r[m] = pd.to_numeric(r[m], errors="coerce").round(NDP)
    n = r.groupby(KEY, dropna=False)[METRICS].nunique().max(axis=1)
    bad = n[n > 1].index
    if len(bad) == 0:
        return d.iloc[0:0]
    return d.set_index(KEY).loc[bad].reset_index().sort_values(KEY + ["recorded_at"])
