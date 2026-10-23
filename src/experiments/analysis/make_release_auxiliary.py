"""Build the two auxiliary per-unit databases used by paper tables.

``raw_all_final.csv`` covers trained model comparisons, while the formula
database covers faithfulness diagnostics.  Two analyses use separate fitted
objects and therefore need their own compact release tables:

* the ground-truth-emotion upper bound (one row per user-domain unit), and
* emotion-importance coefficients (setting x emotion x user-domain unit), and
* the four seed-averaged MLP architecture conditions used by the supplement.

The two-row reference summary is also retained because test--retest
reliability is computed from licensed raw ratings that cannot be redistributed.

This command copies only the paper-relevant columns, validates their complete
unit coverage, and records the source hashes.  Use ``--output-dir`` to build an
isolated verification copy without overwriting the frozen release files.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.experiments.analysis.make_holm_statistics import (
    mlp_architecture_data, mlp_architecture_source_paths,
)


GT_SOURCE = PROJECT_ROOT / "output" / "formulas" / "user_formula_gt.csv"
IMPORTANCE_SOURCE = (
    PROJECT_ROOT / "output" / "stage2_emotion_importance" / "per_unit.csv"
)
BOUNDS_SOURCE = PROJECT_ROOT / "output" / "upper_bounds.csv"
UNIT = ["fold", "domain", "user_id"]
IMPORTANCE_KEY = ["setting", *UNIT, "emotion"]
DOMAINS = {"art", "fashion", "landscape"}
SETTINGS = {"gt", "pred", "delta"}
EMOTIONS = {
    "impressed", "intellectual", "motivated", "amused",
    "nostalgic", "sad", "distasteful",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(data: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    data.to_csv(temporary, index=False, float_format="%.17g")
    for attempt in range(10):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def _validate_units(data: pd.DataFrame, label: str) -> None:
    if len(data) != 387:
        raise ValueError(f"{label}: expected 387 user-domain units, got {len(data)}")
    if data.duplicated(UNIT, keep=False).any():
        raise ValueError(f"{label}: duplicate user-domain keys")
    counts = data.groupby("domain").size().to_dict()
    if counts != {domain: 129 for domain in sorted(DOMAINS)}:
        raise ValueError(f"{label}: unexpected domain counts: {counts}")
    per_user = data.groupby("user_id").size()
    if len(per_user) != 129 or not (per_user == 3).all():
        raise ValueError(f"{label}: every one of 129 users must have three domains")


def _ground_truth_units() -> pd.DataFrame:
    data = pd.read_csv(GT_SOURCE, low_memory=False)
    _validate_units(data, "ground-truth upper bound")
    if set(data["backbone"].astype(str)) != {"qwen8b"}:
        raise ValueError("ground-truth upper bound must use Qwen3-VL 8B")
    if set(data["n_train"].astype(int)) != {100}:
        raise ValueError("ground-truth upper bound must use N=100")
    if set(data["seed"].astype(int)) != {0}:
        raise ValueError("ground-truth upper bound is deterministic and must use seed 0")
    for metric in ["ccc", "srocc", "plcc"]:
        if not np.isfinite(data[metric].to_numpy(float)).all():
            raise ValueError(f"ground-truth upper bound contains non-finite {metric}")
    return data.sort_values(UNIT, kind="stable").reset_index(drop=True)


def _importance_units() -> pd.DataFrame:
    data = pd.read_csv(IMPORTANCE_SOURCE, low_memory=False)
    required = set(IMPORTANCE_KEY + ["w"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"emotion importance is missing columns: {sorted(missing)}")
    data = data[IMPORTANCE_KEY + ["w"]].copy()
    if len(data) != 8_127:
        raise ValueError(f"expected 8,127 emotion-importance rows, got {len(data)}")
    if data.duplicated(IMPORTANCE_KEY, keep=False).any():
        raise ValueError("duplicate emotion-importance keys")
    if set(data["setting"].astype(str)) != SETTINGS:
        raise ValueError(f"unexpected importance settings: {set(data['setting'])}")
    if set(data["domain"].astype(str)) != DOMAINS:
        raise ValueError(f"unexpected importance domains: {set(data['domain'])}")
    if set(data["emotion"].astype(str)) != EMOTIONS:
        raise ValueError(f"unexpected emotions: {set(data['emotion'])}")
    counts = data.groupby(["setting", "emotion"]).size()
    if not (counts == 387).all():
        raise ValueError("every setting-emotion pair must contain 387 units")
    if not np.isfinite(data["w"].to_numpy(float)).all():
        raise ValueError("emotion importance contains non-finite weights")
    return data.sort_values(IMPORTANCE_KEY, kind="stable").reset_index(drop=True)


def _reference_bounds(ground_truth: pd.DataFrame) -> pd.DataFrame:
    data = pd.read_csv(BOUNDS_SOURCE)
    expected_labels = {"GT emotions", "Test--retest reliability"}
    if len(data) != 2 or set(data["label"].astype(str)) != expected_labels:
        raise ValueError("reference bounds must contain GT and test--retest rows")
    gt_row = data[data.label == "GT emotions"].iloc[0]
    for domain in sorted(DOMAINS):
        selected = ground_truth[ground_truth.domain == domain]
        for metric in ["srocc", "plcc"]:
            expected_mean = float(selected[metric].mean())
            expected_sem = float(
                selected[metric].std(ddof=1) / np.sqrt(len(selected))
            )
            if not np.isclose(gt_row[f"{domain}_{metric}_mean"], expected_mean,
                              rtol=0.0, atol=1e-12):
                raise ValueError(f"GT {domain} {metric} mean disagrees with units")
            if not np.isclose(gt_row[f"{domain}_{metric}_sem"], expected_sem,
                              rtol=0.0, atol=1e-12):
                raise ValueError(f"GT {domain} {metric} SEM disagrees with units")

    # Equal-weight participant means are exactly the statistic produced by
    # resampling each participant with all three domains as one block.
    participant_means = ground_truth.groupby("user_id")[["srocc", "plcc"]].mean()
    for metric in ["srocc", "plcc"]:
        expected_mean = float(participant_means[metric].mean())
        rng = np.random.default_rng(0)
        values = participant_means[metric].to_numpy(float)
        sampled = rng.integers(0, len(values), size=(10_000, len(values)))
        expected_sem = float(values[sampled].mean(axis=1).std(ddof=1))
        if not np.isclose(gt_row[f"avg_{metric}_mean"], expected_mean,
                          rtol=0.0, atol=1e-12):
            raise ValueError(f"GT Average {metric} mean disagrees with units")
        if not np.isclose(gt_row[f"avg_{metric}_sem"], expected_sem,
                          rtol=0.0, atol=1e-12):
            raise ValueError(f"GT Average {metric} SEM is not participant-clustered")

    retest = data[data.label == "Test--retest reliability"].iloc[0]
    mean_columns = [c for c in data if c.endswith("_mean")]
    sem_columns = [c for c in data if c.endswith("_sem")]
    if not np.isfinite(retest[mean_columns].to_numpy(float)).all():
        raise ValueError("test--retest reference means must be finite")
    if not np.isnan(retest[sem_columns].to_numpy(float)).all():
        raise ValueError("pooled test--retest correlations have no per-unit SEM")
    return data.reset_index(drop=True)


def build(output_dir: Path) -> None:
    ground_truth = _ground_truth_units()
    importance = _importance_units()
    mlp_architecture = mlp_architecture_data(PROJECT_ROOT, prefer_release=False)
    bounds = _reference_bounds(ground_truth)
    manifest_rows = [
        {
            "kind": "ground_truth_upper_bound",
            "source_file": GT_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
            "source_rows": len(ground_truth),
            "sha256": _sha256(GT_SOURCE),
        },
        {
            "kind": "emotion_importance",
            "source_file": IMPORTANCE_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
            "source_rows": len(importance),
            "sha256": _sha256(IMPORTANCE_SOURCE),
        },
        {
            "kind": "reference_bounds",
            "source_file": BOUNDS_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
            "source_rows": len(bounds),
            "sha256": _sha256(BOUNDS_SOURCE),
        },
    ]
    for source in mlp_architecture_source_paths(PROJECT_ROOT):
        manifest_rows.append({
            "kind": "mlp_architecture",
            "source_file": source.relative_to(PROJECT_ROOT).as_posix(),
            "source_rows": len(pd.read_csv(source, low_memory=False)),
            "sha256": _sha256(source),
        })
    manifest = pd.DataFrame(manifest_rows)
    _atomic_csv(ground_truth, output_dir / "upper_bound_units_final.csv")
    _atomic_csv(importance, output_dir / "emotion_importance_units_final.csv")
    _atomic_csv(mlp_architecture,
                output_dir / "mlp_architecture_units_final.csv")
    _atomic_csv(bounds, output_dir / "reference_bounds_final.csv")
    _atomic_csv(manifest, output_dir / "auxiliary_sources.csv")
    print(f"ground-truth upper-bound units: {len(ground_truth):,}")
    print(f"emotion-importance rows: {len(importance):,}")
    print(f"MLP architecture units: {len(mlp_architecture):,}")
    print(f"reference-bound summary rows: {len(bounds):,}")
    print(f"wrote auxiliary release databases to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output",
        help="directory for the two databases and their source manifest",
    )
    args = parser.parse_args()
    build(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
