"""Build the canonical per-unit result database used by the paper.

Each model run writes an immutable CSV under ``output/<run>/``.  This module
does not train or summarize models.  It consolidates the declared run files,
normalizes their metadata, resolves overlapping cells by the documented source
priority, and validates the complete three-seed/387-unit design before writing
``raw_all_final.csv``.

The output is rebuilt from scratch rather than appended in place.  This makes
the result independent of command order and prevents interrupted or repeated
runs from silently duplicating cells.

Usage
-----
Verify against the currently frozen database without overwriting it::

    uv run python -m src.experiments.make_raw_all_final \
        --output tmp/release_verify/raw_all_final.csv \
        --manifest-output tmp/release_verify/raw_all_sources.csv \
        --compare output/raw_all_final.csv

After verification, write the canonical release database::

    uv run python -m src.experiments.make_raw_all_final
"""
from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
EXPECTED_ROWS = 381_969
EXPECTED_SEEDS = {0, 1, 2}
EXPECTED_UNITS_PER_CONFIGURATION = 387

KEY = [
    "backbone", "head", "variant", "n_train", "mediator",
    "fold", "domain", "user_id", "seed",
]
CONFIG_KEY = ["backbone", "head", "variant", "n_train", "mediator"]
OUTPUT_COLUMNS = [
    "fold", "domain", "user_id", "mediator", "head", "eff_dof",
    "ccc", "srocc", "plcc", "n_train", "seed", "stage2_variant",
    "backbone", "variant", "source_file", "experiment",
]
REQUIRED_INPUT_COLUMNS = {
    "fold", "domain", "user_id", "mediator", "head", "eff_dof",
    "ccc", "srocc", "plcc", "n_train", "seed", "stage2_variant",
}


@dataclass(frozen=True)
class Source:
    path: str
    backbone: str


# Earlier entries win when two runs contain the same fully specified cell.
# The priority is intentional: focused final runs precede broader sweeps that
# also contain preliminary copies of the same configuration.
SOURCES = [
    Source("output/efficiency/clip/raw_B.csv", "clip"),

    # Population-anchored frozen CLIP.
    Source("output/efficiency/clip/raw_C_memo-emo_sd-emo_hist.csv", "clip"),
    Source("output/efficiency/clip/raw_C.csv", "clip"),
    Source("output/norm-2/silver_falcon/1_base-clip/"
           "raw_C_midentity-pca-emo-random-shuffled.csv", "clip"),

    # Population-anchored Qwen3-VL 8B.  The 35-d final run supplies its
    # n=100 cells; the base and distribution sweeps fill the remaining cells.
    Source("output/efficiency/qwen8b/"
           "raw_C_qwen8b_memo-emo_hist-pca35-random35-shuffled35.csv", "qwen8b"),
    Source("output/norm-2/silver_falcon/2_base-qwen8b/"
           "raw_C_qwen8b_midentity-pca-emo-random-shuffled.csv", "qwen8b"),
    Source("output/norm-1/emerald_tiger/10_dist-qwen8b/"
           "raw_C_qwen8b_memo-emo_sd-emo_hist.csv", "qwen8b"),

    # Population-anchored remaining backbones.
    Source("output/norm-2/silver_falcon/3_base-clip-ft/"
           "raw_C_clip_ft_midentity-pca-emo-random-shuffled.csv", "clip_ft"),
    Source("output/norm-1/emerald_tiger/11_dist-clip-ft/"
           "raw_C_clip_ft_memo-emo_sd-emo_hist.csv", "clip_ft"),
    Source("output/norm-2/silver_falcon/4_base-clip-ft-emo/"
           "raw_C_clip_ft_emo_midentity-pca-emo-random-shuffled.csv", "clip_ft_emo"),
    Source("output/norm-1/emerald_tiger/12_dist-clip-ft-emo/"
           "raw_C_clip_ft_emo_memo-emo_sd-emo_hist.csv", "clip_ft_emo"),
    Source("output/norm-2/silver_falcon/5_base-qwen4b/"
           "raw_C_qwen4b_midentity-pca-emo-random-shuffled.csv", "qwen4b"),
    Source("output/norm-1/emerald_tiger/13_dist-qwen4b/"
           "raw_C_qwen4b_memo-emo_sd-emo_hist.csv", "qwen4b"),

    # Unanchored Qwen3-VL 8B and frozen CLIP.  The focused 35-d run wins at
    # n=100, the distribution run supplies emotion targets, and the base run
    # supplies Direct/PCA/Random/Shuffled.
    Source("output/efficiency/qwen8b/"
           "raw_qwen8b_memo-emo_hist-pca35-random35-shuffled35.csv", "qwen8b"),
    Source("output/no_anchor/6_dist-plain-qwen8b/"
           "raw_qwen8b_memo-emo_sd-emo_hist.csv", "qwen8b"),
    Source("output/no_anchor/1_plain-qwen8b/"
           "raw_qwen8b_midentity-pca-emo-random-shuffled.csv", "qwen8b"),
    Source("output/efficiency/clip/"
           "raw_memo-emo_hist-pca35-random35-shuffled35.csv", "clip"),
    Source("output/no_anchor/7_dist-plain-clip/"
           "raw_memo-emo_sd-emo_hist.csv", "clip"),
    Source("output/no_anchor/2_plain-clip/"
           "raw_midentity-pca-emo-random-shuffled.csv", "clip"),

    # Unanchored remaining backbones.
    Source("output/no_anchor/3_plain-clip-ft/"
           "raw_clip_ft_midentity-pca-emo-random-shuffled.csv", "clip_ft"),
    Source("output/no_anchor/4_plain-clip-ft-emo/"
           "raw_clip_ft_emo_midentity-pca-emo-random-shuffled.csv", "clip_ft_emo"),
    Source("output/no_anchor/5_plain-qwen4b/"
           "raw_qwen4b_midentity-pca-emo-random-shuffled.csv", "qwen4b"),

    # Final MLP sweep, one immutable file per seed.
    Source("output/thundering_lobster_modal_all_runs/lr/1_mlp-mse-s0/"
           "raw_C_m_qwen8b_midentity-random-shuffled-pca-emo-emo_mlp-emo_joint.csv",
           "qwen8b"),
    Source("output/thundering_lobster_modal_all_runs/lr/2_mlp-mse-s1/"
           "raw_C_m_qwen8b_midentity-random-shuffled-pca-emo-emo_mlp-emo_joint.csv",
           "qwen8b"),
    Source("output/thundering_lobster_modal_all_runs/lr/3_mlp-mse-s2/"
           "raw_C_m_qwen8b_midentity-random-shuffled-pca-emo-emo_mlp-emo_joint.csv",
           "qwen8b"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(source: Path, destination: Path) -> None:
    """Replace a completed CSV, tolerating short Windows indexer locks."""
    for attempt in range(10):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def _load_source(spec: Source, priority: int) -> tuple[pd.DataFrame, dict]:
    path = PROJECT_ROOT / spec.path
    if not path.is_file():
        raise FileNotFoundError(f"missing declared raw source: {spec.path}")

    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_INPUT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{spec.path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{spec.path}: file is empty")
    if not set(frame["seed"].astype(int)).issubset(EXPECTED_SEEDS):
        raise ValueError(f"{spec.path}: unexpected seed values")

    frame = frame.copy()
    frame["backbone"] = spec.backbone
    frame["variant"] = frame["stage2_variant"]
    frame["source_file"] = spec.path
    frame["experiment"] = "efficiency"
    frame["_source_priority"] = priority
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)

    metadata = {
        "priority": priority,
        "source_file": spec.path,
        "backbone": spec.backbone,
        "input_rows": len(frame),
        "sha256": _sha256(path),
    }
    return frame, metadata


def _validate_final(data: pd.DataFrame) -> None:
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} final rows, got {len(data)}")
    duplicated = data.duplicated(KEY, keep=False)
    if duplicated.any():
        example = data.loc[duplicated, KEY].head().to_dict("records")
        raise ValueError(f"duplicate final keys, e.g. {example}")
    if data[KEY].isna().any().any():
        raise ValueError("final database contains a missing key field")
    if data[["srocc", "plcc"]].isna().any().any():
        raise ValueError("final database contains missing SROCC/PLCC values")

    seed_sets = data.groupby(CONFIG_KEY + ["fold", "domain", "user_id"])["seed"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    bad_seeds = seed_sets[seed_sets != frozenset(EXPECTED_SEEDS)]
    if not bad_seeds.empty:
        raise ValueError(f"incomplete seed sets, first failures:\n{bad_seeds.head()}")

    unit_counts = data.groupby(CONFIG_KEY).size() / len(EXPECTED_SEEDS)
    if not (unit_counts == EXPECTED_UNITS_PER_CONFIGURATION).all():
        raise ValueError(
            "every configuration must have 387 seed-averaged units; first failures:\n"
            + unit_counts[unit_counts != EXPECTED_UNITS_PER_CONFIGURATION].head().to_string()
        )


def _compare_reference(data: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path, low_memory=False)
    paired = reference.merge(
        data,
        on=KEY,
        how="outer",
        suffixes=("_reference", "_rebuilt"),
        indicator=True,
        validate="one_to_one",
    )
    if not (paired["_merge"] == "both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"reference/rebuilt key mismatch: {counts}")
    for metric in ["srocc", "plcc"]:
        left = paired[f"{metric}_reference"].to_numpy(float)
        right = paired[f"{metric}_rebuilt"].to_numpy(float)
        if not np.array_equal(left, right, equal_nan=True):
            maximum = float(np.nanmax(np.abs(left - right)))
            raise ValueError(f"{metric} differs from reference; max abs diff={maximum}")
    print(f"reference check: SROCC/PLCC and all {len(data):,} keys are identical")


def build(output: Path, manifest_output: Path, compare: Path | None = None) -> None:
    frames = []
    manifest_rows = []
    for priority, spec in enumerate(SOURCES):
        frame, metadata = _load_source(spec, priority)
        frames.append(frame)
        manifest_rows.append(metadata)

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.sort_values(
        ["_source_priority", "_source_row"], kind="stable"
    ).drop_duplicates(KEY, keep="first")
    dropped = before - len(combined)
    _validate_final(combined)

    selected_counts = combined.groupby("source_file").size()
    manifest = pd.DataFrame(manifest_rows)
    manifest["selected_rows"] = (
        manifest["source_file"].map(selected_counts).fillna(0).astype(int)
    )

    final = combined[OUTPUT_COLUMNS].sort_values(
        KEY, kind="stable"
    ).reset_index(drop=True)
    if compare is not None:
        _compare_reference(final, compare)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    manifest_tmp = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    final.to_csv(output_tmp, index=False, float_format="%.17g")
    manifest.to_csv(manifest_tmp, index=False)
    _atomic_replace(output_tmp, output)
    _atomic_replace(manifest_tmp, manifest_output)

    print(f"declared source files: {len(SOURCES)}")
    print(f"input rows: {before:,}; overlapping rows resolved: {dropped:,}")
    print(f"final rows: {len(final):,}; configurations: {final.groupby(CONFIG_KEY).ngroups}")
    print(f"wrote {output}")
    print(f"wrote {manifest_output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output" / "raw_all_final.csv"
    )
    parser.add_argument(
        "--manifest-output", type=Path,
        default=PROJECT_ROOT / "output" / "raw_all_sources.csv",
    )
    parser.add_argument(
        "--compare", type=Path,
        help="optional frozen database whose keys/SROCC/PLCC must match exactly",
    )
    args = parser.parse_args()
    build(args.output, args.manifest_output, args.compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
