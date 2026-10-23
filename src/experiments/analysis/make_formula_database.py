"""Consolidate the fitted formulas used by all faithfulness diagnostics.

The fitted model is relational: one table stores a formula per user-domain
unit, while a second stores its 50 evaluation images and predicted emotion
vectors.  Keeping these as two CSVs avoids repeating 47--61 formula fields on
every image row while retaining everything needed to reproduce formula swap,
concept ablation, image swap, weight alignment, and qualitative search.

Usage::

    uv run python -m src.experiments.make_formula_database

Use ``--output-dir`` to build an isolated verification copy.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from src.config import PROJECT_ROOT

UNIT_KEY = ["setting", "fold", "domain", "user_id"]
IMAGE_KEY = UNIT_KEY + ["stimulus_id"]
SETTINGS = {
    "anchored": PROJECT_ROOT / "output" / "formulas",
    "unanchored": PROJECT_ROOT / "output" / "unanchored" / "formulas",
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


def _read(setting: str, base: Path, name: str) -> tuple[pd.DataFrame, dict]:
    path = base / name
    if not path.is_file():
        raise FileNotFoundError(f"missing formula source: {path.relative_to(PROJECT_ROOT)}")
    data = pd.read_csv(path, low_memory=False)
    data.insert(0, "setting", setting)
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    data.insert(1, "source_file", relative)
    metadata = {
        "setting": setting,
        "kind": name.removesuffix(".csv"),
        "source_file": relative,
        "rows": len(data),
        "sha256": _sha256(path),
    }
    return data, metadata


def _validate_units(units: pd.DataFrame) -> None:
    if len(units) != 774:
        raise ValueError(f"expected 774 formula units, got {len(units)}")
    if units.duplicated(UNIT_KEY, keep=False).any():
        raise ValueError("duplicate setting/user-domain formula units")
    counts = units.groupby("setting").size().to_dict()
    if counts != {"anchored": 387, "unanchored": 387}:
        raise ValueError(f"unexpected formula-unit counts: {counts}")
    expected_variants = {"anchored": "C", "unanchored": "plain"}
    for setting, variant in expected_variants.items():
        observed = set(units.loc[units.setting == setting, "variant"].astype(str))
        if observed != {variant}:
            raise ValueError(f"{setting}: expected variant {variant}, got {observed}")


def _validate_images(units: pd.DataFrame, images: pd.DataFrame) -> None:
    if len(images) != 38_700:
        raise ValueError(f"expected 38,700 formula-image rows, got {len(images)}")
    if images.duplicated(IMAGE_KEY, keep=False).any():
        raise ValueError("duplicate setting/user-domain/image rows")
    per_unit = images.groupby(UNIT_KEY).size()
    if not (per_unit == 50).all():
        raise ValueError(f"every formula unit must have 50 images:\n{per_unit[per_unit != 50].head()}")
    image_units = images[UNIT_KEY].drop_duplicates().sort_values(UNIT_KEY).reset_index(drop=True)
    formula_units = units[UNIT_KEY].sort_values(UNIT_KEY).reset_index(drop=True)
    if not image_units.equals(formula_units):
        raise ValueError("formula-unit and formula-image keys do not match")

    # Both settings must evaluate the same held-out images and ratings.
    comparison_key = ["fold", "domain", "user_id", "stimulus_id"]
    anchored = images[images.setting == "anchored"].set_index(comparison_key)
    unanchored = images[images.setting == "unanchored"].set_index(comparison_key)
    if not anchored.index.equals(unanchored.index):
        raise ValueError("anchored and unanchored evaluation-image keys differ")
    if not np.array_equal(
        anchored["y_true"].to_numpy(float),
        unanchored["y_true"].to_numpy(float),
        equal_nan=True,
    ):
        raise ValueError("anchored and unanchored true ratings differ")


def _stage1_accuracy() -> tuple[pd.DataFrame, list[dict]]:
    anchored_path = SETTINGS["anchored"] / "stage1_accuracy.csv"
    unanchored_path = SETTINGS["unanchored"] / "stage1_accuracy.csv"
    anchored = pd.read_csv(anchored_path)
    unanchored = pd.read_csv(unanchored_path)
    key = ["fold", "domain", "emotion"]
    paired = anchored.merge(
        unanchored, on=key, suffixes=("_anchored", "_unanchored"),
        validate="one_to_one",
    )
    for column in ["n_images", "srocc", "plcc"]:
        left = paired[f"{column}_anchored"].to_numpy(float)
        right = paired[f"{column}_unanchored"].to_numpy(float)
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError(f"Stage-1 {column} differs between formula settings")
    if len(anchored) != 105:
        raise ValueError(f"expected 105 fold-domain-emotion rows, got {len(anchored)}")
    metadata = [
        {
            "setting": setting,
            "kind": "stage1_accuracy",
            "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
            "rows": 105,
            "sha256": _sha256(path),
        }
        for setting, path in [
            ("anchored", anchored_path), ("unanchored", unanchored_path)
        ]
    ]
    return anchored.sort_values(key).reset_index(drop=True), metadata


def build(output_dir: Path) -> None:
    unit_parts = []
    image_parts = []
    manifest = []
    for setting, base in SETTINGS.items():
        units, unit_meta = _read(setting, base, "user_formula.csv")
        images, image_meta = _read(setting, base, "per_image.csv")
        images["stimulus_id"] = images["stimulus_id"].astype(str)
        unit_parts.append(units)
        image_parts.append(images)
        manifest.extend([unit_meta, image_meta])

    units = pd.concat(unit_parts, ignore_index=True, sort=False)
    images = pd.concat(image_parts, ignore_index=True, sort=False)
    # Preserve each source file's row order.  Correlation is mathematically
    # permutation-invariant, but changing summation order can move PLCC at the
    # 1e-16 level and prevents byte-identical downstream CSV reproduction.
    units = units.reset_index(drop=True)
    images = images.reset_index(drop=True)
    _validate_units(units)
    _validate_images(units, images)
    stage1, stage1_meta = _stage1_accuracy()
    manifest.extend(stage1_meta)

    manifest_frame = pd.DataFrame(manifest).sort_values(
        ["kind", "setting"], kind="stable"
    ).reset_index(drop=True)
    _atomic_csv(units, output_dir / "formula_units_final.csv")
    _atomic_csv(images, output_dir / "formula_images_final.csv")
    _atomic_csv(stage1, output_dir / "stage1_accuracy.csv")
    _atomic_csv(manifest_frame, output_dir / "formula_sources.csv")

    print(f"formula units: {len(units):,} (387 anchored + 387 unanchored)")
    print(f"formula images: {len(images):,} (50 per unit)")
    print(f"Stage-1 rows: {len(stage1):,}")
    print(f"wrote consolidated formula database to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output",
        help="directory for the four consolidated release CSVs",
    )
    args = parser.parse_args()
    build(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
