"""Rebuild the reported audit artifacts without retraining any model.

The compact release contains the seed-level evaluation database and
formula/diagnostic databases.  This script derives faithfulness CSVs, all
statistical tests, LaTeX tabular fragments, and data figures in an isolated
directory.  It never modifies the frozen files under ``output/``.

Usage:
    uv run python reproduce.py
    uv run python reproduce.py --output-dir tmp/my_release_check
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.experiments.analysis import faithfulness
from src.experiments.analysis.make_holm_statistics import generate as generate_statistics
from src.experiments.presentation import (
    build_table,
    fig_efficiency,
    fig_faithfulness,
    generate_holm_tables,
    generate_importance_table,
    generate_regularizer_table,
    generate_stage1_table,
    load_qualitative,
)
from src.utils.metrics import participant_cluster_bootstrap


ROOT = Path(__file__).resolve().parent
RELEASE_FILES = [
    ROOT / "output" / "raw_all_final.csv",
    ROOT / "output" / "formula_units_final.csv",
    ROOT / "output" / "formula_images_final.csv",
    ROOT / "output" / "stage1_accuracy.csv",
    ROOT / "output" / "upper_bound_units_final.csv",
    ROOT / "output" / "emotion_importance_units_final.csv",
    ROOT / "output" / "mlp_architecture_units_final.csv",
    ROOT / "output" / "reference_bounds_final.csv",
    ROOT / "output" / "regularizer_units_final.csv",
    ROOT / "output" / "supplementary_v2" / "test_retest_units.csv",
    ROOT / "output" / "statistics" / "holm_corrected.csv",
]


def _check_release_files() -> None:
    missing = [str(path.relative_to(ROOT)) for path in RELEASE_FILES if not path.exists()]
    if missing:
        raise SystemExit("Missing compact-release inputs:\n  " + "\n  ".join(missing))


def _verify_statistics(reproduced: Path, frozen: Path) -> None:
    """Require regenerated tests to match the frozen audit row for row."""
    left = pd.read_csv(reproduced).sort_values(
        ["table", "domain", "metric", "contrast"], kind="stable"
    ).reset_index(drop=True)
    right = pd.read_csv(frozen).sort_values(
        ["table", "domain", "metric", "contrast"], kind="stable"
    ).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        raise RuntimeError("Regenerated Holm audit has a different schema or row count")
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                left[column].to_numpy(float), right[column].to_numpy(float),
                rtol=0.0, atol=1e-14, equal_nan=True,
            ):
                delta = np.nanmax(np.abs(
                    left[column].to_numpy(float) - right[column].to_numpy(float)
                ))
                raise RuntimeError(
                    f"Regenerated Holm column {column!r} differs (max |diff|={delta})"
                )
        elif not left[column].fillna("<NA>").equals(right[column].fillna("<NA>")):
            raise RuntimeError(f"Regenerated Holm column {column!r} differs")


def _verify_test_retest_units() -> None:
    """Reaggregate the released within-user reference and verify its freeze.

    The expected tuple is a regression guard, not the calculation itself: the
    mean and interval below are recomputed from all 387 released units.
    """
    source = ROOT / "output" / "supplementary_v2" / "test_retest_units.csv"
    data = pd.read_csv(source)
    required = {"user_id", "domain", "n", "srocc", "plcc"}
    missing = required.difference(data.columns)
    if missing:
        raise RuntimeError(f"test-retest units are missing columns: {sorted(missing)}")
    if len(data) != 387 or data.duplicated(["user_id", "domain"]).any():
        raise RuntimeError("test-retest reference must contain 387 unique user-domain units")
    if set(data["domain"]) != {"art", "fashion", "landscape"}:
        raise RuntimeError("test-retest reference has an unexpected domain set")
    counts = data.groupby("user_id")["domain"].nunique()
    if len(counts) != 129 or not counts.eq(3).all():
        raise RuntimeError("test-retest reference must contain three domains for 129 users")
    result = participant_cluster_bootstrap(data["srocc"], data["user_id"])
    expected = (0.5845714050896298, 0.550453993169238, 0.6169690464357855)
    observed = (result["mean"], result["ci95_lower"], result["ci95_upper"])
    if not np.allclose(observed, expected, rtol=0.0, atol=1e-14):
        raise RuntimeError(f"test-retest reference differs: observed {observed}")


def _verify_faithfulness_scope_checks(faithfulness_dir: Path) -> None:
    """Verify the two descriptive controls reported only in the supplement."""
    peer = pd.read_csv(faithfulness_dir / "weight_alignment_peer_null.csv")
    if len(peer) != 387 or peer.duplicated(["fold", "domain", "user_id"]).any():
        raise RuntimeError("peer alignment must contain 387 unique units")
    if peer["user_id"].eq(peer["peer_user_id"]).any():
        raise RuntimeError("peer alignment contains a self-pair")
    peer_bootstrap = participant_cluster_bootstrap(
        peer["excess_spearman"], peer["user_id"], alternative="greater"
    )
    peer_observed = (
        peer["own_spearman"].mean(),
        peer["peer_spearman"].mean(),
        peer_bootstrap["mean"],
        peer_bootstrap["ci95_lower"],
        peer_bootstrap["ci95_upper"],
    )
    peer_expected = (
        0.5249169435215948,
        0.45330380214101146,
        0.07161314138058326,
        0.04051310446659285,
        0.10151347360649689,
    )
    if not np.allclose(peer_observed, peer_expected, rtol=0.0, atol=1e-14):
        raise RuntimeError(f"peer alignment differs: observed {peer_observed}")

    control = pd.read_csv(
        faithfulness_dir / "image_swap_covariance_control.csv"
    )
    if len(control) != 387 or control.duplicated(
        ["fold", "domain", "user_id"]
    ).any():
        raise RuntimeError("image-swap control must contain 387 unique units")
    observed_mean = control["gaussian_matched_srocc"].mean()
    if not np.isclose(observed_mean, 0.38524577375215807, rtol=1e-2, atol=1e-3):
        raise RuntimeError(
            f"covariance-matched image control differs: observed {observed_mean}"
        )


def _stage_efficiency_slice(raw_path: Path, output_dir: Path) -> Path:
    """Stage only the reported primary-backbone rows used by the curve."""
    raw = pd.read_csv(raw_path, low_memory=False)
    selected = raw[
        (raw["backbone"] == "qwen8b")
        & (raw["variant"] == "C")
        & (raw["head"] == "ridge")
        & raw["mediator"].isin(["population", "identity", "emotion"])
    ]
    if selected.empty:
        raise RuntimeError("raw_all_final.csv has no qwen8b/C/ridge efficiency rows")
    destination = output_dir / "efficiency" / "qwen8b" / "raw.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "tmp" / "release_reproduction",
        help="isolated destination; frozen output/ files are never overwritten",
    )
    args = parser.parse_args(argv)
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    _check_release_files()
    _verify_test_retest_units()

    # Faithfulness reads the two compact formula databases from cfg.output_dir.
    for name in ("formula_units_final.csv", "formula_images_final.csv"):
        shutil.copy2(ROOT / "output" / name, work / name)
    cfg = Config(output_dir=work, figures_dir=work / "figures")
    faithfulness.run(cfg)
    _verify_faithfulness_scope_checks(work / "faithfulness")

    statistics_dir = work / "statistics"
    generate_statistics(
        raw_path=ROOT / "output" / "raw_all_final.csv",
        output_dir=statistics_dir,
        source_root=ROOT,
        faithfulness_dir=work / "faithfulness",
    )
    _verify_statistics(
        statistics_dir / "holm_corrected.csv",
        ROOT / "output" / "statistics" / "holm_corrected.csv",
    )

    tables_dir = work / "tables"
    generate_holm_tables([
        "--raw", str(ROOT / "output" / "raw_all_final.csv"),
        "--holm", str(statistics_dir / "holm_corrected.csv"),
        "--output-dir", str(tables_dir),
        "--faithfulness-dir", str(work / "faithfulness"),
    ])
    generate_stage1_table([
        "--source", str(ROOT / "output" / "stage1_accuracy.csv"),
        "--output", str(tables_dir / "tab4_emotion_accuracy.tex"),
    ])
    generate_importance_table([
        "--source", str(ROOT / "output" / "emotion_importance_units_final.csv"),
        "--output", str(tables_dir / "tab5_emotion_importance.tex"),
    ])
    generate_regularizer_table([
        "--source", str(ROOT / "output" / "regularizer_units_final.csv"),
        "--output", str(tables_dir / "tab_regularizer.tex"),
    ])
    qualitative_images, qualitative_units = load_qualitative()
    build_table(qualitative_images, qualitative_units, top_n=10).to_csv(
        tables_dir / "qualitative_candidates.csv", index=False
    )

    # Plot directly from the isolated reproduction tree. Keeping the figure
    # modules separate makes each output independently testable.
    _stage_efficiency_slice(ROOT / "output" / "raw_all_final.csv", work)
    cfg.backbone = "qwen8b"
    fig_efficiency.run(cfg, "sem")
    fig_faithfulness.run(cfg, "sem")

    print("\nRelease reproduction passed.")
    print(f"  statistics: {statistics_dir}")
    print(f"  tables:     {tables_dir}")
    print(f"  figures:    {work / 'figures'}")
    print("  frozen output/ files were not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
