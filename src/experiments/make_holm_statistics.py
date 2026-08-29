"""Recompute paper significance tests from seed-level per-unit results.

Seeds are averaged within each (fold, domain, user_id) unit before testing.
Domain-specific columns use paired Wilcoxon signed-rank tests.  Pooled Average
columns use a participant-cluster bootstrap so that a participant's three
correlated domain rows are always resampled as one block.

The audited blocks are:
* Table 1: Hybrid versus Population, Direct, Random, and PCA (Ridge only).
  Each domain-metric family contains four tests.
* Table S2 (backbones): Hybrid versus Population for each of five backbones.
  Each domain-metric family contains five tests.  Direct remains a descriptive
  unrestricted-personalization reference and receives no significance marker.
* Anchored/unanchored, efficiency, faithfulness, weight alignment, MLP
  architecture, and distribution-valued mediator results. Their complete
  family definitions are written beside the corrected p-values.

The pooled Average contains 387 user-domain rows but only 129 independent
participants.  It is corrected as its own family rather than mixed into any
domain-specific family.

Usage:
    python -m src.experiments.make_holm_statistics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.utils.metrics import participant_cluster_bootstrap, wilcoxon_paired


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "output" / "raw_all_final.csv"
OUT = ROOT / "output" / "statistics" / "holm_corrected.csv"
TABLE_S2_OUT = ROOT / "output" / "statistics" / "tableS2_holm_corrected.csv"
FAMILY_OUT = ROOT / "output" / "statistics" / "holm_family_summary.csv"
CHANGES_OUT = ROOT / "output" / "statistics" / "holm_dagger_changes.csv"
DEFINITIONS_OUT = ROOT / "output" / "statistics" / "holm_family_definitions.csv"
MLP_OUT = ROOT / "output" / "statistics" / "tableS_mlp_holm_corrected.csv"
DISTRIBUTION_OUT = ROOT / "output" / "statistics" / "tableS_distribution_holm_corrected.csv"

UNIT = ["fold", "domain", "user_id"]
DOMAINS: list[str | None] = ["art", "fashion", "landscape", None]
METRICS = ["srocc", "plcc"]
EXPECTED_SEEDS = {0, 1, 2}

TABLE1_MODELS = ["population", "identity", "random", "pca", "emotion"]
TABLE1_LABELS = {
    "population": "Population",
    "identity": "Direct",
    "random": "Random",
    "pca": "PCA",
    "emotion": "Hybrid",
}

BACKBONES = ["clip", "clip_ft", "clip_ft_emo", "qwen4b", "qwen8b"]
BACKBONE_MODELS = ["population", "identity", "emotion"]
BACKBONE_LABELS = {
    "population": "Population",
    "identity": "Direct",
    "emotion": "Hybrid",
}


def holm_adjust(p_values) -> np.ndarray:
    """Return Holm-adjusted p-values in the input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("Holm correction requires a non-empty finite p-value vector")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")

    order = np.argsort(p, kind="stable")
    ranked = p[order]
    multipliers = len(p) - np.arange(len(p))
    adjusted_ranked = np.minimum(1.0, np.maximum.accumulate(ranked * multipliers))

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def _validate_and_average(
    data: pd.DataFrame,
    model_columns: list[str],
    block: str,
) -> pd.DataFrame:
    """Require exactly three seed rows per model/unit, then average them."""
    keys = model_columns + UNIT
    seed_rows = model_columns + UNIT + ["seed"]

    duplicated = data.duplicated(seed_rows, keep=False)
    if duplicated.any():
        example = data.loc[duplicated, seed_rows].head().to_dict("records")
        raise ValueError(f"{block}: duplicate model/unit/seed rows, e.g. {example}")

    observed = data.groupby(keys, observed=True)["seed"].agg(
        n_rows="size",
        seeds=lambda x: frozenset(int(v) for v in x),
    )
    bad = observed[(observed.n_rows != len(EXPECTED_SEEDS))
                   | (observed.seeds != frozenset(EXPECTED_SEEDS))]
    if not bad.empty:
        raise ValueError(
            f"{block}: expected seeds {sorted(EXPECTED_SEEDS)} exactly once "
            f"for every model/unit; first failures:\n{bad.head()}"
        )

    averaged = data.groupby(keys, as_index=False, observed=True)[METRICS].mean()
    counts = averaged.groupby(model_columns, observed=True).size()
    if not (counts == 387).all():
        raise ValueError(f"{block}: expected 387 averaged units per model; got\n{counts}")
    return averaged


def _paired_row(
    data: pd.DataFrame,
    *,
    table: str,
    block: str,
    domain: str | None,
    metric: str,
    candidate_query: dict[str, str],
    reference_query: dict[str, str],
    candidate_label: str,
    reference_label: str,
    contrast: str,
) -> dict:
    candidate = data
    reference = data
    for column, value in candidate_query.items():
        candidate = candidate[candidate[column] == value]
    for column, value in reference_query.items():
        reference = reference[reference[column] == value]

    if domain is not None:
        candidate = candidate[candidate.domain == domain]
        reference = reference[reference.domain == domain]

    candidate = candidate.set_index(UNIT)
    reference = reference.set_index(UNIT)
    if candidate.index.has_duplicates or reference.index.has_duplicates:
        raise ValueError(f"{table}/{contrast}: duplicated paired-unit index")

    paired = candidate[[metric]].join(
        reference[[metric]],
        how="inner",
        lsuffix="_candidate",
        rsuffix="_reference",
    ).dropna()

    expected_n = 387 if domain is None else 129
    if len(paired) != expected_n:
        raise ValueError(
            f"{table}/{contrast}/{domain or 'average'}/{metric}: "
            f"expected {expected_n} paired units, got {len(paired)}"
        )
    if not candidate.index.equals(reference.index):
        missing_candidate = reference.index.difference(candidate.index)
        missing_reference = candidate.index.difference(reference.index)
        raise ValueError(
            f"{table}/{contrast}: unit mismatch; missing candidate "
            f"{len(missing_candidate)}, missing reference {len(missing_reference)}"
        )

    a = paired[f"{metric}_candidate"]
    b = paired[f"{metric}_reference"]
    bootstrap = None
    if domain is None:
        user_ids = paired.index.get_level_values("user_id")
        bootstrap = participant_cluster_bootstrap(a - b, user_ids)
        p_raw = bootstrap["p_value"]
    else:
        p_raw = wilcoxon_paired(a, b)
    if not np.isfinite(p_raw):
        raise ValueError(f"{table}/{contrast}: test returned a non-finite p-value")

    domain_label = "average" if domain is None else domain
    return {
        "table": table,
        "block": block,
        "test": ("participant_cluster_bootstrap" if domain is None
                 else "paired_wilcoxon"),
        "alternative": "two-sided",
        "domain": domain_label,
        "metric": metric,
        "family_id": f"{table}__{domain_label}__{metric}",
        "contrast": contrast,
        "candidate": candidate_label,
        "reference": reference_label,
        "n_units": len(paired),
        "n_participants": (bootstrap["n_participants"] if bootstrap else len(paired)),
        "candidate_mean": float(a.mean()),
        "reference_mean": float(b.mean()),
        "mean_difference": float((a - b).mean()),
        "difference_sem": (bootstrap["sem"] if bootstrap else np.nan),
        "difference_ci95_lower": (bootstrap["ci95_lower"] if bootstrap else np.nan),
        "difference_ci95_upper": (bootstrap["ci95_upper"] if bootstrap else np.nan),
        "p_raw": p_raw,
    }


def table1_rows(raw: pd.DataFrame) -> list[dict]:
    selected = raw[
        (raw.backbone == "qwen8b")
        & (raw.variant == "C")
        & (raw.n_train == 100)
        & (raw["head"] == "ridge")
        & raw.mediator.isin(TABLE1_MODELS)
    ].copy()
    averaged = _validate_and_average(selected, ["mediator", "head"], "Table 1")

    rows = []
    for domain in DOMAINS:
        for metric in METRICS:
            for reference_model in ["population", "identity", "random", "pca"]:
                rows.append(_paired_row(
                    averaged,
                    table="table1",
                    block="ridge",
                    domain=domain,
                    metric=metric,
                    candidate_query={"mediator": "emotion", "head": "ridge"},
                    reference_query={"mediator": reference_model, "head": "ridge"},
                    candidate_label="Hybrid",
                    reference_label=TABLE1_LABELS[reference_model],
                    contrast=f"Hybrid vs {TABLE1_LABELS[reference_model]}",
                ))
    return rows


def mlp_rows(raw: pd.DataFrame) -> list[dict]:
    """Supplementary MLP rows, all compared with the reported Hybrid Ridge."""
    candidates = [
        ("population", "mlp", "Population MLP"),
        ("identity", "mlp", "Direct MLP"),
        ("random", "mlp", "Random MLP"),
        ("pca", "mlp", "PCA MLP"),
        ("emotion_mlp", "mlp", "Hybrid MLP sequential"),
        ("emotion_joint", "mlp", "Hybrid MLP joint"),
    ]
    keep = {(m, h) for m, h, _ in candidates} | {("emotion", "ridge")}
    selected = raw[
        (raw.backbone == "qwen8b")
        & (raw.variant == "C")
        & (raw.n_train == 100)
        & pd.Series(
            [(m, h) in keep for m, h in zip(raw.mediator, raw["head"])],
            index=raw.index,
        )
    ].copy()
    averaged = _validate_and_average(
        selected, ["mediator", "head"], "Supplementary MLP"
    )

    rows = []
    for domain in DOMAINS:
        for metric in METRICS:
            for mediator, head, label in candidates:
                rows.append(_paired_row(
                    averaged,
                    table="tableS_mlp",
                    block="mlp",
                    domain=domain,
                    metric=metric,
                    candidate_query={"mediator": mediator, "head": head},
                    reference_query={"mediator": "emotion", "head": "ridge"},
                    candidate_label=label,
                    reference_label="Hybrid Ridge",
                    contrast=f"{label} vs Hybrid Ridge",
                ))
    return rows


def mlp_architecture_data(source_root: Path = ROOT) -> pd.DataFrame:
    """Load and seed-average the MLP architecture/loss-weight ablation."""
    frames = []

    for lam in ["0.1", "10"]:
        for seed in [0, 1, 2]:
            path = (
                source_root / "output" / "MLP_SWEEP" / f"joint_lam{lam}_s{seed}"
                / "efficiency" / "qwen8b"
                / "raw_C_m_qwen8b_memo_joint.csv"
            )
            d = pd.read_csv(path, low_memory=False)
            d = d[
                (d.n_train == 100)
                & (d.mediator == "emotion_joint")
                & (d["head"] == "mlp")
            ][UNIT + ["seed"] + METRICS].copy()
            d["condition"] = f"joint_lambda_{lam}"
            frames.append(d)

    lambda_one_paths = [
        source_root / "output" / "thundering_lobster_modal_all_runs" / "lr"
        / "1_mlp-mse-s0"
        / "raw_C_m_qwen8b_midentity-random-shuffled-pca-emo-emo_mlp-emo_joint.csv",
        source_root / "output" / "thundering_lobster_modal_all_runs" / "lr"
        / "2_mlp-mse-s1"
        / "raw_C_m_qwen8b_midentity-random-shuffled-pca-emo-emo_mlp-emo_joint.csv",
        source_root / "output" / "thundering_lobster_modal_all_runs" / "lr"
        / "3_mlp-mse-s2"
        / "raw_C_m_qwen8b_midentity-random-shuffled-pca-emo-emo_mlp-emo_joint.csv",
    ]
    for path in lambda_one_paths:
        d = pd.read_csv(path, low_memory=False)
        d = d[(d.n_train == 100) & (d["head"] == "mlp")]
        for mediator, condition in [
            ("emotion_joint", "joint_lambda_1"),
            ("emotion_mlp", "sequential"),
        ]:
            x = d[d.mediator == mediator][UNIT + ["seed"] + METRICS].copy()
            x["condition"] = condition
            frames.append(x)

    combined = pd.concat(frames, ignore_index=True)
    return _validate_and_average(combined, ["condition"], "MLP architecture")


def mlp_architecture_rows(source_root: Path = ROOT) -> list[dict]:
    """MLP architecture/loss-weight ablation shown in the supplement."""
    averaged = mlp_architecture_data(source_root)
    candidates = [
        ("joint_lambda_0.1", "MLP joint lambda=0.1"),
        ("joint_lambda_1", "MLP joint lambda=1"),
        ("joint_lambda_10", "MLP joint lambda=10"),
    ]
    rows = []
    for domain in DOMAINS:
        for metric in METRICS:
            for condition, label in candidates:
                rows.append(_paired_row(
                    averaged,
                    table="tableS_mlp_architecture",
                    block="mlp_architecture",
                    domain=domain,
                    metric=metric,
                    candidate_query={"condition": condition},
                    reference_query={"condition": "sequential"},
                    candidate_label=label,
                    reference_label="MLP sequential",
                    contrast=f"{label} vs MLP sequential",
                ))
    return rows


def distribution_rows(source_root: Path = ROOT) -> list[dict]:
    """Distribution-valued mediator variants versus Population."""
    path = (
        source_root / "output" / "efficiency" / "clip"
        / "raw_C_memo-emo_sd-emo_hist.csv"
    )
    data = pd.read_csv(path, low_memory=False)
    budgets = [10, 25, 50, 100]
    mediators = ["population", "emotion", "emotion_sd", "emotion_hist"]
    data = data[
        data.n_train.isin(budgets)
        & data.mediator.isin(mediators)
        & (data["head"] == "ridge")
    ].copy()
    averaged = _validate_and_average(
        data, ["n_train", "mediator", "head"], "Distribution-valued mediator"
    )
    labels = {
        "emotion": "Hybrid 7-d",
        "emotion_sd": "Hybrid 14-d",
        "emotion_hist": "Hybrid 35-d",
    }
    rows = []
    for budget in budgets:
        for mediator, label in labels.items():
            rows.append(_paired_row(
                averaged,
                table="tableS_distribution",
                block="distribution_targets",
                domain=None,
                metric="srocc",
                candidate_query={
                    "n_train": budget, "mediator": mediator, "head": "ridge"
                },
                reference_query={
                    "n_train": budget, "mediator": "population", "head": "ridge"
                },
                candidate_label=f"{label} (N={budget})",
                reference_label=f"Population (N={budget})",
                contrast=f"{label} vs Population [N={budget}]",
            ))
    return rows


def anchor_rows(raw: pd.DataFrame) -> list[dict]:
    """Keep anchored and unanchored halves in the same metric family."""
    mediators = ["population", "identity", "random", "pca", "emotion"]
    labels = {
        "population": "Population",
        "identity": "Direct",
        "random": "Random",
        "pca": "PCA",
        "emotion": "Hybrid",
    }
    selected = raw[
        (raw.backbone == "qwen8b")
        & raw.variant.isin(["plain", "C"])
        & (raw.n_train == 100)
        & (raw["head"] == "ridge")
        & raw.mediator.isin(mediators)
    ].copy()
    averaged = _validate_and_average(
        selected, ["variant", "mediator", "head"], "Table 2 anchor"
    )

    rows = []
    for metric in METRICS:
        for variant, variant_label in [("plain", "Unanchored"), ("C", "Anchored")]:
            for reference_model in ["population", "identity", "random", "pca"]:
                rows.append(_paired_row(
                    averaged,
                    table="table2_anchor",
                    block=variant_label.lower(),
                    domain=None,
                    metric=metric,
                    candidate_query={
                        "variant": variant, "mediator": "emotion", "head": "ridge"
                    },
                    reference_query={
                        "variant": variant, "mediator": reference_model, "head": "ridge"
                    },
                    candidate_label=f"Hybrid ({variant_label})",
                    reference_label=f"{labels[reference_model]} ({variant_label})",
                    contrast=(f"Hybrid vs {labels[reference_model]} "
                              f"[{variant_label}]")
                ))
    return rows


def efficiency_rows(raw: pd.DataFrame) -> list[dict]:
    """Displayed personalized rows versus Population at four rating budgets."""
    budgets = [10, 25, 50, 100]
    selected = raw[
        (raw.backbone == "qwen8b")
        & (raw.variant == "C")
        & raw.n_train.isin(budgets)
        & (raw["head"] == "ridge")
        & raw.mediator.isin(["population", "identity", "emotion"])
    ].copy()
    averaged = _validate_and_average(
        selected, ["n_train", "mediator", "head"], "Efficiency"
    )

    rows = []
    for domain in DOMAINS:
        for metric in METRICS:
            for budget in budgets:
                for candidate_model, candidate_label in [
                    ("emotion", "Hybrid"), ("identity", "Direct")
                ]:
                    rows.append(_paired_row(
                        averaged,
                        table="table_efficiency",
                        block="rating_budget",
                        domain=domain,
                        metric=metric,
                        candidate_query={
                            "n_train": budget,
                            "mediator": candidate_model,
                            "head": "ridge",
                        },
                        reference_query={
                            "n_train": budget,
                            "mediator": "population",
                            "head": "ridge",
                        },
                        candidate_label=f"{candidate_label} (N={budget})",
                        reference_label=f"Population (N={budget})",
                        contrast=f"{candidate_label} vs Population [N={budget}]",
                    ))
    return rows


def backbone_rows(raw: pd.DataFrame) -> list[dict]:
    selected = raw[
        (raw.variant == "C")
        & (raw.n_train == 100)
        & (raw["head"] == "ridge")
        & raw.backbone.isin(BACKBONES)
        & raw.mediator.isin(BACKBONE_MODELS)
    ].copy()
    averaged = _validate_and_average(
        selected, ["backbone", "mediator", "head"], "Table S2"
    )

    rows = []
    for domain in DOMAINS:
        for metric in METRICS:
            for backbone in BACKBONES:
                rows.append(_paired_row(
                    averaged,
                    table="tableS2",
                    block="backbones",
                    domain=domain,
                    metric=metric,
                    candidate_query={
                        "backbone": backbone,
                        "mediator": "emotion",
                        "head": "ridge",
                    },
                    reference_query={
                        "backbone": backbone,
                        "mediator": "population",
                        "head": "ridge",
                    },
                    candidate_label=f"Hybrid ({backbone})",
                    reference_label=f"Population ({backbone})",
                    contrast=f"Hybrid vs Population [{backbone}]",
                ))
    return rows


def _validate_per_unit_frame(data: pd.DataFrame, block: str) -> None:
    missing = [c for c in UNIT if c not in data.columns]
    if missing:
        raise ValueError(f"{block}: missing unit columns {missing}")
    if data.duplicated(UNIT).any():
        raise ValueError(f"{block}: duplicate user-domain units")
    counts = data.groupby("domain").size().to_dict()
    if counts != {"art": 129, "fashion": 129, "landscape": 129}:
        raise ValueError(f"{block}: expected 129 units per domain, got {counts}")


def _paired_columns_row(
    data: pd.DataFrame,
    *,
    table: str,
    block: str,
    domain: str | None,
    metric: str,
    candidate_column: str,
    reference_column: str,
    candidate_label: str,
    reference_label: str,
    contrast: str,
) -> dict:
    d = data if domain is None else data[data.domain == domain]
    paired = d[["user_id", candidate_column, reference_column]].dropna()
    expected_n = 387 if domain is None else 129
    if len(paired) != expected_n:
        raise ValueError(
            f"{table}/{contrast}/{domain or 'average'}/{metric}: "
            f"expected {expected_n} paired units, got {len(paired)}"
        )
    a = paired[candidate_column].to_numpy(float)
    b = paired[reference_column].to_numpy(float)
    bootstrap = None
    if domain is None:
        bootstrap = participant_cluster_bootstrap(
            a - b, paired["user_id"].to_numpy()
        )
        p_raw = bootstrap["p_value"]
    else:
        p_raw = wilcoxon_paired(a, b)
    if not np.isfinite(p_raw):
        raise ValueError(f"{table}/{contrast}: test returned a non-finite p-value")
    domain_label = "average" if domain is None else domain
    return {
        "table": table,
        "block": block,
        "test": ("participant_cluster_bootstrap" if domain is None
                 else "paired_wilcoxon"),
        "alternative": "two-sided",
        "domain": domain_label,
        "metric": metric,
        "family_id": f"{table}__{domain_label}__{metric}",
        "contrast": contrast,
        "candidate": candidate_label,
        "reference": reference_label,
        "n_units": len(paired),
        "n_participants": (bootstrap["n_participants"] if bootstrap else len(paired)),
        "candidate_mean": float(np.mean(a)),
        "reference_mean": float(np.mean(b)),
        "mean_difference": float(np.mean(a - b)),
        "difference_sem": (bootstrap["sem"] if bootstrap else np.nan),
        "difference_ci95_lower": (bootstrap["ci95_lower"] if bootstrap else np.nan),
        "difference_ci95_upper": (bootstrap["ci95_upper"] if bootstrap else np.nan),
        "p_raw": p_raw,
    }


def faithfulness_rows(base: Path | None = None) -> list[dict]:
    """All displayed faithfulness panels share one domain-metric family."""
    base = base or ROOT / "output" / "faithfulness"
    ablation = pd.read_csv(base / "ablation.csv")
    swap_anchored = pd.read_csv(base / "formula_swap.csv")
    swap_plain = pd.read_csv(base / "formula_swap_unanchored.csv")
    image = pd.read_csv(base / "image_swap.csv")
    image_plain = pd.read_csv(base / "image_swap_unanchored.csv")
    for name, frame in [
        ("ablation", ablation),
        ("formula swap anchored", swap_anchored),
        ("formula swap unanchored", swap_plain),
        ("image swap", image),
        ("image swap unanchored", image_plain),
    ]:
        _validate_per_unit_frame(frame, name)

    rows = []
    for domain in DOMAINS:
        for metric in METRICS:
            # The detailed table marks both replacement formulas against Own;
            # the text also claims Population mean outperforms Other user.
            for frame, variant in [
                (swap_anchored, "Anchored"),
                (swap_plain, "Unanchored"),
            ]:
                for candidate, reference, candidate_label, reference_label in [
                    ("own", "pop", "Own", "Population mean"),
                    ("own", "other", "Own", "Other user"),
                    ("pop", "other", "Population mean", "Other user"),
                ]:
                    rows.append(_paired_columns_row(
                        frame,
                        table="figure_faithfulness",
                        block="formula_swap",
                        domain=domain,
                        metric=metric,
                        candidate_column=f"{candidate}_{metric}",
                        reference_column=f"{reference}_{metric}",
                        candidate_label=f"{candidate_label} ({variant})",
                        reference_label=f"{reference_label} ({variant})",
                        contrast=(f"{candidate_label} vs {reference_label} "
                                  f"[{variant}]")
                    ))

            for ablated, label in [
                ("top1", "Top-weight emotion held"),
                ("avg1", "Average emotion held"),
                ("bottom1", "Bottom-weight emotion held"),
            ]:
                rows.append(_paired_columns_row(
                    ablation,
                    table="figure_faithfulness",
                    block="concept_ablation",
                    domain=domain,
                    metric=metric,
                    candidate_column=f"full_{metric}",
                    reference_column=f"{ablated}_{metric}",
                    candidate_label="Full model",
                    reference_label=label,
                    contrast=f"Full model vs {label}",
                ))

            for control, label in [
                ("swapped", "Swapped image"),
                ("no_image", "No-image emotion"),
            ]:
                rows.append(_paired_columns_row(
                    image,
                    table="figure_faithfulness",
                    block="image_swap",
                    domain=domain,
                    metric=metric,
                    candidate_column=f"correct_{metric}",
                    reference_column=f"{control}_{metric}",
                    candidate_label="Correct image",
                    reference_label=label,
                    contrast=f"Correct image vs {label}",
                ))
            # The Results and Discussion also compare the two controls
            # directly: image-swapped emotions should be worse than a control
            # carrying no image-specific emotion information.  This contrast
            # uses the same 387 units as the other faithfulness diagnostics,
            # so it belongs in the same Holm family rather than a new family.
            rows.append(_paired_columns_row(
                image,
                table="figure_faithfulness",
                block="image_swap",
                domain=domain,
                metric=metric,
                candidate_column=f"no_image_{metric}",
                reference_column=f"swapped_{metric}",
                candidate_label="No-image emotion",
                reference_label="Swapped image",
                contrast="No-image emotion vs Swapped image",
            ))
            rows.append(_paired_columns_row(
                image_plain,
                table="figure_faithfulness",
                block="image_swap",
                domain=domain,
                metric=metric,
                candidate_column=f"correct_{metric}",
                reference_column=f"swapped_{metric}",
                candidate_label="Correct image (Unanchored)",
                reference_label="Swapped image (Unanchored)",
                contrast="Correct image vs Swapped image [Unanchored]",
            ))
    return rows


def weight_alignment_rows(base: Path | None = None) -> list[dict]:
    """One-sided tests that alignment is greater than zero."""
    base = base or ROOT / "output" / "faithfulness"
    data = pd.read_csv(base / "weight_vs_empirical.csv")
    _validate_per_unit_frame(data, "weight alignment")
    rows = []
    for domain in DOMAINS:
        d = data if domain is None else data[data.domain == domain]
        expected_n = 387 if domain is None else 129
        for metric in ["spearman", "pearson"]:
            finite = np.isfinite(d[metric].to_numpy(float))
            values = d.loc[finite, metric].to_numpy(float)
            if len(values) != expected_n:
                raise ValueError(
                    f"weight alignment/{domain or 'average'}/{metric}: "
                    f"expected {expected_n}, got {len(values)}"
                )
            bootstrap = None
            if domain is None:
                bootstrap = participant_cluster_bootstrap(
                    values, d.loc[finite, "user_id"].to_numpy(), alternative="greater"
                )
                p_raw = bootstrap["p_value"]
            else:
                p_raw = float(wilcoxon(
                    values, np.zeros(len(values)), method="approx", alternative="greater"
                ).pvalue)
            domain_label = "average" if domain is None else domain
            rows.append({
                "table": "table_weight_alignment",
                "block": "alignment_vs_zero",
                "test": ("participant_cluster_bootstrap" if domain is None
                         else "one_sample_wilcoxon"),
                "alternative": "greater",
                "domain": domain_label,
                "metric": metric,
                "family_id": f"table_weight_alignment__{domain_label}__{metric}",
                "contrast": f"{metric.capitalize()} alignment vs zero",
                "candidate": f"{metric.capitalize()} alignment",
                "reference": "Zero",
                "n_units": len(values),
                "n_participants": (bootstrap["n_participants"] if bootstrap else len(values)),
                "candidate_mean": float(np.mean(values)),
                "reference_mean": 0.0,
                "mean_difference": float(np.mean(values)),
                "difference_sem": (bootstrap["sem"] if bootstrap else np.nan),
                "difference_ci95_lower": (bootstrap["ci95_lower"] if bootstrap else np.nan),
                "difference_ci95_upper": (bootstrap["ci95_upper"] if bootstrap else np.nan),
                "p_raw": p_raw,
            })
    return rows


def apply_holm(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    family_sizes = out.groupby("family_id").size()
    out["family_size"] = out.family_id.map(family_sizes).astype(int)
    out["p_holm"] = np.nan

    for _family_id, index in out.groupby("family_id", sort=False).groups.items():
        out.loc[index, "p_holm"] = holm_adjust(out.loc[index, "p_raw"])

    out["reject_holm_0_05"] = out.p_holm < 0.05
    columns = [
        "table", "block", "test", "alternative", "domain", "metric",
        "family_id", "family_size", "contrast", "candidate", "reference",
        "n_units", "n_participants", "candidate_mean", "reference_mean",
        "mean_difference", "difference_sem", "difference_ci95_lower",
        "difference_ci95_upper",
        "p_raw", "p_holm",
        "reject_holm_0_05",
    ]
    return out[columns].sort_values(
        ["table", "domain", "metric", "contrast"], kind="stable"
    ).reset_index(drop=True)


def generate(
    *,
    raw_path: Path = RAW,
    output_dir: Path = OUT.parent,
    source_root: Path = ROOT,
    faithfulness_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate every corrected test without modifying canonical outputs."""
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    source_root = Path(source_root)
    faithfulness_dir = (
        Path(faithfulness_dir) if faithfulness_dir is not None
        else source_root / "output" / "faithfulness"
    )
    raw = pd.read_csv(raw_path, low_memory=False)
    result = apply_holm(
        table1_rows(raw)
        + mlp_architecture_rows(source_root)
        + distribution_rows(source_root)
        + anchor_rows(raw)
        + backbone_rows(raw)
        + efficiency_rows(raw)
        + faithfulness_rows(faithfulness_dir)
        + weight_alignment_rows(faithfulness_dir)
    )

    expected_families = {
        ("table1", 4): 8,
        ("tableS_mlp_architecture", 3): 8,
        ("tableS_distribution", 12): 1,
        ("tableS2", 5): 8,
        ("table2_anchor", 8): 2,
        ("table_efficiency", 8): 8,
        ("figure_faithfulness", 13): 8,
        ("table_weight_alignment", 1): 8,
    }
    observed = (
        result[["table", "family_id", "family_size"]]
        .drop_duplicates()
        .groupby(["table", "family_size"])
        .size()
        .to_dict()
    )
    if observed != expected_families:
        raise ValueError(f"unexpected family layout: {observed}; expected {expected_families}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "holm_corrected.csv"
    table_s2_out = output_dir / "tableS2_holm_corrected.csv"
    mlp_out = output_dir / "tableS_mlp_holm_corrected.csv"
    distribution_out = output_dir / "tableS_distribution_holm_corrected.csv"
    family_out = output_dir / "holm_family_summary.csv"
    changes_out = output_dir / "holm_dagger_changes.csv"
    definitions_out = output_dir / "holm_family_definitions.csv"

    result.to_csv(out, index=False, float_format="%.17g")
    result[result.table == "tableS2"].to_csv(
        table_s2_out, index=False, float_format="%.17g"
    )
    result[result.table == "tableS_mlp_architecture"].to_csv(
        mlp_out, index=False, float_format="%.17g"
    )
    result[result.table == "tableS_distribution"].to_csv(
        distribution_out, index=False, float_format="%.17g"
    )

    result["raw_significant_0_05"] = result.p_raw < 0.05
    family_summary = result.groupby(
        ["table", "domain", "metric", "family_id", "family_size"],
        as_index=False,
    ).agg(
        n_tests=("contrast", "size"),
        n_raw_significant=("raw_significant_0_05", "sum"),
        n_holm_significant=("reject_holm_0_05", "sum"),
        min_p_raw=("p_raw", "min"),
        max_p_raw=("p_raw", "max"),
        min_p_holm=("p_holm", "min"),
        max_p_holm=("p_holm", "max"),
    )
    family_summary.to_csv(family_out, index=False, float_format="%.17g")

    changes = result[
        result.raw_significant_0_05 != result.reject_holm_0_05
    ].copy()
    changes.to_csv(changes_out, index=False, float_format="%.17g")

    definitions = pd.DataFrame([
        {
            "table": "table1",
            "reported": "Main Table 1",
            "source": "output/raw_all_final.csv",
            "family_by": "domain x metric",
            "tests_per_family": 4,
            "contrasts": "Hybrid Ridge vs Population, Direct, Random, PCA",
        },
        {
            "table": "tableS_mlp_architecture",
            "reported": "MLP architecture supplementary table",
            "source": "output/MLP_SWEEP/ and output/thundering_lobster_modal_all_runs/",
            "family_by": "domain x metric",
            "tests_per_family": 3,
            "contrasts": ("joint lambda=0.1, 1 and 10 versus the sequential MLP; "
                          "Ridge is descriptive only"),
        },
        {
            "table": "tableS_distribution",
            "reported": "Distribution-valued Stage-1 targets",
            "source": "output/efficiency/clip/raw_C_memo-emo_sd-emo_hist.csv",
            "family_by": "metric; Average only",
            "tests_per_family": 12,
            "contrasts": ("Hybrid 7-d, 14-d and 35-d versus Population at "
                          "N = 10, 25, 50, 100"),
        },
        {
            "table": "table2_anchor",
            "reported": "Anchored vs unanchored table",
            "source": "output/raw_all_final.csv",
            "family_by": "metric; Average only; both halves combined",
            "tests_per_family": 8,
            "contrasts": ("Hybrid vs Population, Direct, Random, PCA under "
                          "both anchored and unanchored settings"),
        },
        {
            "table": "tableS2",
            "reported": "Backbone table",
            "source": "output/raw_all_final.csv",
            "family_by": "domain x metric",
            "tests_per_family": 5,
            "contrasts": "Hybrid vs Population across five backbones",
        },
        {
            "table": "table_efficiency",
            "reported": "Efficiency table/figure",
            "source": "output/raw_all_final.csv",
            "family_by": "domain x metric",
            "tests_per_family": 8,
            "contrasts": ("Hybrid and Direct vs Population at "
                          "N = 10, 25, 50, 100"),
        },
        {
            "table": "figure_faithfulness",
            "reported": "Faithfulness figure and detail tables",
            "source": "output/faithfulness/*.csv",
            "family_by": "domain x metric; correlated panels combined",
            "tests_per_family": 13,
            "contrasts": ("formula swap (6), concept ablation (3), "
                          "image swap (4)"),
        },
        {
            "table": "table_weight_alignment",
            "reported": "Weight-alignment table",
            "source": "output/faithfulness/weight_vs_empirical.csv",
            "family_by": "domain x metric",
            "tests_per_family": 1,
            "contrasts": "one-sided alignment greater than zero",
        },
        {
            "table": "stage1_accuracy",
            "reported": "Stage-1 emotion accuracy",
            "source": "output/stage1_emotion_acc/",
            "family_by": "descriptive only",
            "tests_per_family": 0,
            "contrasts": "none",
        },
        {
            "table": "emotion_importance",
            "reported": "Emotion importance",
            "source": "output/stage2_emotion_importance_matched/",
            "family_by": "descriptive only",
            "tests_per_family": 0,
            "contrasts": "none",
        },
    ])
    definitions.to_csv(definitions_out, index=False)

    print(f"wrote {len(result)} tests to {out}")
    print(f"wrote Table S2 tests to {table_s2_out}")
    print(f"wrote MLP tests to {mlp_out}")
    print(f"wrote distribution tests to {distribution_out}")
    print(f"wrote family summary to {family_out}")
    print(f"wrote {len(changes)} dagger changes to {changes_out}")
    print("family layout:")
    print(result.groupby(["table", "family_size"]).family_id.nunique().to_string())
    print("paired-unit counts:", sorted(result.n_units.unique()))

    priority = result[
        (result.table == "tableS2")
        & result.contrast.str.startswith("Hybrid vs Population")
    ][["domain", "metric", "contrast", "n_units", "candidate_mean", "reference_mean", "mean_difference", "p_raw", "p_holm",
       "reject_holm_0_05"]]
    print("\nTable S2 priority contrasts:")
    print(priority.to_string(index=False))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=OUT.parent)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--faithfulness-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    generate(
        raw_path=args.raw,
        output_dir=args.output_dir,
        source_root=args.source_root,
        faithfulness_dir=args.faithfulness_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
