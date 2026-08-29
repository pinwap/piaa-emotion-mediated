"""Build manuscript-ready LaTeX tables from seed-averaged unit results.

This module is presentation-only. It never fits a model. All three-seed
outputs are validated and averaged within (fold, domain, user_id) before the
mean, SD, SEM, and table cells are computed. Significance markers are read
from the audited Holm results produced by make_holm_statistics.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.make_holm_statistics import (
    BACKBONE_LABELS,
    BACKBONES,
    METRICS,
    RAW,
    ROOT,
    UNIT,
    _validate_and_average,
    mlp_architecture_data,
)
from src.utils.metrics import participant_cluster_bootstrap


OUT_DIR = ROOT / "output" / "statistics"
HOLM = OUT_DIR / "holm_corrected.csv"
FAITHFULNESS_DIR = ROOT / "output" / "faithfulness"
DISTRIBUTION_RAW = (ROOT / "output" / "efficiency" / "clip" /
                    "raw_C_memo-emo_sd-emo_hist.csv")
DOMAIN_ORDER = [("art", "Artwork"), ("fashion", "Fashion"),
                ("landscape", "Landscape"), (None, "Average")]


def _short(value: float, digits: int = 3) -> str:
    text = f"{value:.{digits}f}"
    if text.startswith("-0."):
        return "-." + text[3:]
    if text.startswith("0."):
        return "." + text[2:]
    return text


def _cell(mean: float, sem: float, *, marker: str = "", bold: bool = False) -> str:
    value = f"{_short(mean)}_{{\\pm {_short(sem)}}}"
    if bold:
        value = f"\\bm{{{value}}}"
    suffix = f"^{{\\{marker}}}" if marker else ""
    return f"${value}{suffix}$"


def _plain_cell(mean: float, *, marker: str = "", bold: bool = False) -> str:
    value = _short(mean)
    if bold:
        value = f"\\bm{{{value}}}"
    suffix = f"^{{\\{marker}}}" if marker else ""
    return f"${value}{suffix}$"


def _p_cell(value: float) -> str:
    return r"$<.0001$" if value < 0.0001 else f"${_short(value, 4)}$"


def _mean_sem(values: pd.Series, participant_ids: pd.Series | None = None) -> tuple[float, float]:
    values = values.dropna().astype(float)
    if len(values) not in {129, 387}:
        raise ValueError(f"expected 129 or 387 unit values, got {len(values)}")
    if len(values) == 387:
        if participant_ids is None:
            raise ValueError("Average SEM requires participant IDs for cluster bootstrap")
        ids = participant_ids.loc[values.index]
        result = participant_cluster_bootstrap(values, ids)
        if result["n_participants"] != 129 or result["rows_per_participant"] != 3:
            raise ValueError("Average must contain 129 participants x 3 domains")
        return float(result["mean"]), float(result["sem"])
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def _stats(data: pd.DataFrame, query: dict, domain: str | None, metric: str):
    selected = data
    for column, value in query.items():
        selected = selected[selected[column] == value]
    if domain is not None:
        selected = selected[selected.domain == domain]
    ids = selected["user_id"] if domain is None else None
    return _mean_sem(selected[metric], ids)


def _sig(holm: pd.DataFrame, table: str, domain: str | None, metric: str,
         contrast: str) -> bool:
    domain_label = "average" if domain is None else domain
    row = holm[
        (holm.table == table)
        & (holm.domain == domain_label)
        & (holm.metric == metric)
        & (holm.contrast == contrast)
    ]
    if len(row) != 1:
        raise ValueError(
            f"expected one Holm row for {table}/{domain_label}/{metric}/{contrast}; "
            f"got {len(row)}"
        )
    return bool(row.iloc[0].reject_holm_0_05)


def _write(name: str, lines: list[str]) -> None:
    (OUT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _performance_header(first: str = "Model") -> list[str]:
    return [
        r"\begin{tabular}{ll cc cc cc cc}" if first == "Backbone" else
        r"\begin{tabular}{l cc cc cc cc}",
        r"\toprule",
        (r"\multirow{2}{*}{Backbone} & \multirow{2}{*}{Model} & "
         r"\multicolumn{2}{c}{Artwork} & \multicolumn{2}{c}{Fashion} & "
         r"\multicolumn{2}{c}{Landscape} & \multicolumn{2}{c}{Average} \\")
        if first == "Backbone" else
        (rf"\multirow{{2}}{{*}}{{{first}}} & \multicolumn{{2}}{{c}}{{Artwork}} & "
         r"\multicolumn{2}{c}{Fashion} & \multicolumn{2}{c}{Landscape} & "
         r"\multicolumn{2}{c}{Average} \\"),
        (r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
         r"\cmidrule(lr){9-10}") if first == "Backbone" else
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        (r"& & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC \\")
        if first == "Backbone" else
        r"& SROCC & PLCC & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC \\",
        r"\midrule",
    ]


def table1(raw: pd.DataFrame, holm: pd.DataFrame) -> None:
    models = [
        ("population", "Population"), ("identity", "Direct"),
        ("random", "Random"), ("pca", "PCA"), ("emotion", "Hybrid"),
    ]
    selected = raw[
        (raw.backbone == "qwen8b") & (raw.variant == "C")
        & (raw.n_train == 100) & (raw["head"] == "ridge")
        & raw.mediator.isin([m for m, _ in models])
    ]
    averaged = _validate_and_average(selected, ["mediator", "head"], "Table 1 LaTeX")
    best = {(d, m): max(_stats(averaged, {"mediator": med}, d, m)[0]
                        for med, _ in models)
            for d, _ in DOMAIN_ORDER for m in METRICS}
    lines = _performance_header("Mediator")
    for mediator, label in models:
        cells = []
        for domain, _ in DOMAIN_ORDER:
            for metric in METRICS:
                mean, sem = _stats(averaged, {"mediator": mediator}, domain, metric)
                marker = ""
                if mediator != "emotion":
                    contrast = f"Hybrid vs {label}"
                    marker = "dagger" if _sig(holm, "table1", domain, metric, contrast) else ""
                cells.append(_cell(mean, sem, marker=marker,
                                   bold=np.isclose(mean, best[(domain, metric)])))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% \dagger: Holm-adjusted p < .05 versus Hybrid Ridge (paired Wilcoxon per domain; participant-cluster bootstrap for Average)."]
    _write("table1_corrected.tex", lines)


def anchor(raw: pd.DataFrame, holm: pd.DataFrame) -> None:
    models = [("population", "Population"), ("identity", "Direct"),
              ("random", "Random"), ("pca", "PCA"), ("emotion", "Hybrid")]
    selected = raw[
        (raw.backbone == "qwen8b") & raw.variant.isin(["plain", "C"])
        & (raw.n_train == 100) & (raw["head"] == "ridge")
        & raw.mediator.isin([m for m, _ in models])
    ]
    averaged = _validate_and_average(
        selected, ["variant", "mediator", "head"], "Anchor LaTeX"
    )
    best = {(v, metric): max(_stats(averaged, {"variant": v, "mediator": med},
                                    None, metric)[0] for med, _ in models)
            for v in ["plain", "C"] for metric in METRICS}
    lines = [
        r"\begin{tabular}{l cc cc c}", r"\toprule",
        r"\multirow{2}{*}{Mediator} & \multicolumn{2}{c}{Unanchored} & \multicolumn{2}{c}{Population-anchored} & \multirow{2}{*}{$\Delta$ SROCC} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"& SROCC & PLCC & SROCC & PLCC & \\", r"\midrule",
    ]
    for mediator, label in models:
        cells = []
        means = {}
        for variant, variant_label in [("plain", "Unanchored"), ("C", "Anchored")]:
            for metric in METRICS:
                mean, sem = _stats(averaged, {"variant": variant, "mediator": mediator}, None, metric)
                means[(variant, metric)] = mean
                marker = ""
                if mediator != "emotion":
                    contrast = f"Hybrid vs {label} [{variant_label}]"
                    marker = "dagger" if _sig(holm, "table2_anchor", None, metric, contrast) else ""
                cells.append(_cell(mean, sem, marker=marker,
                                   bold=np.isclose(mean, best[(variant, metric)])))
        delta = means[("C", "srocc")] - means[("plain", "srocc")]
        lines.append(f"{label} & " + " & ".join(cells) + f" & ${delta:+.3f}$ " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% \dagger: Holm-adjusted participant-cluster bootstrap p < .05 versus Hybrid in the same anchoring condition."]
    _write("table_anchor_corrected.tex", lines)


def backbone(raw: pd.DataFrame, holm: pd.DataFrame) -> None:
    models = [("population", "Population"), ("identity", "Direct"), ("emotion", "Hybrid")]
    selected = raw[
        (raw.variant == "C") & (raw.n_train == 100) & (raw["head"] == "ridge")
        & raw.backbone.isin(BACKBONES) & raw.mediator.isin([m for m, _ in models])
    ]
    averaged = _validate_and_average(selected, ["backbone", "mediator", "head"], "Backbone LaTeX")
    names = {"clip": "CLIP frozen", "clip_ft": "CLIP-ft (score)",
             "clip_ft_emo": "CLIP-ft (emotion)", "qwen4b": "Qwen3-VL 4B",
             "qwen8b": "Qwen3-VL 8B"}
    lines = _performance_header("Backbone")
    for bi, bb in enumerate(BACKBONES):
        best = {(d, m): max(_stats(averaged, {"backbone": bb, "mediator": med}, d, m)[0]
                            for med, _ in models)
                for d, _ in DOMAIN_ORDER for m in METRICS}
        for mi, (mediator, label) in enumerate(models):
            cells = []
            for domain, _ in DOMAIN_ORDER:
                for metric in METRICS:
                    mean, sem = _stats(averaged, {"backbone": bb, "mediator": mediator}, domain, metric)
                    marker = ""
                    if mediator == "emotion":
                        marker = "dagger" if _sig(
                            holm, "tableS2", domain, metric,
                            f"Hybrid vs Population [{bb}]"
                        ) else ""
                    cells.append(_cell(mean, sem, marker=marker,
                                       bold=np.isclose(mean, best[(domain, metric)])))
            prefix = rf"\multirow{{3}}{{*}}{{{names[bb]}}} & " if mi == 0 else "& "
            lines.append(prefix + label + " & " + " & ".join(cells) + r" \\")
        if bi != len(BACKBONES) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% \dagger: Hybrid versus Population, Holm-adjusted p < .05 (paired Wilcoxon per domain; participant-cluster bootstrap for Average). Direct is descriptive."]
    _write("tableS2_backbone_corrected.tex", lines)


def efficiency(raw: pd.DataFrame, holm: pd.DataFrame) -> None:
    models = [("population", "Population"), ("identity", "Direct"), ("emotion", "Hybrid")]
    budgets = [10, 25, 50, 100]
    selected = raw[
        (raw.backbone == "qwen8b") & (raw.variant == "C")
        & raw.n_train.isin(budgets) & (raw["head"] == "ridge")
        & raw.mediator.isin([m for m, _ in models])
    ]
    averaged = _validate_and_average(
        selected, ["n_train", "mediator", "head"], "Efficiency LaTeX"
    )
    lines = [
        r"\begin{tabular}{cc cc cc cc cc}", r"\toprule",
        r"\multirow{2}{*}{$N$} & \multirow{2}{*}{Model} & \multicolumn{2}{c}{Artwork} & \multicolumn{2}{c}{Fashion} & \multicolumn{2}{c}{Landscape} & \multicolumn{2}{c}{Average} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"& & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC \\", r"\midrule",
    ]
    for ni, n_train in enumerate(budgets):
        best = {(d, m): max(_stats(averaged, {"n_train": n_train, "mediator": med}, d, m)[0]
                            for med, _ in models)
                for d, _ in DOMAIN_ORDER for m in METRICS}
        for mi, (mediator, label) in enumerate(models):
            cells = []
            for domain, _ in DOMAIN_ORDER:
                for metric in METRICS:
                    mean, sem = _stats(averaged, {"n_train": n_train, "mediator": mediator}, domain, metric)
                    marker = ""
                    if mediator != "population":
                        contrast = f"{label} vs Population [N={n_train}]"
                        marker = "dagger" if _sig(holm, "table_efficiency", domain, metric, contrast) else ""
                    cells.append(_cell(mean, sem, marker=marker,
                                       bold=np.isclose(mean, best[(domain, metric)])))
            prefix = rf"\multirow{{3}}{{*}}{{{n_train}}} & " if mi == 0 else "& "
            lines.append(prefix + label + " & " + " & ".join(cells) + r" \\")
        if ni != len(budgets) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% Domain SEM uses 129 participants; Average SEM uses a 129-participant cluster bootstrap with three domains per participant.",
              r"% \dagger: Holm-adjusted p < .05 versus Population at the same N (paired Wilcoxon per domain; participant-cluster bootstrap for Average)."]
    _write("table_efficiency_corrected.tex", lines)


def mlp_baseline(raw: pd.DataFrame) -> None:
    """Descriptive Ridge/MLP comparison moved out of the main table."""
    rows = [
        ("population", "ridge", "Population", "Ridge"),
        ("population", "mlp", "Population", "MLP"),
        ("identity", "ridge", "Direct", "Ridge"),
        ("identity", "mlp", "Direct", "MLP"),
        ("random", "ridge", "Random", "Ridge"),
        ("random", "mlp", "Random", "MLP"),
        ("pca", "ridge", "PCA", "Ridge"),
        ("pca", "mlp", "PCA", "MLP"),
        ("emotion", "ridge", "Hybrid (ours)", "Ridge"),
        ("emotion_mlp", "mlp", "Hybrid (ours)", "MLP"),
    ]
    keep = {(m, h) for m, h, _, _ in rows}
    selected = raw[
        (raw.backbone == "qwen8b") & (raw.variant == "C")
        & (raw.n_train == 100)
        & pd.Series([(m, h) in keep for m, h in zip(raw.mediator, raw["head"])],
                    index=raw.index)
    ]
    averaged = _validate_and_average(
        selected, ["mediator", "head"], "MLP baseline LaTeX"
    )
    lines = [
        r"\begin{tabular}{ll cc cc cc cc}", r"\toprule",
        (r"\multirow{2}{*}{Mediator} & \multirow{2}{*}{Head} & "
         r"\multicolumn{2}{c}{Artwork} & \multicolumn{2}{c}{Fashion} & "
         r"\multicolumn{2}{c}{Landscape} & \multicolumn{2}{c}{Average} \\"),
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"& & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC \\",
        r"\midrule",
    ]
    previous = None
    for mediator, head, label, head_label in rows:
        if previous is not None and label != previous:
            lines.append(r"\addlinespace")
        cells = []
        for domain, _ in DOMAIN_ORDER:
            for metric in METRICS:
                mean, sem = _stats(
                    averaged, {"mediator": mediator, "head": head}, domain, metric
                )
                cells.append(_cell(mean, sem))
        shown = rf"\multirow{{2}}{{*}}{{{label}}}" if label != previous else ""
        lines.append(f"{shown} & {head_label} & " + " & ".join(cells) + r" \\")
        previous = label
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% Descriptive table: no significance markers.",
              r"% Qwen3-VL 8B, population-anchored, N=100; seeds averaged within units."]
    _write("tableS_mlp_baseline_corrected.tex", lines)


def mlp_architecture(raw: pd.DataFrame) -> None:
    """Descriptive sequential/joint MLP architecture table."""
    ridge = raw[
        (raw.backbone == "qwen8b") & (raw.variant == "C")
        & (raw.n_train == 100) & (raw.mediator == "emotion")
        & (raw["head"] == "ridge")
    ]
    ridge = _validate_and_average(ridge, ["mediator", "head"],
                                  "MLP architecture Ridge LaTeX")
    mlp = mlp_architecture_data(ROOT)
    rows = [
        ("ridge", "Ridge $\\to$ Ridge (main)"),
        ("joint_lambda_0.1", r"MLP $\to$ MLP, joint $\lambda=0.1$"),
        ("joint_lambda_1", r"MLP $\to$ MLP, joint $\lambda=1$"),
        ("joint_lambda_10", r"MLP $\to$ MLP, joint $\lambda=10$"),
        ("sequential", r"MLP $\to$ MLP, sequential"),
    ]
    lines = [
        r"\begin{tabular}{@{}l cc cc cc cc@{}}", r"\toprule",
        (r"\multirow{2}{*}{Stage-1 $\to$ Stage-2} & "
         r"\multicolumn{2}{c}{Artwork} & \multicolumn{2}{c}{Fashion} & "
         r"\multicolumn{2}{c}{Landscape} & \multicolumn{2}{c}{Average} \\"),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"& SROCC & PLCC & SROCC & PLCC & SROCC & PLCC & SROCC & PLCC \\",
        r"\midrule",
    ]
    for i, (condition, label) in enumerate(rows):
        source = ridge if condition == "ridge" else mlp
        query = {"mediator": "emotion", "head": "ridge"} if condition == "ridge" else {"condition": condition}
        cells = []
        for domain, _ in DOMAIN_ORDER:
            for metric in METRICS:
                mean, sem = _stats(source, query, domain, metric)
                cells.append(_cell(mean, sem, bold=(condition == "ridge")))
        if i in {1, 4}:
            lines.append(r"\addlinespace")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% Descriptive table: no significance markers.",
              r"% Average SEM uses a 129-participant cluster bootstrap."]
    _write("tableS_mlp_architecture_corrected.tex", lines)


def distribution(raw_all: pd.DataFrame, holm: pd.DataFrame) -> None:
    raw = pd.read_csv(DISTRIBUTION_RAW, low_memory=False)
    mediators = [("population", "Population"), ("identity", "Direct"),
                 ("emotion", "Hybrid 7-d"), ("emotion_sd", "Hybrid 14-d"),
                 ("emotion_hist", "Hybrid 35-d")]
    budgets = [10, 25, 50, 100]
    selected = raw[
        raw.n_train.isin(budgets) & (raw["head"] == "ridge")
        & raw.mediator.isin(["population", "emotion", "emotion_sd", "emotion_hist"])
    ]
    averaged = _validate_and_average(
        selected, ["n_train", "mediator", "head"], "Distribution LaTeX"
    )
    direct = raw_all[
        (raw_all.backbone == "clip") & (raw_all.variant == "C")
        & raw_all.n_train.isin(budgets) & (raw_all["head"] == "ridge")
        & (raw_all.mediator == "identity")
    ]
    direct = _validate_and_average(
        direct, ["n_train", "mediator", "head"], "Distribution Direct LaTeX"
    )
    lines = [r"\begin{tabular}{@{}c cc ccc@{}}", r"\toprule",
             r"$N$ & Population & Direct & Hybrid 7-d & Hybrid 14-d & Hybrid 35-d \\",
             r"\midrule"]
    labels = dict(mediators)
    for n_train in budgets:
        cells = []
        for mediator, label in mediators:
            source = direct if mediator == "identity" else averaged
            mean, sem = _stats(
                source,
                {"n_train": n_train, "mediator": mediator},
                None,
                "srocc",
            )
            marker = ""
            if mediator.startswith("emotion"):
                contrast = f"{label} vs Population [N={n_train}]"
                marker = "dagger" if _sig(holm, "tableS_distribution", None, "srocc", contrast) else ""
            cells.append(_cell(mean, sem, marker=marker))
        lines.append(str(n_train) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% Values are mean \pm participant-clustered bootstrap SEM (129 participants, three domains each).",
              r"% \dagger: Holm-adjusted participant-cluster bootstrap p < .05 versus Population.",
              r"% One pooled-Average SROCC family contains all 12 Hybrid comparisons."]
    _write("tableS_distribution_corrected.tex", lines)

    tabular_end = lines.index(r"\end{tabular}") + 1
    full = [
        r"% ----------- distribution: sd 14-d, histogram 35-d",
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{threeparttable}",
        (r"\caption{\textbf{Distribution-valued Stage-1 targets.} CLIP frozen, "
         r"population-anchored, ridge head, three seeds averaged within each of "
         r"387 user--domain rows from 129 participants. Widths are 7 (per-emotion mean), 14 (mean and "
         r"across-rater standard deviation), and 35 (7 emotions $\times$ 5 "
         r"rating-histogram bins).}"),
        r"\label{tab:distribution}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        *lines[:tabular_end],
        r"\begin{tablenotes}[flushleft]\footnotesize",
        (r"\item Values are mean $\pm$ participant-cluster bootstrap SEM over "
         r"129 participants, with their three domains resampled as a block."),
        (r"\item $^\dagger$ Holm-adjusted two-sided participant-cluster bootstrap $p<.05$ "
         r"versus Population. One family contains the 12 Hybrid comparisons: "
         r"three mediator widths at four rating budgets."),
        (r"\item At $N=100$, the 14-d and 7-d variants have nearly identical "
         r"mean SROCC (.434 and .435). Among the Hybrid variants, the 35-d "
         r"histogram has the highest mean from $N=25$ onward and significantly "
         r"outperforms Population at each of these budgets; \cref{tab:rater_sd} "
         r"examines why adding across-rater standard deviation provides no "
         r"apparent gain."),
        (r"\item This experiment was run with frozen CLIP only and is reported "
         r"as a negative control for the choice of Stage-1 target, not as a "
         r"primary result."),
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    _write("tableS_distribution_full_corrected.tex", full)


def faithfulness(holm: pd.DataFrame) -> None:
    base = FAITHFULNESS_DIR
    conditions = [
        ("Anchored", pd.read_csv(base / "formula_swap.csv")),
        ("Unanchored", pd.read_csv(base / "formula_swap_unanchored.csv")),
    ]
    lines = [r"\begin{tabular}{ll cc cc}", r"\toprule",
             r"Setting & Formula & \multicolumn{2}{c}{Average performance} & \multicolumn{2}{c}{Holm-adjusted $p$} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"& & SROCC & PLCC & SROCC & PLCC \\", r"\midrule"]
    for ci, (setting, frame) in enumerate(conditions):
        mapping = [("own", "Own", "own"), ("pop", "Population mean", "population_mean"),
                   ("other", "Other user", "other_user")]
        for ri, (key, label, _) in enumerate(mapping):
            means = [_mean_sem(frame[f"{key}_{m}"], frame["user_id"])
                     for m in METRICS]
            perf = [_cell(mean, sem, bold=(key == "own")) for mean, sem in means]
            if key == "own":
                pvals = ["---", "---"]
            else:
                contrast = f"Own vs {label} [{setting}]"
                rows = holm[(holm.table == "figure_faithfulness")
                            & (holm.domain == "average")
                            & (holm.contrast == contrast)]
                pvals = [_p_cell(float(rows[rows.metric == m].iloc[0].p_holm))
                         for m in METRICS]
            prefix = rf"\multirow{{3}}{{*}}{{{setting}}} & " if ri == 0 else "& "
            lines.append(prefix + label + " & " + " & ".join(perf + pvals) + r" \\")
        if ci == 0:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% Adjusted p-values compare Own with the displayed replacement formula."]
    _write("table_faithfulness_formula_swap_corrected.tex", lines)

    ablation = pd.read_csv(base / "ablation.csv")
    ablations = [("full", "Full", None), ("top1", "Top-weight held", "Top-weight emotion held"),
                 ("avg1", "Average-weight held", "Average emotion held"),
                 ("bottom1", "Bottom-weight held", "Bottom-weight emotion held")]
    lines = [r"\begin{tabular}{l cc}", r"\toprule",
             r"Condition & SROCC & PLCC \\", r"\midrule"]
    for key, label, ref_label in ablations:
        cells = []
        for metric in METRICS:
            mean, sem = _mean_sem(
                ablation[f"{key}_{metric}"], ablation["user_id"]
            )
            marker = ""
            if ref_label:
                marker = "dagger" if _sig(
                    holm, "figure_faithfulness", None, metric,
                    f"Full model vs {ref_label}"
                ) else ""
            cells.append(_cell(mean, sem, marker=marker, bold=(key == "full")))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% \dagger: Holm-adjusted participant-cluster bootstrap p < .05 versus Full."]
    _write("table_faithfulness_ablation_corrected.tex", lines)

    image_tables = [("Anchored", pd.read_csv(base / "image_swap.csv"),
                     [("correct", "Correct image", None), ("swapped", "Swapped image", "Swapped image"),
                      ("no_image", "No-image emotion", "No-image emotion")]),
                    ("Unanchored", pd.read_csv(base / "image_swap_unanchored.csv"),
                     [("correct", "Correct image", None), ("swapped", "Swapped image", "Swapped image")])]
    lines = [r"\begin{tabular}{ll cc}", r"\toprule",
             r"Setting & Emotion input & SROCC & PLCC \\", r"\midrule"]
    for ti, (setting, frame, rowspec) in enumerate(image_tables):
        for ri, (key, label, control) in enumerate(rowspec):
            cells = []
            for metric in METRICS:
                mean, sem = _mean_sem(
                    frame[f"{key}_{metric}"], frame["user_id"]
                )
                marker = ""
                if control:
                    suffix = " [Unanchored]" if setting == "Unanchored" else ""
                    marker = "dagger" if _sig(
                        holm, "figure_faithfulness", None, metric,
                        f"Correct image vs {control}{suffix}"
                    ) else ""
                cells.append(_cell(mean, sem, marker=marker, bold=(key == "correct")))
            prefix = rf"\multirow{{{len(rowspec)}}}{{*}}{{{setting}}} & " if ri == 0 else "& "
            lines.append(prefix + label + " & " + " & ".join(cells) + r" \\")
        if ti == 0:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% \dagger: Holm-adjusted participant-cluster bootstrap p < .05 versus Correct image in the same setting."]
    _write("table_faithfulness_image_swap_corrected.tex", lines)


def weight_alignment(holm: pd.DataFrame) -> None:
    data = pd.read_csv(FAITHFULNESS_DIR / "weight_vs_empirical.csv")
    lines = [r"\begin{tabular}{l cc}", r"\toprule",
             r"Domain & Spearman & Pearson \\", r"\midrule"]
    for domain, label in DOMAIN_ORDER:
        d = data if domain is None else data[data.domain == domain]
        cells = []
        for metric in ["spearman", "pearson"]:
            mean, sem = _mean_sem(
                d[metric], d["user_id"] if domain is None else None
            )
            marker = "dagger" if _sig(
                holm, "table_weight_alignment", domain, metric,
                f"{metric.capitalize()} alignment vs zero"
            ) else ""
            cells.append(_cell(mean, sem, marker=marker))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% \dagger: one-sided Wilcoxon alignment-greater-than-zero p < .05 (family size 1)."]
    _write("table_weight_alignment_corrected.tex", lines)


def main(argv=None) -> int:
    global OUT_DIR, FAITHFULNESS_DIR, DISTRIBUTION_RAW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--holm", type=Path, default=HOLM)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--faithfulness-dir", type=Path,
                        default=FAITHFULNESS_DIR)
    parser.add_argument("--distribution-raw", type=Path,
                        default=DISTRIBUTION_RAW)
    args = parser.parse_args(argv)

    OUT_DIR = args.output_dir
    FAITHFULNESS_DIR = args.faithfulness_dir
    DISTRIBUTION_RAW = args.distribution_raw
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw, low_memory=False)
    holm = pd.read_csv(args.holm)
    table1(raw, holm)
    anchor(raw, holm)
    backbone(raw, holm)
    efficiency(raw, holm)
    mlp_baseline(raw)
    mlp_architecture(raw)
    distribution(raw, holm)
    faithfulness(holm)
    weight_alignment(holm)
    print("wrote manuscript tables to", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
