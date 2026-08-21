r"""Render Table 1 (tab:main) straight from output/raw_all_final.csv.

Every number in the table is read from the merged per-unit results file --
nothing is refitted here -- so the table cannot drift from the runs that
produced it. The upper bounds are the one exception and come from
output/upper_bounds.csv, written by make_upper_bounds.py.

The reported configuration is fixed at the top: Qwen3-VL 8B, the
population-anchored Stage-2, 100 ratings per user, averaged over three seeds.

Row set, after the reviewer settled the design:

  Population (GIAA)     no personalization at all
  Direct                no mediator, per-user head on raw features
  Random                content-free mediator of the same width
  PCA                   unsupervised compression to the same width
  Hybrid ridge->ridge   the paper's design
  Hybrid MLP->MLP seq.  same emotions, both stages an MLP, emotion loss only
  Hybrid MLP->MLP joint same emotions, both stages an MLP, emotion+score loss

Shuffled is no longer a Table 1 row; Random carries the content-free control.

Usage:  python -m src.experiments.make_tab_main [out.tex]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.data import DOMAINS                        # noqa: E402
from src.experiments.paper_paths import TABLES, ensure   # noqa: E402
from src.utils.metrics import sem, wilcoxon_paired       # noqa: E402

try:                        # the repo path has Thai characters in it
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "output" / "raw_all_final.csv"
BOUNDS = ROOT / "output" / "upper_bounds.csv"

BACKBONE, VARIANT, N_TRAIN = "qwen8b", "C", 100
UNIT = ["fold", "domain", "user_id"]

#: (section heading or None, mediator, head, printed label)
#:
#: Only "emotion" has a learned Stage-1, so only it can carry an MLP
#: extractor and a jointly trained one. Direct has no mediator at all,
#: Random is a fixed projection and PCA a fixed unsupervised basis -- there
#: is no extractor there to make nonlinear, so their MLP design is the MLP
#: predictor and there is no third variant. Padding them out would invent a
#: distinction the models do not have.
ROWS = [
    ("No personalization (GIAA)", "population", "ridge", "Population (GIAA)"),

    ("No mediator", "identity", "ridge", "Direct"),
    (None, "identity", "mlp", "Direct"),

    ("Content-free mediator", "random", "ridge", "Random"),
    (None, "random", "mlp", "Random"),

    ("Compressed mediator", "pca", "ridge", "PCA"),
    (None, "pca", "mlp", "PCA"),

    ("Emotion bottleneck (ours)", "emotion", "ridge", "Hybrid"),
    (None, "emotion_mlp", "mlp", "Hybrid"),
    (None, "emotion_joint", "mlp", "Hybrid"),
]

#: what the "Stage 2" column prints, per (mediator, head)
ARROW = chr(92) + 'rightarrow'
DESIGN = {
    ("population", "ridge"): "---",
    ("identity", "ridge"): "Ridge",
    ("identity", "mlp"): "MLP",
    ("random", "ridge"): "Ridge",
    ("random", "mlp"): "MLP",
    ("pca", "ridge"): "Ridge",
    ("pca", "mlp"): "MLP",
    ("emotion", "ridge"): f"Ridge ${ARROW}$ Ridge",
    ("emotion_mlp", "mlp"): f"MLP ${ARROW}$ MLP (seq.)",
    ("emotion_joint", "mlp"): f"MLP ${ARROW}$ MLP (joint)",
}
REFERENCE = ("emotion", "ridge")     # what the dagger tests against

DOMAIN_COLS = [(d, d.capitalize()) for d in DOMAINS] + [(None, "Average")]


def _num(x: float) -> str:
    """.478 -- three decimals, no leading zero (negatives keep the sign)."""
    if not np.isfinite(x):
        return "---"
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def _cell(mean: float, se: float, bold: bool, dagger: bool) -> str:
    body = f"{_num(mean)}_{{\\pm {_num(se)}}}"
    if bold:
        body = r"\bm{" + body + "}"
    if dagger:
        body += r"^\dagger"
    return f"${body}$"


def load() -> pd.DataFrame:
    """Per-unit results for the reported configuration, averaged over seeds.

    Averaging per unit before summarizing (rather than pooling seed-rows) is
    what makes the Wilcoxon test paired over the 387 units it claims.
    """
    d = pd.read_csv(RAW, low_memory=False)
    d = d[(d.backbone == BACKBONE) & (d.variant == VARIANT)
          & (d.n_train == N_TRAIN)]
    keep = {(m, h) for _, m, h, _ in ROWS}
    d = d[[(m, h) in keep for m, h in zip(d.mediator, d["head"])]]
    return d.groupby(["mediator", "head"] + UNIT, as_index=False)[
        ["srocc", "plcc", "eff_dof"]].mean()


def _slice(df, med, head):
    return df[(df.mediator == med) & (df["head"] == head)].set_index(UNIT)


def build(df: pd.DataFrame, strict: bool = True) -> tuple[dict, dict, list]:
    ref = _slice(df, *REFERENCE)
    stats, missing = {}, []
    for _, med, head, _ in ROWS:
        s = _slice(df, med, head)
        if s.empty:
            missing.append(f"{med}+{head}")
            continue
        r = {"n": len(s), "eff_dof": s["eff_dof"].mean()}
        for dom, _label in DOMAIN_COLS:
            d = s if dom is None else s[s.index.get_level_values("domain") == dom]
            for m in ("srocc", "plcc"):
                r[(dom, m, "mean")] = float(d[m].mean())
                r[(dom, m, "sem")] = float(sem(d[m]))
        for m in ("srocc", "plcc"):
            j = s[[m]].merge(ref[[m]], left_index=True, right_index=True,
                             suffixes=("", "_ref")).dropna()
            p = (np.nan if (med, head) == REFERENCE
                 else wilcoxon_paired(j[m], j[f"{m}_ref"]))
            r[("sig", m)] = bool(np.isfinite(p) and p < 0.05)
            r[("p", m)] = p
        stats[(med, head)] = r
    if missing and strict:
        raise SystemExit("missing rows in raw_all_final.csv: " + ", ".join(missing))

    best = {}
    for dom, _label in DOMAIN_COLS:
        for m in ("srocc", "plcc"):
            best[(dom, m)] = max(v[(dom, m, "mean")] for v in stats.values())
    return stats, best, missing


def med_group(med: str) -> str:
    """The emotion rows are one mediator wearing three designs, so the
    mediator column names it once and the Stage-2 column carries the rest.
    """
    return "emotion" if med.startswith("emotion") else med


def render(stats, best, bounds: pd.DataFrame | None) -> str:
    seen: set[str] = set()
    out = [r"\begin{tabular}{llc cc cc cc cc}", r"\toprule",
           r"\multirow{2}{*}{Mediator} & \multirow{2}{*}{Stage 2} & "
           r"\multirow{2}{*}{eff.\ DoF} &",
           " ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}} &"
                    for _, lab in DOMAIN_COLS).rstrip("&") + r"\\",
           r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
           " & & & " + " & ".join(["SROCC", "PLCC"] * len(DOMAIN_COLS)) + r" \\",
           r"\midrule"]

    for section, med, head, label in ROWS:
        shown = label if med_group(med) not in seen else ""
        seen.add(med_group(med))
        if section:
            if med != "population":
                out.append(r"\addlinespace")
            out.append(rf"\multicolumn{{11}}{{l}}{{\textit{{{section}}}}}\\")
        stage2 = DESIGN[(med, head)]
        if (med, head) not in stats:
            # still running -- keep the row so the layout is final
            out.append(f"{shown} & {stage2} & \\na & "
                       + " & ".join([r"\na"] * 2 * len(DOMAIN_COLS)) + r" \\")
            continue
        r = stats[(med, head)]
        dof = "\\na" if not np.isfinite(r["eff_dof"]) else f"{r['eff_dof']:.1f}"
        if med == "population":
            stage2, dof = "---", "\\na"
        cells = []
        for dom, _lab in DOMAIN_COLS:
            for m in ("srocc", "plcc"):
                mean = r[(dom, m, "mean")]
                cells.append(_cell(mean, r[(dom, m, "sem")],
                                   bold=np.isclose(mean, best[(dom, m)]),
                                   dagger=r[("sig", m)]))
        out.append(f"{shown} & {stage2} & {dof} & " + " & ".join(cells) + r" \\")

    if bounds is not None:
        out += [r"\addlinespace", r"\midrule",
                r"\multicolumn{11}{l}{\textit{Upper bounds}}\\"]
        for _, b in bounds.iterrows():
            cells = []
            for dom, _lab in DOMAIN_COLS:
                key = "avg" if dom is None else dom
                for m in ("srocc", "plcc"):
                    mean = b[f"{key}_{m}_mean"]
                    se = b.get(f"{key}_{m}_sem", np.nan)
                    cells.append(f"${_num(mean)}$" if not np.isfinite(se)
                                 else _cell(mean, se, False, False))
            out.append(f"{b['label']} & {b['stage2']} & {b['eff_dof']} & "
                       + " & ".join(cells) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    strict = "--strict" in argv
    argv = [a for a in argv if a != "--strict"]
    df = load()
    stats, best, missing = build(df, strict=strict)
    if missing:
        print("[pending] rows not yet computed, rendered as dashes: "
              + ", ".join(missing), file=sys.stderr)
    bounds = pd.read_csv(BOUNDS) if BOUNDS.exists() else None
    if bounds is None:
        print("[warn] no output/upper_bounds.csv -- table written without the "
              "GT-emotion and test-retest rows", file=sys.stderr)

    for (med, head), r in stats.items():
        print(f"  {med + '+' + head:22s} n={r['n']:4d}  "
              f"SROCC {r[(None, 'srocc', 'mean')]:.4f}  "
              f"PLCC {r[(None, 'plcc', 'mean')]:.4f}  "
              f"p_vs_ref={r[('p', 'srocc')]:.3g}")

    tex = render(stats, best, bounds)
    ensure()
    dest = Path(argv[0]) if argv else TABLES / "tab1_main.tex"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tex, encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
