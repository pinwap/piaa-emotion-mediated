r"""Table: does the emotion bottleneck still pay as the backbone gets stronger?

Three rows per backbone rather than two. Under the population anchor the
interesting question moved: every personalized row now starts from the
population prediction, so "does personalizing beat not personalizing" is no
longer answered by the design and has to be read off the table. Printing
Population alongside Direct and Hybrid answers two questions at once --
whether personalization pays at all on this backbone, and what the bottleneck
costs against unrestricted personalization -- without a second table.

  bold      the better of Direct / Hybrid in that cell
  \dagger   that row differs significantly from Population on the same
            backbone and domain (paired Wilcoxon over user-domain units)

Read from output/raw_all_final.csv, not from a dedicated backbone run. These
three rows are deterministic given the fold split -- the mediator is a ridge,
Direct and Population have no stochastic part, and the seed only moves the
random and shuffled controls -- which is verified before the table is built.

Usage:  python -m src.experiments.make_tab_backbone_final
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.paper_paths import TABLES, VARIANT, N_TRAIN, ensure  # noqa: E402
from src.modeling.backbones import backbone_label                          # noqa: E402
from src.utils.metrics import sem, wilcoxon_paired                         # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "output" / "raw_all_final.csv"
UNIT = ["fold", "domain", "user_id"]

BACKBONES = ["clip", "clip_ft", "clip_ft_emo", "qwen4b", "qwen8b"]
ROWS = [("population", "Population"), ("identity", "Direct"),
        ("emotion", "Hybrid")]
REFERENCE = "population"
COMPARED = ["identity", "emotion"]      # what bold ranks; Population is the ruler

DOMAINS = [("art", "Artwork"), ("fashion", "Fashion"),
           ("landscape", "Landscape"), (None, "Average")]


def _num(x: float) -> str:
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def load() -> pd.DataFrame:
    d = pd.read_csv(RAW, low_memory=False)
    d = d[(d.variant == VARIANT) & (d.n_train == N_TRAIN)
          & (d["head"] == "ridge")
          & d.mediator.isin([m for m, _ in ROWS])]

    # the three rows must not depend on the seed, or averaging them would be
    # hiding variance the table does not report
    for (bb, m), g in d.groupby(["backbone", "mediator"]):
        p = g.pivot_table(index=UNIT, columns="seed", values="srocc")
        if p.shape[1] > 1:
            w = float((p.max(axis=1) - p.min(axis=1)).max())
            if w > 1e-12:
                raise SystemExit(f"{bb}/{m} moves by {w:.2e} across seeds; "
                                 f"this table assumes it does not")
    return d.groupby(["backbone", "mediator"] + UNIT,
                     as_index=False)[["srocc", "plcc"]].mean()


def main() -> int:
    df = load()
    have = [b for b in BACKBONES if b in set(df.backbone)]
    missing = [b for b in BACKBONES if b not in have]
    if missing:
        print(f"[warn] no rows for {missing}; omitted", file=sys.stderr)

    stats = {}
    for bb in have:
        d = df[df.backbone == bb]
        ref = d[d.mediator == REFERENCE].set_index(UNIT)
        for med, _lab in ROWS:
            s = d[d.mediator == med].set_index(UNIT)
            r = {"n": len(s)}
            for dom, _l in DOMAINS:
                sd = s if dom is None else s[s.index.get_level_values("domain") == dom]
                rd = ref if dom is None else ref[ref.index.get_level_values("domain") == dom]
                for m in ("srocc", "plcc"):
                    r[(dom, m, "mean")] = float(sd[m].mean())
                    r[(dom, m, "sem")] = float(sem(sd[m]))
                    j = sd[[m]].merge(rd[[m]], left_index=True, right_index=True,
                                      suffixes=("", "_p")).dropna()
                    p = (np.nan if med == REFERENCE
                         else wilcoxon_paired(j[m], j[f"{m}_p"]))
                    r[(dom, m, "sig")] = bool(np.isfinite(p) and p < 0.05)
            stats[(bb, med)] = r

    out = [r"\begin{tabular}{ll cc cc cc cc}", r"\toprule",
           r"\multirow{2}{*}{Backbone} & \multirow{2}{*}{Model} & "
           + " ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}} &" for _, lab in DOMAINS).rstrip("&")
           + r"\\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
           " & & " + " & ".join(["SROCC", "PLCC"] * len(DOMAINS)) + r" \\",
           r"\midrule"]

    for bi, bb in enumerate(have):
        if bi:
            out.append(r"\addlinespace")
        best = {}
        for dom, _l in DOMAINS:
            for m in ("srocc", "plcc"):
                best[(dom, m)] = max(stats[(bb, c)][(dom, m, "mean")]
                                     for c in COMPARED)
        for ri, (med, lab) in enumerate(ROWS):
            r = stats[(bb, med)]
            cells = []
            for dom, _l in DOMAINS:
                for m in ("srocc", "plcc"):
                    mean = r[(dom, m, "mean")]
                    body = rf"{_num(mean)}_{{\pm {_num(r[(dom, m, 'sem')])}}}"
                    if med in COMPARED and np.isclose(mean, best[(dom, m)]):
                        body = r"\bm{" + body + "}"
                    if r[(dom, m, "sig")]:
                        body += r"^\dagger"
                    cells.append(f"${body}$")
            name = (rf"\multirow{{{len(ROWS)}}}{{*}}{{{backbone_label(bb)}}}"
                    if ri == 0 else "")
            out.append(f"{name} & {lab} & " + " & ".join(cells) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}"]

    ensure()
    dest = TABLES / "tab3_backbone.tex"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"population-anchored, n={N_TRAIN}, 387 units, SROCC (average over domains)")
    print(f"  {'backbone':16s} {'Pop':>7s} {'Direct':>8s} {'Hybrid':>8s}"
          f"   {'Hyb vs Dir p':>12s}")
    for bb in have:
        d = df[df.backbone == bb]
        p = d.pivot_table(index=UNIT, columns="mediator", values="srocc")
        print(f"  {backbone_label(bb):16s} {p.population.mean():7.4f} "
              f"{p.identity.mean():8.4f} {p.emotion.mean():8.4f}   "
              f"{wilcoxon_paired(p.emotion, p.identity):12.3g}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
