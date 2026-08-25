r"""The swap experiment reported twice: with the population anchor and without.

The reviewer asked for both, and the reason shows up in the numbers. The
anchor hands every row the population prediction to start from, so it helps a
row in proportion to how little that row's mediator had already learned. The
content-free control gains most and the emotion bottleneck least, which
compresses exactly the gaps the mediator comparison rests on. Reporting only
the anchored table would hide that; reporting only the unanchored one would
drop the design the rest of the paper uses.

Both halves are the same units, the same seeds and the same selection
protocol -- only the Stage-2 anchor differs. The population row is identical
in both by construction (it uses no personal ratings at all), which is the
check that the two halves are comparable.

Usage:  python -m src.experiments.make_tab_anchor [out.tex]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.paper_paths import TABLES, ensure   # noqa: E402
from src.utils.metrics import sem, wilcoxon_paired       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "output" / "raw_all_final.csv"

BACKBONE, N_TRAIN, HEAD = "qwen8b", 100, "ridge"
UNIT = ["fold", "domain", "user_id"]

ROWS = [
    ("population", "Population (GIAA)"),
    ("identity", "Direct"),
    ("random", "Random"),
    ("shuffled", "Shuffled"),
    ("pca", "PCA"),
    ("emotion", "Hybrid (ours)"),
]
REFERENCE = "emotion"


def _num(x: float) -> str:
    if not np.isfinite(x):
        return "---"
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def load(variant: str) -> pd.DataFrame:
    d = pd.read_csv(RAW, low_memory=False)
    d = d[(d.backbone == BACKBONE) & (d.variant == variant)
          & (d.n_train == N_TRAIN) & (d["head"] == HEAD)
          & d.mediator.isin([m for m, _ in ROWS])]
    return d.groupby(["mediator"] + UNIT, as_index=False)[["srocc", "plcc"]].mean()


def column(variant: str) -> dict:
    df = load(variant)
    ref = df[df.mediator == REFERENCE].set_index(UNIT)
    out = {}
    for med, _label in ROWS:
        s = df[df.mediator == med].set_index(UNIT)
        if s.empty:
            raise SystemExit(f"no rows for {med} under variant {variant}")
        r = {"n": len(s)}
        for m in ("srocc", "plcc"):
            r[f"{m}_mean"] = float(s[m].mean())
            r[f"{m}_sem"] = float(sem(s[m]))
            j = s[[m]].merge(ref[[m]], left_index=True, right_index=True,
                             suffixes=("", "_ref")).dropna()
            p = np.nan if med == REFERENCE else wilcoxon_paired(j[m], j[f"{m}_ref"])
            r[f"{m}_p"] = p
            r[f"{m}_sig"] = bool(np.isfinite(p) and p < 0.05)
        out[med] = r
    return out


def render(plain: dict, anchored: dict) -> str:
    best = {}
    for tag, col in (("plain", plain), ("anch", anchored)):
        for m in ("srocc", "plcc"):
            best[(tag, m)] = max(v[f"{m}_mean"] for v in col.values())

    out = [r"\begin{tabular}{l cc cc c}", r"\toprule",
           r"\multirow{2}{*}{Mediator} & \multicolumn{2}{c}{Unanchored} &"
           r" \multicolumn{2}{c}{Population-anchored} &"
           r" \multirow{2}{*}{$\Delta$ SROCC} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r" & SROCC & PLCC & SROCC & PLCC & \\", r"\midrule"]

    for med, label in ROWS:
        cells = []
        for tag, col in (("plain", plain), ("anch", anchored)):
            r = col[med]
            for m in ("srocc", "plcc"):
                body = f"{_num(r[f'{m}_mean'])}_{{\\pm {_num(r[f'{m}_sem'])}}}"
                if np.isclose(r[f"{m}_mean"], best[(tag, m)]):
                    body = r"\bm{" + body + "}"
                if r[f"{m}_sig"]:
                    body += r"^\dagger"
                cells.append(f"${body}$")
        delta = anchored[med]["srocc_mean"] - plain[med]["srocc_mean"]
        cells.append(f"${delta:+.3f}$".replace("0.", ".", 1))
        out.append(f"{label} & " + " & ".join(cells) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    plain, anchored = column("plain"), column("C")

    pp, pa = plain["population"]["srocc_mean"], anchored["population"]["srocc_mean"]
    if not np.isclose(pp, pa, atol=1e-9):
        raise SystemExit(f"population differs between halves ({pp:.6f} vs "
                         f"{pa:.6f}); the two are not the same experiment")
    print(f"population identical in both halves: {pp:.6f}  OK\n")

    print(f"{'mediator':12s} {'unanchored':>11s} {'anchored':>10s} {'delta':>8s}"
          f"  {'p vs Hybrid (anch)':>19s}")
    for med, _ in ROWS:
        a, b = plain[med]["srocc_mean"], anchored[med]["srocc_mean"]
        p = anchored[med]["srocc_p"]
        print(f"{med:12s} {a:11.4f} {b:10.4f} {b - a:+8.3f}  "
              f"{'--' if not np.isfinite(p) else f'{p:19.3g}'}")

    ensure()
    dest = Path(argv[0]) if argv else TABLES / "tab2_anchor.tex"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render(plain, anchored), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
