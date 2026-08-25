r"""Where the paper's finished tables and figures go.

One place, so every generator agrees and nothing lands in the old mixed
directory by accident. The paper repo is separate from this one: results and
generators live here, the manuscript lives there.
"""
from __future__ import annotations

from pathlib import Path

PAPER_ROOT = Path(r"D:\Pin\STUDY\0CU\แลกเปลี่ยน\JAIST Internship2025"
                  r"\PIAA_project\paper\final")
TABLES = PAPER_ROOT / "tables"
FIGURES = PAPER_ROOT / "figures"

#: the one configuration the paper reports
BACKBONE = "qwen8b"
VARIANT = "C"            # the population-anchored Stage-2
N_TRAIN = 100
HEAD = "ridge"


def ensure() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
