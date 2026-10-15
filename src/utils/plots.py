"""Figures -- font/size settings matched to what's actually used in the paper.

Every figure is 7in wide (full two-column page width), 8pt serif font,
saved as both .pdf (for the paper) and .png (for a quick look).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FULL_WIDTH = 7.0
COL_WIDTH = 3.3
HEIGHT_2PANEL = 1.9
HEIGHT_3PANEL = 2.2

COLOR = {
    "blue": "#185FA5",     # our model
    "blue_mid": "#5B9BD5", 
    "blue_light": "#AFC9E8",  
    "grey": "#9E9E9E",     # baseline
    "grey_mid": "#B0B0B0",
    "grey_light": "#D0D0D0",
    "amber":  "#8C4A0F",    # content-free control
    "amber_mid": "#BA7517",
    "amber_light": "#E8B87A",
    "teal": "#1D9E75"
}

def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.3,
        "legend.frameon": False,
        "pdf.fonttype": 42,          # embed as TrueType, ACM requires this
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    })


def grid(ax) -> None:
    ax.grid(alpha=0.25, linewidth=0.4)


def save(fig, path: str | Path) -> Path:
    """Save both .pdf and .png."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.with_suffix('.pdf').name} and .png")
    return p
