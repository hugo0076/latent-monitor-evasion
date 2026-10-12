"""Shared style for the paper figures: palettes, fonts and sizes.

Every fig_*.py takes its colours from here, so a colour means the same thing in every figure.

    ROLE    the attack arms: no attack, base edit, amplified base edit, model organism.
    MODEL   one colour per model, where a figure compares models (colourblind-safe Okabe-Ito).

The two palettes share no colour. Heatmaps use RdYlBu over [0, 1]: low values are red and
mean that the monitor was evaded.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

ROLE = {
    "none": "#3e4c59",
    "base": "#9aa5b1",
    "amplified": "#5b8ff9",
    "organism": "#e8684a",
    "random": "#c5ccd3",
}

MODEL_COLOR = {
    "llama-3.1-8b": "#0072B2",
    "gemma-3-27b": "#009E73",
    "llama-3.3-70b": "#D55E00",
    "qwen3-14b": "#CC79A7",
}

DETECTOR = {"monitor": "#7c5cbf", "ood": "#2a9d8f"}

HEATMAP = dict(cmap="RdYlBu", vmin=0.0, vmax=1.0)
SCORE_CMAP = "RdYlGn_r"

INK, MUTED, FAINT = "#1f2933", "#7b8794", "#c8cfd6"

FS = dict(
    base=9.0,
    label=9.0,
    tick=8.5,
    title=9.5,
    legend=8.0,
    annot=8.0,
    letter=13,
)
DPI = 200


def apply() -> None:
    """Set the shared rcParams. Call once, before building any figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FS["base"],
            "axes.titlesize": FS["title"],
            "axes.labelsize": FS["label"],
            "xtick.labelsize": FS["tick"],
            "ytick.labelsize": FS["tick"],
            "legend.fontsize": FS["legend"],
            "axes.edgecolor": FAINT,
            "axes.linewidth": 0.8,
            "xtick.color": "#3e4c59",
            "ytick.color": "#3e4c59",
            "pdf.fonttype": 42,
            "savefig.dpi": DPI,
        }
    )
