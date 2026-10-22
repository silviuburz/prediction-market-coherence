"""Shared plotting style for the WC-coherence paper figures.

Conventions enforced here rather than repeated per figure:
  * everything is reported in CENTS (S is carried in probability units; four
    separate display bugs during the correction sprint came from forgetting this)
  * the HOLDOUT series is always emphasised and development always subordinate,
    so no reader can mistake which number is confirmatory
  * Okabe-Ito palette, colour-blind safe and separable in greyscale by lightness
  * IEEE column widths, Type-42 fonts for camera-ready
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.dirname(os.path.abspath(__file__))

C = 100.0                    # probability -> cents

# Okabe-Ito
BLUE = "#0072B2"             # holdout / primary
ORANGE = "#D55E00"           # goals / emphasis
SKY = "#56B4E9"              # development / secondary
GREEN = "#009E73"
YELLOW = "#E69F00"
PURPLE = "#CC79A7"
GREY = "#555555"
LGREY = "#AAAAAA"

COL_WIDTH = 3.5              # IEEE single column, inches
DBL_WIDTH = 7.16             # IEEE double column, inches


def use_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.linewidth": 0.8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.8, "figure.dpi": 150,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def engine(cohort):
    """The frozen engine result. Figures annotate THIS, never a recomputation."""
    return load(f"event_study_{cohort}.json")


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(p, dpi=400)
        print(f"  wrote {p}")
