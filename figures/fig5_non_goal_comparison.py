#!/usr/bin/env python3
"""Draw paper Figure 5 from the recorded non-goal comparison draws."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt                        # noqa: E402
from _style import (BLUE, DBL_WIDTH, GREY, ORANGE, SKY,   # noqa: E402
                    despine, load, save, use_style)

use_style()

N = load("randomization_null.json")
fig, axes = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.15), sharey=False)
fig.subplots_adjust(left=0.062, right=0.99, top=0.86, bottom=0.185, wspace=0.20)

for k, coh in enumerate(("dev", "holdout")):
    ax = axes[k]
    d = N[coh]
    draws = np.array(d["draws_cents"])
    obs = d["observed_cents"]
    ax.hist(draws, bins=60, color=SKY if coh == "dev" else BLUE, alpha=0.85,
            lw=0)
    ax.axvline(obs, color=ORANGE, lw=1.8, zorder=5)
    ax.axvline(draws.max(), color=GREY, lw=0.9, ls=":", zorder=4)
    ax.annotate(f"observed\n{obs:+.3f} c", xy=(obs, ax.get_ylim()[1] * 0.62),
                xytext=(obs * 0.60, ax.get_ylim()[1] * 0.82), fontsize=6.8,
                color=ORANGE, ha="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ORANGE))
    ax.text(draws.max(), ax.get_ylim()[1] * 0.34,
            f"  largest of {d['n_draws']:,}\n  draws {draws.max():+.3f} c",
            fontsize=6.2, color=GREY, va="center")
    ax.set_xlim(min(draws.min() * 1.3, -0.06), obs * 1.16)
    ax.set_xlabel("$D$ at within-match non-goal comparison times  (cents)")
    ax.set_title(f"({'ab'[k]}) {coh}, registered specification  $p={d['p']:.5f}$",
                 fontsize=8, loc="left", pad=4)
    if k == 0:
        ax.set_ylabel("null draws")
    despine(ax)

save(fig, "fig8_randomization_null")
print("\nrandomization null:")
for coh in ("dev", "holdout"):
    d = N[coh]
    dr = np.array(d["draws_cents"])
    print(f"  {coh:<8} obs {d['observed_cents']:+.4f}c   null max "
          f"{dr.max():+.4f}c   ratio {d['observed_cents']/dr.max():.1f}x   "
          f"exceed {d['n_exceed']}/{d['n_draws']}   p={d['p']:.5f}")
