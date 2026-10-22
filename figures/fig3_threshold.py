#!/usr/bin/env python3
"""Figure 3 — threshold response: the effect is not sub-tick dust.

The mean effect is 0.44 median ticks, which invites the reading that this is a
uniform, negligible drift. It is not. This figure plots the complementary CDF of
the per-goal window mean of S, before and after the goal, so the change in the
whole distribution is visible: the post-goal distribution has a far heavier tail.

UNITS. The engine's threshold table is EVENT-level (one observation per isolated
goal, the window mean of S), so the event series is plotted as the primary and
reproduces that table exactly. The match-weighted series is overlaid because the
primary estimand is match-weighted, and the instant-level series is deliberately
NOT plotted -- it answers a different question and was a genuine defect in the
first version of the figure inputs.

Run: python figures/wc_coherence/fig3_threshold.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt                        # noqa: E402
from _style import (BLUE, C, COL_WIDTH, GREY, LGREY, ORANGE, SKY, despine,     # noqa: E402
                    engine, load, save, use_style)

use_style()
T = load("threshold_ccdf.json")

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.7))
fig.subplots_adjust(left=0.155, right=0.975, top=0.90, bottom=0.16)

h = T["holdout"]
x = np.array(h["threshold_cents"])
keep = x > 0

ax.axvline(h["median_tick_cents"], color=GREY, lw=0.8, ls="--", zorder=1)
ax.text(h["median_tick_cents"] * 1.08, 0.93, "median\ntick", fontsize=6.3,
        color=GREY, va="top")

# match-weighted band first, then the event-level primary on top
mw = h["match_weighted"]
ax.fill_between(x[keep], np.array(mw["pre_lo"])[keep],
                np.array(mw["pre_hi"])[keep], color=SKY, alpha=0.20, lw=0)
ax.fill_between(x[keep], np.array(mw["post_lo"])[keep],
                np.array(mw["post_hi"])[keep], color=ORANGE, alpha=0.18, lw=0)

ax.plot(x[keep], np.array(h["event"]["pre"])[keep], color=BLUE, lw=1.8,
        label="pre-goal  $[-60,-6]$ s", zorder=5)
ax.plot(x[keep], np.array(h["event"]["post"])[keep], color=ORANGE, lw=1.8,
        label="post-goal  $[+6,+30]$ s", zorder=5)

# The two thresholds quoted in the text. The MARKERS AND LABELS come from the
# frozen engine's threshold table, not from this figure's log grid: the grid's
# nearest sample point can differ by one event in 125 (0.8pp), which would put a
# number in the figure that is not the number in the paper.
eng_thr = {r["threshold"]: r for r in engine("holdout")["thresholds"]}
CALLOUT = {"0.25c": (2.3, 0.74), "1.00c": (2.3, 0.44)}
for name, tc in (("0.25c", 0.25), ("1.00c", 1.00)):
    e = eng_thr[name]
    p0, p1 = e["pre"] / 100.0, e["post"] / 100.0
    ax.plot([tc, tc], [p0, p1], color="k", lw=0.7, zorder=6)
    ax.plot([tc], [p0], "o", ms=3.4, color=BLUE, mec="k", mew=0.4, zorder=7)
    ax.plot([tc], [p1], "o", ms=3.4, color=ORANGE, mec="k", mew=0.4, zorder=7)
    ax.annotate(f"at {tc:.2f} c:  {e['pre']:.1f}% \u2192 {e['post']:.1f}%",
                xy=(tc * 1.06, (p0 + p1) / 2), xytext=CALLOUT[name],
                fontsize=6.6, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=GREY,
                                connectionstyle="arc3,rad=0.18"))

ax.set_xscale("log")
ax.set_xlim(0.02, 12)
ax.set_ylim(0, 1.0)
ax.set_xlabel("violation threshold  (cents, log scale)")
ax.set_ylabel("share of isolated goals exceeding")
ax.set_title("holdout: 51 matches, 125 goals", fontsize=7.5, loc="left", pad=4)
ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderpad=0.2,
          bbox_to_anchor=(-0.012, -0.018))
despine(ax)
save(fig, "fig3_threshold")

# the figure's own curve should still agree with the engine to within one event
print("\ncross-check: figure curve vs engine table (1 event = 0.8pp)")
for name, tc in (("0.10c", 0.10), ("0.25c", 0.25), ("0.50c", 0.50),
                 ("1.00c", 1.00)):
    j = int(np.argmin(np.abs(x - tc)))
    fp, fq = 100 * h["event"]["pre"][j], 100 * h["event"]["post"][j]
    e = eng_thr[name]
    ok = abs(fp - e["pre"]) <= 0.81 and abs(fq - e["post"]) <= 0.81
    print(f"  {name}: curve {fp:5.1f}->{fq:5.1f}   engine "
          f"{e['pre']:5.1f}->{e['post']:5.1f}   "
          f"{'within 1 event' if ok else '*** DIFFERS BY >1 EVENT ***'}")
