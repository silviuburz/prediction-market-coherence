#!/usr/bin/env python3
"""Figure 5 — event-class control forest.

Effects for every event class, ordered by how much the event revises beliefs
about the FINAL SCORE. The point is not that the controls are non-significant --
it is that they are BOUNDED: every control's confidence interval lies far below
the goal effect, so a generic salience or liquidity account, which predicts a
comparable response at the half-time break, is not consistent with the data.

Underpowered rows are drawn in grey, starred and given no interval, so they
cannot be over-read. A row is treated as descriptive when it has fewer than five
contributing matches, or when its interval is WIDER than the goal effect it is
being compared against -- such a row cannot separate "no response" from "larger
than a goal", so plotting it as an ordinary point in an ordered forest invites
the dose-response reading the design does not support. On the holdout the second
clause catches only red cards (interval 0.636c against a 0.436c goal effect).

Run: python figures/wc_coherence/fig5_controls.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt                        # noqa: E402
from _style import (BLUE, C, COL_WIDTH, GREY, LGREY, ORANGE, SKY, despine,     # noqa: E402
                    load, save, use_style)

use_style()
R = load("placebo_suite.json")

ORDER = [("goal", "goal", "large"),
         ("red_card", "red card", "substantial"),
         ("penalty_nogoal", "penalty missed/saved", "high attention, no score"),
         ("yellow_card", "yellow card", "negligible"),
         ("substitution", "substitution", "~none"),
         ("delay", "delay (injury or drinks)", "~none"),
         ("halftime", "end of first half", "none (scheduled)"),
         ("start_2nd_half", "start of second half", "none (scheduled)")]

fig, ax = plt.subplots(figsize=(COL_WIDTH, 3.05))
fig.subplots_adjust(left=0.375, right=0.975, top=0.86, bottom=0.145)

ax.axvline(0, color=GREY, lw=0.8, ls=":", zorder=1)

gh = R["holdout"]["goal"]["result"]
ax.axvline(C * gh["D"], color=BLUE, lw=0.9, ls="--", zorder=1)
ax.text(C * gh["D"] + 0.04, len(ORDER) - 0.45, "goal effect", fontsize=6.4,
    color=BLUE, va="top", ha="left")

yticks, ylabels = [], []
for i, (key, nice, note) in enumerate(ORDER):
    y = len(ORDER) - 1 - i
    yticks.append(y)
    ylabels.append(nice)
    for coh, off, ms, col_ok in (("holdout", 0.0, 4.6, True),
                                 ("dev", -0.30, 3.0, False)):
        e = R.get(coh, {}).get(key, {})
        r = e.get("result")
        if not r:
            continue
        goal_d = R.get(coh, {}).get("goal", {}).get("result", {}).get("D")
        wide = (r.get("ci") is not None and goal_d is not None
                and key != "goal"
                and (r["ci"][1] - r["ci"][0]) > goal_d)
        under = r.get("ci") is None or wide
        col = (BLUE if coh == "holdout" else SKY) if key == "goal" else (
            ORANGE if coh == "holdout" else "#F0A882")
        if under:
            col = LGREY
        if r.get("ci") is not None and not under:
            ax.plot([C * r["ci"][0], C * r["ci"][1]], [y + off, y + off],
                    color=col, lw=1.5, solid_capstyle="round", zorder=4)
        ax.plot(C * r["D"], y + off, "o", ms=ms, color=col, mec="k", mew=0.35,
                zorder=5)
        if coh == "holdout":
            n = r["n_events"]
            tag = f"n={n}" + ("*" if under else "")
            ax.text(1.70, y + 0.02, tag, fontsize=6.0, color=GREY,
                    va="center", ha="right")

ax.text(1.70, -0.62, "* descriptive only; see caption", fontsize=5.8,
        color=GREY, ha="right", va="center", style="italic")

ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=7)
ax.set_ylim(-1.05, len(ORDER) - 0.35)
ax.set_xlim(-1.75, 1.78)
ax.set_xlabel("change in required widening  $D$  (cents)")
ax.set_title("holdout (large markers), development (small)",
             fontsize=7.2, loc="left", pad=4)
despine(ax, keep=("bottom",))
ax.tick_params(left=False)
save(fig, "fig5_controls")

print("\ncontrol bounds as a share of the goal effect (holdout):")
g = C * gh["D"]
for key, nice, _ in ORDER:
    r = R["holdout"].get(key, {}).get("result")
    if not r or r.get("ci") is None or key == "goal":
        continue
    print(f"  {nice:<24} D {C*r['D']:+7.3f}c   CI upper {C*r['ci'][1]:+7.3f}c"
          f"   = {100*r['ci'][1]/gh['D']:6.1f}% of goal")
