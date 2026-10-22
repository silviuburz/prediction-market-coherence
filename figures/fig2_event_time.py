#!/usr/bin/env python3
"""Figure 2 — the main figure: event-time coherence around goals.

Panel A  mean S in event time for goals, holdout emphasised and development
         subordinate, with match-clustered 95% bootstrap ribbons and the
         pre/post estimation windows shaded.

Panel B  the same axes and the same y-scale, comparing goals against the
         lower-score-relevance control events. This panel carries the argument:
         if the estimator were picking up generic salience or liquidity
         withdrawal, the end of the first half -- a scheduled,
         universally-known, high-attention discontinuity -- would step too.
         It does not.

HARD RULE (FINDINGS 13.5). The profiles are indexed by grid LABEL, so averaging
profile columns reproduces the estimator only approximately. The annotated effect
size is therefore taken from the frozen engine output, never recomputed from the
plotted columns. The script prints both so the residual stays visible.

Run: python figures/wc_coherence/fig2_event_time.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt                        # noqa: E402
from _style import (BLUE, C, DBL_WIDTH, GREY, LGREY, ORANGE, PURPLE, SKY,      # noqa: E402
                    YELLOW, despine, engine, load, save, use_style)

use_style()
P = load("event_time_profiles.json")

PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
XLIM = (-180, 240)


def series(cohort, cls):
    d = P[cohort][cls]
    o = np.array(d["offsets"], dtype=float)
    m = np.array([np.nan if v is None else v for v in d["mean_cents"]])
    lo = np.array([np.nan if v is None else v for v in d["lo_cents"]])
    hi = np.array([np.nan if v is None else v for v in d["hi_cents"]])
    return o, m, lo, hi, d["n_events"], d["n_matches"]


fig, axes = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.75), sharey=True)
fig.subplots_adjust(left=0.062, right=0.995, top=0.87, bottom=0.155, wspace=0.07)

# ── panel A: goals, both cohorts ──────────────────────────────────────────
ax = axes[0]
for lo_, hi_ in ((PRE_LO, PRE_HI), (POST_LO, POST_HI)):
    ax.axvspan(lo_, hi_, color=LGREY, alpha=0.28, lw=0, zorder=0)
ax.axvline(0, color="k", lw=0.9, zorder=2)
ax.axhline(0, color=GREY, lw=0.5, ls=":", zorder=1)

o, m, lo, hi, nev_d, nmt_d = series("dev", "goal")
ax.fill_between(o, lo, hi, color=SKY, alpha=0.22, lw=0, zorder=3)
ax.plot(o, m, color=SKY, lw=1.1, zorder=4,
        label=f"development ({nmt_d} matches, {nev_d} goals)")

o, m, lo, hi, nev_h, nmt_h = series("holdout", "goal")
ax.fill_between(o, lo, hi, color=BLUE, alpha=0.26, lw=0, zorder=5)
ax.plot(o, m, color=BLUE, lw=2.0, zorder=6,
        label=f"holdout ({nmt_h} matches, {nev_h} goals)")

D_h = C * engine("holdout")["isolated"]["D"]
ci_h = [C * v for v in engine("holdout")["isolated"]["ci"]]
D_d = C * engine("dev")["isolated"]["D"]

ax.annotate(f"holdout $D={D_h:+.3f}$ c\n[{ci_h[0]:+.3f}, {ci_h[1]:+.3f}]",
            xy=(18, 0.60), xytext=(74, 0.90), fontsize=7.2, color=BLUE,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=BLUE,
                            connectionstyle="arc3,rad=-0.2"))
ax.text(PRE_LO + 2, -0.09, "pre", fontsize=6.5, color=GREY)
ax.text(POST_LO + 1, -0.09, "post", fontsize=6.5, color=GREY)
ax.set_title("(a)  goals", fontsize=8.5, loc="left", pad=5)
ax.set_ylabel("mean required widening $S$  (cents)")
ax.set_xlabel("seconds from goal")
ax.legend(loc="upper left", frameon=False, handlelength=1.5,
          borderpad=0.15, labelspacing=0.3)
despine(ax)

# ── panel B: goals vs control events, holdout ─────────────────────────────
ax = axes[1]
for lo_, hi_ in ((PRE_LO, PRE_HI), (POST_LO, POST_HI)):
    ax.axvspan(lo_, hi_, color=LGREY, alpha=0.28, lw=0, zorder=0)
ax.axvline(0, color="k", lw=0.9, zorder=2)
ax.axhline(0, color=GREY, lw=0.5, ls=":", zorder=1)

o, m, lo, hi, nev, nmt = series("holdout", "goal")
ax.fill_between(o, lo, hi, color=BLUE, alpha=0.26, lw=0, zorder=5)
ax.plot(o, m, color=BLUE, lw=2.0, zorder=6, label=f"goal ({nev})")

CONTROLS = [("halftime", ORANGE, "-", 1.5),
            ("substitution", PURPLE, "--", 1.1),
            ("yellow_card", YELLOW, "-.", 1.1)]
NICE = {"halftime": "end of first half", "substitution": "substitution",
        "yellow_card": "yellow card"}
for cls, col, ls, lw in CONTROLS:
    if cls not in P["holdout"]:
        continue
    o, m, _l, _h, nev_c, _ = series("holdout", cls)
    ax.plot(o, m, color=col, lw=lw, ls=ls, zorder=4,
            label=f"{NICE[cls]} ({nev_c})")

ax.set_title("(b)  goals vs. lower-score-relevance controls   (holdout)",
             fontsize=8.5, loc="left", pad=5)
ax.set_xlabel("seconds from event")
ax.legend(loc="upper left", frameon=False, handlelength=1.9,
          borderpad=0.15, labelspacing=0.3)
despine(ax, keep=("bottom",))
ax.tick_params(left=False)

for a in axes:
    a.set_xlim(*XLIM)
    a.set_xticks([-180, -120, -60, 0, 60, 120, 180, 240])
axes[0].set_ylim(-0.17, 1.45)

save(fig, "fig2_event_time")

# ── the residual that the hard rule exists to keep visible ────────────────
print("\nannotation provenance (must come from the engine, not the columns):")
for coh, D_rep in (("dev", D_d), ("holdout", D_h)):
    o, m, *_ = series(coh, "goal")
    ok = np.isfinite(m)
    pre = np.nanmean(m[ok & (o >= PRE_LO) & (o <= PRE_HI)])
    post = np.nanmean(m[ok & (o >= POST_LO) & (o <= POST_HI)])
    print(f"  {coh:<8} engine {D_rep:+.4f}c   column arithmetic "
          f"{post-pre:+.4f}c   (grid-label approximation, NOT annotated)")

# ── pre-window asymmetry: the engine's slice is the conservative one ─────
# searchsorted with an inclusive upper index makes the engine's effective
# pre-window [-60, -5] rather than [-60, -6]. The grid point labelled -5 is
# already partly contaminated by the goal and is therefore HIGH, so including it
# RAISES the baseline and LOWERS D. Printed so the direction stays on record.
print("\npre-window convention (higher baseline => smaller, more conservative D):")
for coh in ("dev", "holdout"):
    o, m, *_ = series(coh, "goal")
    a = np.nanmean(m[(o >= PRE_LO) & (o <= PRE_HI)])
    b = np.nanmean(m[(o >= PRE_LO) & (o <= -5)])
    print(f"  {coh:<8} [-60,-6] {a:+.4f}c   [-60,-5] (engine) {b:+.4f}c   "
          f"engine baseline is {'HIGHER' if b > a else 'LOWER'} by {abs(b-a):.4f}c")
