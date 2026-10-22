#!/usr/bin/env python3
"""Figure 4 — specification curve.

300 window/isolation specifications x 2 weightings x 2 cohorts. Every cell is
bootstrapped over MATCHES, and the whole curve is computed from the persisted
coherence series with the engine's absolute-time window semantics, so the primary
specification reproduces the frozen estimate exactly (diff 0.00e+00).

The honest reading is stated in the figure rather than hidden: all cells are
positive and significant, and the pre-registered primary sits ABOVE the
median across all specifications, not below it. An earlier version of this analysis described
the primary as conservative; that was an artifact of computing the curve from
grid-labelled long-form rows and has been withdrawn.

Run: python figures/wc_coherence/fig4_spec_curve.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt                        # noqa: E402
from _style import (BLUE, C, DBL_WIDTH, GREY, LGREY, ORANGE, despine,          # noqa: E402
                    engine, load, save, use_style)

use_style()
S = load("spec_curve_v3.json")
rows = [r for r in S["holdout"] if r["weighting"] == "equal_match"]
rows.sort(key=lambda r: r["D"])
n = len(rows)

D = C * np.array([r["D"] for r in rows])
lo = C * np.array([r["ci"][0] for r in rows])
hi = C * np.array([r["ci"][1] for r in rows])
sig = np.array([r["sig"] for r in rows])
prim = int(np.argmax([r["primary"] for r in rows]))
D_eng = C * engine("holdout")["isolated"]["D"]

fig = plt.figure(figsize=(DBL_WIDTH, 3.5))
gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1.0], hspace=0.10,
                      left=0.075, right=0.985, top=0.92, bottom=0.11)

# ── top: sorted estimates with intervals ──────────────────────────────────
ax = fig.add_subplot(gs[0])
xs = np.arange(n)
ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.20, lw=0, step="mid")
ax.plot(xs, D, color=BLUE, lw=1.4, zorder=4)
ax.axhline(0, color="k", lw=0.8, zorder=3)
ax.axhline(np.median(D), color=GREY, lw=0.8, ls=":", zorder=3)
ax.text(n * 0.985, np.median(D) - 0.028, f"median of all specifications {np.median(D):.3f} c",
        fontsize=6.4, color=GREY, ha="right", va="top")

ax.plot([prim], [D[prim]], "o", ms=6, color=ORANGE, mec="k", mew=0.5, zorder=6)
pct = 100.0 * (D < D[prim]).mean()
ax.annotate(f"registered specification\n$D={D_eng:+.3f}$ c   ({pct:.0f}th percentile)",
            xy=(prim, D[prim]), xytext=(n * 0.30, 0.115), fontsize=7.0,
            color=ORANGE, ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=ORANGE,
                            connectionstyle="arc3,rad=0.2"))

ax.set_ylabel("$D$  (cents)")
ax.set_xlim(-3, n + 2)
ax.set_ylim(0.0, 0.70)
ax.set_xticks([])
ax.set_title(f"holdout, match-equal weighting: {int(sig.sum())} of {n} "
             f"specifications positive and significant",
             fontsize=8, loc="left", pad=5)
despine(ax, keep=("left",))

# ── bottom: which analyst choice each spec makes ──────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax)
DIMS = [("pre_lo", [-120, -90, -60, -45, -30], "pre start"),
        ("post_hi", [20, 30, 45, 60, 90], "post end"),
        ("iso_gap", [180, 300, 600], "isolation")]
# label positions are recorded in the SAME loop that draws the markers, so the
# two cannot drift apart (they did in the first version)
placed, yrow = [], 0.0
for key, vals, nice in DIMS:
    for v in vals:
        on = np.array([r[key] == v for r in rows])
        ax2.plot(xs[on], np.full(on.sum(), yrow), "s", ms=1.5,
                 color=BLUE, alpha=0.85, mew=0)
        if rows[prim][key] == v:
            ax2.plot([prim], [yrow], "s", ms=3.4, color=ORANGE, mew=0)
        placed.append((yrow, f"{nice}  {v}"))
        yrow += 1.0
    yrow += 0.8

for y, lab in placed:
    ax2.text(-5, y, lab, fontsize=5.8, ha="right", va="center", color=GREY)
ax2.set_ylim(-1.0, yrow - 0.4)
ax2.set_xlabel("specifications, ordered by estimated effect")
ax2.set_xticks([])
ax2.set_yticks([])
despine(ax2, keep=())
ax2.tick_params(left=False, bottom=False)

save(fig, "fig4_spec_curve")

print(f"\nholdout equal_match: {int(sig.sum())}/{n} positive and significant")
print(f"  range   {D.min():+.3f} .. {D.max():+.3f} cents")
print(f"  median  {np.median(D):+.3f} cents")
print(f"  primary {D[prim]:+.3f} cents at the {pct:.0f}th percentile")
print(f"  engine  {D_eng:+.3f} cents   diff {D[prim]-D_eng:+.2e}")
for w in ("equal_match", "event"):
    for coh in ("dev", "holdout"):
        rr = [r for r in S[coh] if r["weighting"] == w]
        print(f"  {coh:<8}{w:<12} {sum(r['sig'] for r in rr)}/{len(rr)} sig")
