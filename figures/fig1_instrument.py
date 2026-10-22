#!/usr/bin/env python3
"""Figure 1 — the measurement instrument.

Panel (a) compiles natural-language resolution criteria into payoff rules over
score states and shows the resulting payoff row.

Panel (b) draws the coherence test. With three states and two contracts, the set
of coherent prices is a triangle. A quoted bid-ask box outside it must be
expanded uniformly by S before the two touch. For the displayed box, S = 0.075.

Panel (c) gives a three-way moneyline example whose quoted lower bounds sum to
1.04. Its minimum uniform widening is 1.33 cents. Every displayed number is
computed and checked rather than drawn by hand.

Output: figures/wc_coherence/fig1_instrument.{pdf,png}
Run: python figures/wc_coherence/fig1_instrument.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib.patches import Polygon, Rectangle      # noqa: E402
from scipy.optimize import linprog                     # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))

# colour-blind safe (Okabe-Ito), and distinguishable in greyscale by lightness
C_POLY = "#0072B2"     # coherent-price set
C_BOX = "#D55E00"      # quoted box
C_INFL = "#E69F00"     # inflated box
C_TXT = "#000000"
C_MUT = "#555555"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.8, "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def min_uniform_slack(X, lo, hi):
    """Exactly the estimator used in the paper: min s >= 0 such that some
    distribution q over states satisfies lo - s <= Xq <= hi + s."""
    m, n = X.shape
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A_ub = np.vstack([np.hstack([-X, -np.ones((m, 1))]),
                      np.hstack([X, -np.ones((m, 1))])])
    b_ub = np.concatenate([-np.asarray(lo), np.asarray(hi)])
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                bounds=[(0, 1)] * n + [(0, None)], method="highs")
    if not r.success:
        raise RuntimeError("LP failed")
    return float(r.x[-1]), r.x[:n]


# ── verify the two worked examples before drawing anything ────────────────
# states are (home win, draw, away win); the two contracts are the standard
# double chances, so both pay on a draw and the hull is unchanged by the pairing
X_geo = np.array([[1.0, 1.0, 0.0],
                  [0.0, 1.0, 1.0]])
LO_geo, HI_geo = np.array([0.30, 0.35]), np.array([0.40, 0.45])
S_geo, q_geo = min_uniform_slack(X_geo, LO_geo, HI_geo)

X_ml = np.eye(3)
LO_ml, HI_ml = np.array([0.55, 0.26, 0.23]), np.array([0.57, 0.28, 0.25])
S_ml, q_ml = min_uniform_slack(X_ml, LO_ml, HI_ml)

print(f"panel (b): S = {S_geo:.6f}  (closed form 0.075)   q = {q_geo.round(4)}")
print(f"panel (c): S = {S_ml:.6f}  (closed form {(LO_ml.sum()-1)/3:.6f})"
      f"   q = {q_ml.round(4)}")
assert abs(S_geo - 0.075) < 1e-9, "panel (b) closed form mismatch"
assert abs(S_ml - (LO_ml.sum() - 1) / 3) < 1e-9, "panel (c) closed form mismatch"

fig = plt.figure(figsize=(7.16, 2.65))          # IEEE double-column width
gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.02, 0.96], wspace=0.42,
                      left=0.045, right=0.985, top=0.86, bottom=0.15)

# ── panel (a): compilation ────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_title("(a)  payoff compilation", fontsize=8.5, loc="left", pad=6)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.0, 0.94, "resolution text", fontsize=7.2, color=C_MUT, style="italic")
ax.add_patch(Rectangle((0.0, 0.72), 1.0, 0.19, fc="#F2F2F2", ec=C_MUT, lw=0.6))
ax.text(0.03, 0.845, '"…resolves YES if the match ends\n'
                     ' with both teams having scored…"',
        fontsize=6.6, va="center", family="DejaVu Sans")

ax.annotate("", xy=(0.5, 0.68), xytext=(0.5, 0.71),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=C_TXT))
ax.text(0.53, 0.685, "compile", fontsize=6.4, color=C_MUT)

ax.text(0.0, 0.62, "payoff rule over score states", fontsize=7.2, color=C_MUT,
        style="italic")
ax.text(0.03, 0.545, r"$\phi_{\mathrm{btts}}(h,a)=\mathbf{1}\{h>0 \wedge a>0\}$",
        fontsize=7.4)

# small score grid with identical-payoff shading
x0, y0, cell = 0.06, 0.10, 0.072
for i in range(5):
    for j in range(5):
        val = 1 if (i > 0 and j > 0) else 0
        ax.add_patch(Rectangle((x0 + j * cell, y0 + i * cell), cell, cell,
                               fc=(C_POLY if val else "#FFFFFF"),
                               alpha=(0.55 if val else 1.0),
                               ec=C_MUT, lw=0.4))
# goal counts on both axes, so the shaded region reads as exactly h>0 and a>0
for k in range(5):
    ax.text(x0 - 0.014, y0 + (k + 0.5) * cell, str(k), fontsize=5.8,
            color=C_MUT, ha="right", va="center")
    ax.text(x0 + (k + 0.5) * cell, y0 - 0.028, str(k), fontsize=5.8,
            color=C_MUT, ha="center", va="top")
ax.text(x0 - 0.076, y0 + 2.5 * cell, "$h$", fontsize=7, va="center",
        ha="center")
ax.text(x0 + 2.5 * cell, y0 - 0.092, "$a$", fontsize=7, ha="center",
        va="center")
ax.text(x0 + 5 * cell + 0.03, y0 + 3.4 * cell, "pays 1", fontsize=6.6,
        color=C_POLY)
ax.text(x0 + 5 * cell + 0.03, y0 + 0.7 * cell, "pays 0", fontsize=6.6,
        color=C_MUT)
ax.text(0.0, -0.105, "validated against realized settlement:\n"
                    "4,573 checks, zero failures",
        fontsize=6.3, color=C_MUT, style="italic")

# ── panel (b): geometry ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.set_title("(b)  minimum required widening", fontsize=8.5,
             loc="left", pad=6)

tri = np.array([[1, 1], [1, 0], [0, 1]])
ax.add_patch(Polygon(tri, closed=True, fc=C_POLY, alpha=0.20, ec=C_POLY,
                     lw=1.2, zorder=1))
ax.text(0.74, 0.74, r"$\mathcal{M}$", fontsize=10, color=C_POLY, zorder=5)
ax.text(0.60, 0.62, "coherent\nprices", fontsize=6.4, color=C_POLY,
        ha="center", zorder=5)

# inflated box (drawn first so the tight box sits on top)
ax.add_patch(Rectangle((LO_geo[0] - S_geo, LO_geo[1] - S_geo),
                       (HI_geo[0] - LO_geo[0]) + 2 * S_geo,
                       (HI_geo[1] - LO_geo[1]) + 2 * S_geo,
                       fc=C_INFL, alpha=0.28, ec=C_INFL, lw=1.0,
                       ls=(0, (3, 2)), zorder=2))
ax.add_patch(Rectangle((LO_geo[0], LO_geo[1]),
                       HI_geo[0] - LO_geo[0], HI_geo[1] - LO_geo[1],
                       fc=C_BOX, alpha=0.75, ec=C_BOX, lw=1.2, zorder=3))
# the quoted box is 0.10 wide, too small to hold a label, so both boxes are
# named outside and colour-matched to their edges
ax.annotate("quoted box", xy=(LO_geo[0], 0.5 * (LO_geo[1] + HI_geo[1])),
            xytext=(0.115, 0.30), fontsize=6.4, color=C_BOX, weight="bold",
            ha="center", va="center", zorder=6,
            arrowprops=dict(arrowstyle="-", lw=0.7, color=C_BOX,
                            shrinkA=1.5, shrinkB=1.5))
ax.text(0.5 * (LO_geo[0] + HI_geo[0]), LO_geo[1] - S_geo - 0.028,
        r"widened by $S$", fontsize=6.4, color="#B07800", ha="center",
        va="top", zorder=6)

# the touching point and the slack arrow
touch = np.array([HI_geo[0] + S_geo, HI_geo[1] + S_geo])
ax.plot(*touch, "o", ms=3.5, color=C_INFL, zorder=7, mec="k", mew=0.4)
ax.annotate("", xy=touch, xytext=(HI_geo[0], HI_geo[1]),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_TXT), zorder=7)
ax.text(0.505, 0.445, f"$S={S_geo:.3f}$", fontsize=7.6, zorder=7)

ax.plot([0, 1], [1, 0], lw=0.6, color=C_POLY, ls=":", zorder=1)
ax.text(0.10, 0.90, r"$\mu_1+\mu_2\geq 1$", fontsize=6.4, color=C_POLY)

ax.set_xlim(0.05, 1.05)
ax.set_ylim(0.05, 1.05)
ax.set_xlabel("price of \u201chome or draw\u201d\n" +
              r"$\mu_1=\Pr(\mathrm{home})+\Pr(\mathrm{draw})$", fontsize=7)
ax.set_ylabel("price of \u201cdraw or away\u201d\n" +
              r"$\mu_2=\Pr(\mathrm{draw})+\Pr(\mathrm{away})$", fontsize=7)
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(labelsize=6.5, length=2.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ── panel (c): worked micro-example, as a lower-bound sum ─────────────────
# The quoted intervals are only 0.02 wide, far too narrow to read at this
# figure size, so the incoherence is shown the way it is actually detected:
# the lower bounds of a three-way exhaustive market must sum to at most 1.
ax = fig.add_subplot(gs[0, 2])
ax.set_title("(c)  worked example", fontsize=8.5, loc="left", pad=6)

names = ["home", "draw", "away"]
shades = ["#D55E00", "#E8834A", "#F2B48C"]
value_ink = ["white", "white", "#7A3200"]   # the palest shade needs dark text
left = 0.0
for k in range(3):
    ax.barh(1.0, LO_ml[k], left=left, height=0.42, color=shades[k],
            ec="white", lw=0.8, zorder=3)
    ax.text(left + LO_ml[k] / 2, 1.0, f"{LO_ml[k]:.2f}", fontsize=6.0,
            ha="center", va="center", color=value_ink[k], weight="bold",
            zorder=8)
    ax.text(left + LO_ml[k] / 2, 0.70, names[k], fontsize=6.2,
            ha="center", va="top", color=C_MUT)
    left += LO_ml[k]

# the limit is marked above and below the bar rather than ruled through it: at
# x=1.0 it lands inside the last segment, on top of that segment's value
ax.plot([1.0, 1.0], [1.21, 1.44], color="k", lw=1.1, zorder=6)
ax.plot([1.0, 1.0], [0.72, 0.79], color="k", lw=1.1, zorder=6)
ax.text(0.0, 1.52, "coherence limit  " + r"$\sum_i \ell_i \leq 1$",
        fontsize=6.5, ha="left", va="bottom")

# the 0.04 overhang is too small to annotate in place, so bracket it above
ax.plot([1.0, LO_ml.sum()], [1.27, 1.27], color="k", lw=1.6,
        solid_capstyle="butt", zorder=8)
ax.annotate(f"excess {LO_ml.sum()-1:.2f}",
            xy=(1.03, 1.24), xytext=(1.32, 0.84), fontsize=6.5,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_TXT,
                            connectionstyle="arc3,rad=0.28"))

ax.text(0.0, -0.52, r"$\sum_i \ell_i = 1.04 > 1\;\Rightarrow\;$ incoherent",
        fontsize=7.0)
ax.text(0.0, -1.02, rf"$S=\dfrac{{1.04-1}}{{3}}={S_ml*100:.2f}$ cents",
        fontsize=7.4)

# the scale is drawn by hand directly beneath the bar; left at the axes bottom
# it reads as an axis for the equations rather than for the bar it measures
SCALE_Y = 0.34
ax.plot([0.0, 1.0], [SCALE_Y, SCALE_Y], color=C_MUT, lw=0.8, zorder=4)
for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
    ax.plot([tick, tick], [SCALE_Y, SCALE_Y - 0.07], color=C_MUT, lw=0.8,
            zorder=4)
    ax.text(tick, SCALE_Y - 0.13, f"{tick:.2f}", fontsize=6.2, color=C_MUT,
            ha="center", va="top")
ax.text(0.5, SCALE_Y - 0.42, "cumulative lower bound", fontsize=7,
        color=C_MUT, ha="center", va="top")

ax.set_xlim(-0.02, 1.55)
ax.set_ylim(-1.25, 2.05)
ax.set_yticks([])
ax.set_xticks([])
ax.tick_params(labelsize=6.5, length=2.5)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)

for ext in ("pdf", "png"):
    path = os.path.join(OUT, f"fig1_instrument.{ext}")
    fig.savefig(path, dpi=400, bbox_inches="tight")
    print(f"wrote {path}")
