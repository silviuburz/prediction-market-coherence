#!/usr/bin/env python3
"""How precise does the goal anchor have to be?

Motivating question: could an openly licensed source replace ESPN as the event
anchor? Wikipedia, FIFA match reports and football-data.org all publish goal
times to the MINUTE. StatsBomb's open data has second-resolution event times but
covers World Cups 2018 and 2022, not 2026. So the practical question is whether a
minute-resolution anchor would still recover the effect.

Rather than assert it would not, this measures the attenuation. Each goal's
anchor is perturbed by uniform noise of width J seconds and the whole match-
weighted estimator is recomputed, N_DRAW times per J. J = 60 is the situation a
minute-resolution source puts us in, and that is OPTIMISTIC: it assumes the
match-minute can be mapped to wallclock exactly, which stoppage time and clock
drift make untrue in practice.

This doubles as a robustness result: it bounds how much of the measured effect
depends on the precision of the anchor.

Everything is reported in CENTS. S is stored in probability units.

Run: /home/azureuser/.venv/bin/python -u code/anchor_precision.py
"""
import gzip
import json
import os
import sys
from datetime import datetime

import numpy as np

OUT = "/home/azureuser/projects/wc_coherence"
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
ISO_GAP, WIN, SEED = 300, 300, 29
C = 100.0
JITTERS = [0, 5, 10, 20, 30, 60, 120]   # full width, seconds
N_DRAW = 400


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def prefix_sums(a):
    ok = ~np.isnan(a)
    v = np.where(ok, a, 0.0)
    return (np.concatenate([[0.0], np.cumsum(v)]),
            np.concatenate([[0.0], np.cumsum(ok.astype(float))]))


def win_mean(cs, cn, gt, centre, lo, hi):
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = min(int(np.searchsorted(gt, centre + hi)) + 1, len(gt))
    if i1 <= i0:
        return None
    n = cn[i1] - cn[i0]
    if n <= 0:
        return None
    return float((cs[i1] - cs[i0]) / n)


results = {}
for cohort in ("dev", "holdout"):
    p = f"{OUT}/results/{cohort}/match_stats.json.gz"
    if not os.path.exists(p):
        log(f"!! missing {p}")
        continue
    with gzip.open(p, "rt") as f:
        MS = json.load(f)

    prepped, goals = {}, []
    for m, st in MS.items():
        Sc = np.array(st["Sc"], dtype=float)
        gt = st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"])
        cs, cn = prefix_sums(Sc)
        prepped[m] = (cs, cn, gt)
        gw = sorted(st["goal_w"])
        for w in gw:
            if w - WIN < st["t_start"] or w + WIN > st["t_end"]:
                continue
            if not all(abs(w - o) > ISO_GAP for o in gw if abs(o - w) > 1e-9):
                continue
            goals.append((m, w))

    def estimate(offsets):
        acc = {}
        for (m, w), off in zip(goals, offsets):
            cs, cn, gt = prepped[m]
            a = win_mean(cs, cn, gt, w + off, PRE_LO, PRE_HI)
            b = win_mean(cs, cn, gt, w + off, POST_LO, POST_HI)
            if a is not None and b is not None:
                acc.setdefault(m, []).append(b - a)
        return float(np.mean([np.mean(v) for v in acc.values()])) if acc else np.nan

    base = estimate(np.zeros(len(goals)))
    log(f"{cohort}: {len(goals)} isolated goals, exact anchor D = {C*base:+.4f}c")

    rng = np.random.default_rng(SEED)
    rows = []
    for J in JITTERS:
        if J == 0:
            vals = np.array([base])
        else:
            vals = np.array([estimate(rng.uniform(-J / 2, J / 2, len(goals)))
                             for _ in range(N_DRAW)])
        rec = {"jitter_s": J, "mean_c": C * float(vals.mean()),
               "sd_c": C * float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
               "p05_c": C * float(np.percentile(vals, 5)),
               "p95_c": C * float(np.percentile(vals, 95)),
               "retained_pct": 100.0 * float(vals.mean()) / base}
        rows.append(rec)
        log(f"  jitter +/-{J/2:>5.1f}s (width {J:>3}s)  "
            f"D = {rec['mean_c']:+.4f}c  sd {rec['sd_c']:.4f}  "
            f"[{rec['p05_c']:+.4f},{rec['p95_c']:+.4f}]  "
            f"retains {rec['retained_pct']:5.1f}%")
    results[cohort] = {"n_goals": len(goals), "exact_c": C * base, "rows": rows}

dest = f"{OUT}/results/anchor_precision.json"
with open(dest, "w") as f:
    json.dump({"params": {"N_DRAW": N_DRAW, "SEED": SEED, "JITTERS_s": JITTERS,
                          "PRE": [PRE_LO, PRE_HI], "POST": [POST_LO, POST_HI]},
               "cohorts": results}, f, indent=2)
log(f"wrote {dest}")
