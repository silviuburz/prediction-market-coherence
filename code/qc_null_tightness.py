#!/usr/bin/env python3
"""QC on the coherence series and the randomization null.

The corrected randomization test returns a null with standard deviation ~2e-4
cents against an observed effect of +0.667 cents. That ratio is large enough to
be suspicious: a near-degenerate null would make the test look far stronger than
it is. This script asks whether the null is tight because the series is genuinely
flat away from goals, or because of a defect (e.g. S constant, or the placebo
windows overlapping).

Reports, per cohort:
  * the distribution of S over all in-play grid points, and the share exactly 0
  * the distribution of the SAME pre/post window contrast evaluated at every
    eligible placebo anchor -- this is the per-draw ingredient of the null
  * how that per-anchor contrast compares with the contrast at real goals
  * the implied null sd under independent sampling, as a cross-check on the
    Monte Carlo sd actually observed

Run: /home/azureuser/.venv/bin/python -u qc_null_tightness.py
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
ISO_GAP, WIN = 300, 300


def win_mean(arr, gt, centre, lo, hi):
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = int(np.searchsorted(gt, centre + hi))
    seg = arr[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if len(seg) else None


for cohort in ("dev", "holdout"):
    p = f"{OUT}/results/{cohort}/match_stats.json.gz"
    if not os.path.exists(p):
        print(f"!! {p} missing")
        continue
    with gzip.open(p, "rt") as f:
        MS = json.load(f)
    print("\n" + "=" * 96)
    print(f"COHORT {cohort.upper()} — {len(MS)} matches")
    print("=" * 96)

    allS, anchor_d, goal_d, per_match_anchor = [], [], [], {}
    for m, st in MS.items():
        Sc = np.array(st["Sc"], dtype=float)
        gt = st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"])
        gw = sorted(st["goal_w"])
        fin = Sc[np.isfinite(Sc)]
        allS.append(fin)
        ds = []
        for i in st["elig"]:
            t = gt[i]
            a = win_mean(Sc, gt, t, PRE_LO, PRE_HI)
            b = win_mean(Sc, gt, t, POST_LO, POST_HI)
            if a is not None and b is not None:
                ds.append(b - a)
        if ds:
            per_match_anchor[m] = np.array(ds)
            anchor_d.extend(ds)
        for w in gw:
            if w - WIN < st["t_start"] or w + WIN > st["t_end"]:
                continue
            if not all(abs(w - o) > ISO_GAP for o in gw if abs(o - w) > 1e-9):
                continue
            a = win_mean(Sc, gt, w, PRE_LO, PRE_HI)
            b = win_mean(Sc, gt, w, POST_LO, POST_HI)
            if a is not None and b is not None:
                goal_d.append(b - a)

    S = np.concatenate(allS) if allS else np.array([])
    ad = np.array(anchor_d)
    gd = np.array(goal_d)

    print(f"\n  S over all in-play grid points   n={len(S):,}")
    if len(S):
        print(f"    exactly zero            : {100*np.mean(S <= 1e-12):5.1f}%")
        print(f"    > 0.001c                : {100*np.mean(S > 1e-5):5.1f}%")
        print(f"    > 0.01c                 : {100*np.mean(S > 1e-4):5.1f}%")
        q = np.percentile(S, [50, 75, 90, 99])
        print(f"    median {100*q[0]:.4f}c   p75 {100*q[1]:.4f}c   "
              f"p90 {100*q[2]:.4f}c   p99 {100*q[3]:.4f}c   "
              f"max {100*S.max():.4f}c")

    print(f"\n  placebo-anchor contrast (post-pre)   n={len(ad):,}")
    if len(ad):
        print(f"    mean {100*ad.mean():+.5f}c   sd {100*ad.std(ddof=1):.5f}c   "
              f"exactly zero {100*np.mean(np.abs(ad) < 1e-12):.1f}%")
        q = np.percentile(ad, [1, 25, 50, 75, 99])
        print(f"    p1 {100*q[0]:+.4f}c  p25 {100*q[1]:+.4f}c  med {100*q[2]:+.4f}c  "
              f"p75 {100*q[3]:+.4f}c  p99 {100*q[4]:+.4f}c")
        print(f"    min {100*ad.min():+.4f}c   max {100*ad.max():+.4f}c")

    print(f"\n  REAL GOAL contrast (post-pre)   n={len(gd):,}")
    if len(gd):
        print(f"    mean {100*gd.mean():+.4f}c   sd {100*gd.std(ddof=1):.4f}c")
        q = np.percentile(gd, [1, 25, 50, 75, 99])
        print(f"    p1 {100*q[0]:+.4f}c  p25 {100*q[1]:+.4f}c  med {100*q[2]:+.4f}c  "
              f"p75 {100*q[3]:+.4f}c  p99 {100*q[4]:+.4f}c")

    if len(ad) and len(gd):
        print(f"\n  goal sd / anchor sd = {gd.std(ddof=1)/ad.std(ddof=1):.1f}x")
        print(f"  share of placebo anchors exceeding the mean goal contrast: "
              f"{100*np.mean(ad >= gd.mean()):.3f}%")
        ks = [len(v) for v in per_match_anchor.values()]
        nm = len(per_match_anchor)
        approx = np.mean([v.std(ddof=1) for v in per_match_anchor.values()
                          if len(v) > 1])
        print(f"\n  cross-check on the Monte Carlo null sd")
        print(f"    matches {nm}, mean eligible anchors/match {np.mean(ks):.0f}")
        print(f"    mean within-match anchor sd {100*approx:.5f}c")
        print(f"    if each match contributes k~2.5 draws, expected null sd ~ "
              f"{100*approx/np.sqrt(2.5)/np.sqrt(nm):.5f}c")
        print("    (compare with the 'null sd' printed by event_study.py)")
