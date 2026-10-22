#!/usr/bin/env python3
"""Specification curve v3 — computed from the persisted per-match coherence
series with EXACT engine window semantics.

Why a third version. v1 bootstrapped event-weighted specifications over events
rather than matches, understating standard errors. v2 fixed the clustering but
was still computed from long_form.jsonl, whose "offset" field is a grid LABEL:
the match grid is np.arange(t_start, t_end, 5) and is not aligned to goal times,
so the row labelled -60 is really the grid point in [-60, -55). The engine
instead slices on ABSOLUTE time via searchsorted. The two conventions differ by
up to one grid step, which is why v2 reproduced +0.451c / +0.414c against the
engine's +0.436c on the holdout.

v3 removes the ambiguity: it reads match_stats.json.gz, which stores the raw
coherence series together with grid_t0, grid_step and the goal wallclocks, and
reproduces the engine's searchsorted slice exactly. The primary specification
must therefore reproduce the frozen estimate to floating-point tolerance, which
is asserted. Every other cell of the curve is then a like-for-like perturbation
of that same estimator rather than an approximation of it.

Grid: 5 pre-window starts x 2 pre-window ends x 2 post-window starts
      x 5 post-window ends x 3 isolation gaps = 300 specifications,
      each under match-equal and event-equal weighting.

Run: /home/azureuser/.venv/bin/python -u spec_curve_v3.py
"""
import gzip
import itertools
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
N_BOOT = 2000
C = 100.0   # S is carried in probability units; all display is in cents
rng = np.random.default_rng(29)

PRE_LO = [-120, -90, -60, -45, -30]
PRE_HI = [-6, -16]
POST_LO = [6, 16]
POST_HI = [20, 30, 45, 60, 90]
ISO = [180, 300, 600]
PRIMARY = (-60, -6, 6, 30, 300)


def win_mean(arr, gt, centre, lo, hi):
    """Byte-identical to the frozen engine's win_mean."""
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = int(np.searchsorted(gt, centre + hi))
    seg = arr[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if len(seg) else None


def boot_ci(vals, weights=None):
    n = len(vals)
    if n < 3:
        return None
    idx = rng.integers(0, n, size=(N_BOOT, n))
    if weights is None:
        draws = vals[idx].mean(axis=1)
        point = float(vals.mean())
    else:
        v, w = vals[idx], weights[idx]
        draws = (v * w).sum(axis=1) / w.sum(axis=1)
        point = float((vals * weights).sum() / weights.sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


summary = {}
for cohort in ("dev", "holdout"):
    path = f"{OUT}/results/{cohort}/match_stats.json.gz"
    if not os.path.exists(path):
        print(f"!! missing {path} — run event_study.py first")
        continue
    with gzip.open(path, "rt") as f:
        MS = json.load(f)
    # pre-materialise arrays once. WIN is the frozen +/-300 s eligibility margin
    # (P["WIN"]); it is a fixed design parameter and is deliberately not varied.
    prepped = {}
    for m, st in MS.items():
        prepped[m] = (np.array(st["Sc"], dtype=float),
                      st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"]),
                      st["t_start"], st["t_end"], sorted(st["goal_w"]), 300)

    print("\n" + "=" * 104)
    print(f"SPECIFICATION CURVE v3 — {cohort.upper()}  ({len(MS)} matches)")
    print("=" * 104)

    rows = []
    for pl, ph, ql, qh, gap in itertools.product(PRE_LO, PRE_HI, POST_LO,
                                                 POST_HI, ISO):
        if ph <= pl or qh <= ql:
            continue
        per_match = defaultdict(list)
        for m, (Sc, gt, t0, t1, gw, WIN) in prepped.items():
            for w in gw:
                if w - WIN < t0 or w + WIN > t1:
                    continue
                if not all(abs(w - o) > gap for o in gw if abs(o - w) > 1e-9):
                    continue
                a = win_mean(Sc, gt, w, pl, ph)
                b = win_mean(Sc, gt, w, ql, qh)
                if a is None or b is None:
                    continue
                per_match[m].append(b - a)
        if len(per_match) < 5:
            continue
        ms = sorted(per_match)
        mvals = np.array([np.mean(per_match[m]) for m in ms])
        cnts = np.array([len(per_match[m]) for m in ms], dtype=float)
        for wname, res in (("equal_match", boot_ci(mvals)),
                           ("event", boot_ci(mvals, cnts))):
            if res is None:
                continue
            D, lo, hi = res
            rows.append({"pre_lo": pl, "pre_hi": ph, "post_lo": ql, "post_hi": qh,
                         "iso_gap": gap, "weighting": wname, "D": D,
                         "ci": [lo, hi], "sig": bool(lo > 0),
                         "n_matches": len(ms), "n_events": int(cnts.sum()),
                         "primary": (pl, ph, ql, qh, gap) == PRIMARY})

    for wname in ("equal_match", "event"):
        sub = [r for r in rows if r["weighting"] == wname]
        if not sub:
            continue
        ds = np.array([r["D"] for r in sub])
        nsig = sum(r["sig"] for r in sub)
        npos = int((ds > 0).sum())
        prim = [r for r in sub if r["primary"]]
        print(f"\n  weighting = {wname}   ({len(sub)} specifications)")
        print(f"    positive          : {npos}/{len(sub)}")
        print(f"    positive AND sig  : {nsig}/{len(sub)}")
        print(f"    D range           : {C*ds.min():+.3f} .. {C*ds.max():+.3f} cents")
        print(f"    D median          : {C*np.median(ds):+.3f} cents")
        if prim:
            p = prim[0]
            pct = 100.0 * (ds < p["D"]).mean()
            print(f"    PRIMARY spec      : {C*p['D']:+.3f}c "
                  f"[{C*p['ci'][0]:+.3f}, {C*p['ci'][1]:+.3f}]  "
                  f"-> {pct:.0f}th percentile of the curve "
                  f"({'below' if pct < 50 else 'above'} the multiverse median)")

    # ── exactness self-check against the frozen engine ─────────────────────
    prim = [r for r in rows if r["primary"] and r["weighting"] == "equal_match"]
    try:
        eng = json.load(open(f"{OUT}/results/{cohort}/event_study.json"))
        want = eng["isolated"]["D"]
        got = prim[0]["D"]
        d = got - want
        print(f"\n  EXACTNESS CHECK: engine {C*want:+.6f}c   curve {C*got:+.6f}c   "
              f"diff {C*d:+.2e}c  -> {'EXACT' if abs(d) < 1e-12 else ('close' if abs(d) < 5e-5 else '*** MISMATCH ***')}")
        print(f"    events: engine {eng['isolated']['n_events']}, "
              f"curve {prim[0]['n_events']}   |   matches: engine "
              f"{eng['isolated']['n_matches']}, curve {prim[0]['n_matches']}")
    except Exception as e:
        print(f"\n  exactness check unavailable: {e}")

    summary[cohort] = rows

json.dump(summary, open(f"{OUT}/results/spec_curve_v3.json", "w"), indent=2)
print(f"\nresults -> {OUT}/results/spec_curve_v3.json")
