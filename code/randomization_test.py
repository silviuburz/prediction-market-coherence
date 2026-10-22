#!/usr/bin/env python3
"""Randomization inference, recomputed offline with an exactly-matched estimator.

This supersedes the randomization block inside event_study.py, which had two
defects that the persisted series now lets us fix without recomputing S.

DEFECT 1 -- units in the diagnostic print. S is carried in probability units and
only multiplied by 100 at print time for D. The rand_diag print introduced during
the correction sprint labelled the null moments "c" without converting, making
the null look 100x tighter than it is. The stored values were correct; only the
label was wrong. Everything here is reported in cents explicitly.

DEFECT 2 -- the null did not use the same estimator as the observation. The
observed contrast slices the series with searchsorted on absolute time, giving a
12-point pre-window for [-60, -6]. The null sliced with floor division of the
offsets (-6 // 5 = -2), giving an 11-point pre-window. The null was therefore a
slightly different statistic than the thing it was meant to calibrate. Here both
use the identical win_mean.

The test: for each match, draw as many placebo anchors as that match contributed
isolated goals, from the eligible in-play anchors (>= PLACEBO_GAP from any goal
and >= WIN from either end of the match). Average within match, then across
matches, exactly as the observed statistic does. Repeat N_RAND times.

Run: /home/azureuser/.venv/bin/python -u randomization_test.py
"""
import gzip
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
ISO_GAP, WIN, N_RAND, SEED = 300, 300, 5000, 29
C = 100.0  # probability -> cents


def win_mean(arr, gt, centre, lo, hi):
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = int(np.searchsorted(gt, centre + hi))
    seg = arr[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if len(seg) else None


out = {}
for cohort in ("dev", "holdout"):
    p = f"{OUT}/results/{cohort}/match_stats.json.gz"
    if not os.path.exists(p):
        print(f"!! missing {p}")
        continue
    with gzip.open(p, "rt") as f:
        MS = json.load(f)

    prepped, obs_per_match = {}, defaultdict(list)
    for m, st in MS.items():
        Sc = np.array(st["Sc"], dtype=float)
        gt = st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"])
        gw = sorted(st["goal_w"])
        prepped[m] = (Sc, gt, st["elig"])
        for w in gw:
            if w - WIN < st["t_start"] or w + WIN > st["t_end"]:
                continue
            if not all(abs(w - o) > ISO_GAP for o in gw if abs(o - w) > 1e-9):
                continue
            a = win_mean(Sc, gt, w, PRE_LO, PRE_HI)
            b = win_mean(Sc, gt, w, POST_LO, POST_HI)
            if a is not None and b is not None:
                obs_per_match[m].append(b - a)

    obs_k = {m: len(v) for m, v in obs_per_match.items() if v}
    obs = float(np.mean([np.mean(obs_per_match[m]) for m in sorted(obs_k)]))
    usable = [m for m in sorted(obs_k) if len(prepped[m][2]) >= obs_k[m]]
    dropped = sorted(set(obs_k) - set(usable))

    random.seed(SEED)
    null = np.empty(N_RAND)
    for b_ in range(N_RAND):
        pm = []
        for m in usable:
            Sc, gt, elig = prepped[m]
            ds = []
            for i in random.sample(elig, obs_k[m]):
                t = gt[i]
                a = win_mean(Sc, gt, t, PRE_LO, PRE_HI)
                c_ = win_mean(Sc, gt, t, POST_LO, POST_HI)
                if a is not None and c_ is not None:
                    ds.append(c_ - a)
            if ds:
                pm.append(np.mean(ds))
        null[b_] = np.mean(pm) if pm else np.nan
    null = null[np.isfinite(null)]

    n_exceed = int((null >= obs).sum())
    p_rand = (1.0 + n_exceed) / (1.0 + len(null))
    q = np.percentile(null, [50, 90, 95, 99, 99.9])

    print("\n" + "=" * 96)
    print(f"RANDOMIZATION INFERENCE — {cohort.upper()}")
    print("=" * 96)
    print(f"  observed D            : {C*obs:+.4f} cents")
    print(f"  matches / events      : {len(obs_k)} / {sum(obs_k.values())}")
    print(f"  matches in null       : {len(usable)}"
          + (f"   DROPPED {dropped}" if dropped else "   (aligned)"))
    print(f"  null draws            : {len(null)}")
    print(f"  null mean             : {C*null.mean():+.4f} cents")
    print(f"  null sd               : {C*null.std(ddof=1):.4f} cents")
    print(f"  null p50/p90/p99/p99.9: {C*q[0]:+.4f} / {C*q[1]:+.4f} / "
          f"{C*q[3]:+.4f} / {C*q[4]:+.4f} cents")
    print(f"  null max              : {C*null.max():+.4f} cents")
    print(f"  draws >= observed     : {n_exceed}")
    print(f"  p (1+B)/(1+N)         : {p_rand:.5f}")
    print(f"  observed / null sd    : {obs/null.std(ddof=1):.1f}"
          "   (descriptive only; a 5000-draw Monte Carlo test cannot")
    print("                          resolve tail probabilities at that scale)")

    out[cohort] = {"observed_cents": C * obs, "n_matches": len(obs_k),
                   "n_events": int(sum(obs_k.values())),
                   "n_matches_null": len(usable), "dropped": dropped,
                   "n_null": int(len(null)), "n_exceed": n_exceed,
                   "p": p_rand, "null_mean_cents": C * float(null.mean()),
                   "null_sd_cents": C * float(null.std(ddof=1)),
                   "null_max_cents": C * float(null.max()),
                   "null_quantiles_cents": {k: C * float(v) for k, v in
                                            zip([50, 90, 95, 99, 99.9], q)},
                   "seed": SEED, "estimator": "(1+#{null>=obs})/(1+N)",
                   "windows": {"pre": [PRE_LO, PRE_HI], "post": [POST_LO, POST_HI],
                               "iso_gap": ISO_GAP, "win": WIN}}

    try:
        eng = json.load(open(f"{OUT}/results/{cohort}/event_study.json"))
        w_ = C * eng["isolated"]["D"] if abs(eng["isolated"]["D"]) < 1 else \
            eng["isolated"]["D"]
        print(f"\n  self-check: engine D {w_:+.4f}c vs here {C*obs:+.4f}c -> "
              f"{'MATCH' if abs(w_ - C*obs) < 5e-3 else '*** MISMATCH ***'}")
    except Exception as e:
        print(f"\n  self-check unavailable: {e}")

json.dump(out, open(f"{OUT}/results/randomization_test.json", "w"), indent=2)
print(f"\nresults -> {OUT}/results/randomization_test.json")
