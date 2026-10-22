#!/usr/bin/env python3
"""Covariate-matched placebo-time test.

The pre-specified placebo test samples anchors from any eligible in-play instant
at least PLACEBO_GAP seconds from a goal. It does not match on match minute,
period, score state or pre-event slack, so a critic can argue the null draws are
not comparable to goal instants -- goals cluster in particular game phases and
score states, and those phases may differ in baseline volatility.

This script rebuilds the null with anchors matched to each goal on observable
pre-treatment covariates, under progressively stricter schemes:

  unmatched   the pre-specified test, reproduced here as a control
  time        |elapsed(anchor) - elapsed(goal)| <= TIME_TOL
  score       same number of goals already scored in the match
  time+score  both of the above
  full        both, plus pre-window slack within SLACK_TOL cents

Matching on the pre-window slack is reported separately because it conditions on
the level of the outcome immediately before the window. That balances mean
reversion between goals and placebos, which is the point, but it is a stronger
conditioning choice than the others and should be read as a sensitivity rather
than the headline.

Each scheme draws one anchor per goal from that goal's own candidate pool, so the
null preserves the observed distribution of goals across matches exactly. Goals
with an empty pool are reported and excluded from that scheme.

Everything is reported in CENTS. S is stored in probability units.

Run: /home/azureuser/.venv/bin/python -u matched_placebo.py 2>&1 | tee logs/matched_placebo.log
"""
import gzip
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

OUT = "/home/azureuser/projects/wc_coherence"
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
ISO_GAP, WIN, N_RAND, SEED = 300, 300, 5000, 29
C = 100.0                 # probability -> cents
TIME_TOL = 600.0          # seconds of match-elapsed time
SLACK_TOL = 0.25 / C      # 0.25 cents, expressed in probability units

SCHEMES = ("unmatched", "time", "score", "time+score", "full")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def prefix_sums(a):
    """NaN-aware prefix sums so window means are O(1)."""
    ok = ~np.isnan(a)
    v = np.where(ok, a, 0.0)
    return np.concatenate([[0.0], np.cumsum(v)]), \
        np.concatenate([[0.0], np.cumsum(ok.astype(float))])


def win_mean_fast(cs, cn, gt, centre, lo, hi):
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = int(np.searchsorted(gt, centre + hi))
    i1 = min(i1 + 1, len(gt))
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
    log(f"loading {cohort}")
    with gzip.open(p, "rt") as f:
        MS = json.load(f)

    # ---- per-match arrays and per-goal covariates ------------------------
    prepped = {}
    goals = []            # (match, centre, d, elapsed, n_prior, pre_slack)
    for m, st in MS.items():
        Sc = np.array(st["Sc"], dtype=float)
        gt = st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"])
        cs, cn = prefix_sums(Sc)
        prepped[m] = (cs, cn, gt, st["elig"], st["t_start"])
        gw = sorted(st["goal_w"])
        for w in gw:
            if w - WIN < st["t_start"] or w + WIN > st["t_end"]:
                continue
            if not all(abs(w - o) > ISO_GAP for o in gw if abs(o - w) > 1e-9):
                continue
            a = win_mean_fast(cs, cn, gt, w, PRE_LO, PRE_HI)
            b = win_mean_fast(cs, cn, gt, w, POST_LO, POST_HI)
            if a is None or b is None:
                continue
            goals.append((m, w, b - a, w - st["t_start"],
                          sum(1 for o in gw if o < w - 1e-9), a))

    matches = sorted({g[0] for g in goals})
    per_match_obs = {}
    for m in matches:
        per_match_obs[m] = float(np.mean([g[2] for g in goals if g[0] == m]))
    obs = float(np.mean([per_match_obs[m] for m in matches]))
    log(f"{cohort}: {len(matches)} matches, {len(goals)} isolated goals, "
        f"observed D = {C*obs:+.4f}c")

    # ---- candidate anchor pools, per goal, per scheme --------------------
    pools = {s: [] for s in SCHEMES}
    for (m, w, _d, elapsed, n_prior, pre_slack) in goals:
        cs, cn, gt, elig, t_start = prepped[m]
        gw = sorted(MS[m]["goal_w"])
        cand = {s: [] for s in SCHEMES}
        for i in elig:
            t = gt[i]
            a_el = t - t_start
            a_prior = sum(1 for o in gw if o < t - 1e-9)
            ok_t = abs(a_el - elapsed) <= TIME_TOL
            ok_s = a_prior == n_prior
            cand["unmatched"].append(i)
            if ok_t:
                cand["time"].append(i)
            if ok_s:
                cand["score"].append(i)
            if ok_t and ok_s:
                cand["time+score"].append(i)
                a_pre = win_mean_fast(cs, cn, gt, t, PRE_LO, PRE_HI)
                if a_pre is not None and abs(a_pre - pre_slack) <= SLACK_TOL:
                    cand["full"].append(i)
        for s in SCHEMES:
            pools[s].append((m, cand[s]))

    # ---- Monte Carlo per scheme -----------------------------------------
    results[cohort] = {"observed_c": C * obs, "n_matches": len(matches),
                       "n_goals": len(goals), "schemes": {}}
    for s in SCHEMES:
        t0 = time.time()
        usable = [(m, c) for (m, c) in pools[s] if c]
        n_drop = len(pools[s]) - len(usable)
        pool_sizes = [len(c) for _m, c in usable]
        if not usable:
            log(f"  {s:<11} NO usable goals, skipped")
            continue
        # observed statistic restricted to the goals this scheme can calibrate
        keep = [g for g, (_m, c) in zip(goals, pools[s]) if c]
        by_match = {}
        for g in keep:
            by_match.setdefault(g[0], []).append(g[2])
        obs_s = float(np.mean([np.mean(v) for v in by_match.values()]))

        random.seed(SEED)
        null = np.empty(N_RAND)
        for b_ in range(N_RAND):
            acc = {}
            for (m, cand) in usable:
                cs, cn, gt, _e, _ts = prepped[m]
                t = gt[random.choice(cand)]
                a = win_mean_fast(cs, cn, gt, t, PRE_LO, PRE_HI)
                c_ = win_mean_fast(cs, cn, gt, t, POST_LO, POST_HI)
                if a is not None and c_ is not None:
                    acc.setdefault(m, []).append(c_ - a)
            null[b_] = np.mean([np.mean(v) for v in acc.values()]) if acc else np.nan
        null = null[np.isfinite(null)]
        n_exceed = int((null >= obs_s).sum())
        p = (1.0 + n_exceed) / (1.0 + len(null))

        rec = {"goals_used": len(usable), "goals_dropped": n_drop,
               "median_pool": float(np.median(pool_sizes)),
               "min_pool": int(np.min(pool_sizes)),
               "observed_c": C * obs_s,
               "null_mean_c": C * float(null.mean()),
               "null_sd_c": C * float(null.std(ddof=1)),
               "null_max_c": C * float(null.max()),
               "n_draws": int(len(null)), "n_exceed": n_exceed, "p": p,
               "ratio_obs_to_max": float(obs_s / null.max()) if null.max() > 0 else None}
        results[cohort]["schemes"][s] = rec
        log(f"  {s:<11} goals {len(usable):>3} (drop {n_drop:>2})  "
            f"pool med {rec['median_pool']:>5.0f} min {rec['min_pool']:>3}  "
            f"obs {rec['observed_c']:+.4f}c  null max {rec['null_max_c']:+.4f}c  "
            f"sd {rec['null_sd_c']:.4f}  p={p:.5f}  [{time.time()-t0:.0f}s]")

os.makedirs(f"{OUT}/results", exist_ok=True)
dest = f"{OUT}/results/matched_placebo.json"
with open(dest, "w") as f:
    json.dump({"params": {"TIME_TOL_s": TIME_TOL, "SLACK_TOL_c": C * SLACK_TOL,
                          "N_RAND": N_RAND, "SEED": SEED,
                          "PRE": [PRE_LO, PRE_HI], "POST": [POST_LO, POST_HI]},
               "cohorts": results}, f, indent=2)
log(f"wrote {dest}")
