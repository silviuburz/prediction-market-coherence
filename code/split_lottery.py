#!/usr/bin/env python3
"""Was the dev/holdout split lucky, or chosen to be lucky?

The registered rule is sha256(slug)[0] in {0..6} plus three already-inspected
matches forced into development. It is a deterministic function of the match
identifier and no outcome data enters it. But a sceptical reader can still ask
whether the RULE was searched -- whether several candidate splits were tried and
the flattering one kept.

Git ordering answers part of it (split registered, then engine frozen, then
holdout run once). This adds a post-holdout label-sensitivity check: draw many
alternative partitions of the same 104 matches into 49/55 and recompute the
holdout estimate under the FROZEN estimand. It does not re-select windows or
re-tune the engine on each alternative development cohort, so it is not a
simulation of alternative end-to-end research workflows. An upper-tail
registered value would be consistent with favourable label selection; a low-tail
value shows that the registered labels did not inflate this fixed statistic.

Reported in CENTS. S is stored in probability units.

Run: /home/azureuser/.venv/bin/python -u code/split_lottery.py
"""
import gzip
import json
import os
from datetime import datetime

import numpy as np

OUT = "/home/azureuser/projects/wc_coherence"
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
ISO_GAP, WIN, SEED = 300, 300, 29
C = 100.0
N_DRAW = 20000


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def win_mean(arr, gt, centre, lo, hi):
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = int(np.searchsorted(gt, centre + hi))
    seg = arr[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if len(seg) else None


# ── per-match statistic for every match in the study ──────────────────────
per_match, cohort_of = {}, {}
for cohort in ("dev", "holdout"):
    with gzip.open(f"{OUT}/results/{cohort}/match_stats.json.gz", "rt") as f:
        MS = json.load(f)
    for m, st in MS.items():
        cohort_of[m] = cohort
        Sc = np.array(st["Sc"], dtype=float)
        gt = st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"])
        gw = sorted(st["goal_w"])
        ds = []
        for w in gw:
            if w - WIN < st["t_start"] or w + WIN > st["t_end"]:
                continue
            if not all(abs(w - o) > ISO_GAP for o in gw if abs(o - w) > 1e-9):
                continue
            a = win_mean(Sc, gt, w, PRE_LO, PRE_HI)
            b = win_mean(Sc, gt, w, POST_LO, POST_HI)
            if a is not None and b is not None:
                ds.append(b - a)
        if ds:
            per_match[m] = float(np.mean(ds))

split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
dev_reg = [m for m in split["dev"] if m in per_match]
hold_reg = [m for m in split["holdout"] if m in per_match]
n_dev_all, n_hold_all = len(split["dev"]), len(split["holdout"])

obs_hold = float(np.mean([per_match[m] for m in hold_reg]))
obs_dev = float(np.mean([per_match[m] for m in dev_reg]))
all_matches = sorted(per_match)
overall = float(np.mean([per_match[m] for m in all_matches]))

log(f"matches contributing an isolated goal : {len(all_matches)}")
log(f"registered holdout D  : {C*obs_hold:+.4f}c   ({len(hold_reg)} matches)")
log(f"registered dev     D  : {C*obs_dev:+.4f}c   ({len(dev_reg)} matches)")
log(f"pooled over all       : {C*overall:+.4f}c")

# ── lottery over alternative partitions of the same 104 matches ───────────
rng = np.random.default_rng(SEED)
universe = sorted(cohort_of)          # all 104, including those with no goal
vals = np.empty(N_DRAW)
for i in range(N_DRAW):
    perm = rng.permutation(universe)
    hold = [m for m in perm[n_dev_all:] if m in per_match]
    vals[i] = np.mean([per_match[m] for m in hold]) if hold else np.nan
vals = vals[np.isfinite(vals)]

pct = 100.0 * float((vals < obs_hold).mean())
frac_higher = 100.0 * float((vals >= obs_hold).mean())

print("\n" + "=" * 74)
print("SPLIT LOTTERY")
print("=" * 74)
print(f"  alternative partitions drawn : {len(vals)}")
print(f"  null mean                    : {C*vals.mean():+.4f}c")
print(f"  null sd                      : {C*vals.std(ddof=1):.4f}c")
for q in (1, 5, 25, 50, 75, 95, 99):
    print(f"  p{q:<3}                         : {C*np.percentile(vals, q):+.4f}c")
print(f"\n  REGISTERED SPLIT             : {C*obs_hold:+.4f}c")
print(f"  percentile of registered     : {pct:.1f}")
print(f"  random splits scoring HIGHER : {frac_higher:.1f}%")
print(f"  fraction of draws positive   : {100.0*float((vals > 0).mean()):.1f}%")
print(f"  minimum over all draws       : {C*vals.min():+.4f}c")

dest = f"{OUT}/results/split_lottery.json"
with open(dest, "w") as f:
    json.dump({"n_draw": int(len(vals)), "seed": SEED,
            "registered_holdout_c": C * obs_hold,
            "registered_dev_c": C * obs_dev,
            "pooled_c": C * overall,
            "null_mean_c": C * float(vals.mean()),
            "null_sd_c": C * float(vals.std(ddof=1)),
            "percentile_of_registered": pct,
            "pct_random_splits_higher": frac_higher,
            "pct_draws_positive": 100.0 * float((vals > 0).mean()),
            "min_draw_c": C * float(vals.min()),
            "scope": "post-holdout label sensitivity conditional on frozen estimand",
            "retunes_analysis_per_split": False}, f, indent=2)
log(f"wrote {dest}")
