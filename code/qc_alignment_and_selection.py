#!/usr/bin/env python3
"""Two residual validity checks that the audits have not covered.

CHECK 1 -- GOAL TIMESTAMP ALIGNMENT.
Goal times come from ESPN keyEvents wallclock. That the key events reproduce the
final score 104/104 validates the SCORE, not the TIMESTAMP. If the wallclock lags
the real event by tens of seconds, the pre-goal window [-60,-6] could contain the
actual goal, which would contaminate the baseline and misplace the event-time
curve. The event-time profile is itself the diagnostic: locate the offset at
which the largest jump in mean S occurs. A step at or near 0 confirms alignment;
a step at +40 s would mean the timestamps lag and the windows are misplaced.

CHECK 2 -- ISOLATION SELECTION.
The primary estimand uses only isolated goals (no other goal within 300 s), which
discards roughly a quarter of goals. If clustered goals behave differently, the
primary is a selected subsample. The engine already computes the all-goals
contrast; compare them.

Run: /home/azureuser/.venv/bin/python -u qc_alignment_and_selection.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
C = 100.0

print("=" * 96)
print("CHECK 2 — ISOLATION SELECTION (isolated primary vs all eligible goals)")
print("=" * 96)
print(f"  {'cohort':<10}{'set':<12}{'matches':>9}{'events':>8}{'D (cents)':>12}"
      f"{'95% CI (cents)':>24}")
for cohort in ("dev", "holdout"):
    p = f"{OUT}/results/{cohort}/event_study.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    for tag, key in (("isolated", "isolated"), ("all goals", "all")):
        r = d.get(key)
        if not r:
            continue
        ci = f"[{C*r['ci'][0]:+.3f}, {C*r['ci'][1]:+.3f}]"
        print(f"  {cohort:<10}{tag:<12}{r['n_matches']:>9}{r['n_events']:>8}"
              f"{C*r['D']:>+12.3f}{ci:>24}")

print("\n" + "=" * 96)
print("CHECK 1 — GOAL TIMESTAMP ALIGNMENT (where does the step occur?)")
print("=" * 96)
pf = f"{OUT}/figs/event_time_profiles.json"
if not os.path.exists(pf):
    print(f"  !! {pf} not present yet")
    sys.exit(0)
P = json.load(open(pf))
for cohort in ("dev", "holdout"):
    g = P.get(cohort, {}).get("goal")
    if not g:
        continue
    off = np.array(g["offsets"], dtype=float)
    mean = np.array([np.nan if v is None else v for v in g["mean_cents"]],
                    dtype=float)
    ok = np.isfinite(mean)
    o, m = off[ok], mean[ok]
    d1 = np.diff(m)
    j = int(np.argmax(d1))
    # centre of mass of the rise, robust to a single noisy step
    base = np.nanmean(m[o < -60])
    peak = np.nanmax(m[(o >= 0) & (o <= 120)])
    half = base + 0.5 * (peak - base)
    cross = o[1:][(m[1:] >= half) & (o[1:] >= -120)]
    print(f"\n  {cohort.upper()}  ({g['n_events']} events, {g['n_matches']} matches)")
    print(f"    pre-goal baseline (offset < -60 s) : {base:+.4f}c")
    print(f"    peak within [0, +120] s            : {peak:+.4f}c")
    print(f"    largest single-step rise at offset : {o[j]:+.0f} s "
          f"(+{d1[j]:.4f}c)")
    print(f"    first offset at half of the rise   : "
          f"{cross[0]:+.0f} s" if len(cross) else "    half-rise not reached")
    for lo, hi in ((-120, -61), (-60, -31), (-30, -6), (-5, 5), (6, 30),
                   (31, 60), (61, 120)):
        sel = (o >= lo) & (o <= hi)
        if sel.any():
            print(f"      mean S on [{lo:>+5},{hi:>+5}] s : {np.nanmean(m[sel]):+.4f}c")

print("\nINTERPRETATION")
print("  A step centred at 0 s confirms the keyEvents wallclock is the real")
print("  event time. A step centred well after 0 s would mean the timestamps")
print("  lag, the pre-window is contaminated, and RQ3 latencies are offset.")
