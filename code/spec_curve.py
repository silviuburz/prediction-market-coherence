#!/usr/bin/env python3
"""SPECIFICATION CURVE (multiverse) over analysis choices.

Pre-empts "why that window?" by re-estimating the effect under every reasonable
combination of analyst choices, on BOTH cohorts, and reporting the distribution.

Choices varied:
  pre-window     5 options
  post-window    7 options
  metric         S_cross (payoff-only) | S_state (known-state conditioned)
  sample         isolated goals | all eligible goals
  weighting      equal-match (primary) | event-weighted

=> 5 x 7 x 2 x 2 x 2 = 280 specifications per cohort.

Uses ONLY the persisted long-form rows, so no LP is re-solved and no new data is
touched. The pre-specified primary remains (-60,-6) vs (+6,+30), S_cross,
isolated, equal-match; it is highlighted rather than replaced.

Run: /home/azureuser/.venv/bin/python -u spec_curve.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

OUT = "/home/azureuser/projects/wc_coherence"
N_BOOT = 2000
np.random.seed(31)

PRE = [(-60, -6), (-120, -6), (-180, -6), (-120, -31), (-300, -61)]
POST = [(6, 30), (0, 30), (6, 60), (11, 40), (6, 20), (16, 45), (0, 60)]
METRICS = ["S_cross", "S_state"]
SAMPLES = ["isolated", "all"]
WEIGHTS = ["equal_match", "event"]
PRIMARY = ((-60, -6), (6, 30), "S_cross", "isolated", "equal_match")


def load(cohort):
    """-> {(match,event): {'isolated':bool, 'S_cross':{off:v}, 'S_state':{off:v}}}"""
    path = f"{OUT}/results/{cohort}/long_form.jsonl"
    if not os.path.exists(path):
        return None
    ev = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = (r["match"], r["event"])
            d = ev.setdefault(k, {"isolated": r["isolated"],
                                  "S_cross": {}, "S_state": {}})
            for m in METRICS:
                if r.get(m) is not None:
                    d[m][int(r["offset"])] = float(r[m])
    return ev


def win_mean(series, lo, hi):
    vals = [v for off, v in series.items() if lo <= off <= hi]
    return float(np.mean(vals)) if vals else None


def estimate(ev, pre, post, metric, sample, weight):
    per_match = defaultdict(list)
    for (match, _eid), d in ev.items():
        if sample == "isolated" and not d["isolated"]:
            continue
        s = d[metric]
        a = win_mean(s, pre[0], pre[1])
        b = win_mean(s, post[0], post[1])
        if a is None or b is None:
            continue
        per_match[match].append(b - a)
    if not per_match:
        return None
    if weight == "equal_match":
        vals = np.array([np.mean(v) for v in per_match.values()])
    else:
        vals = np.array([x for v in per_match.values() for x in v])
    if len(vals) < 5:
        return None
    D = float(vals.mean())
    boot = np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"D": D, "lo": float(lo), "hi": float(hi), "n": len(vals),
            "n_events": sum(len(v) for v in per_match.values()),
            "n_matches": len(per_match)}


print("=" * 96)
print("SPECIFICATION CURVE — robustness to analyst choices")
print("=" * 96)

summary = {}
for cohort in ("dev", "holdout"):
    ev = load(cohort)
    if ev is None:
        print(f"\n{cohort}: long-form not found, skipped")
        continue
    specs = []
    for pre in PRE:
        for post in POST:
            for metric in METRICS:
                for sample in SAMPLES:
                    for weight in WEIGHTS:
                        r = estimate(ev, pre, post, metric, sample, weight)
                        if r is None:
                            continue
                        r.update({"pre": pre, "post": post, "metric": metric,
                                  "sample": sample, "weight": weight,
                                  "is_primary": (pre, post, metric, sample, weight) == PRIMARY})
                        specs.append(r)
    Ds = np.array([s["D"] for s in specs])
    pos = sum(1 for s in specs if s["D"] > 0)
    sig = sum(1 for s in specs if s["lo"] > 0)
    prim = next((s for s in specs if s["is_primary"]), None)
    summary[cohort] = {"n_specs": len(specs), "pos": pos, "sig": sig,
                       "min": float(Ds.min()), "median": float(np.median(Ds)),
                       "max": float(Ds.max()),
                       "primary": prim["D"] if prim else None}
    print(f"\n--- {cohort.upper()} ({len(ev)} events) ---")
    print(f"  specifications evaluated : {len(specs)}")
    print(f"  D > 0                    : {pos}/{len(specs)}  ({100*pos/len(specs):.1f}%)")
    print(f"  95% CI excludes 0        : {sig}/{len(specs)}  ({100*sig/len(specs):.1f}%)")
    print(f"  D range (cents)          : min {Ds.min()*100:+.3f}  "
          f"median {np.median(Ds)*100:+.3f}  max {Ds.max()*100:+.3f}")
    if prim:
        print(f"  PRE-SPECIFIED PRIMARY    : {prim['D']*100:+.3f} cents "
              f"[{prim['lo']*100:+.3f}, {prim['hi']*100:+.3f}]  "
              f"(matches={prim['n_matches']}, events={prim['n_events']})")
        pct = 100.0 * float((Ds < prim["D"]).mean())
        print(f"  primary percentile in cloud: {pct:.0f}th "
              f"({'conservative' if pct < 50 else 'optimistic'} half)")

    # worst-case specifications
    print(f"\n  five WEAKEST specifications:")
    for s in sorted(specs, key=lambda z: z["D"])[:5]:
        print(f"    {s['D']*100:+.3f}c [{s['lo']*100:+.3f},{s['hi']*100:+.3f}]  "
              f"pre{s['pre']} post{s['post']} {s['metric']} {s['sample']} {s['weight']}")

    # marginal effect of each choice
    print(f"\n  mean D by choice (cents):")
    for dim, opts in (("pre", PRE), ("post", POST), ("metric", METRICS),
                      ("sample", SAMPLES), ("weight", WEIGHTS)):
        parts = []
        for o in opts:
            v = [s["D"] for s in specs if s[dim] == o]
            if v:
                parts.append(f"{o}={np.mean(v)*100:+.2f}")
        print(f"    {dim:<8} " + "  ".join(parts))

    json.dump(specs, open(f"{OUT}/results/{cohort}/spec_curve.json", "w"), indent=2)

print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
for c, s in summary.items():
    print(f"  {c:<9} {s['sig']}/{s['n_specs']} specifications significant; "
          f"D in [{s['min']*100:+.3f}, {s['max']*100:+.3f}] cents")
print("\n  If ~all specifications are positive and significant, the window choice")
print("  is immaterial and the 'researcher degrees of freedom' objection is moot.")
