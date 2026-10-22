#!/usr/bin/env python3
"""SPECIFICATION CURVE v2 — corrected.

Fixes three defects in v1:
  1. v1 bootstrapped EVENTS for event-weighted specs, ignoring within-match
     dependence. v2 always resamples MATCHES (clusters) for every specification.
  2. v1 called the +0.451 vs +0.436 primary gap "sampling noise". It is fully
     DETERMINISTIC: the engine's inclusive slice `arr[i0:i1+1]` makes its
     pre-window [-60,-5] whereas offset-selection gives [-60,-10]. Including the
     -5 s offset (nearest the goal, already slightly elevated) raises the
     baseline and lowers D -- so the engine value is the more conservative one.
     v2 reports BOTH window conventions and reconciles exactly.
  3. v1 pooled 140 equal-match and 140 event-weighted specs into one "280/280"
     headline. v2 separates them and labels the whole exercise POST-HOLDOUT
     DESCRIPTIVE ROBUSTNESS -- not a second confirmation.

Run: /home/azureuser/.venv/bin/python -u spec_curve_v2.py
"""
import json
import os
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
            d = ev.setdefault(k, {"isolated": r["isolated"], "S_cross": {}, "S_state": {}})
            for m in METRICS:
                if r.get(m) is not None:
                    d[m][int(r["offset"])] = float(r[m])
    return ev


def win_mean(series, lo, hi, engine_slice):
    """engine_slice=True mimics the engine's inclusive upper bound (adds the
    first grid point at or above `hi`), reproducing the frozen primary exactly."""
    offs = sorted(series)
    if engine_slice:
        above = [o for o in offs if o >= hi]
        hi_eff = min(above) if above else hi
    else:
        hi_eff = hi
    vals = [v for off, v in series.items() if lo <= off <= hi_eff]
    return float(np.mean(vals)) if vals else None


def estimate(ev, pre, post, metric, sample, weight, engine_slice=False):
    per_match = defaultdict(list)
    for (match, _e), d in ev.items():
        if sample == "isolated" and not d["isolated"]:
            continue
        s = d[metric]
        a = win_mean(s, pre[0], pre[1], engine_slice)
        b = win_mean(s, post[0], post[1], engine_slice)
        if a is None or b is None:
            continue
        per_match[match].append(b - a)
    if len(per_match) < 5:
        return None
    ms = sorted(per_match)
    if weight == "equal_match":
        vals = np.array([np.mean(per_match[m]) for m in ms])
        boot = np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                         for _ in range(N_BOOT)])
        D = float(vals.mean())
    else:
        # event-weighted, but STILL cluster-resampled by match
        D = float(np.mean([x for m in ms for x in per_match[m]]))
        boot = []
        for _ in range(N_BOOT):
            pick = np.random.choice(len(ms), len(ms), replace=True)
            xs = [x for i in pick for x in per_match[ms[i]]]
            boot.append(np.mean(xs))
        boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"D": D, "lo": float(lo), "hi": float(hi),
            "n_matches": len(ms), "n_events": sum(len(v) for v in per_match.values())}


print("=" * 100)
print("SPECIFICATION CURVE v2 (POST-HOLDOUT DESCRIPTIVE ROBUSTNESS -- not confirmation)")
print("=" * 100)

report = {}
for cohort in ("dev", "holdout"):
    ev = load(cohort)
    if ev is None:
        continue
    print(f"\n{'='*100}\n{cohort.upper()}  ({len(ev)} events)\n{'='*100}")

    # ---- exact reconciliation of the frozen primary ------------------------
    p_off = estimate(ev, *PRIMARY[:2], PRIMARY[2], PRIMARY[3], PRIMARY[4], False)
    p_eng = estimate(ev, *PRIMARY[:2], PRIMARY[2], PRIMARY[3], PRIMARY[4], True)
    print("  RECONCILIATION of the registered primary specification")
    print(f"    offset-selection  pre=[-60,-10] : {p_off['D']*100:+.3f} c")
    print(f"    engine slice      pre=[-60, -5] : {p_eng['D']*100:+.3f} c  <- frozen primary")
    print(f"    difference is deterministic (extra -5 s point raises the baseline)")

    # ---- specification cloud, split by weighting ---------------------------
    for weight in WEIGHTS:
        specs = []
        for pre in PRE:
            for post in POST:
                for metric in METRICS:
                    for sample in SAMPLES:
                        r = estimate(ev, pre, post, metric, sample, weight, True)
                        if r:
                            r.update({"pre": pre, "post": post, "metric": metric,
                                      "sample": sample, "weight": weight})
                            specs.append(r)
        Ds = np.array([s["D"] for s in specs])
        pos = int((Ds > 0).sum())
        sig = sum(1 for s in specs if s["lo"] > 0)
        print(f"\n  [{weight}]  {len(specs)} specifications (match-clustered bootstrap)")
        print(f"    D > 0                : {pos}/{len(specs)}")
        print(f"    lower bound > 0      : {sig}/{len(specs)}")
        print(f"    D range (cents)      : {Ds.min()*100:+.3f} .. {Ds.max()*100:+.3f} "
              f"(median {np.median(Ds)*100:+.3f})")
        worst = min(specs, key=lambda z: z["D"])
        print(f"    weakest spec         : {worst['D']*100:+.3f}c "
              f"[{worst['lo']*100:+.3f},{worst['hi']*100:+.3f}]  "
              f"pre{worst['pre']} post{worst['post']} {worst['metric']} {worst['sample']}")
        report[f"{cohort}_{weight}"] = {
            "n": len(specs), "pos": pos, "sig": sig,
            "min": float(Ds.min()), "median": float(np.median(Ds)),
            "max": float(Ds.max())}
        json.dump(specs, open(f"{OUT}/results/{cohort}/spec_curve_{weight}.json", "w"),
                  indent=2)
    report[f"{cohort}_primary_offset"] = p_off["D"]
    report[f"{cohort}_primary_engine"] = p_eng["D"]

json.dump(report, open(f"{OUT}/results/spec_curve_v2_summary.json", "w"), indent=2)
print("\n" + "=" * 100)
print("NOTE: these specifications were evaluated AFTER the holdout result was known.")
print("They are descriptive robustness, NOT an independent confirmation, and the")
print("intervals are unadjusted for multiplicity.")
print("=" * 100)
