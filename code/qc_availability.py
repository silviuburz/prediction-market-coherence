#!/usr/bin/env python3
"""Availability confound check.

S is the minimum uniform slack over the CURRENTLY QUOTED contracts. A contract
with no fresh two-sided quote contributes the vacuous bound [0, 1] and therefore
constrains nothing. Availability is thus mechanically coupled to S:

    more contracts quoted  ->  more binding constraints  ->  weakly larger S
    fewer contracts quoted ->  weaker system            ->  weakly smaller S

Two opposite worries follow, and only one of them is dangerous:

  (a) DANGEROUS. If availability systematically RISES after a goal, part of the
      measured increase in S is mechanical rather than informational.
  (b) REASSURING. If availability FALLS after a goal (market makers pulling
      quotes), then the liquidity-withdrawal story predicts S should DECREASE.
      Observing an increase anyway means the effect survives a force pushing
      against it.

This script settles which case holds, and then re-estimates the effect on the
subsample of goals where availability did NOT increase, which removes the
mechanical channel by construction.

Everything is computed from long_form.jsonl so that S and availability come from
exactly the same rows. Note this uses the offset-GRID LABEL convention, so the
absolute D differs slightly from the engine's absolute-time estimate (see
FINDINGS 12.4); the comparison ACROSS availability strata is internally
consistent and is the object of interest.

Run: /home/azureuser/.venv/bin/python -u qc_availability.py
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
N_BOOT, C = 5000, 100.0
rng = np.random.default_rng(29)


def boot(per_match):
    ms = sorted(per_match)
    v = np.array([float(np.mean(per_match[m])) for m in ms])
    if len(v) < 3:
        return None
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    d = v[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"D": float(v.mean()), "ci": [float(lo), float(hi)],
            "n_matches": len(ms),
            "n_events": int(sum(len(x) for x in per_match.values())),
            "sig": bool(lo > 0 or hi < 0)}


for cohort in ("dev", "holdout"):
    path = f"{OUT}/results/{cohort}/long_form.jsonl"
    if not os.path.exists(path):
        print(f"!! missing {path}")
        continue
    ev = defaultdict(lambda: {"pre_S": [], "post_S": [],
                              "pre_A": [], "post_A": [], "iso": False})
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("isolated"):
                continue
            k = (r["match"], r["event"])
            o = r["offset"]
            ev[k]["iso"] = True
            if PRE_LO <= o <= PRE_HI:
                if r["S_cross"] is not None:
                    ev[k]["pre_S"].append(r["S_cross"])
                ev[k]["pre_A"].append(r["avail"])
            elif POST_LO <= o <= POST_HI:
                if r["S_cross"] is not None:
                    ev[k]["post_S"].append(r["S_cross"])
                ev[k]["post_A"].append(r["avail"])

    rows = []
    for (m, e), d in ev.items():
        if not (d["pre_S"] and d["post_S"] and d["pre_A"] and d["post_A"]):
            continue
        rows.append({"match": m, "event": e,
                     "dS": float(np.mean(d["post_S"]) - np.mean(d["pre_S"])),
                     "preA": float(np.mean(d["pre_A"])),
                     "postA": float(np.mean(d["post_A"]))})
    if not rows:
        continue
    dA = np.array([r["postA"] - r["preA"] for r in rows])
    preA = np.array([r["preA"] for r in rows])
    postA = np.array([r["postA"] for r in rows])

    print("\n" + "=" * 96)
    print(f"AVAILABILITY CONFOUND CHECK — {cohort.upper()}   ({len(rows)} isolated goals)")
    print("=" * 96)
    print(f"  contracts quoted, pre-window : mean {preA.mean():.2f}  "
          f"min {preA.min():.1f}  max {preA.max():.1f}")
    print(f"  contracts quoted, post-window: mean {postA.mean():.2f}  "
          f"min {postA.min():.1f}  max {postA.max():.1f}")
    print(f"  change (post - pre)          : mean {dA.mean():+.3f}  "
          f"median {np.median(dA):+.3f}  sd {dA.std(ddof=1):.3f}")
    print(f"  events where availability ROSE   : {(dA > 1e-9).sum():>4} "
          f"({100*(dA > 1e-9).mean():.1f}%)")
    print(f"  events where availability FELL   : {(dA < -1e-9).sum():>4} "
          f"({100*(dA < -1e-9).mean():.1f}%)")
    print(f"  events where availability UNCHANGED: {(np.abs(dA) <= 1e-9).sum():>4} "
          f"({100*(np.abs(dA) <= 1e-9).mean():.1f}%)")

    # per-match effect, full sample and availability-restricted subsamples
    def sub(mask, label):
        pm = defaultdict(list)
        for r, keep in zip(rows, mask):
            if keep:
                pm[r["match"]].append(r["dS"])
        b = boot(pm)
        if b is None:
            print(f"  {label:<34} insufficient")
            return
        print(f"  {label:<34}{C*b['D']:>+9.3f}c  "
              f"[{C*b['ci'][0]:+.3f}, {C*b['ci'][1]:+.3f}]  "
              f"{b['n_events']:>4} ev {b['n_matches']:>3} mt  "
              f"{'SIG' if b['sig'] else '.'}")

    print("\n  effect by availability stratum (grid-label convention):")
    sub(np.ones(len(rows), dtype=bool), "all isolated goals")
    sub(dA <= 1e-9, "availability did NOT increase")
    sub(np.abs(dA) <= 1e-9, "availability exactly unchanged")
    sub(dA < -1e-9, "availability FELL")
    sub(dA > 1e-9, "availability ROSE")

    # correlation between the availability change and the measured effect
    dS = np.array([r["dS"] for r in rows])
    if dA.std() > 0:
        c = float(np.corrcoef(dA, dS)[0, 1])
        print(f"\n  corr(delta availability, delta S) = {c:+.3f}")
        print("  (a strong positive value would indicate a mechanical channel)")
    else:
        print("\n  availability is constant across all windows: "
              "no mechanical channel is possible")
