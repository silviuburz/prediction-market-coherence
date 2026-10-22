#!/usr/bin/env python3
"""Quoted-contract census from the persisted long-form engine output.

No LP work and no re-run: event_study.py already records `avail` -- the number
of contracts carrying at least one real bound -- at every 5s offset of every
event it emitted. This reads that directly, so the census is by construction the
same quantity the engine gated on (avail >= MIN_AVAIL) rather than a
reconstruction that could drift from it.

Reports the distribution over the pre-goal [-60,-6]s and post-goal [+6,+30]s
estimation windows of isolated goals, which is the population the headline
estimate is computed on.

Run: /home/azureuser/.venv/bin/python -u availability_census.py
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np

OUT = "/home/azureuser/projects/wc_coherence"
PRE, POST = (-60, -6), (6, 30)
MIN_AVAIL = 10
out = {}

for cohort in ("dev", "holdout"):
    path = f"{OUT}/results/{cohort}/long_form.jsonl"
    pre, post, allw = [], [], []
    n_rows = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            n_rows += 1
            if not d.get("isolated"):
                continue
            # only instants the engine actually analysed contribute to the
            # headline; an instant below the gate produced no S at all
            if d.get("S_cross") is None:
                continue
            off, av = d.get("offset"), d.get("avail")
            if av is None:
                continue
            allw.append(av)
            if PRE[0] <= off <= PRE[1]:
                pre.append(av)
            elif POST[0] <= off <= POST[1]:
                post.append(av)

    rec = {"n_long_rows": n_rows}
    for tag, arr in (("pre", pre), ("post", post), ("analysed_any_offset", allw)):
        a = np.array(arr, dtype=float)
        if not len(a):
            continue
        q = np.percentile(a, [1, 5, 25, 50, 75, 95])
        rec[tag] = {"n": int(len(a)), "min": float(a.min()), "p01": float(q[0]),
                    "p05": float(q[1]), "q1": float(q[2]), "median": float(q[3]),
                    "q3": float(q[4]), "p95": float(q[5]), "max": float(a.max()),
                    "mean": float(a.mean()),
                    "pct_at_or_below_15": float(100.0 * (a <= 15).mean()),
                    "pct_at_or_above_30": float(100.0 * (a >= 30).mean()),
                    "pct_at_or_above_40": float(100.0 * (a >= 40).mean())}
    out[cohort] = rec

    print("=" * 92)
    print(f"QUOTED-CONTRACT CENSUS — {cohort}  (gate: avail >= {MIN_AVAIL})")
    print("=" * 92)
    print(f"  {'window':<22}{'n':>10}{'min':>6}{'p05':>6}{'q1':>6}{'median':>8}"
          f"{'q3':>6}{'p95':>6}{'max':>6}{'>=30':>8}{'>=40':>8}")
    for tag in ("pre", "post", "analysed_any_offset"):
        if tag not in rec:
            continue
        r = rec[tag]
        print(f"  {tag:<22}{r['n']:>10,}{r['min']:>6.0f}{r['p05']:>6.0f}"
              f"{r['q1']:>6.0f}{r['median']:>8.0f}{r['q3']:>6.0f}{r['p95']:>6.0f}"
              f"{r['max']:>6.0f}{r['pct_at_or_above_30']:>7.1f}%"
              f"{r['pct_at_or_above_40']:>7.1f}%")
    print()

dst = f"{OUT}/results/availability_census.json"
out["generated_at"] = datetime.now(timezone.utc).isoformat()
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {dst}")
