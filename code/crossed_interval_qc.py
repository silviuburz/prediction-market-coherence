#!/usr/bin/env python3
"""QC: how often do canonical intervals cross, and does it change around goals?

Section 4.3 of the paper discloses that when a contract's YES range and its
reflected NO range fail to intersect, the contract is dropped to the vacuous bound
[0,1] rather than reconciled. That is conservative for the cross-contract
statistic, but the paper could not say how often it happens. This measures it.

Note what the frozen engine does with a crossed book: it appends [0,1] AND does
not increment `avail`. So a crossed contract is already counted as unavailable,
which means the reported availability stability across goals is itself an upper
bound on any post-goal spike in crossing. This script confirms that directly.

Only the estimation windows of isolated goals are evaluated -- the same instants
the estimator uses -- so this is a QC of the measured quantity, not of the whole
grid. Book loading dominates the runtime, so expect roughly the cost of one
engine pass.

Run (from /home/azureuser/projects/wc_coherence):
  nohup /home/azureuser/.venv/bin/python -u code/crossed_interval_qc.py \
      --cohort holdout > logs/crossed_qc_holdout.log 2>&1 &
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_study import (OUT, P, build_contracts, canonical_interval,  # noqa: E402
                         goals_of, iso, load_manifest, load_slug_books,
                         token_quote_env)


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--cohort", choices=["dev", "holdout"], required=True)
a = ap.parse_args()
split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
matches = split[a.cohort]
log(f"cohort {a.cohort}: {len(matches)} matches")

G, WIN = P["GRID"], P["WIN"]
per_goal, per_contract_cross = [], {}
n_match = 0

for mi, match in enumerate(matches, 1):
    t0 = time.time()
    M, _ = load_manifest()
    contracts, info = build_contracts(M, match)
    if contracts is None:
        log(f"  [{mi}/{len(matches)}] {match}: no contracts, skipped")
        continue
    rows = info["espn_rows"]
    tin = [iso(d.get("captured_at")) for d in rows
           if str(d.get("status")) in ("in", "STATUS_IN_PROGRESS")
           and (d.get("period") or 0) and int(d["period"]) <= 2]
    tin = [t for t in tin if t]
    if len(tin) < 20:
        continue
    t_start, t_end = min(tin), max(tin)
    goals = [g for g in goals_of(info["game_id"])
             if t_start - 60 <= g["w"] <= t_end + 60]
    gw = sorted(g["w"] for g in goals)

    ser = {}
    for c in contracts:
        books = load_slug_books(c["slug"])
        for key in ("canon_token", "comp_token"):
            tok = c[key]
            if not tok:
                continue
            got = books.get(tok)
            if got is None:
                cand = [k for k in books if k.startswith(tok[:16])]
                got = books[cand[0]] if len(cand) == 1 else None
            if got:
                ser[(c["slug"], key)] = got

    def window_cross(lo_off, hi_off, centre):
        """(crossed, quoted_both_sides) summed over the window."""
        cross = both = 0
        for t in np.arange(centre + lo_off, centre + hi_off + 1e-9, G):
            for c in contracts:
                cs = ser.get((c["slug"], "canon_token"))
                ps = ser.get((c["slug"], "comp_token"))
                can = token_quote_env(cs[0], cs[1], t) if cs else None
                comp = token_quote_env(ps[0], ps[1], t) if ps else None
                if can is None or comp is None:
                    continue
                both += 1
                if canonical_interval(can, comp) is None:
                    cross += 1
                    per_contract_cross[c["slug"]] = \
                        per_contract_cross.get(c["slug"], 0) + 1
        return cross, both

    n_g = 0
    for g in goals:
        w = g["w"]
        if w - WIN < t_start or w + WIN > t_end:
            continue
        if not all(abs(w - o) > P["ISO_GAP"] for o in gw if abs(o - w) > 1e-9):
            continue
        pc, pb = window_cross(P["PRE_LO"], P["PRE_HI"], w)
        qc, qb = window_cross(P["POST_LO"], P["POST_HI"], w)
        per_goal.append({"match": match, "w": w,
                         "pre_cross": pc, "pre_both": pb,
                         "post_cross": qc, "post_both": qb})
        n_g += 1
    n_match += 1
    log(f"  [{mi}/{len(matches)}] {match}: {n_g} isolated goals "
        f"[{time.time()-t0:.0f}s]")

pre_c = sum(r["pre_cross"] for r in per_goal)
pre_b = sum(r["pre_both"] for r in per_goal)
post_c = sum(r["post_cross"] for r in per_goal)
post_b = sum(r["post_both"] for r in per_goal)
rate = lambda c, b: 100.0 * c / b if b else float("nan")   # noqa: E731

per_goal_delta = [
    (100.0 * r["post_cross"] / r["post_both"] if r["post_both"] else 0.0)
    - (100.0 * r["pre_cross"] / r["pre_both"] if r["pre_both"] else 0.0)
    for r in per_goal]

summary = {
    "cohort": a.cohort, "n_matches": n_match, "n_goals": len(per_goal),
    "pre_crossed": pre_c, "pre_evaluations": pre_b, "pre_rate_pct": rate(pre_c, pre_b),
    "post_crossed": post_c, "post_evaluations": post_b,
    "post_rate_pct": rate(post_c, post_b),
    "delta_pp": rate(post_c, post_b) - rate(pre_c, pre_b),
    "goals_with_any_cross": sum(1 for r in per_goal
                                if r["pre_cross"] or r["post_cross"]),
    "mean_per_goal_delta_pp": float(np.mean(per_goal_delta)) if per_goal else None,
    "contracts_ever_crossed": len(per_contract_cross),
    "top_contracts": sorted(per_contract_cross.items(),
                            key=lambda kv: -kv[1])[:15],
}

print("\n" + "=" * 78)
print(f"CROSSED CANONICAL INTERVALS — {a.cohort.upper()}")
print("=" * 78)
for k, v in summary.items():
    if k != "top_contracts":
        print(f"  {k:<26}{v}")
print("  top contracts by crossings:")
for slug, n in summary["top_contracts"]:
    print(f"    {n:>6}  {slug}")

d = f"{OUT}/results/{a.cohort}"
os.makedirs(d, exist_ok=True)
with open(f"{d}/crossed_interval_qc.json", "w") as f:
    json.dump({"summary": summary, "per_goal": per_goal}, f, indent=2)
log(f"wrote {d}/crossed_interval_qc.json")
