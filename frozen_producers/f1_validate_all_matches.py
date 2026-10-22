#!/usr/bin/env python3
"""GATE F1 — semantic validation of the payoff compiler across ALL 104 matches.

G1 previously validated 3 matches (138 contract settlements). A referee will ask
whether the compiler is correct tournament-wide. This extends validation to every
match and reports:

  * fixture resolution rate
  * primary contracts compiled per family
  * settlement validation against the REGULATION-time score (period<=2)
  * unknown/failed-closed template counts, with examples
  * payoff-equivalence class counts and saturation checks per match
  * the saturation bound DERIVED from observed thresholds (not assumed)

HOLDOUT-SAFE: uses only manifest resolution prices and ESPN scores. No order-book
quotes are read, so this does not consume the holdout.

Run: /home/azureuser/.venv/bin/python -u f1_validate_all_matches.py
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compiler import (CAP, build_contracts, load_manifest, match_markets,   # noqa: E402
                      saturation_ok, state_classes)

OUT = "/home/azureuser/projects/wc_coherence"

print("=" * 96)
print("GATE F1 — SEMANTIC VALIDATION ACROSS ALL MATCHES")
print("=" * 96)
M, mp = load_manifest()

all_matches = sorted({re.match(r"fifwc-([a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2})",
                               (mk.get("slug") or "").lower()).group(1)
                      for lst in M.get("markets_by_event", {}).values() for mk in lst
                      if re.match(r"fifwc-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}",
                                  (mk.get("slug") or "").lower())})
print(f"matches in manifest: {len(all_matches)}")

# ── threshold inventory -> derived saturation bound --------------------------
th = defaultdict(set)
for lst in M.get("markets_by_event", {}).values():
    for mk in lst:
        s = (mk.get("slug") or "").lower()
        if re.search(r"(corners|shots|assists|saves|first-half|second-half|halftime)", s):
            continue
        m = re.search(r"-total-(\d+)pt5$", s)
        if m:
            th["total"].add(int(m.group(1)))
        m = re.search(r"-team-total-(home|away)-(\d+)pt5$", s)
        if m:
            th["team-total"].add(int(m.group(2)))
        m = re.search(r"-spread-(home|away)-(\d+)pt5$", s)
        if m:
            th["spread"].add(int(m.group(2)))
        m = re.search(r"-exact-score-(\d+)-(\d+)$", s)
        if m:
            th["exact"].add(max(int(m.group(1)), int(m.group(2))))
maxthr = max(max(v) for v in th.values() if v)
print(f"observed primary thresholds: " +
      ", ".join(f"{k}<= {max(v)}" for k, v in sorted(th.items())))
print(f"max primary threshold = {maxthr}  ->  saturation bound must exceed it; "
      f"CAP in use = {CAP}  ({'OK' if CAP > maxthr + 1 else 'TOO SMALL'})")

rows, fam_tot, unknown_examples = [], Counter(), Counter()
tot_checked = tot_fail = 0
t0 = time.time()

for i, match in enumerate(all_matches, 1):
    contracts, info = build_contracts(M, match)
    if contracts is None:
        rows.append({"match": match, "status": "UNRESOLVED"})
        continue
    H, A = info["H"], info["A"]
    fams = Counter(c["family"] for c in contracts)
    fam_tot.update(fams)

    checked = fails = 0
    fail_detail = []
    for c in contracts:
        opx = c["outcome_prices"]
        if not opx or str(opx[0]) not in ("0", "1"):
            continue
        realized = 1.0 if str(opx[0]) == "1" else 0.0
        pred = c["rule"](H, A)
        checked += 1
        if abs(pred - realized) > 1e-9:
            fails += 1
            fail_detail.append((c["slug"], c["family"], pred, realized))
    tot_checked += checked
    tot_fail += fails

    reps = state_classes(contracts)
    sat = saturation_ok(contracts)

    # record unknown templates that failed closed
    for mk in match_markets(M, match):
        s = (mk.get("slug") or "").lower()
        body = s[len(f"fifwc-{match}-"):] if s.startswith(f"fifwc-{match}-") else ""
        if not body:
            continue
        if not any(body == c["slug"][len(f"fifwc-{match}-"):] for c in contracts):
            key = re.sub(r"\d+", "<n>", body)
            unknown_examples[key] += 1

    rows.append({"match": match, "status": "OK" if fails == 0 else "SETTLEMENT_FAIL",
                 "reg": f"{H}-{A}", "n_contracts": len(contracts),
                 "checked": checked, "fails": fails, "states": len(reps),
                 "saturation": sat, "families": dict(fams),
                 "failed_closed": info["n_failed_closed"],
                 "fail_detail": fail_detail[:5]})
    if i % 20 == 0:
        print(f"  ... {i}/{len(all_matches)} ({time.time()-t0:.0f}s)", flush=True)

ok = [r for r in rows if r["status"] == "OK"]
bad = [r for r in rows if r["status"] not in ("OK",)]
print("\n" + "=" * 96)
print("RESULT")
print("=" * 96)
print(f"  matches resolved+validated : {len(ok)}/{len(rows)}")
print(f"  settlement checks          : {tot_checked:,}")
print(f"  settlement FAILURES        : {tot_fail}")
print(f"  saturation holds everywhere: {all(r.get('saturation') for r in ok)}")
ncon = [r["n_contracts"] for r in ok]
nst = [r["states"] for r in ok]
print(f"  primary contracts/match    : min {min(ncon)} median "
      f"{sorted(ncon)[len(ncon)//2]} max {max(ncon)}")
print(f"  payoff-equivalence classes : min {min(nst)} median "
      f"{sorted(nst)[len(nst)//2]} max {max(nst)}")
print(f"\n  contracts by family (all matches): {dict(fam_tot)}")

if bad:
    print(f"\n  PROBLEM MATCHES ({len(bad)}):")
    for r in bad[:12]:
        print(f"    {r['match']:<24} {r['status']}  " +
              (f"fails={r.get('fails')} e.g. {r.get('fail_detail')[:2]}"
               if r["status"] == "SETTLEMENT_FAIL" else ""))

print(f"\n  most common failed-closed (non-primary) templates:")
for k, v in unknown_examples.most_common(12):
    print(f"    {v:>6}  {k}")

os.makedirs(f"{OUT}/results", exist_ok=True)
json.dump({"n_matches": len(rows), "n_ok": len(ok), "settlement_checks": tot_checked,
           "settlement_failures": tot_fail, "max_threshold": maxthr, "CAP": CAP,
           "families": dict(fam_tot),
           "rows": [{k: v for k, v in r.items() if k != "fail_detail"} for r in rows]},
          open(f"{OUT}/results/f1_validate_all.json", "w"), indent=2)
print(f"\nresults -> {OUT}/results/f1_validate_all.json")
print("=" * 96)
print(f"GATE F1: {'PASS' if (tot_fail == 0 and len(ok) == len(rows)) else 'REVIEW'}")
print("=" * 96)
