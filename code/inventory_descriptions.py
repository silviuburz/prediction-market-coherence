#!/usr/bin/env python3
"""Inventory the ACTUAL contingent-settlement clauses for every primary contract.

Descriptions live in the FROZEN corpus at
    _baseline/markets/<condition_id_without_0x>.json.gz
so this is fully reproducible and needs no live API.

For each primary family it extracts the cancellation/postponement clause and
classifies the contingent payout, so the cancellation state added to the payoff
matrix is grounded in real rules rather than assumption.

Also reports availability of `acceptingOrders`, `feeSchedule`/`feeType` (both
relevant to limitations previously believed unrecoverable).

Run: /home/azureuser/.venv/bin/python -u inventory_descriptions.py
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compiler import build_contracts, load_manifest        # noqa: E402

ROOT = "/home/azureuser/polymarket/analysis/data/world_cup"
MKT = f"{ROOT}/_baseline/markets"
OUT = "/home/azureuser/projects/wc_coherence"


def norm(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def load_market(cid):
    c = (cid or "").replace("0x", "").lower()
    p = f"{MKT}/{c}.json.gz"
    if not os.path.exists(p):
        return None
    try:
        o = json.load(gzip.open(p, "rt"))
        return o[0] if isinstance(o, list) and o else o
    except Exception:
        return None


CANCEL = re.compile(r"(if the game is canceled[^.]*\.)", re.I)
POSTPONE = re.compile(r"(if the game is postponed[^.]*\.)", re.I)


def classify(desc):
    """-> ('50-50' | 'No' | 'Yes' | '0-0' | 'other' | 'none', clause_text)"""
    m = CANCEL.search(desc or "")
    if not m:
        return "none", ""
    cl = norm(m.group(1))
    low = cl.lower()
    if "50-50" in low or "50/50" in low:
        return "50-50", cl
    if re.search(r'resolve to "?no"?', low):
        return "No", cl
    if re.search(r'resolve to "?yes"?', low):
        return "Yes", cl
    if "0-0" in low:
        return "0-0", cl
    return "other", cl


print("=" * 100)
print("CONTINGENT-SETTLEMENT CLAUSE INVENTORY (from frozen corpus)")
print("=" * 100)
print(f"market description files: {len(glob.glob(f'{MKT}/*.json.gz')):,}")

M, _ = load_manifest()
split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
matches = split["dev"] + split["holdout"]

fam_clause = defaultdict(Counter)
fam_example = {}
missing = Counter()
flags = Counter()
fee_types = Counter()
acc_seen = Counter()

for match in matches:
    contracts, info = build_contracts(M, match)
    if contracts is None:
        continue
    for c in contracts:
        mk = load_market(c.get("condition_id") or "")
        if mk is None:
            # fall back: find condition_id from manifest entry
            missing[c["family"]] += 1
            continue
        desc = norm(mk.get("description"))
        kind, clause = classify(desc)
        fam_clause[c["family"]][kind] += 1
        if (c["family"], kind) not in fam_example and clause:
            fam_example[(c["family"], kind)] = clause
        if mk.get("acceptingOrders") is not None:
            acc_seen[str(mk.get("acceptingOrders"))] += 1
        ft = mk.get("feeType")
        if ft is not None:
            fee_types[str(ft)] += 1
        if mk.get("feeSchedule") not in (None, "", {}):
            flags["has_feeSchedule"] += 1

print("\n" + "=" * 100)
print("CANCELLATION CLAUSE BY FAMILY")
print("=" * 100)
for fam in sorted(fam_clause):
    tot = sum(fam_clause[fam].values())
    print(f"\n  {fam}  (n={tot})")
    for kind, n in fam_clause[fam].most_common():
        print(f"     {kind:<8} {n:>6}  ({100*n/tot:.1f}%)")
        ex = fam_example.get((fam, kind))
        if ex:
            print(f"        e.g. {ex[:190]}")
if missing:
    print(f"\n  markets with no description file: {dict(missing)}")

print("\n" + "=" * 100)
print("OTHER RECOVERABLE FIELDS (previously believed unavailable)")
print("=" * 100)
print(f"  acceptingOrders values seen : {dict(acc_seen)}")
print(f"  feeType values seen         : {dict(fee_types)}")
print(f"  markets with feeSchedule    : {flags['has_feeSchedule']}")

json.dump({"family_clauses": {k: dict(v) for k, v in fam_clause.items()},
           "examples": {f"{k[0]}|{k[1]}": v for k, v in fam_example.items()},
           "missing": dict(missing), "acceptingOrders": dict(acc_seen),
           "feeType": dict(fee_types)},
          open(f"{OUT}/results/clause_inventory.json", "w"), indent=2)
print(f"\nresults -> {OUT}/results/clause_inventory.json")
