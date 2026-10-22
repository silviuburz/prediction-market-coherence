#!/usr/bin/env python3
"""Saturation lemma — machine-checked proof that truncating the score state
space at CAP goals per team loses NO information.

CLAIM. Every compiled rule is a Boolean combination of predicates drawn from

    h == k,  a == k,        (exact-score)          k <= T
    h > a, h < a, h == a,   (moneyline)
    h + a > k + 0.5,        (total-goals)          k <= T
    h > 0 and a > 0,        (btts)
    h > k, a > k,           (team-total)           k <= T
    h - a > k, a - h > k,   (spread)               k <= T

Let E, H, D and U be the largest integer literals in exact-score, team-total,
spread and total-goals rules, respectively, and L = max(E, H). Rules cannot
distinguish exact/team score levels above L+1, score differences beyond D+1, or
goal totals beyond U+1. Every attainable combination of those clipped features
has a representative whose team scores are bounded by

    B = max(U + 1, L + D + 2).

Thus CAP >= B is a sufficient corpus-specific bound. For this corpus E=3, H=3,
D=5 and U=9, so B=10 and CAP=15 is conservative. The machine check below then
guards against parser mistakes by comparing CAP with a much larger box.

VERIFICATION. Asserting the lemma is not enough, because the rules are compiled
from slugs and a single mis-compiled rule would break it silently. This script
therefore checks it exhaustively and empirically for every match: it enumerates
the payoff vectors over the box [0, CAP_TEST]^2 with CAP_TEST far beyond CAP, and
requires that the distinct-vector set is identical to the one obtained at CAP.
Any match where a new payoff vector appears above CAP is reported as a FAILURE.

Run: /home/azureuser/.venv/bin/python -u saturation_lemma.py
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compiler import CAP, build_contracts, load_manifest, state_classes  # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
CAP_TEST = 40


def family_thresholds(contracts):
    """Return the largest integer literals by rule family."""
    out = {"exact": 0, "team": 0, "spread": 0, "total": 0}
    for contract in contracts:
        slug = contract["slug"]
        family = contract["family"]
        if family == "exact-score":
            match = re.search(r"exact-score-(\d+)-(\d+)$", slug)
            if match:
                out["exact"] = max(out["exact"], int(match.group(1)),
                                   int(match.group(2)))
        elif family == "team-total":
            match = re.search(r"-(\d+)pt5$", slug)
            if match:
                out["team"] = max(out["team"], int(match.group(1)))
        elif family == "spread":
            match = re.search(r"-(\d+)pt5$", slug)
            if match:
                out["spread"] = max(out["spread"], int(match.group(1)))
        elif family == "total-goals":
            match = re.search(r"-(\d+)pt5$", slug)
            if match:
                out["total"] = max(out["total"], int(match.group(1)))
    return out


M, _ = load_manifest()
split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
matches = sorted(set(split["dev"]) | set(split["holdout"]))

print("=" * 100)
print(f"SATURATION LEMMA — exhaustive check, CAP={CAP} vs CAP_TEST={CAP_TEST}")
print("=" * 100)

n_ok = n_fail = n_skip = 0
max_thresholds = {"exact": 0, "team": 0, "spread": 0, "total": 0}
max_required_cap = 0
class_counts = Counter()
failures = []

for match in matches:
    contracts, info = build_contracts(M, match)
    if contracts is None:
        n_skip += 1
        continue
    thresholds = family_thresholds(contracts)
    for family, value in thresholds.items():
        max_thresholds[family] = max(max_thresholds[family], value)
    level = max(thresholds["exact"], thresholds["team"])
    required_cap = max(thresholds["total"] + 1,
                       level + thresholds["spread"] + 2)
    max_required_cap = max(max_required_cap, required_cap)

    def vec(h, a):
        return tuple(c["rule"](h, a) for c in contracts)

    small = {}
    for h in range(CAP + 1):
        for a in range(CAP + 1):
            small.setdefault(vec(h, a), (h, a))
    big = {}
    for h in range(CAP_TEST + 1):
        for a in range(CAP_TEST + 1):
            big.setdefault(vec(h, a), (h, a))

    extra = set(big) - set(small)
    class_counts[len(small)] += 1
    reps = state_classes(contracts)
    if extra:
        n_fail += 1
        ex = sorted(big[v] for v in extra)[:4]
        failures.append({"match": match, "thresholds": thresholds,
                         "required_cap": required_cap,
                         "n_small": len(small), "n_big": len(big),
                         "n_extra": len(extra), "example_states": ex})
        print(f"  FAIL {match}: required_cap={required_cap} "
              f"classes {len(small)} -> {len(big)} "
              f"(+{len(extra)}), first new states {ex}")
    else:
        n_ok += 1
    if len(reps) != len(small):
        print(f"  NOTE {match}: state_classes returned {len(reps)} reps but "
              f"{len(small)} distinct payoff vectors exist")

print(f"\nmatches checked      : {n_ok + n_fail}  (skipped {n_skip})")
print(f"saturation holds     : {n_ok}")
print(f"saturation FAILS     : {n_fail}")
print(f"max thresholds       : {max_thresholds}")
print(f"required CAP         : {max_required_cap}; CAP = {CAP}  "
    f"{'OK' if CAP >= max_required_cap else '*** VIOLATED ***'}")
print(f"margin               : {CAP - max_required_cap} goals")
print("\ndistinct payoff-vector classes per match:")
for k, v in sorted(class_counts.items()):
    print(f"  {k:>3} classes : {v:>4} matches")

res = {"CAP": CAP, "CAP_TEST": CAP_TEST,
    "max_thresholds": max_thresholds,
    "required_CAP": max_required_cap,
    "cap_sufficient": bool(CAP >= max_required_cap),
       "n_ok": n_ok, "n_fail": n_fail, "n_skipped": n_skip,
       "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
       "failures": failures}
json.dump(res, open(f"{OUT}/results/saturation_lemma.json", "w"), indent=2)
print(f"\nresults -> {OUT}/results/saturation_lemma.json")
print("\nVERDICT: " + ("truncation at CAP is EXACT for every match checked"
                       if n_fail == 0 else "*** TRUNCATION LOSES STATES ***"))
