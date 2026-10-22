"""Contract counts by family and cohort, and what the dropped shapes price.

The shape grouping below is a pattern ruleset, not a hand review of 3,252
shapes. Its value is that it is exhaustive and inspectable: every shape lands in
a named bucket or in `unclassified`, and the full inventory ships so a reader can
disagree with any bucket.
"""
import json
import re
import sys
from collections import Counter, defaultdict

ROOT = "/home/azureuser/projects/wc_coherence"
sys.path.insert(0, f"{ROOT}/code")
import compiler as cp  # noqa: E402

M, _ = cp.load_manifest()
split = json.load(open(f"{ROOT}/registrations/dev_holdout_split.json"))

per_cohort = {c: defaultdict(list) for c in ("dev", "holdout")}
for cohort in ("dev", "holdout"):
    for match in sorted(split[cohort]):
        contracts, info = cp.build_contracts(M, match)
        if contracts is None:
            continue
        counts = Counter(c["family"] for c in contracts)
        for fam in ("exact-score", "moneyline", "total-goals", "team-total",
                    "spread", "btts"):
            per_cohort[cohort][fam].append(counts.get(fam, 0))
        per_cohort[cohort]["ALL"].append(len(contracts))

out = {"by_family": {}}
print(f"{'family':<14}{'dev median':>12}{'dev range':>14}"
      f"{'holdout median':>16}{'holdout range':>16}")
for fam in ("exact-score", "moneyline", "total-goals", "team-total", "spread",
            "btts", "ALL"):
    row = {}
    line = f"{fam:<14}"
    for cohort in ("dev", "holdout"):
        v = sorted(per_cohort[cohort][fam])
        med = v[len(v) // 2]
        row[cohort] = {"median": med, "min": v[0], "max": v[-1]}
        line += f"{med:>12}{f'{v[0]}-{v[-1]}':>14}" if cohort == "dev" \
            else f"{med:>16}{f'{v[0]}-{v[-1]}':>16}"
    out["by_family"][fam] = row
    print(line)

# grouping of the unsupported shapes, by what they price
fcb = json.load(open(f"{ROOT}/results/fail_closed_breakdown.json"))
RULES = [
    ("corners", r"corner"),
    ("cards and discipline", r"card|booking|foul|offside"),
    ("shots and saves", r"shot|save"),
    ("player goal or assist props", r"^goals-|goals-plus|assist"),
    ("other player props", r"pass|tackle|cross|dribble|touch|minutes"),
    ("first or last scorer", r"first-to-score|last-to-score|scorer|first-goal"),
    ("tournament progression", r"advance|winner|qualify|group|to-lift|champion"),
    ("penalties and VAR", r"penalt|var\b"),
    ("timing and periods", r"minute|half|period|extra-time|stoppage"),
]
tax, unmatched = Counter(), {}
for shape, info in fcb["unsupported_shapes"].items():
    for label, pattern in RULES:
        if re.search(pattern, shape, re.I):
            tax[label] += info["count"]
            break
    else:
        tax["unclassified"] += info["count"]
        unmatched[shape] = info["count"]

total = sum(tax.values())
print(f"\ntaxonomy of {total:,} dropped slugs in "
      f"{fcb['n_unsupported_shapes']:,} shapes")
for label, n in tax.most_common():
    print(f"  {label:<32}{n:>8,}  {100.0 * n / total:5.1f}%")
if unmatched:
    print(f"\n  unclassified shapes ({len(unmatched)}):")
    for s, n in sorted(unmatched.items(), key=lambda kv: -kv[1])[:20]:
        print(f"    {s:<44}{n:>6}")

out["dropped_taxonomy"] = dict(tax)
out["unclassified_shapes"] = unmatched
json.dump(out, open(f"{ROOT}/results/system_composition.json", "w"), indent=2)
print(f"\nwrote {ROOT}/results/system_composition.json")
