"""Break the fail-closed counter into its three causes.

`compiler.build_contracts` increments one `unknown` counter for every dropped
slug, but `compile_rule` returns None for three different reasons:

  1. the slug belongs to a different match          (not a parser failure)
  2. the slug is a half-time market, excluded by design as "D3"
  3. the slug matches no supported template          (the real unknown)

The manuscript's fail-closed discussion needs those separated. This reruns the
frozen compiler's own predicates over the full manifest without modifying it,
and additionally reports the distinct shapes behind reason 3, which the reviewer
noted are never enumerated anywhere.
"""
import json
import re
import sys
from collections import Counter, defaultdict

ROOT = "/home/azureuser/projects/wc_coherence"
sys.path.insert(0, f"{ROOT}/code")
import compiler as CP  # noqa: E402

M, _ = CP.load_manifest()
split = json.load(open(f"{ROOT}/registrations/dev_holdout_split.json"))
cohort_of = {m: "dev" for m in split["dev"]}
cohort_of.update({m: "holdout" for m in split["holdout"]})

reasons = defaultdict(Counter)
shapes = Counter()
shape_examples = defaultdict(list)
per_match = defaultdict(lambda: Counter())

for match in sorted(cohort_of):
    contracts, info = CP.build_contracts(M, match)
    if contracts is None:
        continue
    coh = cohort_of[match]
    pref = f"fifwc-{match}-"
    home = info["home_code"]
    away = match.split("-")[1] if home == match.split("-")[0] \
        else match.split("-")[0]
    for mkt in CP.match_markets(M, match):
        slug = (mkt.get("slug") or "").lower()
        if CP.compile_rule(slug, match, home, away) is not None:
            reasons[coh]["compiled"] += 1
            per_match[match]["compiled"] += 1
            continue
        if not slug.startswith(pref):
            reasons[coh]["other match"] += 1
            per_match[match]["other match"] += 1
        elif CP.HALF.search(slug[len(pref):]):
            reasons[coh]["half-time (design exclusion D3)"] += 1
            per_match[match]["half-time"] += 1
        else:
            reasons[coh]["unsupported template"] += 1
            per_match[match]["unsupported"] += 1
            body = slug[len(pref):]
            # collapse digits so families of thresholds group into one shape
            shape = re.sub(r"\d+", "N", body)
            shapes[shape] += 1
            if len(shape_examples[shape]) < 3:
                shape_examples[shape].append(body)

print("Dropped-slug causes, by cohort\n")
order = ["compiled", "other match", "half-time (design exclusion D3)",
         "unsupported template"]
print(f"{'cause':<36}{'dev':>10}{'holdout':>10}{'total':>10}")
for k in order:
    d, h = reasons["dev"][k], reasons["holdout"][k]
    print(f"{k:<36}{d:>10,}{h:>10,}{d + h:>10,}")
tot_d = sum(reasons["dev"].values())
tot_h = sum(reasons["holdout"].values())
print(f"{'TOTAL':<36}{tot_d:>10,}{tot_h:>10,}{tot_d + tot_h:>10,}")

drop_d = tot_d - reasons["dev"]["compiled"]
drop_h = tot_h - reasons["holdout"]["compiled"]
print(f"\nOf all dropped slugs, the share that is a genuine unsupported "
      f"template:")
for coh, drop in (("dev", drop_d), ("holdout", drop_h)):
    uns = reasons[coh]["unsupported template"]
    pct = 100.0 * uns / drop if drop else 0.0
    print(f"  {coh:<10}{uns:>6} of {drop:>6} dropped  ({pct:.1f}%)")

print(f"\nDistinct unsupported template shapes: {len(shapes)}")
print(f"{'shape':<44}{'count':>7}  examples")
for shape, n in shapes.most_common():
    ex = ", ".join(shape_examples[shape][:2])
    print(f"{shape:<44}{n:>7}  {ex[:60]}")

worst = sorted(per_match.items(),
               key=lambda kv: -kv[1]["unsupported"])[:8]
print("\nMatches with the most unsupported templates:")
for m, c in worst:
    print(f"  {m:<26} unsupported={c['unsupported']:>3}  "
          f"compiled={c['compiled']:>3}  half-time={c['half-time']:>3}")

out = {
    "by_cohort": {c: dict(v) for c, v in reasons.items()},
    "unsupported_shapes": {k: {"count": v, "examples": shape_examples[k]}
                           for k, v in shapes.items()},
    "n_unsupported_shapes": len(shapes),
}
dest = f"{ROOT}/results/fail_closed_breakdown.json"
json.dump(out, open(dest, "w"), indent=2)
print(f"\nwrote {dest}")
