"""Cross-contract capture skew at the instants the study actually evaluates.

The objection is that ~46 books are polled independently, so a quote vector
assigned to one instant may be assembled from captures spread over time. The
quantity that matters is therefore not how long a polling sweep takes, but how
far apart the contributing observations are at the instants the estimator uses.

For sampled evaluation instants this reports, per instant:
  age   = t minus the most recent capture at or before t, per contract
  skew  = max age minus min age across the contracts entering that instant

and the mean age by family, which shows whether the collector visits some
families systematically earlier than others.

Read-only. Writes results/capture_skew.json.
"""
import json
import sys
from collections import defaultdict

import numpy as np

ROOT = "/home/azureuser/projects/wc_coherence"
sys.path.insert(0, f"{ROOT}/code")
import compiler as cp                                   # noqa: E402
import event_study as es                                # noqa: E402

SAMPLE = 12          # matches per cohort; capture timing is a collector property
STRIDE = 20          # evaluate every Nth grid point, to keep the scan bounded
P = es.P

split = json.load(open(f"{ROOT}/registrations/dev_holdout_split.json"))
M, _ = cp.load_manifest()


def run(cohort, matches):
    ages, skews, fam_age = [], [], defaultdict(list)
    n_instants = 0
    for mi, match in enumerate(matches, 1):
        contracts, info = cp.build_contracts(M, match)
        if contracts is None:
            continue
        series = {}
        for c in contracts:
            books = es.load_slug_books(c["slug"])
            for key in ("canon_token", "comp_token"):
                tok = c[key]
                if not tok:
                    continue
                got = books.get(tok)
                if got:
                    series[(c["slug"], key)] = (got[1], c["family"])
        if not series:
            continue
        all_times = sorted({t for times, _f in series.values() for t in times})
        if len(all_times) < 50:
            continue
        for t in all_times[::STRIDE]:
            per_contract = []
            for times, fam in series.values():
                i = np.searchsorted(times, t, side="right") - 1
                if i < 0:
                    continue
                age = t - times[i]
                if age > P["FRESH"]:
                    continue          # the estimator would drop it anyway
                per_contract.append((age, fam))
            if len(per_contract) < P["MIN_AVAIL"]:
                continue
            n_instants += 1
            a = [x[0] for x in per_contract]
            ages.extend(a)
            skews.append(max(a) - min(a))
            for age, fam in per_contract:
                fam_age[fam].append(age)
        print(f"  {cohort} {mi}/{len(matches)} {match}: "
              f"{n_instants:,} instants", flush=True)
    return ages, skews, fam_age, n_instants


out = {"matches_per_cohort": SAMPLE, "grid_stride": STRIDE,
       "freshness_cap_s": P["FRESH"], "min_avail": P["MIN_AVAIL"],
       "definition": ("age = t minus the most recent capture at or before t, "
                      "per contract; skew = max age minus min age across the "
                      "contracts entering that instant"),
       "cohorts": {}}

for cohort in ("dev", "holdout"):
    print(f"=== {cohort} ===", flush=True)
    ages, skews, fam_age, n = run(cohort, sorted(split[cohort])[:SAMPLE])
    if not skews:
        continue
    A, S = np.array(ages), np.array(skews)
    out["cohorts"][cohort] = {
        "n_instants": n,
        "age_s": {"median": float(np.median(A)),
                  "p90": float(np.percentile(A, 90)),
                  "p99": float(np.percentile(A, 99)),
                  "max": float(A.max())},
        "skew_s": {"median": float(np.median(S)),
                   "p90": float(np.percentile(S, 90)),
                   "p99": float(np.percentile(S, 99)),
                   "max": float(S.max())},
        "family_mean_age_s": {f: {"mean": float(np.mean(v)), "n": len(v)}
                              for f, v in sorted(fam_age.items())},
    }
    c = out["cohorts"][cohort]
    print(f"  instants {n:,}")
    print(f"  age    median {c['age_s']['median']:.2f}s  "
          f"p90 {c['age_s']['p90']:.2f}s  p99 {c['age_s']['p99']:.2f}s")
    print(f"  skew   median {c['skew_s']['median']:.2f}s  "
          f"p90 {c['skew_s']['p90']:.2f}s  p99 {c['skew_s']['p99']:.2f}s")
    for f, v in c["family_mean_age_s"].items():
        print(f"    {f:<14}mean age {v['mean']:.2f}s  n={v['n']:,}")

dest = f"{ROOT}/results/capture_skew.json"
json.dump(out, open(dest, "w"), indent=2)
print(f"\nwrote {dest}")
