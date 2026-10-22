"""Cycle period of the live collector, recovered from the archive.

captured_at is stamped once per polling cycle, so consecutive distinct stamps
within a live window give the cycle PERIOD. The loop sleeps cadence minus
elapsed, so period >= duration: a period sitting at the cadence is a cycle that
finished inside it, and a period above the cadence is a cycle that ran long and
equals its own duration.

That converts the cadence argument in the supplement into a measurement, and it
needs no instrumentation the collector did not already have.
"""
import glob
import gzip
import json
import os
import statistics as st
from collections import defaultdict
from datetime import datetime

ROOT = "/home/azureuser/polymarket/analysis/data/world_cup"
LIVE = f"{ROOT}/live"
OUT = "/home/azureuser/projects/wc_coherence/results/cycle_period.json"
CADENCE = 10.0
SLUGS_PER_WINDOW = 8


def iso(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def stamps_of(slug_dir):
    files = sorted(glob.glob(f"{slug_dir}/*.jsonl.gz"))
    if not files:
        return set()
    out = set()
    with gzip.open(files[0], "rt", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("captured_at"):
                out.add(r["captured_at"])
    return out


by_window = defaultdict(list)
for d in sorted(glob.glob(f"{LIVE}/*/*/fifwc-*")):
    by_window[os.path.dirname(d)].append(d)

gaps, shared_fracs, n_windows = [], [], 0
for win, slugs in sorted(by_window.items()):
    sets = [s for s in (stamps_of(d) for d in slugs[:SLUGS_PER_WINDOW]) if s]
    if len(sets) < 2:
        continue
    n_windows += 1
    union = set().union(*sets)
    shared_fracs.append(len(set.intersection(*sets)) / len(union))
    ts = sorted(iso(s) for s in union)
    gaps.extend(b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 3600)

gaps.sort()


def q(p):
    return gaps[min(len(gaps) - 1, int(p * len(gaps)))]


over = sum(1 for g in gaps if g > CADENCE + 0.5)
res = {
    "cadence_s": CADENCE,
    "n_windows": n_windows,
    "slugs_sampled_per_window": SLUGS_PER_WINDOW,
    "n_consecutive_pairs": len(gaps),
    "period_s": {"median": st.median(gaps), "p90": q(0.90), "p99": q(0.99),
                 "max": gaps[-1]},
    "n_over_cadence": over,
    "share_over_cadence_pct": 100.0 * over / len(gaps),
    "share_within_cadence_pct": 100.0 * (len(gaps) - over) / len(gaps),
    "stamp_sharing": {"mean": st.mean(shared_fracs),
                      "median": st.median(shared_fracs),
                      "min": min(shared_fracs)},
}

print(f"{n_windows:,} live windows, {len(gaps):,} consecutive stamp pairs")
p = res["period_s"]
print(f"cycle period  median {p['median']:.1f}s  p90 {p['p90']:.1f}s  "
      f"p99 {p['p99']:.1f}s  max {p['max']:.0f}s")
print(f"within the {CADENCE:.0f}s cadence: {res['share_within_cadence_pct']:.2f}%"
      f"   over it: {over:,} ({res['share_over_cadence_pct']:.2f}%)")
s = res["stamp_sharing"]
print(f"contracts sharing a stamp  mean {100 * s['mean']:.1f}%  "
      f"median {100 * s['median']:.1f}%  min {100 * s['min']:.1f}%")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
