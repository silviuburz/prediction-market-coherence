#!/usr/bin/env python3
"""Measure the actual order-book poll cadence from the frozen corpus.

The draft states a 5-second evaluation grid but never states how often the books
were actually polled. That number has to come from the data, not from memory.
Samples book files at random and reports the distribution of gaps between
consecutive captures.
"""
import glob
import gzip
import json
import random
from datetime import datetime

BASE = "/home/azureuser/polymarket/analysis/data/world_cup/live"
N_FILES = 120


def iso(s):
    """Match the frozen engine's timestamp parse."""
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None

random.seed(7)
files = glob.glob(BASE + "/*/*/*/*.jsonl.gz")
print(f"book files in corpus : {len(files)}")
sample = random.sample(files, min(N_FILES, len(files)))

gaps = []
per_file_median = []
for p in sample:
    ts = []
    try:
        with gzip.open(p, "rt", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                v = iso(r.get("captured_at"))
                if v is not None:
                    ts.append(v)
    except Exception:
        continue
    ts.sort()
    g = [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 600]
    if g:
        gaps += g
        g.sort()
        per_file_median.append(g[len(g) // 2])

gaps.sort()
n = len(gaps)
print(f"files parsed         : {len(per_file_median)}")
print(f"consecutive gaps     : {n}")
if n:
    def q(p):
        return gaps[int(p * (n - 1))]
    print(f"  p05    {q(.05):6.2f} s")
    print(f"  p25    {q(.25):6.2f} s")
    print(f"  median {q(.50):6.2f} s")
    print(f"  p75    {q(.75):6.2f} s")
    print(f"  p95    {q(.95):6.2f} s")
    print(f"  mean   {sum(gaps)/n:6.2f} s")
    within = sum(1 for g in gaps if 8.0 <= g <= 12.0) / n
    print(f"  share in [8,12] s : {within:.3f}")
if per_file_median:
    per_file_median.sort()
    m = per_file_median[len(per_file_median) // 2]
    print(f"median of per-file medians : {m:.2f} s")
