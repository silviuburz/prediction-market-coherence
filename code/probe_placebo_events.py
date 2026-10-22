#!/usr/bin/env python3
"""Probe: how are half-time and non-scoring key events encoded in the frozen ESPN feed?

Needed before building the placebo suite:
  (a) half-time  -- a SCHEDULED discontinuity carrying no new information about
      the final score. If coherence degrades at half-time, the event-study is
      picking up liquidity withdrawal / attention, not information processing.
  (b) non-scoring key events (cards, substitutions, VAR) -- a dose-response
      ladder: red cards should move beliefs, substitutions should barely.
"""
import glob
import gzip
import json
import lzma
from collections import Counter
from datetime import datetime

LIVE = "/home/azureuser/polymarket/analysis/data/world_cup/live"


def iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


print("=" * 100)
print("A. SCOREBOARD STATUS / PERIOD / SHORT-DETAIL VOCABULARY")
print("=" * 100)
stat, per, short_by_stat = Counter(), Counter(), {}
paths = sorted(glob.glob(f"{LIVE}/*/*/_espn.jsonl.gz"))
print(f"scoreboard files: {len(paths)}")
for p in paths[:60]:
    try:
        with gzip.open(p, "rt", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                s = str(d.get("status"))
                stat[s] += 1
                per[(s, d.get("period"))] += 1
                short_by_stat.setdefault(s, Counter())[str(d.get("short_detail"))[:40]] += 1
    except Exception:
        continue
print("\nstatus values:")
for k, v in stat.most_common():
    print(f"  {k:<28} {v:>9,}")
print("\n(status, period) pairs:")
for k, v in per.most_common(20):
    print(f"  {str(k):<36} {v:>9,}")
print("\nshort_detail by status (top 4 each):")
for s, c in short_by_stat.items():
    print(f"  [{s}]")
    for k, v in c.most_common(4):
        print(f"      {k:<44} {v:>8,}")

print("\n" + "=" * 100)
print("B. KEY-EVENT TYPE VOCABULARY (non-scoring)")
print("=" * 100)
types, scoring_types = Counter(), Counter()
sample = {}
sp = sorted(glob.glob(f"{LIVE}/*/*/_espn_summary.jsonl.xz") +
            glob.glob(f"{LIVE}/*/*/_espn_summary.jsonl.gz"))
print(f"summary files: {len(sp)}")
n_games = 0
for p in sp:
    op = lzma.open if p.endswith(".xz") else gzip.open
    last = None
    try:
        with op(p, "rt", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
    except Exception:
        continue
    if not last:
        continue
    try:
        d = json.loads(last)
    except Exception:
        continue
    ke = d.get("keyEvents") or []
    if not ke:
        continue
    n_games += 1
    for e in ke:
        t = e.get("type") or {}
        key = (str(t.get("id")), str(t.get("text"))[:30])
        if e.get("scoringPlay"):
            scoring_types[key] += 1
        else:
            types[key] += 1
            if key not in sample:
                sample[key] = {k: e.get(k) for k in
                               ("wallclock", "clock", "period", "text", "shortText")}
print(f"games with keyEvents: {n_games}")
print("\nNON-scoring key-event types:")
for k, v in types.most_common(25):
    print(f"  id={k[0]:<6} {k[1]:<32} {v:>6}")
print("\nscoring key-event types:")
for k, v in scoring_types.most_common(10):
    print(f"  id={k[0]:<6} {k[1]:<32} {v:>6}")
print("\nsample payloads:")
for k in list(sample)[:8]:
    print(f"  id={k[0]} {k[1]}")
    print(f"      {json.dumps(sample[k], default=str)[:250]}")

print("\n" + "=" * 100)
print("C. CAN WE LOCATE THE HALF-TIME WHISTLE PER MATCH?")
print("=" * 100)
shown = 0
for p in paths:
    rows = []
    try:
        with gzip.open(p, "rt", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = iso(d.get("captured_at"))
                if t:
                    rows.append((t, str(d.get("status")), d.get("period"),
                                 str(d.get("short_detail"))[:22], str(d.get("game_id"))))
    except Exception:
        continue
    if not rows:
        continue
    for gid in sorted({r[4] for r in rows}):
        rr = sorted([r for r in rows if r[4] == gid])
        p1 = [r for r in rr if r[2] == 1 and r[1] in ("in", "STATUS_IN_PROGRESS")]
        p2 = [r for r in rr if r[2] == 2 and r[1] in ("in", "STATUS_IN_PROGRESS")]
        ht = [r for r in rr if "half" in r[1].lower() or "HALFTIME" in r[1].upper()
              or "HT" == r[3].strip()]
        if not (p1 and p2):
            continue
        gap = p2[0][0] - p1[-1][0]
        print(f"  game {gid}: last-P1 {datetime.utcfromtimestamp(p1[-1][0]):%H:%M:%S} "
              f"first-P2 {datetime.utcfromtimestamp(p2[0][0]):%H:%M:%S} "
              f"gap={gap/60:.1f}min  n_HT_rows={len(ht)}"
              + (f"  HT status='{ht[0][1]}' detail='{ht[0][3]}'" if ht else ""))
        shown += 1
        if shown >= 12:
            break
    if shown >= 12:
        break
