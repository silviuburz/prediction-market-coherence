#!/usr/bin/env python3
"""GATE G3 — goal event study on DEVELOPMENT matches only.

Locks a deterministic dev/holdout split (D14), builds a goal event table from
ESPN `keyEvents.wallclock` (regulation only), and measures the primary coherence
metric S_inf on a 5-second grid around each shock, against placebo controls.

Design (frozen decisions):
  D5  metric      S_inf = min uniform quote widening, in cents
  D7  quotes      canonical orientation, complementary bounds combined
  D9  timing      ESPN wallclock anchor; conservative ENVELOPE quote bound
  D14 inference   match is the clustering unit; dev/holdout split; dev only here

Placebos: in-play instants at least PLACEBO_GAP_S away from any goal, sampled per
match, evaluated on the same grid shape.

Run: /home/azureuser/.venv/bin/python -u g3_event_study.py
"""
import glob
import gzip
import hashlib
import json
import lzma
import os
import random
import re
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402
from scipy.optimize import linprog                     # noqa: E402
from compiler import (LIVE, ROOT, build_contracts, load_manifest,         # noqa: E402
                      payoff_matrix, state_classes)

random.seed(23)
np.random.seed(23)

GRID_STEP = 5                 # seconds
PRE, POST = 300, 300          # window around the shock
FRESH = 30.0
TOL = 1e-4
PLACEBO_GAP_S = 240
PLACEBOS_PER_MATCH = 6
DEV_FRACTION_HEX = set("0123456")     # ~44% of matches -> development
FORCE_DEV = {"eng-arg-2026-07-15", "arg-cvi-2026-07-03", "eng-hrv-2026-06-17"}
MAX_DEV_MATCHES = 24          # pilot budget for this stage
OUTDIR = "/home/azureuser/projects/wc_coherence"


def iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def bts(b):
    v = b.get("timestamp")
    if v is None:
        return None
    try:
        f = float(v)
        return f / 1000.0 if f > 1e11 else f
    except Exception:
        return iso(v)


def best_bid_ask(book):
    bid = ask = None
    try:
        for lv in book.get("bids") or []:
            p, s = float(lv["price"]), float(lv["size"])
            if s > 0 and (bid is None or p > bid):
                bid = p
        for lv in book.get("asks") or []:
            p, s = float(lv["price"]), float(lv["size"])
            if s > 0 and (ask is None or p < ask):
                ask = p
    except Exception:
        return None, None
    return bid, ask


_cache = {}


def load_slug_books(slug):
    if slug in _cache:
        return _cache[slug]
    books = defaultdict(list)
    for p in glob.glob(f"{LIVE}/*/*/{slug}/*.jsonl.gz"):
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
                    b = d.get("book")
                    if t is None or not isinstance(b, dict):
                        continue
                    aid = str(b.get("asset_id") or d.get("token_id") or "")
                    if not aid:
                        continue
                    bid, ask = best_bid_ask(b)
                    books[aid].append((t, bid, ask, str(b.get("hash")), bts(b)))
        except Exception:
            continue
    out = {}
    for aid, rows in books.items():
        rows.sort(key=lambda r: r[0])
        out[aid] = (rows, [r[0] for r in rows])
    _cache[slug] = out
    return out


def token_quote_env(series, times, t):
    """Conservative envelope quote at t (absorbs movement between polls)."""
    i = bisect_right(times, t) - 1
    if i < 0:
        return None
    pt, pb, pa, _ph, _pts = series[i]
    if t - pt > FRESH:
        return None
    if i + 1 < len(series):
        nb, na = series[i + 1][1], series[i + 1][2]
        bid = min([v for v in (pb, nb) if v is not None], default=None)
        ask = max([v for v in (pa, na) if v is not None], default=None)
        return bid, ask
    return pb, pa


def canonical_interval(can, comp):
    los, his = [], []
    if can:
        b, a = can
        if b is not None:
            los.append(b)
        if a is not None:
            his.append(a)
    if comp:
        b, a = comp
        if a is not None:
            los.append(1.0 - a)
        if b is not None:
            his.append(1.0 - b)
    lo = max(los) if los else 0.0
    hi = min(his) if his else 1.0
    if lo > hi + 1e-9:
        return None
    return max(0.0, lo), min(1.0, hi), (1 if los else 0) + (1 if his else 0)


def solve_S(X, lows, highs):
    m, n = X.shape
    nv = n + 1
    c = np.zeros(nv)
    c[-1] = 1.0
    A_ub, b_ub = [], []
    for j in range(m):
        r1 = np.zeros(nv)
        r1[:n] = -X[j]
        r1[-1] = -1.0
        A_ub.append(r1)
        b_ub.append(-lows[j])
        r2 = np.zeros(nv)
        r2[:n] = X[j]
        r2[-1] = -1.0
        A_ub.append(r2)
        b_ub.append(highs[j])
    A_eq = np.zeros((1, nv))
    A_eq[0, :n] = 1.0
    try:
        r = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq,
                    b_eq=np.array([1.0]), bounds=[(0, 1)] * n + [(0, None)],
                    method="highs")
    except Exception:
        return None, 0
    if not r.success:
        return None, 0
    return float(r.x[-1]), 1


def goals_from_summary(game_id):
    """Regulation-time scoring plays with ESPN's own wallclock."""
    out = []
    for p in (glob.glob(f"{LIVE}/*/*/_espn_summary.jsonl.xz") +
              glob.glob(f"{LIVE}/*/*/_espn_summary.jsonl.gz")):
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
        if str(d.get("espn_game_id")) != str(game_id):
            continue
        for e in (d.get("keyEvents") or []):
            try:
                if not e.get("scoringPlay"):
                    continue
                per = int((e.get("period") or {}).get("number") or 0)
                if per < 1 or per > 2:            # regulation only
                    continue
                w = iso(e.get("wallclock"))
                if w:
                    out.append({"wallclock": w, "period": per,
                                "clock": (e.get("clock") or {}).get("displayValue"),
                                "text": str(e.get("text"))[:90],
                                "type": ((e.get("type") or {}).get("text"))})
            except Exception:
                continue
        if out:
            break
    out.sort(key=lambda g: g["wallclock"])
    dedup = []
    for g in out:
        if not dedup or g["wallclock"] - dedup[-1]["wallclock"] > 30:
            dedup.append(g)
    return dedup


# ── deterministic dev/holdout split ─────────────────────────────────────────
M, mp = load_manifest()
all_matches = sorted({re.match(r"fifwc-([a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2})",
                               (mk.get("slug") or "").lower()).group(1)
                      for lst in M.get("markets_by_event", {}).values() for mk in lst
                      if re.match(r"fifwc-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}",
                                  (mk.get("slug") or "").lower())})


def is_dev(m):
    if m in FORCE_DEV:
        return True
    return hashlib.sha256(m.encode()).hexdigest()[0] in DEV_FRACTION_HEX


dev = [m for m in all_matches if is_dev(m)]
hold = [m for m in all_matches if not is_dev(m)]
os.makedirs(f"{OUTDIR}/registrations", exist_ok=True)
json.dump({"rule": "sha256(match)[0] in 0123456, plus forced dev trio",
           "n_all": len(all_matches), "dev": dev, "holdout": hold},
          open(f"{OUTDIR}/registrations/dev_holdout_split.json", "w"), indent=2)

print("=" * 92)
print("GATE G3 — GOAL EVENT STUDY (development matches only)")
print("=" * 92)
print(f"matches total {len(all_matches)} | dev {len(dev)} | holdout {len(hold)} "
      f"(holdout NOT touched)")

use = dev[:MAX_DEV_MATCHES]
grid = list(range(-PRE, POST + 1, GRID_STEP))
ev_curves, pb_curves = defaultdict(list), defaultdict(list)
n_goals = n_pb = 0
per_match_goal = {}

for match in use:
    t0 = time.time()
    contracts, info = build_contracts(M, match)
    if contracts is None:
        continue
    reps = state_classes(contracts)
    X = np.array(payoff_matrix(contracts, reps), dtype=float)
    rows = info["espn_rows"]
    tin = [iso(d.get("captured_at")) for d in rows
           if str(d.get("status")) in ("in", "STATUS_IN_PROGRESS")
           and (d.get("period") or 0) and int(d["period"]) <= 2]
    tin = [t for t in tin if t]
    if len(tin) < 20:
        continue
    t_start, t_end = min(tin), max(tin)
    goals = [g for g in goals_from_summary(info["game_id"])
             if t_start - 60 <= g["wallclock"] <= t_end + 60]
    per_match_goal[match] = len(goals)

    ser = {}
    for c in contracts:
        books = load_slug_books(c["slug"])
        for key in ("canon_token", "comp_token"):
            tok = c[key]
            if not tok:
                continue
            got = books.get(tok)
            if got is None:
                cand = [a for a in books if a.startswith(tok[:16])]
                got = books[cand[0]] if len(cand) == 1 else None
            if got:
                ser[(c["slug"], key)] = got

    def S_at(t):
        lows, highs, avail = [], [], 0
        for c in contracts:
            cs = ser.get((c["slug"], "canon_token"))
            ps = ser.get((c["slug"], "comp_token"))
            can = token_quote_env(cs[0], cs[1], t) if cs else None
            comp = token_quote_env(ps[0], ps[1], t) if ps else None
            ci = canonical_interval(can, comp)
            if ci is None:
                lows.append(0.0)
                highs.append(1.0)
                continue
            lo, hi, ns = ci
            lows.append(lo)
            highs.append(hi)
            if ns > 0:
                avail += 1
        S, ok = solve_S(X, lows, highs)
        return (S, avail) if ok else (None, avail)

    for g in goals:
        w = g["wallclock"]
        if w - PRE < t_start or w + POST > t_end:
            continue
        n_goals += 1
        for d in grid:
            S, av = S_at(w + d)
            if S is not None and av >= 10:
                ev_curves[d].append(S)

    # placebos: in-play, far from any goal
    cand = []
    tt = t_start + PRE
    while tt < t_end - POST:
        if all(abs(tt - g["wallclock"]) > PLACEBO_GAP_S for g in goals):
            cand.append(tt)
        tt += 30
    for tt in random.sample(cand, min(PLACEBOS_PER_MATCH, len(cand))):
        n_pb += 1
        for d in grid:
            S, av = S_at(tt + d)
            if S is not None and av >= 10:
                pb_curves[d].append(S)

    print(f"--- {match}: goals={len(goals)} ({time.time()-t0:.0f}s)", flush=True)

print("\n" + "=" * 92)
print(f"EVENT STUDY — {n_goals} goals, {n_pb} placebo windows, "
      f"{len(per_match_goal)} matches")
print("=" * 92)
print(f"  {'t (s)':>7}{'n':>7}{'mean S':>10}{'med S':>9}{'%S>0':>8}   | "
      f"{'PLACEBO mean':>13}{'%S>0':>8}")
buckets = [(-300, -181), (-180, -61), (-60, -6), (-5, 5), (6, 30), (31, 60),
           (61, 120), (121, 180), (181, 300)]
rows_out = []
for lo, hi in buckets:
    ev = [s for d in grid if lo <= d <= hi for s in ev_curves[d]]
    pb = [s for d in grid if lo <= d <= hi for s in pb_curves[d]]
    if not ev:
        continue
    e = np.array(ev)
    p = np.array(pb) if pb else np.array([0.0])
    print(f"  {f'{lo}..{hi}':>7}{len(e):>7,}{e.mean():>10.4f}{np.median(e):>9.4f}"
          f"{100*(e>TOL).mean():>7.1f}%   | {p.mean():>13.4f}{100*(p>TOL).mean():>7.1f}%")
    rows_out.append({"lo": lo, "hi": hi, "n": len(e), "mean": float(e.mean()),
                     "median": float(np.median(e)), "frac_pos": float((e > TOL).mean()),
                     "pb_mean": float(p.mean()), "pb_frac_pos": float((p > TOL).mean())})

pre = [s for d in grid if -300 <= d <= -61 for s in ev_curves[d]]
post = [s for d in grid if 6 <= d <= 120 for s in ev_curves[d]]
if pre and post:
    pre_a, post_a = np.array(pre), np.array(post)
    print(f"\n  PRE(-300..-61)  mean={pre_a.mean():.4f}  %S>0={100*(pre_a>TOL).mean():.1f}%")
    print(f"  POST(+6..+120)  mean={post_a.mean():.4f}  %S>0={100*(post_a>TOL).mean():.1f}%")
    print(f"  lift            mean x{post_a.mean()/max(pre_a.mean(),1e-9):.2f}  "
          f"+{100*((post_a>TOL).mean()-(pre_a>TOL).mean()):.1f}pp incidence")

os.makedirs(f"{OUTDIR}/results", exist_ok=True)
json.dump({"n_goals": n_goals, "n_placebo": n_pb, "matches": per_match_goal,
           "buckets": rows_out,
           "params": {"GRID_STEP": GRID_STEP, "PRE": PRE, "POST": POST,
                      "FRESH": FRESH, "mode": "envelope"}},
          open(f"{OUTDIR}/results/g3_event_study.json", "w"), indent=2)
print(f"\nresults -> {OUTDIR}/results/g3_event_study.json")
