#!/usr/bin/env python3
"""GATE G2b — is the observed S>0 REAL, or an artefact of stale/asynchronous quotes?

The G2 pilot found small but frequently positive coherence slack (S_med 0-0.13c,
S_p90 <=0.5c). Because quotes are polled every 10s and combined across contracts,
an apparent violation can be manufactured by comparing states observed at
slightly different times. This script rules that in or out.

KEY IDEA — a certainty rule using book hashes. At instant t, let the previous
observation (t_prev <= t) have hash H_prev and the next (t_next > t) have H_next.
If H_prev == H_next then the book provably did not change across t, so the state
at t is KNOWN EXACTLY. If the hashes differ, the change may have occurred either
side of t, so the contract is weakened to [0,1] (conservative) -- it can then
only ever REDUCE measured incoherence, never create it.

Modes compared:
  naive_30 / naive_10   last obs within FRESH sec (G2 behaviour)
  certain_30/certain_10 same, but non-certain contracts weakened to [0,1]

If S>0 survives `certain_*`, the incoherence is not a staleness artefact.
Also reports detection power BY FAMILY.

Run: /home/azureuser/.venv/bin/python -u g2b_staleness.py
"""
import glob
import gzip
import json
import os
import random
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402
from scipy.optimize import linprog                     # noqa: E402
from compiler import (LIVE, build_contracts, load_manifest,               # noqa: E402
                      payoff_matrix, state_classes)

random.seed(17)
np.random.seed(17)

MATCHES = ["eng-arg-2026-07-15", "arg-cvi-2026-07-03", "eng-hrv-2026-06-17",
           "nzl-egy-2026-06-21", "jor-alg-2026-06-22", "ger-civ-2026-06-20"]
STEP_SEC = 60
TOL = 1e-4
DELTAS = (0.01, 0.02, 0.05)
MODES = [("naive_30", 30.0, False), ("naive_10", 10.0, False),
         ("certain_30", 30.0, True), ("certain_10", 10.0, True)]


def iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


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
    """-> {asset_id: (rows, times)}; rows = (ts, bid, ask, hash)"""
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
                    books[aid].append((t, bid, ask, str(b.get("hash"))))
        except Exception:
            continue
    out = {}
    for aid, rows in books.items():
        rows.sort(key=lambda r: r[0])
        out[aid] = (rows, [r[0] for r in rows])
    _cache[slug] = out
    return out


def quote_at(series, times, t, fresh, certainty):
    """-> (bid, ask) or None if unusable/uncertain."""
    i = bisect_right(times, t) - 1
    if i < 0:
        return None
    ts, bid, ask, h = series[i]
    if t - ts > fresh:
        return None
    if certainty:
        if i + 1 >= len(series):
            return None                       # no bracketing next obs
        if series[i + 1][3] != h:
            return None                       # book changed across t -> uncertain
    return bid, ask


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
        row = np.zeros(nv)
        row[:n] = -X[j]
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(-lows[j])
        row = np.zeros(nv)
        row[:n] = X[j]
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(highs[j])
    A_eq = np.zeros((1, nv))
    A_eq[0, :n] = 1.0
    try:
        r = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq,
                    b_eq=np.array([1.0]), bounds=[(0, 1)] * n + [(0, None)],
                    method="highs")
    except Exception:
        return None, None
    if not r.success:
        return None, None
    return float(r.x[-1]), r.x[:n]


print("=" * 92)
print("GATE G2b — STALENESS SENSITIVITY (is S>0 real?)")
print("=" * 92)
M, _ = load_manifest()

agg = defaultdict(lambda: {"n": 0, "pos": 0, "S": [], "avail": [], "crossed": 0})
fam_power = defaultdict(lambda: defaultdict(lambda: [0, 0]))

for match in MATCHES:
    t0 = time.time()
    contracts, info = build_contracts(M, match)
    if contracts is None:
        print(f"--- {match}: SKIP")
        continue
    reps = state_classes(contracts)
    X = np.array(payoff_matrix(contracts, reps), dtype=float)
    rows = info["espn_rows"]
    tin = [iso(d.get("captured_at")) for d in rows
           if str(d.get("status")) in ("in", "STATUS_IN_PROGRESS")
           and (d.get("period") or 0) and int(d["period"]) <= 2]
    tin = [t for t in tin if t]
    if len(tin) < 10:
        print(f"--- {match}: SKIP (no window)")
        continue
    t_start, t_end = min(tin), max(tin)

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

    instants = list(np.arange(t_start, t_end, STEP_SEC))
    for mname, fresh, certainty in MODES:
        for t in instants:
            lows, highs, avail = [], [], 0
            crossed = False
            for c in contracts:
                cs = ser.get((c["slug"], "canon_token"))
                ps = ser.get((c["slug"], "comp_token"))
                can = quote_at(cs[0], cs[1], t, fresh, certainty) if cs else None
                comp = quote_at(ps[0], ps[1], t, fresh, certainty) if ps else None
                ci = canonical_interval(can, comp)
                if ci is None:
                    crossed = True
                    lows.append(0.0)
                    highs.append(1.0)
                    continue
                lo, hi, ns = ci
                lows.append(lo)
                highs.append(hi)
                if ns > 0:
                    avail += 1
            S, pi = solve_S(X, lows, highs)
            if S is None:
                continue
            a = agg[mname]
            a["n"] += 1
            a["S"].append(S)
            a["avail"].append(avail)
            if S > TOL:
                a["pos"] += 1
            if crossed:
                a["crossed"] += 1

            # family-level power, computed only in the strictest mode
            if certainty and fresh == 30.0 and pi is not None and avail >= 10 and random.random() < 0.25:
                p0 = np.array(pi, dtype=float)
                if p0.sum() <= 0:
                    continue
                p0 = p0 / p0.sum()
                pstar = X @ p0
                w = np.array([h - l for l, h in zip(lows, highs)])
                lo0 = np.clip(pstar - w / 2, 0, 1)
                hi0 = np.clip(pstar + w / 2, 0, 1)
                S0, _ = solve_S(X, list(lo0), list(hi0))
                if S0 is None or S0 > TOL:
                    continue
                for j in random.sample(range(len(contracts)), min(6, len(contracts))):
                    fam = contracts[j]["family"]
                    for d in DELTAS:
                        l1, h1 = lo0.copy(), hi0.copy()
                        l1[j] = min(1.0, lo0[j] + d)
                        h1[j] = min(1.0, hi0[j] + d)
                        if h1[j] <= l1[j]:
                            continue
                        S1, _ = solve_S(X, list(l1), list(h1))
                        if S1 is None:
                            continue
                        fam_power[fam][d][1] += 1
                        if S1 > TOL:
                            fam_power[fam][d][0] += 1
    print(f"--- {match}: done ({time.time()-t0:.0f}s)", flush=True)

print("\n" + "=" * 92)
print("MODE COMPARISON (pooled over matches)")
print("=" * 92)
print(f"  {'mode':<13}{'instants':>10}{'avail_med':>11}{'%S>0':>8}{'S_med':>10}{'S_p90':>10}{'S_max':>10}{'crossed':>9}")
for mname, _f, _c in MODES:
    a = agg[mname]
    if not a["n"]:
        continue
    S = np.array(a["S"])
    print(f"  {mname:<13}{a['n']:>10,}{np.median(a['avail']):>11.0f}"
          f"{100*a['pos']/a['n']:>7.1f}%{np.median(S):>10.4f}"
          f"{np.percentile(S,90):>10.4f}{S.max():>10.4f}{a['crossed']:>9,}")

print("\n" + "=" * 92)
print("DETECTION POWER BY FAMILY (mode = certain_30)")
print("=" * 92)
print(f"  {'family':<14}" + "".join(f"{int(d*100):>7}c" for d in DELTAS) + f"{'trials':>10}")
for fam in sorted(fam_power):
    row = f"  {fam:<14}"
    tot = 0
    for d in DELTAS:
        det, n = fam_power[fam][d]
        tot = max(tot, n)
        row += f"{(100*det/n if n else 0):>7.0f}%"
    print(row + f"{tot:>10,}")

out = "/home/azureuser/projects/wc_coherence/results/g2b_staleness.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({m: {"n": agg[m]["n"], "pos": agg[m]["pos"],
               "S_med": float(np.median(agg[m]["S"])) if agg[m]["n"] else None,
               "S_p90": float(np.percentile(agg[m]["S"], 90)) if agg[m]["n"] else None,
               "avail_med": float(np.median(agg[m]["avail"])) if agg[m]["n"] else None,
               "crossed": agg[m]["crossed"]}
           for m, _f, _c in MODES},
          open(out, "w"), indent=2)
print(f"\nresults -> {out}")
