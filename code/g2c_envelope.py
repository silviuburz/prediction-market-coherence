#!/usr/bin/env python3
"""GATE G2c — does S>0 survive quote-timing uncertainty?  (corrected test)

G2b's certainty rule ("book unchanged across t") was far too strict: actively
quoted books change most 10s intervals, so availability collapsed to ~1 contract
and S was trivially 0. That tested nothing.

The right question is whether measured incoherence survives a bound that ABSORBS
any price movement between polls. Three modes, increasingly conservative:

  naive     use the last observation at or before t (G2 behaviour)
  envelope  bracket t by the observations before AND after, and widen each token
            quote to the union: bid_env = min(bid_prev,bid_next),
            ask_env = max(ask_prev,ask_next). Any monotone move between polls is
            absorbed, so surviving slack cannot be produced by asynchronous
            observation of a moving price.
  exact     use only contracts whose state at t is provably known from
            book.timestamp: either the next observation's last change occurred
            at/BEFORE t (so its state was already in force), or the book did not
            change at all across the bracket.

If S>0 persists under `envelope`, the incoherence is not a timing artefact.

Run: /home/azureuser/.venv/bin/python -u g2c_envelope.py
"""
import glob
import gzip
import json
import os
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

MATCHES = ["eng-arg-2026-07-15", "arg-cvi-2026-07-03", "eng-hrv-2026-06-17",
           "nzl-egy-2026-06-21", "jor-alg-2026-06-22", "ger-civ-2026-06-20"]
STEP_SEC = 60
FRESH = 30.0
TOL = 1e-4
MODES = ("naive", "envelope", "exact")


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
    """-> {asset_id: (rows, times)}; rows = (cap_ts, bid, ask, hash, book_ts)"""
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


def token_quote(series, times, t, mode):
    """-> (bid, ask) or None"""
    i = bisect_right(times, t) - 1
    if i < 0:
        return None
    pt, pb, pa, ph, pts = series[i]
    if t - pt > FRESH:
        return None
    nxt = series[i + 1] if i + 1 < len(series) else None
    if mode == "naive":
        return pb, pa
    if mode == "envelope":
        if nxt is None:
            return pb, pa
        nb, na = nxt[1], nxt[2]
        bid = min([v for v in (pb, nb) if v is not None], default=None)
        ask = max([v for v in (pa, na) if v is not None], default=None)
        return bid, ask
    # mode == exact
    if nxt is None:
        return None
    nts, nh = nxt[4], nxt[3]
    if nh == ph:
        return pb, pa                      # no change across the bracket
    if nts is not None and nts <= t:
        return nxt[1], nxt[2]              # next state already in force at t
    return None


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
        return None
    return float(r.x[-1]) if r.success else None


print("=" * 96)
print("GATE G2c — DOES MEASURED INCOHERENCE SURVIVE QUOTE-TIMING UNCERTAINTY?")
print("=" * 96)
M, _ = load_manifest()
agg = defaultdict(lambda: {"n": 0, "pos": 0, "S": [], "avail": []})
per_match = defaultdict(lambda: defaultdict(list))

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

    for t in np.arange(min(tin), max(tin), STEP_SEC):
        for mode in MODES:
            lows, highs, avail = [], [], 0
            for c in contracts:
                cs = ser.get((c["slug"], "canon_token"))
                ps = ser.get((c["slug"], "comp_token"))
                can = token_quote(cs[0], cs[1], t, mode) if cs else None
                comp = token_quote(ps[0], ps[1], t, mode) if ps else None
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
            S = solve_S(X, lows, highs)
            if S is None:
                continue
            a = agg[mode]
            a["n"] += 1
            a["S"].append(S)
            a["avail"].append(avail)
            if S > TOL:
                a["pos"] += 1
            per_match[match][mode].append(S)
    print(f"--- {match}: done ({time.time()-t0:.0f}s)", flush=True)

print("\n" + "=" * 96)
print("POOLED MODE COMPARISON")
print("=" * 96)
print(f"  {'mode':<11}{'instants':>10}{'avail_med':>11}{'%S>0':>9}{'S_med':>10}"
      f"{'S_p90':>10}{'S_p99':>10}{'S_max':>10}")
for mode in MODES:
    a = agg[mode]
    if not a["n"]:
        continue
    S = np.array(a["S"])
    print(f"  {mode:<11}{a['n']:>10,}{np.median(a['avail']):>11.0f}"
          f"{100*a['pos']/a['n']:>8.1f}%{np.median(S):>10.4f}"
          f"{np.percentile(S,90):>10.4f}{np.percentile(S,99):>10.4f}{S.max():>10.4f}")

print("\n" + "=" * 96)
print("PER-MATCH %S>0 BY MODE")
print("=" * 96)
print(f"  {'match':<24}" + "".join(f"{m:>12}" for m in MODES))
for match in MATCHES:
    if match not in per_match:
        continue
    row = f"  {match:<24}"
    for m in MODES:
        v = per_match[match][m]
        row += f"{(100*sum(1 for s in v if s>TOL)/len(v) if v else 0):>11.0f}%"
    print(row)

env = agg["envelope"]
verdict = env["n"] and (env["pos"] / env["n"]) > 0.05
print("\n" + "=" * 96)
print(f"VERDICT: incoherence {'SURVIVES' if verdict else 'DOES NOT survive'} "
      f"the conservative envelope bound")
print("=" * 96)

out = "/home/azureuser/projects/wc_coherence/results/g2c_envelope.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({m: {"n": agg[m]["n"], "pos": agg[m]["pos"],
               "S_med": float(np.median(agg[m]["S"])) if agg[m]["n"] else None,
               "S_p90": float(np.percentile(agg[m]["S"], 90)) if agg[m]["n"] else None,
               "S_p99": float(np.percentile(agg[m]["S"], 99)) if agg[m]["n"] else None,
               "avail_med": float(np.median(agg[m]["avail"])) if agg[m]["n"] else None}
           for m in MODES}, open(out, "w"), indent=2)
print(f"results -> {out}")
