#!/usr/bin/env python3
"""RQ3 — heterogeneous propagation across linked families (common implied marginals).

Per D8, the response endpoint is MOVEMENT IN A COMMON IMPLIED MARGINAL, not the
first book-hash change. This avoids the family-size hazard (a 17-contract family
mutates more often than a 3-contract family purely by chance) and compares like
with like: the SAME probability, priced two ways.

Targets (each expressible both directly and via the exact-score grid):
    home_win   direct: moneyline home      implied: sum of grid cells with h > a
    over_2_5   direct: total-2pt5 (Over)   implied: sum of grid cells with h + a > 2.5
    btts       direct: btts (Yes)          implied: sum of grid cells with h>0 and a>0

For the implied side the identified interval is obtained by min/max LP over the
exact-score contracts only. For the direct side it is simply that contract's
canonical quote interval.

Response = first time after the goal that the interval midpoint departs from its
pre-goal mean by >= RESP (default 2 cents). Non-responders are right-censored.
Primary statistic is the PAIRED difference (implied - direct) per goal, then
aggregated to match level with a match-block bootstrap.

Run: /home/azureuser/.venv/bin/python -u rq3_propagation.py
"""
import glob
import gzip
import json
import lzma
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
                      state_classes)

GRID = 5
PRE_LO, PRE_HI = -60, -6
MAXLAG = 180
RESP = 0.02                    # 2 cents departure from pre-goal mean
FRESH = 30.0
OUT = "/home/azureuser/projects/wc_coherence"
np.random.seed(37)


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


def load_slug_books(slug):
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
                    books[aid].append((t, bid, ask))
        except Exception:
            continue
    return {a: (sorted(r), [x[0] for x in sorted(r)]) for a, r in books.items()}


def quote_env(series, times, t):
    i = bisect_right(times, t) - 1
    if i < 0:
        return None
    pt, pb, pa = series[i]
    if t - pt > FRESH:
        return None
    if i + 1 < len(series):
        nb, na = series[i + 1][1], series[i + 1][2]
        return (min([v for v in (pb, nb) if v is not None], default=None),
                max([v for v in (pa, na) if v is not None], default=None))
    return pb, pa


def canon_int(can, comp):
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
    return max(0.0, lo), min(1.0, hi)


def minmax_target(Xg, lows, highs, w):
    """min/max of w.pi subject to grid-family quote bounds; pi in simplex."""
    m, n = Xg.shape
    A_ub = np.vstack([-Xg, Xg])
    b_ub = np.concatenate([-np.array(lows), np.array(highs)])
    A_eq = np.ones((1, n))
    out = []
    for sign in (1.0, -1.0):
        try:
            r = linprog(sign * w, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                        b_eq=np.array([1.0]), bounds=[(0, 1)] * n, method="highs")
        except Exception:
            return None
        if not r.success:
            return None
        out.append(float(w @ r.x))
    return min(out), max(out)


def goals_of(game_id):
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
        out, seen = [], set()
        for e in (d.get("keyEvents") or []):
            try:
                if not e.get("scoringPlay"):
                    continue
                per = int((e.get("period") or {}).get("number") or 0)
                if per < 1 or per > 2:
                    continue
                w = iso(e.get("wallclock"))
                eid = str(e.get("id"))
                if w and eid not in seen:
                    seen.add(eid)
                    out.append({"id": eid, "w": w})
            except Exception:
                continue
        return sorted(out, key=lambda z: z["w"])
    return []


def first_response(times, mids, w, pre_mean):
    """First offset in (0, MAXLAG] where |mid - pre_mean| >= RESP; else None."""
    for t, m in zip(times, mids):
        off = t - w
        if off <= 0 or off > MAXLAG:
            continue
        if m is not None and abs(m - pre_mean) >= RESP:
            return off
    return None


split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
cohort_of = {m: "dev" for m in split["dev"]}
cohort_of.update({m: "holdout" for m in split["holdout"]})

print("=" * 96)
print("RQ3 — PROPAGATION ORDERING VIA COMMON IMPLIED MARGINALS")
print("=" * 96)
M, _ = load_manifest()
pairs = defaultdict(list)       # (cohort,target) -> list of (match, d_implied - d_direct)
resp_raw = defaultdict(list)    # (cohort,target,source) -> response times
cens = defaultdict(lambda: [0, 0])

for mi, match in enumerate(sorted(cohort_of), 1):
    t0 = time.time()
    contracts, info = build_contracts(M, match)
    if contracts is None:
        continue
    coh = cohort_of[match]
    reps = state_classes(contracts)
    grid_c = [c for c in contracts if c["family"] == "exact-score"]
    if len(grid_c) < 10:
        continue
    Xg = np.array([[c["rule"](h, a) for (h, a) in reps] for c in grid_c], dtype=float)
    w_home = np.array([1.0 if h > a else 0.0 for (h, a) in reps])
    w_ov = np.array([1.0 if h + a > 2.5 else 0.0 for (h, a) in reps])
    w_btts = np.array([1.0 if (h > 0 and a > 0) else 0.0 for (h, a) in reps])

    direct = {}
    for c in contracts:
        s = c["slug"]
        if c["family"] == "moneyline" and c["rule"](1, 0) == 1.0 and c["rule"](0, 1) == 0.0:
            direct["home_win"] = c
        if s.endswith("-total-2pt5"):
            direct["over_2_5"] = c
        if s.endswith("-btts"):
            direct["btts"] = c
    if len(direct) < 3:
        continue

    rows = info["espn_rows"]
    tin = [iso(d.get("captured_at")) for d in rows
           if str(d.get("status")) in ("in", "STATUS_IN_PROGRESS")
           and (d.get("period") or 0) and int(d["period"]) <= 2]
    tin = [t for t in tin if t]
    if len(tin) < 20:
        continue
    t_start, t_end = min(tin), max(tin)
    goals = [g for g in goals_of(info["game_id"]) if t_start <= g["w"] <= t_end]
    if not goals:
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

    def direct_mid(c, t):
        cs = ser.get((c["slug"], "canon_token"))
        ps = ser.get((c["slug"], "comp_token"))
        ci = canon_int(quote_env(cs[0], cs[1], t) if cs else None,
                       quote_env(ps[0], ps[1], t) if ps else None)
        return None if ci is None else 0.5 * (ci[0] + ci[1])

    def grid_bounds(t):
        lows, highs, ok = [], [], 0
        for c in grid_c:
            cs = ser.get((c["slug"], "canon_token"))
            ps = ser.get((c["slug"], "comp_token"))
            ci = canon_int(quote_env(cs[0], cs[1], t) if cs else None,
                           quote_env(ps[0], ps[1], t) if ps else None)
            if ci is None:
                lows.append(0.0)
                highs.append(1.0)
            else:
                lows.append(ci[0])
                highs.append(ci[1])
                ok += 1
        return (lows, highs) if ok >= 10 else None

    for g in goals:
        w = g["w"]
        if w + MAXLAG > t_end or w + PRE_LO < t_start:
            continue
        ts = np.arange(w + PRE_LO, w + MAXLAG + 1, GRID)
        series = {("direct", k): [] for k in direct}
        series.update({("implied", k): [] for k in direct})
        for t in ts:
            for k, c in direct.items():
                series[("direct", k)].append(direct_mid(c, t))
            gb = grid_bounds(t)
            for k, wv in (("home_win", w_home), ("over_2_5", w_ov), ("btts", w_btts)):
                if gb is None:
                    series[("implied", k)].append(None)
                else:
                    mm = minmax_target(Xg, gb[0], gb[1], wv)
                    series[("implied", k)].append(None if mm is None
                                                  else 0.5 * (mm[0] + mm[1]))
        for k in direct:
            resp = {}
            for src in ("direct", "implied"):
                v = series[(src, k)]
                pre = [x for t, x in zip(ts, v)
                       if PRE_LO <= t - w <= PRE_HI and x is not None]
                if len(pre) < 3:
                    resp[src] = None
                    continue
                pm = float(np.mean(pre))
                r = first_response(ts, v, w, pm)
                resp[src] = r
                cens[(coh, k, src)][1] += 1
                if r is None:
                    cens[(coh, k, src)][0] += 1
                else:
                    resp_raw[(coh, k, src)].append(r)
            if resp.get("direct") is not None and resp.get("implied") is not None:
                pairs[(coh, k)].append((match, resp["implied"] - resp["direct"]))
    if mi % 20 == 0:
        print(f"  ... {mi}/{len(cohort_of)} ({time.time()-t0:.0f}s)", flush=True)

print("\n" + "=" * 96)
print("FIRST-RESPONSE TIME (seconds after goal) BY SOURCE")
print("=" * 96)
print(f"  {'cohort':<9}{'target':<11}{'source':<9}{'n':>6}{'median':>9}{'mean':>8}{'censored':>10}")
out_rows = []
for coh in ("dev", "holdout"):
    for k in ("home_win", "over_2_5", "btts"):
        for src in ("direct", "implied"):
            v = resp_raw[(coh, k, src)]
            c0, c1 = cens[(coh, k, src)]
            if not v:
                continue
            print(f"  {coh:<9}{k:<11}{src:<9}{len(v):>6}{np.median(v):>9.0f}"
                  f"{np.mean(v):>8.0f}{f'{c0}/{c1}':>10}")
            out_rows.append({"cohort": coh, "target": k, "source": src,
                             "n": len(v), "median": float(np.median(v)),
                             "mean": float(np.mean(v)), "censored": c0, "total": c1})

print("\n" + "=" * 96)
print("PAIRED LAG (implied - direct), match-clustered")
print("=" * 96)
print(f"  {'cohort':<9}{'target':<11}{'pairs':>7}{'matches':>9}{'mean lag':>11}{'95% CI':>22}")
paired = []
for coh in ("dev", "holdout"):
    for k in ("home_win", "over_2_5", "btts"):
        P = pairs[(coh, k)]
        if len(P) < 5:
            continue
        bym = defaultdict(list)
        for m, d in P:
            bym[m].append(d)
        vals = np.array([np.mean(v) for v in bym.values()])
        D = float(vals.mean())
        boot = np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                         for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {coh:<9}{k:<11}{len(P):>7}{len(vals):>9}{D:>10.1f}s"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>22}")
        paired.append({"cohort": coh, "target": k, "n_pairs": len(P),
                       "n_matches": len(vals), "mean_lag": D,
                       "ci": [float(lo), float(hi)]})

json.dump({"response": out_rows, "paired": paired,
           "params": {"RESP": RESP, "MAXLAG": MAXLAG, "GRID": GRID,
                      "PRE": [PRE_LO, PRE_HI]}},
          open(f"{OUT}/results/rq3_propagation.json", "w"), indent=2)
print(f"\nresults -> {OUT}/results/rq3_propagation.json")
print("\nPositive lag => the exact-score grid reprices LATER than the direct contract.")
