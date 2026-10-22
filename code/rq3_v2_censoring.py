#!/usr/bin/env python3
"""RQ3 v2 — censoring-aware propagation analysis.

v1 computed a paired mean lag only for goals where BOTH sources responded within
180 s. That is a complete-case analysis and is biased when censoring differs by
source -- which it does badly here (direct BTTS failed to respond in ~25% of
goals versus ~10% for the grid-implied series).

v2 keeps every event by switching the endpoint to RESPONSE-BY-HORIZON indicators:

    R_src(h) = 1 if the source's implied marginal departs from its pre-goal mean
               by >= RESP at any point within h seconds of the goal, else 0

For each goal we form the paired difference  R_implied(h) - R_direct(h), average
within match, then average across matches with a match-block bootstrap. Non-
responders contribute 0 rather than being dropped, so no event is discarded.

Horizons: 10, 30, 60, 180 s.  Response thresholds: 1, 2 and 5 cents.
Per-event rows are persisted so the analysis can be redone without recomputation.

Run: /home/azureuser/.venv/bin/python -u rq3_v2_censoring.py
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
from compiler import (LIVE, build_contracts, load_manifest, state_classes)  # noqa: E402

GRID = 5
PRE_LO, PRE_HI = -60, -6
MAXLAG = 180
HORIZONS = [10, 30, 60, 180]
RESP_LEVELS = [0.01, 0.02, 0.05]
PRIMARY_RESP = 0.02
FRESH = 30.0
OUT = "/home/azureuser/projects/wc_coherence"
np.random.seed(41)


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


def minmax(Xg, lows, highs, w):
    m, n = Xg.shape
    A_ub = np.vstack([-Xg, Xg])
    b_ub = np.concatenate([-np.array(lows), np.array(highs)])
    A_eq = np.ones((1, n))
    got = []
    for sign in (1.0, -1.0):
        try:
            r = linprog(sign * w, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                        b_eq=np.array([1.0]), bounds=[(0, 1)] * n, method="highs")
        except Exception:
            return None
        if not r.success:
            return None
        got.append(float(w @ r.x))
    return min(got), max(got)


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


split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
cohort_of = {m: "dev" for m in split["dev"]}
cohort_of.update({m: "holdout" for m in split["holdout"]})

print("=" * 100)
print("RQ3 v2 — CENSORING-AWARE PROPAGATION (response-by-horizon indicators)")
print("=" * 100)
M, _ = load_manifest()
rows = []

for mi, match in enumerate(sorted(cohort_of), 1):
    contracts, info = build_contracts(M, match)
    if contracts is None:
        continue
    coh = cohort_of[match]
    reps = state_classes(contracts)
    grid_c = [c for c in contracts if c["family"] == "exact-score"]
    if len(grid_c) < 10:
        continue
    Xg = np.array([[c["rule"](h, a) for (h, a) in reps] for c in grid_c], dtype=float)
    W = {"home_win": np.array([1.0 if h > a else 0.0 for (h, a) in reps]),
         "over_2_5": np.array([1.0 if h + a > 2.5 else 0.0 for (h, a) in reps]),
         "btts": np.array([1.0 if (h > 0 and a > 0) else 0.0 for (h, a) in reps])}
    direct = {}
    for c in contracts:
        if c["family"] == "moneyline" and c["rule"](1, 0) == 1.0 and c["rule"](0, 1) == 0.0:
            direct["home_win"] = c
        if c["slug"].endswith("-total-2pt5"):
            direct["over_2_5"] = c
        if c["slug"].endswith("-btts"):
            direct["btts"] = c
    if len(direct) < 3:
        continue
    esp = info["espn_rows"]
    tin = [iso(d.get("captured_at")) for d in esp
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

    def dmid(c, t):
        cs, ps = ser.get((c["slug"], "canon_token")), ser.get((c["slug"], "comp_token"))
        ci = canon_int(quote_env(cs[0], cs[1], t) if cs else None,
                       quote_env(ps[0], ps[1], t) if ps else None)
        return None if ci is None else 0.5 * (ci[0] + ci[1])

    def gbounds(t):
        lows, highs, ok = [], [], 0
        for c in grid_c:
            cs, ps = ser.get((c["slug"], "canon_token")), ser.get((c["slug"], "comp_token"))
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
        cache = {}
        for t in ts:
            gb = gbounds(t)
            for k in W:
                cache[("direct", k, t)] = dmid(direct[k], t)
                cache[("implied", k, t)] = (None if gb is None else
                                            (lambda mm: None if mm is None else 0.5 * (mm[0] + mm[1]))
                                            (minmax(Xg, gb[0], gb[1], W[k])))
        for k in W:
            for src in ("direct", "implied"):
                pre = [cache[(src, k, t)] for t in ts
                       if PRE_LO <= t - w <= PRE_HI and cache[(src, k, t)] is not None]
                if len(pre) < 3:
                    continue
                pm = float(np.mean(pre))
                first = {lvl: None for lvl in RESP_LEVELS}
                for t in ts:
                    off = t - w
                    if off <= 0 or off > MAXLAG:
                        continue
                    v = cache[(src, k, t)]
                    if v is None:
                        continue
                    for lvl in RESP_LEVELS:
                        if first[lvl] is None and abs(v - pm) >= lvl:
                            first[lvl] = float(off)
                rows.append({"cohort": coh, "match": match, "event": g["id"],
                             "target": k, "source": src, "pre_mean": pm,
                             "first_resp": {str(l): first[l] for l in RESP_LEVELS}})
    if mi % 25 == 0:
        print(f"  ... {mi}/{len(cohort_of)}", flush=True)

os.makedirs(f"{OUT}/results", exist_ok=True)
with open(f"{OUT}/results/rq3_v2_events.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"\nper-event rows persisted: {len(rows):,}")

# ── analysis: paired response-by-horizon, match-clustered ───────────────────
def paired(cohort, target, horizon, lvl):
    by = defaultdict(dict)
    for r in rows:
        if r["cohort"] != cohort or r["target"] != target:
            continue
        by[(r["match"], r["event"])][r["source"]] = r["first_resp"][str(lvl)]
    per_match = defaultdict(list)
    for (m, _e), d in by.items():
        if "direct" not in d or "implied" not in d:
            continue
        ri = 1.0 if (d["implied"] is not None and d["implied"] <= horizon) else 0.0
        rd = 1.0 if (d["direct"] is not None and d["direct"] <= horizon) else 0.0
        per_match[m].append((ri, rd))
    if len(per_match) < 5:
        return None
    ms = sorted(per_match)
    di = np.array([np.mean([x[0] for x in per_match[m]]) for m in ms])
    dd = np.array([np.mean([x[1] for x in per_match[m]]) for m in ms])
    diff = di - dd
    boot = np.array([np.mean(np.random.choice(diff, len(diff), replace=True))
                     for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_matches": len(ms), "n_events": sum(len(v) for v in per_match.values()),
            "implied_rate": float(di.mean()), "direct_rate": float(dd.mean()),
            "diff": float(diff.mean()), "ci": [float(lo), float(hi)]}


print("\n" + "=" * 100)
print(f"PAIRED RESPONSE RATE BY HORIZON (threshold {PRIMARY_RESP*100:.0f} cents)")
print("=" * 100)
print(f"  {'cohort':<9}{'target':<10}{'horiz':>7}{'implied':>9}{'direct':>9}"
      f"{'diff':>9}{'95% CI':>20}{'matches':>9}")
out = []
for coh in ("dev", "holdout"):
    for k in ("home_win", "over_2_5", "btts"):
        for h in HORIZONS:
            r = paired(coh, k, h, PRIMARY_RESP)
            if not r:
                continue
            ci_txt = f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
            print(f"  {coh:<9}{k:<10}{h:>6}s{r['implied_rate']:>9.2f}"
                  f"{r['direct_rate']:>9.2f}{r['diff']:>+9.3f}"
                  f"{ci_txt:>20}{r['n_matches']:>9}")
            r.update({"cohort": coh, "target": k, "horizon": h, "level": PRIMARY_RESP})
            out.append(r)

print("\n" + "=" * 100)
print("THRESHOLD SENSITIVITY at 180 s")
print("=" * 100)
print(f"  {'cohort':<9}{'target':<10}{'level':>7}{'implied':>9}{'direct':>9}{'diff':>9}")
for coh in ("dev", "holdout"):
    for k in ("home_win", "over_2_5", "btts"):
        for lvl in RESP_LEVELS:
            r = paired(coh, k, 180, lvl)
            if r:
                print(f"  {coh:<9}{k:<10}{lvl*100:>6.0f}c{r['implied_rate']:>9.2f}"
                      f"{r['direct_rate']:>9.2f}{r['diff']:>+9.3f}")

json.dump(out, open(f"{OUT}/results/rq3_v2_summary.json", "w"), indent=2)
print(f"\nresults -> {OUT}/results/rq3_v2_summary.json")
print("\nPositive diff => the exact-score grid responds MORE OFTEN by that horizon")
print("than the direct contract. Non-responders are retained (no complete-case bias).")
