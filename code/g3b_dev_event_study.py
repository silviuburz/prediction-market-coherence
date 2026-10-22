#!/usr/bin/env python3
"""G3b — DEFINITIVE DEVELOPMENT event study (holdout untouched).

Addresses the findings audit in full:
  * DEV ONLY -- 49 development matches. Holdout quote outcomes are never computed.
  * Event-level then MATCH-level estimand (no pooling of 5s grid cells as n):
        d_gm = mean(S | +6..+30) - mean(S | -60..-6)      per goal
        d_m  = mean_g d_gm                                 per match
        D    = mean_m d_m                                  equal-match primary
  * ABSOLUTE cents primary; fold-change only descriptive.
  * ISOLATED-goal primary sample (no other goal inside the window).
  * MATCH-BLOCK bootstrap CI + within-match RANDOMIZATION inference.
  * Leave-one-match-out influence.
  * Threshold sensitivity for incidence.
  * Two nested metrics:
        S_cross  internal coherence (payoff-only)            [primary]
        S_state  additionally requires final total goals >= goals already scored
                 (a valid, attribution-free known-state restriction)
  * Long-form per-(match,event,offset) output for reproducible re-analysis.

Efficiency: S is computed ONCE on a regular grid spanning each match's in-play
window; goal windows, placebos and randomization draws all index into that cache.

Run: /home/azureuser/.venv/bin/python -u g3b_dev_event_study.py
"""
import glob
import gzip
import json
import lzma
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

random.seed(29)
np.random.seed(29)

GRID = 5
PRE_LO, PRE_HI = -60, -6          # pre-window (frozen on development data)
POST_LO, POST_HI = 6, 30          # post-window
WIN = 300                         # +/- window for eligibility & curves
FRESH = 30.0
TOL = 1e-4
ISO_GAP = 300                     # isolated goal: no other goal within +/- this
PLACEBO_GAP = 240
N_BOOT = 5000
N_RAND = 5000
OUT = "/home/azureuser/projects/wc_coherence"
THRESHOLDS = [("certified", TOL), ("0.10c", 0.0010), ("0.25c", 0.0025),
              ("0.50c", 0.0050), ("1.00c", 0.0100)]


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


def token_quote_env(series, times, t):
    """Adjacent-poll endpoint envelope (absorbs monotone movement between polls)."""
    i = bisect_right(times, t) - 1
    if i < 0:
        return None
    pt, pb, pa = series[i]
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


def solve_S(X, lows, highs, mask=None):
    """min eps s.t. lows-eps <= X^T pi <= highs+eps, pi in simplex (masked)."""
    m, n = X.shape
    cols = np.arange(n) if mask is None else np.where(mask)[0]
    if len(cols) == 0:
        return None
    Xm = X[:, cols]
    nv = len(cols) + 1
    c = np.zeros(nv)
    c[-1] = 1.0
    A_ub = np.zeros((2 * m, nv))
    b_ub = np.zeros(2 * m)
    A_ub[:m, :-1] = -Xm
    A_ub[:m, -1] = -1.0
    b_ub[:m] = -np.array(lows)
    A_ub[m:, :-1] = Xm
    A_ub[m:, -1] = -1.0
    b_ub[m:] = np.array(highs)
    A_eq = np.zeros((1, nv))
    A_eq[0, :-1] = 1.0
    try:
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([1.0]),
                    bounds=[(0, 1)] * len(cols) + [(0, None)], method="highs")
    except Exception:
        return None
    return float(r.x[-1]) if r.success else None


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
        out = []
        for e in (d.get("keyEvents") or []):
            try:
                if not e.get("scoringPlay"):
                    continue
                per = int((e.get("period") or {}).get("number") or 0)
                if per < 1 or per > 2:
                    continue
                w = iso(e.get("wallclock"))
                if w:
                    out.append({"id": str(e.get("id")), "w": w, "period": per,
                                "clock": (e.get("clock") or {}).get("displayValue")})
            except Exception:
                continue
        seen, ded = set(), []
        for g in sorted(out, key=lambda z: z["w"]):
            if g["id"] in seen:
                continue
            seen.add(g["id"])
            ded.append(g)
        return ded
    return []


split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
DEV = split["dev"]
print("=" * 96)
print("G3b — DEFINITIVE DEVELOPMENT EVENT STUDY")
print("=" * 96)
print(f"development matches: {len(DEV)} | holdout: {len(split['holdout'])} "
      f"(NEVER computed here)")

M, _ = load_manifest()
long_rows = []
match_stats = {}

for mi, match in enumerate(DEV, 1):
    t0 = time.time()
    contracts, info = build_contracts(M, match)
    if contracts is None:
        continue
    reps = state_classes(contracts)
    X = np.array(payoff_matrix(contracts, reps), dtype=float)
    totals = np.array([h + a for (h, a) in reps])
    rows = info["espn_rows"]
    tin = [iso(d.get("captured_at")) for d in rows
           if str(d.get("status")) in ("in", "STATUS_IN_PROGRESS")
           and (d.get("period") or 0) and int(d["period"]) <= 2]
    tin = [t for t in tin if t]
    if len(tin) < 20:
        continue
    t_start, t_end = min(tin), max(tin)
    goals = [g for g in goals_of(info["game_id"]) if t_start - 60 <= g["w"] <= t_end + 60]

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

    # ---- cache S on a regular grid across the whole in-play window ----------
    gt = np.arange(t_start, t_end, GRID)
    Sc = np.full(len(gt), np.nan)
    Ss = np.full(len(gt), np.nan)
    Av = np.zeros(len(gt))
    gw = sorted(g["w"] for g in goals)
    for i, t in enumerate(gt):
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
        Av[i] = avail
        if avail < 10:
            continue
        v = solve_S(X, lows, highs)
        if v is not None:
            Sc[i] = v
        nb = sum(1 for w in gw if w <= t)          # goals already scored
        mask = totals >= nb
        v2 = solve_S(X, lows, highs, mask=mask)
        if v2 is not None:
            Ss[i] = v2

    def win_mean(arr, centre, lo, hi):
        i0 = int(np.searchsorted(gt, centre + lo))
        i1 = int(np.searchsorted(gt, centre + hi))
        seg = arr[i0:i1 + 1]
        seg = seg[~np.isnan(seg)]
        return float(seg.mean()) if len(seg) else None

    evs = []
    for g in goals:
        w = g["w"]
        if w - WIN < t_start or w + WIN > t_end:
            continue
        isolated = all(abs(w - o) > ISO_GAP for o in gw if o != w)
        pre_c, post_c = win_mean(Sc, w, PRE_LO, PRE_HI), win_mean(Sc, w, POST_LO, POST_HI)
        pre_s, post_s = win_mean(Ss, w, PRE_LO, PRE_HI), win_mean(Ss, w, POST_LO, POST_HI)
        if pre_c is None or post_c is None:
            continue
        evs.append({"id": g["id"], "w": w, "clock": g["clock"], "isolated": isolated,
                    "d_cross": post_c - pre_c, "pre_cross": pre_c, "post_cross": post_c,
                    "d_state": (post_s - pre_s) if (pre_s is not None and post_s is not None) else None})
        for off in range(-WIN, WIN + 1, GRID):
            i = int(np.searchsorted(gt, w + off))
            if 0 <= i < len(gt):
                long_rows.append({"match": match, "event": g["id"], "offset": off,
                                  "S_cross": None if np.isnan(Sc[i]) else float(Sc[i]),
                                  "S_state": None if np.isnan(Ss[i]) else float(Ss[i]),
                                  "avail": int(Av[i]), "isolated": isolated})

    # eligible placebo/randomization anchor indices (same eligibility as goals)
    elig = [i for i, t in enumerate(gt)
            if t - WIN >= t_start and t + WIN <= t_end
            and all(abs(t - w) > PLACEBO_GAP for w in gw)]
    match_stats[match] = {"n_goals_total": len(goals), "n_events": len(evs),
                          "events": evs, "elig": elig,
                          "gt0": float(gt[0]) if len(gt) else None,
                          "Sc": Sc.tolist(), "n_grid": len(gt)}
    print(f"[{mi:>2}/{len(DEV)}] {match}: goals={len(goals)} events={len(evs)} "
          f"({time.time()-t0:.0f}s)", flush=True)

# ── estimands ───────────────────────────────────────────────────────────────
def match_effects(isolated_only):
    out = {}
    for m, st in match_stats.items():
        ds = [e["d_cross"] for e in st["events"]
              if (e["isolated"] or not isolated_only)]
        if ds:
            out[m] = float(np.mean(ds))
    return out


def summarize(tag, isolated_only):
    me = match_effects(isolated_only)
    if not me:
        print(f"\n{tag}: no eligible events")
        return None
    ms = sorted(me)
    vals = np.array([me[m] for m in ms])
    D = float(vals.mean())
    boot = np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    loo = [float(np.mean(np.delete(vals, i))) for i in range(len(vals))]
    n_ev = sum(1 for m in ms for e in match_stats[m]["events"]
               if e["isolated"] or not isolated_only)
    print(f"\n{tag}")
    print(f"  matches={len(ms)}  events={n_ev}")
    print(f"  equal-match effect D = {D*100:+.3f} cents   95% match-block CI "
          f"[{lo*100:+.3f}, {hi*100:+.3f}]")
    print(f"  median match effect  = {np.median(vals)*100:+.3f} cents")
    print(f"  leave-one-match-out  = [{min(loo)*100:+.3f}, {max(loo)*100:+.3f}] cents")
    print(f"  matches with positive effect: {int((vals>0).sum())}/{len(vals)}")
    return {"D": D, "ci": [float(lo), float(hi)], "n_matches": len(ms),
            "n_events": n_ev, "loo_min": min(loo), "loo_max": max(loo),
            "median": float(np.median(vals)),
            "frac_pos_matches": float((vals > 0).mean())}


print("\n" + "=" * 96)
print("PRIMARY ESTIMAND — equal-match, absolute cents")
print("=" * 96)
res_all = summarize("ALL eligible goals", False)
res_iso = summarize("ISOLATED goals only (primary)", True)

# ── randomization inference (within-match pseudo-goals) ─────────────────────
print("\n" + "=" * 96)
print("RANDOMIZATION INFERENCE (within-match pseudo-goal anchors)")
print("=" * 96)
obs = res_iso["D"] if res_iso else (res_all["D"] if res_all else None)
if obs is not None:
    null = []
    usable = [m for m in match_stats
              if match_stats[m]["events"] and len(match_stats[m]["elig"]) > 5]
    for _ in range(N_RAND):
        per_match = []
        for m in usable:
            st = match_stats[m]
            k = sum(1 for e in st["events"] if e["isolated"]) or len(st["events"])
            if k == 0:
                continue
            Sc = np.array(st["Sc"])
            gt0, n = st["gt0"], st["n_grid"]
            picks = random.sample(st["elig"], min(k, len(st["elig"])))
            ds = []
            for i in picks:
                a0 = i + PRE_LO // GRID
                a1 = i + PRE_HI // GRID
                b0 = i + POST_LO // GRID
                b1 = i + POST_HI // GRID
                if a0 < 0 or b1 >= n:
                    continue
                pre = Sc[a0:a1 + 1]
                post = Sc[b0:b1 + 1]
                pre = pre[~np.isnan(pre)]
                post = post[~np.isnan(post)]
                if len(pre) and len(post):
                    ds.append(post.mean() - pre.mean())
            if ds:
                per_match.append(np.mean(ds))
        if per_match:
            null.append(np.mean(per_match))
    null = np.array(null)
    p = float((null >= obs).mean())
    print(f"  observed D = {obs*100:+.3f} cents")
    print(f"  null mean  = {null.mean()*100:+.3f} cents   sd {null.std()*100:.3f}")
    print(f"  null p95   = {np.percentile(null,95)*100:+.3f} cents")
    print(f"  one-sided randomization p = {p:.4f}   (draws={len(null)})")
else:
    p = None

# ── threshold sensitivity for incidence ─────────────────────────────────────
print("\n" + "=" * 96)
print("INCIDENCE BY THRESHOLD (isolated goals; pre vs post window)")
print("=" * 96)
print(f"  {'threshold':<12}{'pre %':>9}{'post %':>9}{'delta pp':>11}")
thr_out = []
for name, thr in THRESHOLDS:
    pre_v, post_v = [], []
    for m, st in match_stats.items():
        for e in st["events"]:
            if not e["isolated"]:
                continue
            pre_v.append(e["pre_cross"] > thr)
            post_v.append(e["post_cross"] > thr)
    if pre_v:
        a, b = 100 * np.mean(pre_v), 100 * np.mean(post_v)
        print(f"  {name:<12}{a:>8.1f}%{b:>8.1f}%{b-a:>10.1f}pp")
        thr_out.append({"threshold": name, "pre": a, "post": b, "delta": b - a})

# ── state-conditioned metric ────────────────────────────────────────────────
ds_state = [e["d_state"] for st in match_stats.values() for e in st["events"]
            if e["isolated"] and e["d_state"] is not None]
if ds_state:
    print("\n" + "=" * 96)
    print("STATE-CONDITIONED METRIC (final total goals >= goals already scored)")
    print("=" * 96)
    print(f"  events with S_state: {len(ds_state)}   mean d_state = "
          f"{np.mean(ds_state)*100:+.3f} cents")

os.makedirs(f"{OUT}/results", exist_ok=True)
json.dump({"all": res_all, "isolated": res_iso, "rand_p": p,
           "thresholds": thr_out,
           "per_match": {m: {"n_goals": st["n_goals_total"], "n_events": st["n_events"],
                             "d": [e["d_cross"] for e in st["events"]],
                             "isolated": [e["isolated"] for e in st["events"]]}
                         for m, st in match_stats.items()},
           "params": {"PRE": [PRE_LO, PRE_HI], "POST": [POST_LO, POST_HI],
                      "GRID": GRID, "ISO_GAP": ISO_GAP, "N_BOOT": N_BOOT,
                      "N_RAND": N_RAND}},
          open(f"{OUT}/results/g3b_dev_event_study.json", "w"), indent=2)
with open(f"{OUT}/results/g3b_long_form.jsonl", "w") as f:
    for r in long_rows:
        f.write(json.dumps(r) + "\n")
print(f"\nlong-form rows: {len(long_rows):,}")
print(f"results -> {OUT}/results/g3b_dev_event_study.json")
