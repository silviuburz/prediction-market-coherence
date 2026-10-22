#!/usr/bin/env python3
"""GATE G2 — coherence measurement + informativeness + falsifiability pilot.

For each pilot match:
  1. compile primary contracts (compiler.py, validated at G1)
  2. build payoff-equivalence state space + payoff matrix
  3. extract per-token best bid/ask series from the 10s book capture
  4. at sampled instants, build CANONICAL quote intervals (D7):
         [max(b_can, 1-a_comp), min(a_can, 1-b_comp)]
  5. solve the primary metric (D5)  S_inf = min eps  s.t.
         b_j - eps <= (X^T pi)_j <= a_j + eps,  pi in simplex
  6. report the informativeness panel (D6)
  7. run the falsifiability simulation: inject 1/2/5-cent incoherence into a
     coherent system with the OBSERVED interval geometry and measure detection

G2 passes only if injected violations are detected at useful rates. Otherwise a
null result would be uninformative and the project pivots (per v3 gates).

Read-only w.r.t. the frozen corpus. Run:
  /home/azureuser/.venv/bin/python -u g2_coherence_pilot.py
"""
import glob
import gzip
import json
import os
import random
import statistics as st
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                    # noqa: E402
from scipy.optimize import linprog                    # noqa: E402
from compiler import (LIVE, build_contracts, load_manifest, payoff_matrix,  # noqa: E402
                      saturation_ok, state_classes)

random.seed(11)
np.random.seed(11)

PILOT = [
    "eng-arg-2026-07-15", "arg-cvi-2026-07-03", "eng-hrv-2026-06-17",   # dev trio
    "esp-ksa-2026-06-21", "bra-hai-2026-06-19", "ger-civ-2026-06-20",
    "mex-kr-2026-06-18", "nzl-egy-2026-06-21", "jor-alg-2026-06-22",
    "prt-uzb-2026-06-23",
]
FRESH_SEC = 30.0          # max age of the last observation used at an instant
STEP_SEC = 60             # evaluation cadence within the in-play window
TOL = 1e-4                # S_inf above this counts as detected incoherence
DELTAS = (0.01, 0.02, 0.05)
N_INJECT_CONTRACTS = 8


def iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def best_bid_ask(book):
    """Best prices using positive size only. -> (bid, ask)"""
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


_slug_cache = {}


def load_slug_books(slug):
    """Read every book file under this market's dir and key the series by the
    EMBEDDED `book.asset_id` -- the authoritative token identity (the on-disk
    filename is only a 16-char prefix of the asset id, and the historical rekey
    incident makes filename-based identity unsafe).

    -> {asset_id: (series, times)} with series = [(captured_ts, bid, ask), ...]
    """
    if slug in _slug_cache:
        return _slug_cache[slug]
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
    out = {}
    for aid, rows in books.items():
        rows.sort(key=lambda r: r[0])
        out[aid] = (rows, [r[0] for r in rows])
    _slug_cache[slug] = out
    return out


def obs_at(series, times, t):
    """last observation with captured_at <= t, subject to FRESH_SEC"""
    i = bisect_right(times, t) - 1
    if i < 0:
        return None
    ts, bid, ask = series[i]
    if t - ts > FRESH_SEC:
        return None
    return bid, ask


def canonical_interval(can, comp):
    """Combine canonical + complementary quotes into one canonical bound (D7).
    Never sums depth. Returns (lo, hi, n_sides) or None if crossed/unusable."""
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
    n_sides = (1 if los else 0) + (1 if his else 0)
    if lo > hi + 1e-9:
        return None                                   # crossed -> QC, fail closed
    return max(0.0, lo), min(1.0, hi), n_sides


def solve_S(X, lows, highs):
    """min eps s.t. lows - eps <= X^T pi <= highs + eps, pi in simplex.
    Returns (S, pi) or (None, None)."""
    m, n = X.shape
    nv = n + 1
    c = np.zeros(nv)
    c[-1] = 1.0
    A_ub, b_ub = [], []
    for j in range(m):
        if lows[j] is not None:
            row = np.zeros(nv)
            row[:n] = -X[j]
            row[-1] = -1.0
            A_ub.append(row)
            b_ub.append(-lows[j])
        if highs[j] is not None:
            row = np.zeros(nv)
            row[:n] = X[j]
            row[-1] = -1.0
            A_ub.append(row)
            b_ub.append(highs[j])
    A_eq = np.zeros((1, nv))
    A_eq[0, :n] = 1.0
    try:
        r = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                    A_eq=A_eq, b_eq=np.array([1.0]),
                    bounds=[(0, 1)] * n + [(0, None)], method="highs")
    except Exception:
        return None, None
    if not r.success:
        return None, None
    return float(r.x[-1]), r.x[:n]


print("=" * 80)
print("GATE G2 — COHERENCE, INFORMATIVENESS, FALSIFIABILITY")
print("=" * 80)
M, mp = load_manifest()
print(f"manifest: {os.path.basename(os.path.dirname(mp))}\n")

summary = []
power_acc = defaultdict(lambda: [0, 0])       # delta -> [detected, trials]

for match in PILOT:
    t0 = time.time()
    contracts, info = build_contracts(M, match)
    if contracts is None:
        print(f"--- {match}: SKIP ({info.get('error')})")
        continue
    reps = state_classes(contracts)
    X = np.array(payoff_matrix(contracts, reps), dtype=float)
    sat = saturation_ok(contracts)

    # in-play window from ESPN (regulation only)
    rows = info["espn_rows"]
    tin = [iso(d.get("captured_at")) for d in rows
           if str(d.get("status")) in ("in", "STATUS_IN_PROGRESS")
           and (d.get("period") or 0) and int(d["period"]) <= 2]
    tin = [t for t in tin if t]
    if len(tin) < 10:
        print(f"--- {match}: SKIP (no in-play window)")
        continue
    t_start, t_end = min(tin), max(tin)

    # quote series, keyed by authoritative asset_id
    series = {}
    mapped = unmapped = 0
    for c in contracts:
        books = load_slug_books(c["slug"])
        for key in ("canon_token", "comp_token"):
            tok = c[key]
            if not tok:
                continue
            got = books.get(tok)
            if got is None:
                # fallback: match on the 16-char on-disk prefix convention
                cands = [a for a in books if a.startswith(tok[:16]) or tok.startswith(a[:16])]
                got = books[cands[0]] if len(cands) == 1 else None
            if got is None:
                unmapped += 1
            else:
                mapped += 1
                series[(c["slug"], key)] = got

    instants = list(np.arange(t_start, t_end, STEP_SEC))
    S_vals, panel = [], []
    for t in instants:
        lows, highs, nsides, avail = [], [], [], 0
        bad = False
        for c in contracts:
            cs, ct = series.get((c["slug"], "canon_token"), ([], []))
            ps, pt = series.get((c["slug"], "comp_token"), ([], []))
            can = obs_at(cs, ct, t) if cs else None
            comp = obs_at(ps, pt, t) if ps else None
            ci = canonical_interval(can, comp)
            if ci is None:
                bad = True
                lows.append(0.0)
                highs.append(1.0)
                nsides.append(0)
                continue
            lo, hi, ns = ci
            lows.append(lo)
            highs.append(hi)
            nsides.append(ns)
            if ns > 0:
                avail += 1
        S, pi = solve_S(X, lows, highs)
        if S is None:
            continue
        widths = [h - l for l, h in zip(lows, highs)]
        S_vals.append(S)
        panel.append({"t": t, "S": S, "avail": avail, "two_sided": sum(1 for n in nsides if n == 2),
                      "median_width": float(np.median(widths)), "crossed": bad,
                      "lows": lows, "highs": highs, "widths": widths, "pi": pi})

    if not panel:
        print(f"--- {match}: SKIP (no solvable instants)")
        continue

    # ── falsifiability: inject incoherence into a COHERENT system that has the
    #    OBSERVED interval geometry, and see whether S_inf detects it ──────────
    usable = [p for p in panel if p["pi"] is not None and p["avail"] >= 10]
    probe_pts = random.sample(usable, min(25, len(usable))) if usable else []
    for p in probe_pts:
        pi0 = np.array(p["pi"], dtype=float)
        if pi0.sum() <= 0:
            continue
        pi0 = pi0 / pi0.sum()
        pstar = X @ pi0                                # coherent price vector
        w = np.array(p["widths"], dtype=float)
        lo0 = np.clip(pstar - w / 2, 0, 1)
        hi0 = np.clip(pstar + w / 2, 0, 1)
        S0, _ = solve_S(X, list(lo0), list(hi0))
        if S0 is None or S0 > TOL:                     # sanity: must be coherent
            continue
        idxs = random.sample(range(len(contracts)), min(N_INJECT_CONTRACTS, len(contracts)))
        for d in DELTAS:
            for j in idxs:
                lo1, hi1 = lo0.copy(), hi0.copy()
                lo1[j] = min(1.0, lo0[j] + d)
                hi1[j] = min(1.0, hi0[j] + d)
                if hi1[j] <= lo1[j]:
                    continue
                S1, _ = solve_S(X, list(lo1), list(hi1))
                if S1 is None:
                    continue
                power_acc[d][1] += 1
                if S1 > TOL:
                    power_acc[d][0] += 1

    Sarr = np.array(S_vals)
    summary.append({
        "match": match, "contracts": len(contracts), "states": len(reps),
        "sat": sat, "instants": len(panel),
        "S_med": float(np.median(Sarr)), "S_p90": float(np.percentile(Sarr, 90)),
        "S_max": float(Sarr.max()), "frac_pos": float((Sarr > TOL).mean()),
        "avail_med": float(np.median([p["avail"] for p in panel])),
        "two_sided_med": float(np.median([p["two_sided"] for p in panel])),
        "width_med": float(np.median([p["median_width"] for p in panel])),
        "rank": int(np.linalg.matrix_rank(X)),
    })
    print(f"--- {match}: {len(contracts)} contracts, {len(reps)} states, "
          f"rank={summary[-1]['rank']}, {len(panel)} instants, "
          f"tokens mapped={mapped}/{mapped+unmapped}, "
          f"S_med={summary[-1]['S_med']:.4f}, {time.time()-t0:.0f}s", flush=True)

print("\n" + "=" * 80)
print("INFORMATIVENESS PANEL (D6)")
print("=" * 80)
print(f"  {'match':<22}{'ctr':>5}{'st':>4}{'rank':>5}{'inst':>6}{'avail':>7}"
      f"{'2side':>7}{'width':>8}{'S_med':>9}{'S_p90':>9}{'%S>0':>7}")
for s in summary:
    print(f"  {s['match']:<22}{s['contracts']:>5}{s['states']:>4}{s['rank']:>5}"
          f"{s['instants']:>6}{s['avail_med']:>7.0f}{s['two_sided_med']:>7.0f}"
          f"{s['width_med']:>8.3f}{s['S_med']:>9.4f}{s['S_p90']:>9.4f}"
          f"{100*s['frac_pos']:>6.0f}%")
print(f"\n  saturation lemma holds for every match: "
      f"{all(s['sat'] for s in summary)}")

print("\n" + "=" * 80)
print("FALSIFIABILITY — detection of injected incoherence")
print("=" * 80)
print(f"  {'injected':>10}{'trials':>9}{'detected':>10}{'power':>9}")
gate = {}
for d in DELTAS:
    det, tot = power_acc[d]
    pw = det / tot if tot else 0.0
    gate[d] = pw
    print(f"  {d*100:>9.0f}c{tot:>9,}{det:>10,}{pw*100:>8.1f}%")

print("\n" + "=" * 80)
ok2 = gate.get(0.02, 0) >= 0.80
print(f"GATE G2 (>=80% power at 2 cents): {'PASS' if ok2 else 'REVIEW'}")
print("=" * 80)

out = "/home/azureuser/projects/wc_coherence/results/g2_pilot_summary.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"summary": summary,
           "power": {str(k): {"detected": v[0], "trials": v[1]} for k, v in power_acc.items()},
           "params": {"FRESH_SEC": FRESH_SEC, "STEP_SEC": STEP_SEC, "TOL": TOL}},
          open(out, "w"), indent=2)
print(f"results -> {out}")
