#!/usr/bin/env python3
"""Shared event-study engine + frozen parameters.

Both the development analysis and the one-shot holdout run call `run_cohort()`
in THIS file, so the two cannot drift apart. Parameters below were chosen on
DEVELOPMENT data and are frozen: they must not be edited after holdout results
have been seen.

Usage:
  python event_study.py --cohort dev
  python event_study.py --cohort holdout --unlock-holdout
"""
import argparse
import glob
import gzip
import json
import lzma
import os
import random
import subprocess
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402
from scipy.optimize import linprog                     # noqa: E402
from compiler import (LIVE, build_contracts, load_manifest,               # noqa: E402
                      payoff_matrix, payoff_matrix_with_cancel, state_classes)

# ─────────────── FROZEN PARAMETERS (chosen on development data) ─────────────
P = {
    "GRID": 5,
    "PRE_LO": -60, "PRE_HI": -6,
    "POST_LO": 6, "POST_HI": 30,
    "WIN": 300,
    "FRESH": 30.0,
    "TOL": 1e-4,
    "ISO_GAP": 300,
    "PLACEBO_GAP": 240,
    "N_BOOT": 5000,
    "N_RAND": 5000,
    "MIN_AVAIL": 10,
    "SEED": 29,
}
THRESHOLDS = [("certified", P["TOL"]), ("0.10c", 0.0010), ("0.25c", 0.0025),
              ("0.50c", 0.0050), ("1.00c", 0.0100)]
OUT = "/home/azureuser/projects/wc_coherence"


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
    i = bisect_right(times, t) - 1
    if i < 0:
        return None
    pt, pb, pa = series[i]
    if t - pt > P["FRESH"]:
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
                    out.append({"id": eid, "w": w, "period": per,
                                "clock": (e.get("clock") or {}).get("displayValue")})
            except Exception:
                continue
        return sorted(out, key=lambda z: z["w"])
    return []


def run_cohort(matches, label, cancel_state=False):
    """cancel_state=False reproduces the frozen pre-holdout behaviour exactly.
    cancel_state=True appends the verified match-cancelled payoff state, which
    enlarges the coherent set and can only reduce measured slack (sensitivity)."""
    random.seed(P["SEED"])
    np.random.seed(P["SEED"])
    M, _ = load_manifest()
    match_stats, long_rows = {}, []
    G, WIN = P["GRID"], P["WIN"]

    for mi, match in enumerate(matches, 1):
        t0 = time.time()
        contracts, info = build_contracts(M, match)
        if contracts is None:
            continue
        reps = state_classes(contracts)
        if cancel_state:
            X = np.array(payoff_matrix_with_cancel(contracts, reps), dtype=float)
            totals = np.array([h + a for (h, a) in reps] + [10 ** 6])
        else:
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
        goals = [g for g in goals_of(info["game_id"])
                 if t_start - 60 <= g["w"] <= t_end + 60]

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

        gt = np.arange(t_start, t_end, G)
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
            if avail < P["MIN_AVAIL"]:
                continue
            v = solve_S(X, lows, highs)
            if v is not None:
                Sc[i] = v
            nb = sum(1 for w in gw if w <= t)
            v2 = solve_S(X, lows, highs, mask=(totals >= nb))
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
            isolated = all(abs(w - o) > P["ISO_GAP"] for o in gw if o != w)
            pre_c = win_mean(Sc, w, P["PRE_LO"], P["PRE_HI"])
            post_c = win_mean(Sc, w, P["POST_LO"], P["POST_HI"])
            pre_s = win_mean(Ss, w, P["PRE_LO"], P["PRE_HI"])
            post_s = win_mean(Ss, w, P["POST_LO"], P["POST_HI"])
            if pre_c is None or post_c is None:
                continue
            evs.append({"id": g["id"], "w": w, "clock": g["clock"],
                        "isolated": isolated, "d_cross": post_c - pre_c,
                        "pre_cross": pre_c, "post_cross": post_c,
                        "d_state": (post_s - pre_s)
                        if (pre_s is not None and post_s is not None) else None})
            for off in range(-WIN, WIN + 1, G):
                i = int(np.searchsorted(gt, w + off))
                if 0 <= i < len(gt):
                    long_rows.append({"match": match, "event": g["id"], "offset": off,
                                      "S_cross": None if np.isnan(Sc[i]) else float(Sc[i]),
                                      "S_state": None if np.isnan(Ss[i]) else float(Ss[i]),
                                      "avail": int(Av[i]), "isolated": isolated})

        elig = [i for i, t in enumerate(gt)
                if t - WIN >= t_start and t + WIN <= t_end
                and all(abs(t - w) > P["PLACEBO_GAP"] for w in gw)]
        match_stats[match] = {"n_goals_total": len(goals), "events": evs,
                              "elig": elig, "Sc": Sc.tolist(), "n_grid": len(gt),
                              # absolute-time anchors so the placebo suite, window
                              # respecifications and the specification curve can be
                              # recomputed offline with the SAME searchsorted-on-
                              # absolute-time semantics the engine uses here.
                              "grid_t0": float(gt[0]), "grid_step": G,
                              "t_start": float(t_start), "t_end": float(t_end),
                              "goal_w": [float(w) for w in gw],
                              "game_id": str(info["game_id"])}
        print(f"[{mi:>2}/{len(matches)}] {match}: goals={len(goals)} "
              f"events={len(evs)} ({time.time()-t0:.0f}s)", flush=True)

    # ── estimands ───────────────────────────────────────────────────────────
    def summarize(isolated_only):
        me = {}
        for m, st in match_stats.items():
            ds = [e["d_cross"] for e in st["events"] if e["isolated"] or not isolated_only]
            if ds:
                me[m] = float(np.mean(ds))
        if not me:
            return None
        ms = sorted(me)
        vals = np.array([me[m] for m in ms])
        D = float(vals.mean())
        boot = np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                         for _ in range(P["N_BOOT"])])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        loo = [float(np.mean(np.delete(vals, i))) for i in range(len(vals))]
        n_ev = sum(1 for m in ms for e in match_stats[m]["events"]
                   if e["isolated"] or not isolated_only)
        return {"D": D, "ci": [float(lo), float(hi)], "n_matches": len(ms),
                "n_events": n_ev, "median": float(np.median(vals)),
                "loo_min": min(loo), "loo_max": max(loo),
                "n_pos": int((vals > 0).sum())}

    res_all = summarize(False)
    res_iso = summarize(True)

    # randomization inference
    # AMENDMENT (post-freeze, 2026-07-25): the original null drew from matches
    # with >5 eligible anchors and fell back to k=len(events) when a match had
    # no isolated event, so the null averaged over a DIFFERENT match population
    # than the observed statistic. The null now uses exactly the match set and
    # the per-match event counts that enter res_iso. The p-value also uses the
    # conservative (1+#{null>=obs})/(1+N) estimator instead of the raw mean, so
    # it can never be reported as exactly zero. Both changes are strictly more
    # conservative and neither alters the test statistic or the design.
    obs = res_iso["D"] if res_iso else None
    p_rand, rand_diag = None, None
    if obs is not None:
        obs_k = {}
        for m, st in match_stats.items():
            k = sum(1 for e in st["events"] if e["isolated"])
            if k:
                obs_k[m] = k
        usable = [m for m in sorted(obs_k) if len(match_stats[m]["elig"]) >= obs_k[m]]
        dropped = sorted(set(obs_k) - set(usable))
        null = []
        for _ in range(P["N_RAND"]):
            pm = []
            for m in usable:
                st = match_stats[m]
                Sc = np.array(st["Sc"])
                n = st["n_grid"]
                ds = []
                for i in random.sample(st["elig"], obs_k[m]):
                    a0, a1 = i + P["PRE_LO"] // G, i + P["PRE_HI"] // G
                    b0, b1 = i + P["POST_LO"] // G, i + P["POST_HI"] // G
                    if a0 < 0 or b1 >= n:
                        continue
                    pre, post = Sc[a0:a1 + 1], Sc[b0:b1 + 1]
                    pre, post = pre[~np.isnan(pre)], post[~np.isnan(post)]
                    if len(pre) and len(post):
                        ds.append(post.mean() - pre.mean())
                if ds:
                    pm.append(np.mean(ds))
            if pm:
                null.append(np.mean(pm))
        null = np.array(null)
        n_exceed = int((null >= obs).sum())
        p_rand = (1.0 + n_exceed) / (1.0 + len(null))
        rand_diag = {
            "estimator": "(1 + #{null >= obs}) / (1 + n_null)",
            "observed": float(obs), "n_null": int(len(null)), "n_exceed": n_exceed,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)) if len(null) > 1 else None,
            "null_quantiles": {q: float(np.percentile(null, q))
                               for q in (50, 90, 95, 99, 99.9)},
            "null_max": float(null.max()), "seed": P["SEED"], "n_rand": P["N_RAND"],
            "matches_observed": sorted(obs_k), "matches_in_null": usable,
            "matches_dropped_from_null": dropped, "per_match_k": obs_k,
            "n_matches_observed": len(obs_k), "n_matches_null": len(usable)}

    thr_out = []
    for name, thr in THRESHOLDS:
        pre_v, post_v = [], []
        for st in match_stats.values():
            for e in st["events"]:
                if not e["isolated"]:
                    continue
                pre_v.append(e["pre_cross"] > thr)
                post_v.append(e["post_cross"] > thr)
        if pre_v:
            thr_out.append({"threshold": name, "pre": 100 * float(np.mean(pre_v)),
                            "post": 100 * float(np.mean(post_v)),
                            "delta": 100 * (float(np.mean(post_v)) - float(np.mean(pre_v)))})

    ds_state = [e["d_state"] for st in match_stats.values() for e in st["events"]
                if e["isolated"] and e["d_state"] is not None]

    try:
        commit = subprocess.check_output(
            ["git", "-C", "/home/azureuser/polymarket", "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        commit = None

    out = {"cohort": label, "params": P, "git_commit": commit,
           "generated_at": datetime.utcnow().isoformat() + "Z",
           "all": res_all, "isolated": res_iso, "rand_p": p_rand,
           "rand_diag": rand_diag, "thresholds": thr_out,
           "d_state_mean": float(np.mean(ds_state)) if ds_state else None,
           "n_state_events": len(ds_state),
           "per_match": {m: {"n_goals": st["n_goals_total"],
                             "d": [e["d_cross"] for e in st["events"]],
                             "isolated": [e["isolated"] for e in st["events"]]}
                         for m, st in match_stats.items()}}

    d = f"{OUT}/results/{label}"
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(f"{d}/event_study.json", "w"), indent=2)
    with open(f"{d}/long_form.jsonl", "w") as f:
        for r in long_rows:
            f.write(json.dumps(r) + "\n")
    # per-match coherence series + eligible placebo anchors, so the randomization
    # test and any window respecification can be redone without recomputing S.
    with gzip.open(f"{d}/match_stats.json.gz", "wt") as f:
        json.dump(match_stats, f)

    print("\n" + "=" * 92)
    print(f"COHORT: {label.upper()}   (git {commit[:8] if commit else '?'})")
    print("=" * 92)
    for tag, r in (("ALL eligible goals", res_all), ("ISOLATED (primary)", res_iso)):
        if not r:
            continue
        print(f"\n{tag}: matches={r['n_matches']} events={r['n_events']}")
        print(f"  D = {r['D']*100:+.3f} cents   95% CI [{r['ci'][0]*100:+.3f}, "
              f"{r['ci'][1]*100:+.3f}]")
        print(f"  median {r['median']*100:+.3f}c | LOO [{r['loo_min']*100:+.3f}, "
              f"{r['loo_max']*100:+.3f}]c | positive {r['n_pos']}/{r['n_matches']}")
    if rand_diag:
        # S is carried in probability units; convert to cents for display, as the
        # D print does. Authoritative randomization inference is produced by
        # randomization_test.py, which slices the null with the same searchsorted
        # window as the observed statistic rather than floor-divided offsets.
        print(f"\nrandomization p = {p_rand:.5f}   "
              f"[{rand_diag['n_exceed']} of {rand_diag['n_null']} draws >= observed; "
              f"(1+B)/(1+N) estimator]")
        print(f"  null mean {100*rand_diag['null_mean']:+.4f}c  "
              f"sd {100*rand_diag['null_sd']:.4f}c  "
              f"max {100*rand_diag['null_max']:+.4f}c  "
              f"99.9th pct {100*rand_diag['null_quantiles'][99.9]:+.4f}c")
        print(f"  matches: observed {rand_diag['n_matches_observed']}, "
              f"null {rand_diag['n_matches_null']}"
              + (f", DROPPED {rand_diag['matches_dropped_from_null']}"
                 if rand_diag['matches_dropped_from_null'] else " (aligned)"))
    else:
        print(f"\nrandomization p = {p_rand}")
    print(f"state-conditioned mean = "
          f"{(np.mean(ds_state)*100 if ds_state else float('nan')):+.3f} cents")
    print(f"\n  {'threshold':<12}{'pre %':>9}{'post %':>9}{'delta pp':>11}")
    for t in thr_out:
        print(f"  {t['threshold']:<12}{t['pre']:>8.1f}%{t['post']:>8.1f}%{t['delta']:>10.1f}pp")
    print(f"\nresults -> {d}/event_study.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["dev", "holdout"], required=True)
    ap.add_argument("--unlock-holdout", action="store_true")
    ap.add_argument("--cancel-state", action="store_true",
                    help="sensitivity: add the verified match-cancelled payoff state")
    a = ap.parse_args()
    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    if a.cohort == "holdout" and not a.unlock_holdout:
        sys.exit("REFUSED: holdout requires --unlock-holdout (one-shot resource).")
    lbl = a.cohort + ("_cancel" if a.cancel_state else "")
    run_cohort(split["dev"] if a.cohort == "dev" else split["holdout"], lbl,
               cancel_state=a.cancel_state)
