#!/usr/bin/env python3
"""Freshness sensitivity and cross-goal straddle guard.

The referee objection: FRESH=30s exceeds the post-goal window [+6,+30]s, and the
interval is an ENVELOPE spanning the observations bracketing t on each side. At
10-second polling a post-goal instant can therefore be bracketed on its left by a
PRE-goal observation, so pre-goal price levels enter a post-goal measurement.
Heterogeneity in which contracts are in that state inflates S for reasons of
measurement rather than market behaviour, and the monotonicity lemma does not
cover it because it is not a componentwise widening.

The leak is symmetric and the manuscript should say so: at t = -6s the RIGHT
bracketing observation can be post-goal, pushing post-goal levels into the pre
window. That direction inflates the pre-goal baseline and therefore SHRINKS the
estimate, so the two leaks do not both flatter the result.

Variants:
  base          FRESH=30, no guard                  must reproduce the engine
  fresh20/10    shorter staleness tolerance         quotes drop out, avail falls
  no-straddle   both bracketing observations required to lie on the same side of
                the goal as the instant being evaluated; a contract that straddles
                is treated as ABSENT, i.e. handed [0,1], which is unconstraining
                and lowers S rather than inventing a value for it
  no-straddle+fresh10   both at once

An instant is still only analysed when >= MIN_AVAIL contracts carry a bound, so
guarding can drop instants entirely; the retention rate is reported because a
surviving effect on a decimated sample is not the same claim.

Run: /home/azureuser/.venv/bin/python -u freshness_and_straddle.py --cohort holdout
"""
import argparse
import json
import os
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402
import event_study as ES                               # noqa: E402
from compiler import build_contracts, load_manifest, state_classes  # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
P = ES.P
G, WIN = P["GRID"], P["WIN"]
C = 100.0
np.random.seed(P["SEED"])

# (label, freshness seconds, guard against straddling the goal)
VARIANTS = [("base", 30.0, False), ("fresh20", 20.0, False),
            ("fresh10", 10.0, False), ("no-straddle", 30.0, True),
            ("no-straddle+fresh10", 10.0, True)]


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)


def quote_env(series, times, t, fresh, tmin, tmax):
    """ES.token_quote_env with an explicit freshness and an optional requirement
    that BOTH bracketing observations lie within [tmin, tmax]."""
    i = bisect_right(times, t) - 1
    if i < 0:
        return None
    pt, pb, pa = series[i]
    if t - pt > fresh:
        return None
    if tmin is not None and pt < tmin:
        return None
    if i + 1 < len(series):
        nt, nb, na = series[i + 1]
        if tmax is not None and nt > tmax:
            return None
        bid = min([v for v in (pb, nb) if v is not None], default=None)
        ask = max([v for v in (pa, na) if v is not None], default=None)
        return bid, ask
    return pb, pa


def run(cohort):
    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    matches = sorted(split[cohort])
    M, _ = load_manifest()
    log(f"FRESHNESS + STRADDLE — cohort {cohort} ({len(matches)} matches)")

    dv = {v[0]: defaultdict(list) for v in VARIANTS}
    seen = {v[0]: [0, 0] for v in VARIANTS}   # [instants analysed, instants offered]
    t0 = time.time()

    for mi, match in enumerate(matches, 1):
        contracts, info = build_contracts(M, match)
        if contracts is None:
            continue
        reps = state_classes(contracts)
        X = np.array([[c["rule"](h, a) for (h, a) in reps] for c in contracts],
                     dtype=float)
        ser = {}
        for c in contracts:
            books = ES.load_slug_books(c["slug"])
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

        esp = info["espn_rows"]
        tin = [ES.iso(d.get("captured_at")) for d in esp
               if str(d.get("status")) in ("in", "STATUS_IN_PROGRESS")
               and (d.get("period") or 0) and int(d["period"]) <= 2]
        tin = [t for t in tin if t]
        if len(tin) < 20:
            continue
        t_start, t_end = min(tin), max(tin)
        goals = [g for g in ES.goals_of(info["game_id"]) if t_start <= g["w"] <= t_end]
        gw = sorted(g["w"] for g in goals)
        gt = np.arange(t_start, t_end, G)

        for g in goals:
            w = g["w"]
            if w - WIN < t_start or w + WIN > t_end:
                continue
            if not all(abs(w - o) > P["ISO_GAP"] for o in gw if o != w):
                continue
            idx = {}
            for tag, lo, hi in (("pre", P["PRE_LO"], P["PRE_HI"]),
                                ("post", P["POST_LO"], P["POST_HI"])):
                i0 = int(np.searchsorted(gt, w + lo))
                i1 = int(np.searchsorted(gt, w + hi))
                idx[tag] = [i for i in range(i0, i1 + 1) if 0 <= i < len(gt)]

            vals = {vn: {"pre": [], "post": []} for vn, _, _ in VARIANTS}
            for vn, fresh, guard in VARIANTS:
                for tag in ("pre", "post"):
                    # a pre-goal instant may not look past the goal, and a
                    # post-goal instant may not look back before it
                    tmin = w if (guard and tag == "post") else None
                    tmax = w if (guard and tag == "pre") else None
                    for i in idx[tag]:
                        t = gt[i]
                        seen[vn][1] += 1
                        lows, highs, avail = [], [], 0
                        for c in contracts:
                            cs = ser.get((c["slug"], "canon_token"))
                            ps = ser.get((c["slug"], "comp_token"))
                            can = quote_env(cs[0], cs[1], t, fresh, tmin, tmax) if cs else None
                            comp = quote_env(ps[0], ps[1], t, fresh, tmin, tmax) if ps else None
                            ival = ES.canonical_interval(can, comp)
                            if ival is None:
                                lows.append(0.0)
                                highs.append(1.0)
                                continue
                            lo_, hi_, ns = ival
                            lows.append(lo_)
                            highs.append(hi_)
                            if ns > 0:
                                avail += 1
                        if avail < P["MIN_AVAIL"]:
                            continue
                        s = ES.solve_S(X, lows, highs)
                        if s is None:
                            continue
                        seen[vn][0] += 1
                        vals[vn][tag].append(s)
            for vn, _, _ in VARIANTS:
                if vals[vn]["pre"] and vals[vn]["post"]:
                    dv[vn][match].append(float(np.mean(vals[vn]["post"])
                                               - np.mean(vals[vn]["pre"])))
        if mi % 10 == 0:
            log(f"  {mi}/{len(matches)} matches, {time.time()-t0:.0f}s")

    def boot(pm):
        ms = sorted(pm)
        v = np.array([float(np.mean(pm[m])) for m in ms])
        if len(v) < 3:
            return None
        b = np.array([np.mean(np.random.choice(v, len(v), replace=True))
                      for _ in range(P["N_BOOT"])])
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {"D": float(v.mean()), "ci": [float(lo), float(hi)],
                "n_matches": len(ms), "n_events": sum(len(x) for x in pm.values()),
                "n_pos": int((v > 0).sum())}

    res = {vn: boot(dv[vn]) for vn, _, _ in VARIANTS}
    log("")
    log("=" * 104)
    log(f"FRESHNESS + STRADDLE — {cohort}")
    log("=" * 104)
    log(f"  {'variant':<22}{'n_ev':>6}{'D (cents)':>12}{'95% CI (cents)':>24}"
        f"{'pos/mat':>10}{'instants kept':>15}{'retained':>10}")
    base_keep = seen["base"][0]
    for vn, _, _ in VARIANTS:
        r = res[vn]
        if not r:
            log(f"  {vn:<22}{'insufficient':>18}")
            continue
        ci = f"[{C*r['ci'][0]:+.3f}, {C*r['ci'][1]:+.3f}]"
        keep = seen[vn][0]
        pos_txt = f"{r['n_pos']}/{r['n_matches']}"
        pct = 100.0 * keep / base_keep if base_keep else 0.0
        log(f"  {vn:<22}{r['n_events']:>6}{C*r['D']:>+12.3f}{ci:>24}"
            f"{pos_txt:>10}{keep:>15,}{pct:>9.1f}%")

    eng = json.load(open(f"{OUT}/results/{cohort}/event_study.json"))
    want, got = eng["isolated"]["D"], res["base"]["D"]
    log("")
    log(f"  baseline check: harness {C*got:+.4f}c vs frozen engine {C*want:+.4f}c")
    if abs(got - want) > 2e-5:
        log("  BASELINE MISMATCH — harness has drifted from the engine; results void")
        sys.exit(1)
    log("  baseline reproduces the frozen engine")

    payload = {"cohort": cohort, "generated_at": datetime.now(timezone.utc).isoformat(),
               "variants": {vn: res[vn] for vn, _, _ in VARIANTS},
               "instants_analysed": {vn: seen[vn][0] for vn, _, _ in VARIANTS},
               "instants_offered": {vn: seen[vn][1] for vn, _, _ in VARIANTS}}
    dst = f"{OUT}/results/freshness_straddle_{cohort}.json"
    with open(dst, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["dev", "holdout"])
    run(ap.parse_args().cohort)
