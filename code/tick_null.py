#!/usr/bin/env python3
"""Tick-discretization null.

The referee objection: the headline effect is +0.436c against a median tick of
1 cent. A genuinely coherent belief vector, quoted with real spreads and rounded
onto a 1-cent lattice, will generically require SOME nonzero relaxation. The low
pre-goal baseline is evidence against this mattering, but it is not the
experiment, because post-goal spreads are wider and rounding behaviour under
wider spreads need not match rounding under tighter ones.

Design. The cleanest version holds everything at its observed value except the
one thing under test. At each analysed instant we take the observed canonical
intervals, replace their CENTRES with a coherent price vector, keep the observed
half-widths and each contract's active tick, re-round onto the lattice, and
recompute S.

  1. observed interval [lo_i, hi_i] for every contract carrying a real bound
  2. mid_i = (lo_i + hi_i)/2, halfwidth_i = (hi_i - lo_i)/2
  3. project the mids onto the marginal polytope: choose q on the simplex
     minimising max_i |(Xq)_i - mid_i|. The minimiser p* = Xq is coherent BY
     CONSTRUCTION and is the closest coherent vector to what the market showed,
     so the synthetic book resembles the real one rather than a random one.
    4. synthetic interval = p*_i +/- halfwidth_i, rounded to the active tick
  5. S on the synthetic book

Any S it finds is produced purely by rounding and spread geometry, because the
underlying belief is exactly coherent. Two rounding rules are reported:

  nearest    lo and hi each rounded to the closest tick -- what a book looks like
    narrowest  lo rounded UP and hi rounded DOWN -- the adversarial case. If the
                         interval contains no lattice point, both bounds use the nearest
                         lattice point to the coherent price.

If the null's pre-to-post change is near zero while the observed change is
+0.436c, discretization cannot account for the result. Note the null is expected
to be LOWER post-goal, since observed spreads widen after goals and S is monotone
in width (4.5); that the observed effect runs the other way is the point.

Run: /home/azureuser/.venv/bin/python -u tick_null.py --cohort holdout
     /home/azureuser/.venv/bin/python -u tick_null.py --cohort holdout \
         --subsystem no-exact-score
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402
from scipy.optimize import linprog                     # noqa: E402
import event_study as ES                               # noqa: E402
from compiler import build_contracts, load_manifest, state_classes  # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
P = ES.P
G, WIN = P["GRID"], P["WIN"]
C = 100.0
SUBSYSTEMS = {
    "full": None,
    "no-exact-score": {"moneyline", "btts", "total-goals", "team-total", "spread"},
}
MIN_SUBSYSTEM_QUOTED = 3
np.random.seed(P["SEED"])


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)


def project_to_polytope(X, targets, mask):
    """q on the simplex minimising max_i |(Xq)_i - targets_i| over masked rows.
    Returns the coherent price vector Xq over ALL rows, or None."""
    rows = np.where(mask)[0]
    if len(rows) == 0:
        return None
    Xm = X[rows]
    m, n = Xm.shape
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A = np.zeros((2 * m, n + 1))
    b = np.zeros(2 * m)
    A[:m, :-1] = -Xm
    A[:m, -1] = -1.0
    b[:m] = -np.asarray(targets)[rows]
    A[m:, :-1] = Xm
    A[m:, -1] = -1.0
    b[m:] = np.asarray(targets)[rows]
    Aeq = np.zeros((1, n + 1))
    Aeq[0, :-1] = 1.0
    try:
        r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=np.array([1.0]),
                    bounds=[(0, 1)] * n + [(0, None)], method="highs")
    except Exception:
        return None
    if not r.success:
        return None
    return X @ np.asarray(r.x[:-1])


def load_slug_books_with_ticks(slug):
    """Load quote and tick histories in one pass over a contract's raw books."""
    quotes = defaultdict(list)
    ticks = defaultdict(list)
    for path in glob.glob(f"{ES.LIVE}/*/*/{slug}/*.jsonl.gz"):
        try:
            with gzip.open(path, "rt", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    captured = ES.iso(record.get("captured_at"))
                    book = record.get("book")
                    if captured is None or not isinstance(book, dict):
                        continue
                    asset_id = str(book.get("asset_id") or record.get("token_id") or "")
                    if not asset_id:
                        continue
                    bid, ask = ES.best_bid_ask(book)
                    quotes[asset_id].append((captured, bid, ask))
                    try:
                        tick = float(book.get("tick_size"))
                    except (TypeError, ValueError):
                        continue
                    if tick > 0:
                        ticks[asset_id].append((captured, tick))
        except Exception:
            continue

    quote_series = {}
    for asset_id, rows in quotes.items():
        rows.sort(key=lambda row: row[0])
        quote_series[asset_id] = (rows, [row[0] for row in rows])
    tick_series = {}
    for asset_id, rows in ticks.items():
        rows.sort(key=lambda row: row[0])
        tick_series[asset_id] = (rows, [row[0] for row in rows])
    return quote_series, tick_series


def value_at(series, times, instant):
    """Return the last observed tick state at or before an instant."""
    index = bisect_right(times, instant) - 1
    return None if index < 0 else series[index][1]


def token_stream(mapping, token):
    """Resolve a token id with the engine's unique-prefix fallback."""
    found = mapping.get(token)
    if found is not None:
        return found
    candidates = [asset_id for asset_id in mapping if asset_id.startswith(token[:16])]
    return mapping[candidates[0]] if len(candidates) == 1 else None


def run(cohort, subsystem):
    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    matches = sorted(split[cohort])
    M, _ = load_manifest()
    log(f"TICK NULL — cohort {cohort}, subsystem {subsystem} "
        f"({len(matches)} matches)")

    d_obs = defaultdict(list)
    d_near = defaultdict(list)
    d_narrow = defaultdict(list)
    lvl = {"obs": {"pre": [], "post": []}, "nearest": {"pre": [], "post": []},
           "narrowest": {"pre": [], "post": []}}
    tick_counts = defaultdict(int)
    n_tick_mismatch = 0
    n_inward_fallback = 0
    t0 = time.time()

    for mi, match in enumerate(matches, 1):
        contracts, info = build_contracts(M, match)
        if contracts is None:
            continue
        keep_families = SUBSYSTEMS[subsystem]
        keep_mask = np.array([
            keep_families is None or c["family"] in keep_families
            for c in contracts
        ], dtype=bool)
        selected_contracts = [c for c, keep in zip(contracts, keep_mask) if keep]
        reps = state_classes(selected_contracts)
        X = np.array([[c["rule"](h, a) for (h, a) in reps]
                      for c in selected_contracts],
                     dtype=float)
        ser = {}
        for c in contracts:
            books, ticks = load_slug_books_with_ticks(c["slug"])
            for key in ("canon_token", "comp_token"):
                tok = c[key]
                if not tok:
                    continue
                quote_stream = token_stream(books, tok)
                tick_stream = token_stream(ticks, tok)
                if quote_stream:
                    ser[(c["slug"], key)] = (quote_stream, tick_stream)

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
            acc = {k: {"pre": [], "post": []} for k in ("obs", "nearest", "narrowest")}

            for tag in ("pre", "post"):
                for i in idx[tag]:
                    t = gt[i]
                    lows, highs = [], []
                    active_ticks = np.full(len(contracts), np.nan)
                    bounded = np.zeros(len(contracts), dtype=bool)
                    for ci_, c in enumerate(contracts):
                        cs = ser.get((c["slug"], "canon_token"))
                        ps = ser.get((c["slug"], "comp_token"))
                        can = ES.token_quote_env(cs[0][0], cs[0][1], t) if cs else None
                        comp = ES.token_quote_env(ps[0][0], ps[0][1], t) if ps else None
                        ival = ES.canonical_interval(can, comp)
                        if ival is None:
                            lows.append(0.0)
                            highs.append(1.0)
                            continue
                        lo_, hi_, ns = ival
                        lows.append(lo_)
                        highs.append(hi_)
                        if ns > 0:
                            bounded[ci_] = True
                            if not keep_mask[ci_]:
                                continue
                            observed_ticks = []
                            for stream in (cs, ps):
                                if stream and stream[1]:
                                    tick = value_at(stream[1][0], stream[1][1], t)
                                    if tick is not None:
                                        observed_ticks.append(float(tick))
                            if not observed_ticks:
                                raise RuntimeError(
                                    f"missing active tick for {match} {c['slug']} at {t}")
                            if max(observed_ticks) - min(observed_ticks) > 1e-12:
                                n_tick_mismatch += 1
                            active_ticks[ci_] = min(observed_ticks)
                            tick_counts[f"{active_ticks[ci_]:.6g}"] += 1
                    if int(bounded.sum()) < P["MIN_AVAIL"]:
                        continue
                    bounded = bounded[keep_mask]
                    if subsystem != "full" and int(bounded.sum()) < MIN_SUBSYSTEM_QUOTED:
                        continue
                    lows = np.asarray(lows)[keep_mask]
                    highs = np.asarray(highs)[keep_mask]
                    active_ticks = active_ticks[keep_mask]
                    s_obs = ES.solve_S(X, list(lows), list(highs))
                    if s_obs is None:
                        continue

                    mid = 0.5 * (lows + highs)
                    half = 0.5 * (highs - lows)
                    pstar = project_to_polytope(X, mid, bounded)
                    if pstar is None:
                        continue
                    acc["obs"][tag].append(s_obs)

                    for rule in ("nearest", "narrowest"):
                        nl = np.zeros_like(lows)
                        nh = np.ones_like(highs)
                        raw_lo = pstar - half
                        raw_hi = pstar + half
                        row_ticks = active_ticks[bounded]
                        if rule == "nearest":
                            q_lo = np.round(raw_lo[bounded] / row_ticks) * row_ticks
                            q_hi = np.round(raw_hi[bounded] / row_ticks) * row_ticks
                        else:
                            q_lo = np.ceil(raw_lo[bounded] / row_ticks - 1e-9) * row_ticks
                            q_hi = np.floor(raw_hi[bounded] / row_ticks + 1e-9) * row_ticks
                        q_lo = np.clip(q_lo, 0.0, 1.0)
                        q_hi = np.clip(q_hi, 0.0, 1.0)
                        # If an interval contains no lattice point, use the
                        # nearest lattice point to the coherent price. Never
                        # manufacture an off-lattice midpoint.
                        bad = q_lo > q_hi
                        if bad.any():
                            fallback = np.round(pstar[bounded] / row_ticks) * row_ticks
                            fallback = np.clip(fallback, 0.0, 1.0)
                            q_lo = np.where(bad, fallback, q_lo)
                            q_hi = np.where(bad, fallback, q_hi)
                            if rule == "narrowest":
                                n_inward_fallback += int(bad.sum())
                        nl[bounded] = q_lo
                        nh[bounded] = q_hi
                        s = ES.solve_S(X, list(nl), list(nh))
                        if s is not None:
                            acc[rule][tag].append(s)

            for key, store in (("obs", d_obs), ("nearest", d_near),
                               ("narrowest", d_narrow)):
                if acc[key]["pre"] and acc[key]["post"]:
                    store[match].append(float(np.mean(acc[key]["post"])
                                              - np.mean(acc[key]["pre"])))
                    lvl[key]["pre"].extend(acc[key]["pre"])
                    lvl[key]["post"].extend(acc[key]["post"])
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

    res = {"observed": boot(d_obs), "nearest": boot(d_near),
           "narrowest": boot(d_narrow)}
    log("")
    log("=" * 100)
    log(f"TICK-DISCRETIZATION NULL — {cohort}, subsystem {subsystem}")
    log("=" * 100)
    log(f"  {'series':<14}{'pre level':>12}{'post level':>12}{'D (cents)':>12}"
        f"{'95% CI (cents)':>24}{'pos/mat':>10}")
    for key, lab in (("obs", "observed"), ("nearest", "null nearest"),
                     ("narrowest", "null narrowest")):
        r = res["observed" if key == "obs" else key]
        if not r:
            continue
        pre = C * float(np.mean(lvl[key]["pre"])) if lvl[key]["pre"] else float("nan")
        post = C * float(np.mean(lvl[key]["post"])) if lvl[key]["post"] else float("nan")
        ci = f"[{C*r['ci'][0]:+.3f}, {C*r['ci'][1]:+.3f}]"
        pos_txt = f"{r['n_pos']}/{r['n_matches']}"
        log(f"  {lab:<14}{pre:>11.3f}c{post:>11.3f}c{C*r['D']:>+12.3f}{ci:>24}"
            f"{pos_txt:>10}")

    if subsystem == "full":
        baseline_path = f"{OUT}/results/{cohort}/event_study.json"
        baseline_key = "event_study.isolated"
        baseline = json.load(open(baseline_path))
        want = baseline["isolated"]["D"]
    else:
        baseline_path = f"{OUT}/results/subsystem_availability_{cohort}.json"
        baseline_key = f"subsystems.{subsystem}"
        baseline = json.load(open(baseline_path))
        want = baseline["subsystems"][subsystem]["D"]
    got = res["observed"]["D"]
    log("")
    log(f"  baseline check ({baseline_key}): harness {C*got:+.4f}c "
        f"vs persisted {C*want:+.4f}c")
    if abs(got - want) > 2e-5:
        log("  BASELINE MISMATCH — harness has drifted; results void")
        sys.exit(1)
    log("  baseline reproduces the persisted estimate")
    log(f"  active tick evaluations: {dict(sorted(tick_counts.items()))}")
    log(f"  complementary tick mismatches: {n_tick_mismatch}")
    log(f"  inward intervals with no lattice point: {n_inward_fallback}")

    payload = {"cohort": cohort,
               "subsystem": subsystem,
               "families": sorted(SUBSYSTEMS[subsystem])
               if SUBSYSTEMS[subsystem] is not None else "all",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "results": res,
               "tick_counts": dict(sorted(tick_counts.items())),
               "n_tick_mismatch": n_tick_mismatch,
               "n_inward_fallback": n_inward_fallback,
               "levels": {k: {"pre_mean": float(np.mean(v["pre"])) if v["pre"] else None,
                              "post_mean": float(np.mean(v["post"])) if v["post"] else None,
                              "n_pre": len(v["pre"]), "n_post": len(v["post"])}
                          for k, v in lvl.items()}}
    suffix = "" if subsystem == "full" else "_no_exact_score"
    dst = f"{OUT}/results/tick_null{suffix}_{cohort}.json"
    with open(dst, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["dev", "holdout"])
    ap.add_argument("--subsystem", choices=sorted(SUBSYSTEMS), default="full")
    args = ap.parse_args()
    run(args.cohort, args.subsystem)
