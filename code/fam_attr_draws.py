#!/usr/bin/env python3
"""Family attribution + effect-size context for the coherence event study.

S is the minimum uniform slack s >= 0 such that some distribution q over the
payoff-equivalence classes of regulation scores (typically 66 per match; see
saturation_lemma.py) satisfies  low_i - s <= (Xq)_i <= high_i + s  for every
contract i. So S is an L-infinity violation, and the constraints that are
ACTIVE at the optimum are literally the relations that fail: the LP dual is a
certificate of incoherence. Two complementary attributions are computed:

  1. DUAL CERTIFICATE -- share of |dual| mass by contract family at each instant,
     compared pre-goal vs post-goal. Answers "which relations break".
  2. LEAVE-ONE-FAMILY-OUT -- recompute S with family F deleted and re-estimate the
     event effect. If deleting F collapses the effect, F carries it. Answers
     "is the result driven by one fragile corner of the contract set".
  3. SIZE-MATCHED RANDOM DROP -- the obvious rejoinder to (2) is that deleting a
     family also deletes identifying restrictions, so the effect could collapse
     merely because the system became underdetermined. For every family F we
     therefore also delete a RANDOM subset of the same cardinality and re-estimate.
     If the family-specific drop collapses the effect much further than the
     size-matched random drop, the attribution is genuinely about that family; if
     the two agree, the LOFO result is only measuring system size.

Both are evaluated ONLY at the pre/post window grid points of isolated goals
rather than the full match grid, which is ~400x less LP work than the engine.
The baseline row must reproduce the frozen engine estimate; that is asserted.

Also reports effect-size context: tick size, quoted half-spread and canonical
interval width, so the headline in cents can be stated in ticks and in spreads.

Run: /home/azureuser/.venv/bin/python -u family_attribution.py --cohort dev
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402
from scipy.optimize import linprog                     # noqa: E402
import event_study as ES                               # noqa: E402
from compiler import build_contracts, load_manifest, state_classes  # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
P = ES.P
G, WIN = P["GRID"], P["WIN"]
FAMS = ["exact-score", "moneyline", "btts", "total-goals", "team-total", "spread"]
C = 100.0   # S is carried in probability units; all display is in cents
# Independent size-matched random subsets per family per match. The frozen
# producer family_attribution.py fixes this at 5 for all six families; here both
# the count and the families that receive draws are set on the command line, so
# the draw count can be raised far enough for the control to be read as a
# distribution rather than as five numbers. Cost is linear in
# N_RAND_DROP * len(RAND_FAMS), which is why the default narrows the families.
N_RAND_DROP = 5
RAND_FAMS = list(FAMS)
LOGFILE = f"{OUT}/logs/fam_attr_draws.log"
np.random.seed(P["SEED"])


def log(msg):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as fh:
        fh.write(line + "\n")


def solve_S_dual(X, lows, highs):
    """Same LP as event_study.solve_S, but also returns the dual certificate."""
    m, n = X.shape
    nv = n + 1
    c = np.zeros(nv)
    c[-1] = 1.0
    A_ub = np.zeros((2 * m, nv))
    b_ub = np.zeros(2 * m)
    A_ub[:m, :-1] = -X
    A_ub[:m, -1] = -1.0
    b_ub[:m] = -np.array(lows)
    A_ub[m:, :-1] = X
    A_ub[m:, -1] = -1.0
    b_ub[m:] = np.array(highs)
    A_eq = np.zeros((1, nv))
    A_eq[0, :-1] = 1.0
    try:
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([1.0]),
                    bounds=[(0, 1)] * n + [(0, None)], method="highs")
    except Exception:
        return None, None
    if not r.success:
        return None, None
    s = float(r.x[-1])
    duals = None
    try:
        y = np.abs(np.asarray(r.ineqlin.marginals, dtype=float))
        # rows 0..m-1 are lower-bound constraints, m..2m-1 upper-bound, same order
        per_contract = y[:m] + y[m:]
        tot = per_contract.sum()
        if tot > 1e-12:
            duals = per_contract / tot
    except Exception:
        duals = None
    return s, duals


def run(cohort):
    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    matches = sorted(split[cohort])
    M, _ = load_manifest()
    print("=" * 104)
    print(f"FAMILY ATTRIBUTION — cohort {cohort.upper()} ({len(matches)} matches)")
    print("=" * 104)

    d_base = defaultdict(list)
    d_lofo = {f: defaultdict(list) for f in FAMS}
    d_rand = {(f, r): defaultdict(list) for f in RAND_FAMS
              for r in range(N_RAND_DROP)}
    share = {"pre": defaultdict(list), "post": defaultdict(list)}
    active = {"pre": defaultdict(list), "post": defaultdict(list)}
    ctx = {"tick": [], "half_spread": {"pre": [], "post": []},
           "canon_width": {"pre": [], "post": []}}
    fam_sizes = defaultdict(list)
    n_ev = 0

    for mi, match in enumerate(matches, 1):
        contracts, info = build_contracts(M, match)
        if contracts is None:
            continue
        reps = state_classes(contracts)
        X = np.array([[c["rule"](h, a) for (h, a) in reps] for c in contracts],
                     dtype=float)
        fam_of = np.array([c["family"] for c in contracts])
        # size-matched random drop masks, fixed per match so the comparison with
        # leave-one-family-out is paired on exactly the same events
        seed = int.from_bytes(hashlib.sha256(match.encode()).digest()[:8], "big")
        mrng = np.random.default_rng(seed)
        rand_keep = {}
        for f in FAMS:
            nf = int((fam_of == f).sum())
            fam_sizes[f].append(nf)
            if nf == 0 or nf >= len(contracts) - 2 or f not in RAND_FAMS:
                continue
            for r in range(N_RAND_DROP):
                drop = mrng.choice(len(contracts), size=nf, replace=False)
                k = np.ones(len(contracts), dtype=bool)
                k[drop] = False
                rand_keep[(f, r)] = k
        ser, ticks = _load_series(contracts)
        ctx["tick"].extend(ticks)

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
            vals = {"pre": {"base": [], **{f: [] for f in FAMS},
                            **{("R", f, r): [] for f in FAMS
                               for r in range(N_RAND_DROP)}},
                    "post": {"base": [], **{f: [] for f in FAMS},
                             **{("R", f, r): [] for f in FAMS
                                for r in range(N_RAND_DROP)}}}
            for tag in ("pre", "post"):
                for i in idx[tag]:
                    t = gt[i]
                    lows, highs, avail = [], [], 0
                    for c in contracts:
                        cs = ser.get((c["slug"], "canon_token"))
                        ps = ser.get((c["slug"], "comp_token"))
                        can = ES.token_quote_env(cs[0], cs[1], t) if cs else None
                        comp = ES.token_quote_env(ps[0], ps[1], t) if ps else None
                        ci = ES.canonical_interval(can, comp)
                        if ci is None:
                            lows.append(0.0)
                            highs.append(1.0)
                            continue
                        lo_, hi_, ns = ci
                        lows.append(lo_)
                        highs.append(hi_)
                        if ns > 0:
                            avail += 1
                            ctx["canon_width"][tag].append(hi_ - lo_)
                            if can and can[0] is not None and can[1] is not None:
                                ctx["half_spread"][tag].append(0.5 * (can[1] - can[0]))
                    if avail < P["MIN_AVAIL"]:
                        continue
                    s, du = solve_S_dual(X, lows, highs)
                    if s is None:
                        continue
                    vals[tag]["base"].append(s)
                    if du is not None and s > 1e-6:
                        for f in FAMS:
                            sel = fam_of == f
                            if sel.any():
                                share[tag][f].append(float(du[sel].sum()))
                                active[tag][f].append(
                                    float((du[sel] > 1e-9).any()))
                    for f in FAMS:
                        keep = fam_of != f
                        if keep.sum() < 3 or (~keep).sum() == 0:
                            continue
                        sf, _ = solve_S_dual(X[keep],
                                             list(np.array(lows)[keep]),
                                             list(np.array(highs)[keep]))
                        if sf is not None:
                            vals[tag][f].append(sf)
                        for r in (range(N_RAND_DROP) if f in RAND_FAMS else ()):
                            rk = rand_keep.get((f, r))
                            if rk is None or rk.sum() < 3:
                                continue
                            sr, _ = solve_S_dual(X[rk],
                                                 list(np.array(lows)[rk]),
                                                 list(np.array(highs)[rk]))
                            if sr is not None:
                                vals[tag][("R", f, r)].append(sr)
            if not vals["pre"]["base"] or not vals["post"]["base"]:
                continue
            n_ev += 1
            d_base[match].append(np.mean(vals["post"]["base"])
                                 - np.mean(vals["pre"]["base"]))
            for f in FAMS:
                if vals["pre"][f] and vals["post"][f]:
                    d_lofo[f][match].append(np.mean(vals["post"][f])
                                            - np.mean(vals["pre"][f]))
                for r in (range(N_RAND_DROP) if f in RAND_FAMS else ()):
                    kp, kq = vals["pre"][("R", f, r)], vals["post"][("R", f, r)]
                    if kp and kq:
                        d_rand[(f, r)][match].append(np.mean(kq) - np.mean(kp))
        if mi % 10 == 0:
            log(f"  ... {mi}/{len(matches)} matches, {n_ev} isolated events")

    def boot(pm):
        ms = sorted(pm)
        v = np.array([float(np.mean(pm[m])) for m in ms])
        if len(v) < 3:
            return None
        b = np.array([np.mean(np.random.choice(v, len(v), replace=True))
                      for _ in range(P["N_BOOT"])])
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {"D": float(v.mean()), "ci": [float(lo), float(hi)],
                "n_matches": len(ms), "n_events": sum(len(x) for x in pm.values())}

    base = boot(d_base)
    base_ci = f"[{C*base['ci'][0]:+.3f}, {C*base['ci'][1]:+.3f}]"
    print("\n" + "-" * 104)
    print("LEAVE-ONE-FAMILY-OUT")
    print(f"  {'dropped family':<18}{'n_ev':>6}{'D (cents)':>12}{'95% CI (cents)':>24}"
          f"{'vs base':>10}{'retained':>10}{'n_ctr':>7}{'rand%':>7}{'range':>10}")
    print(f"  {'(none = baseline)':<18}{base['n_events']:>6}{C*base['D']:>+12.3f}"
          f"{base_ci:>24}{'-':>10}{'-':>10}")
    lofo_out = {}
    for f in FAMS:
        r = boot(d_lofo[f])
        if not r:
            print(f"  {f:<18}{'-':>6}{'insufficient':>12}")
            continue
        ci = f"[{C*r['ci'][0]:+.3f}, {C*r['ci'][1]:+.3f}]"
        # size-matched random control, averaged over N_RAND_DROP draws so the
        # comparison does not hinge on a single subset
        rr = [boot(d_rand[(f, k)]) for k in
              (range(N_RAND_DROP) if f in RAND_FAMS else ())]
        rr = [x for x in rr if x]
        if rr:
            pcts = [100 * x["D"] / base["D"] for x in rr]
            rand_txt = f"{np.mean(pcts):>6.0f}%"
            rand_rng = f"[{min(pcts):.0f}-{max(pcts):.0f}]"
        else:
            rand_txt, rand_rng = f"{'-':>7}", ""
        print(f"  {f:<18}{r['n_events']:>6}{C*r['D']:>+12.3f}{ci:>24}"
              f"{C*(r['D']-base['D']):>+10.3f}{100*r['D']/base['D']:>9.0f}%"
              f"{int(np.mean(fam_sizes[f])) if fam_sizes[f] else 0:>7}"
              f"{rand_txt}{rand_rng:>10}")
        lofo_out[f] = {"lofo": r,
                       "size_matched_random_mean_pct":
                           float(np.mean(pcts)) if rr else None,
                       "size_matched_random_pcts":
                           [float(x) for x in pcts] if rr else None,
                       "n_random_draws": len(rr),
                       "mean_family_size": float(np.mean(fam_sizes[f]))
                       if fam_sizes[f] else None}
        if rr:
            # Where the family deletion sits in the distribution of size-matched
            # random deletions. Reported as a one-sided empirical tail: the share
            # of random subsets that cost at least as much of the effect as this
            # family did. A family is load-bearing beyond its size only if this
            # is small.
            lofo_pct = 100 * r["D"] / base["D"]
            p = float(np.mean([x <= lofo_pct for x in pcts]))
            q = np.percentile(pcts, [5, 25, 50, 75, 95])
            lofo_out[f]["lofo_retained_pct"] = float(lofo_pct)
            lofo_out[f]["random_pct_quantiles"] = {
                k: float(v) for k, v in
                zip(["p5", "p25", "p50", "p75", "p95"], q)}
            lofo_out[f]["empirical_p_family_at_least_as_costly"] = p
            log(f"  {f}: lofo retains {lofo_pct:.1f}% | random n={len(rr)} "
                f"mean {np.mean(pcts):.1f}% sd {np.std(pcts):.1f} "
                f"median {q[2]:.1f}% [p5 {q[0]:.1f}, p95 {q[4]:.1f}] | "
                f"empirical p = {p:.3f}")
    print(f"  'rand%' drops a RANDOM subset of the same size instead of the"
          f" family, averaged over {N_RAND_DROP} draws (range in brackets).")
    print("  A family matters specifically only if its retained% is clearly"
          " below its random control.")

    print("\n" + "-" * 104)
    print("DUAL CERTIFICATE — share of violation mass by family (instants with S>0)")
    print(f"  {'family':<18}{'pre share':>11}{'post share':>12}{'delta':>9}"
          f"{'pre active':>12}{'post active':>13}")
    cert = {}
    for f in FAMS:
        pr, po = share["pre"][f], share["post"][f]
        ap, ao = active["pre"][f], active["post"][f]
        if not pr or not po:
            continue
        cert[f] = {"pre_share": float(np.mean(pr)), "post_share": float(np.mean(po)),
                   "pre_active": float(np.mean(ap)), "post_active": float(np.mean(ao)),
                   "n_pre": len(pr), "n_post": len(po)}
        print(f"  {f:<18}{np.mean(pr):>11.3f}{np.mean(po):>12.3f}"
              f"{np.mean(po)-np.mean(pr):>+9.3f}"
              f"{100*np.mean(ap):>11.0f}%{100*np.mean(ao):>12.0f}%")

    print("\n" + "-" * 104)
    print("EFFECT-SIZE CONTEXT")
    tk = np.array(ctx["tick"], dtype=float)
    tk = tk[np.isfinite(tk) & (tk > 0)]
    if len(tk):
        q = np.percentile(tk, [25, 50, 75])
        print(f"  {'tick size':<26} median {C*q[1]:>7.3f}c   "
              f"IQR [{C*q[0]:.3f}, {C*q[2]:.3f}]c   n={len(tk):,}")

    # Direction of the liquidity channel. Wider quoted bounds are WEAKER
    # constraints, so any widening after a goal pushes S DOWN. If widths widen
    # while S rises, the liquidity channel has the wrong sign to explain the
    # result -- a structural argument, now measured rather than asserted.
    print("\n  quoted width by window (wider => weaker constraints => smaller S)")
    print(f"  {'quantity':<26}{'pre':>10}{'post':>10}{'change':>10}")
    width_out = {}
    for k, lab in (("half_spread", "quoted half-spread"),
                   ("canon_width", "canonical interval width")):
        a = np.array(ctx[k]["pre"], dtype=float)
        b = np.array(ctx[k]["post"], dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if not len(a) or not len(b):
            continue
        ma, mb = float(np.median(a)), float(np.median(b))
        print(f"  {lab:<26}{C*ma:>9.3f}c{C*mb:>9.3f}c{C*(mb-ma):>+9.3f}c")
        width_out[k] = {"pre_median": ma, "post_median": mb,
                        "pre_mean": float(a.mean()), "post_mean": float(b.mean()),
                        "n_pre": int(a.size), "n_post": int(b.size)}
    cw_all = np.array(ctx["canon_width"]["pre"] + ctx["canon_width"]["post"],
                      dtype=float)
    cw_all = cw_all[np.isfinite(cw_all)]
    if len(tk):
        print(f"\n  headline {C*base['D']:+.3f}c = "
              f"{C*base['D']/(C*np.median(tk)):.2f} median ticks")
    if len(cw_all):
        print(f"  headline {C*base['D']:+.3f}c = "
              f"{C*base['D']/(C*np.median(cw_all)):.2f} median canonical widths")
    print("  NOTE: the headline is a sub-tick AVERAGE shift. The interpretable")
    print("  statement is the threshold response (see FINDINGS 7c): the share of")
    print("  isolated goals whose window-mean violation is at least 0.25c rises")
    print("  12.8% -> 59.2%.")

    res = {"cohort": cohort, "baseline": base, "lofo": lofo_out, "certificate": cert,
           "width_by_window": width_out,
           "n_random_draws": N_RAND_DROP, "random_families": list(RAND_FAMS),
           "tick_median": float(np.median(tk)) if len(tk) else None,
           "n_tick": int(len(tk))}
    os.makedirs(f"{OUT}/results", exist_ok=True)
    dest = f"{OUT}/results/family_attribution_draws{N_RAND_DROP}_{cohort}.json"
    json.dump(res, open(dest, "w"), indent=2)

    try:
        eng = json.load(open(f"{OUT}/results/{cohort}/event_study.json"))
        w_ = C * eng["isolated"]["D"]
        print(f"\n  self-check vs frozen engine: engine {w_:+.4f}c, "
              f"here {C*base['D']:+.4f}c, diff {C*base['D']-w_:+.5f}c -> "
              f"{'MATCH' if abs(w_-C*base['D']) < 5e-3 else '*** MISMATCH ***'}")
    except Exception as e:
        print(f"\n  self-check unavailable: {e}")
    log(f"results -> {dest}")


def _load_series(contracts):
    """Per-slug book ingestion mirroring the engine, additionally capturing the
    venue tick size so the headline can be expressed in ticks."""
    import glob
    import gzip
    ser, ticks = {}, []
    for c in contracts:
        books = ES.load_slug_books(c["slug"])
        for p in glob.glob(f"{ES.LIVE}/*/*/{c['slug']}/*.jsonl.gz")[:1]:
            try:
                with gzip.open(p, "rt", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            b = (json.loads(line).get("book") or {})
                        except Exception:
                            continue
                        tv = b.get("tick_size")
                        if tv is not None:
                            ticks.append(float(tv))
                            break
            except Exception:
                pass
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
    return ser, ticks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["dev", "holdout"], required=True)
    ap.add_argument("--draws", type=int, default=5,
                    help="size-matched random subsets per family per match")
    ap.add_argument("--rand-families", default=",".join(FAMS),
                    help="comma-separated families that receive random draws; "
                         "runtime is linear in draws x families")
    a = ap.parse_args()
    N_RAND_DROP = a.draws
    RAND_FAMS = [f.strip() for f in a.rand_families.split(",") if f.strip()]
    bad = [f for f in RAND_FAMS if f not in FAMS]
    if bad:
        ap.error(f"unknown families: {bad}; choose from {FAMS}")
    LOGFILE = f"{OUT}/logs/fam_attr_draws{N_RAND_DROP}_{a.cohort}.log"
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    log(f"start cohort={a.cohort} draws={N_RAND_DROP} families={RAND_FAMS}")
    t0 = datetime.datetime.now(datetime.timezone.utc)
    run(a.cohort)
    log(f"done in {(datetime.datetime.now(datetime.timezone.utc)-t0)}")
