#!/usr/bin/env python3
"""L1 aggregation variant of the incoherence functional. DEVELOPMENT ONLY.

Why this exists. S as defined in 4.4 is a uniform (Chebyshev, L-infinity)
relaxation: the single worst-violated bound sets the value. A referee reasonably
asks whether the reported event response is a property of the MARKET or of that
particular choice of aggregation, since an L1 objective, a violated-bound count,
or a per-contract weighting would each define an equally defensible notion of
"how incoherent" the system is.

The L1 variant gives every quoted interval its own slack and minimises the total:

    S1 = min over q, s >= 0 of  sum_i s_i
         s.t.  l_i - s_i <= (X q)_i <= u_i + s_i  for all i,  q in the simplex.

S_inf and S1 are not on the same scale -- S1 accumulates across roughly 46 bounds
while S_inf is cents per bound -- so the comparison of interest is the SIGN and
the match-level agreement of the event response, not the magnitude.

WHY DEVELOPMENT ONLY. The registered split is the paper's strongest
methodological asset. This is an unregistered functional introduced after the
holdout was unblinded, so running it on the holdout would spend that asset on a
robustness check. The script refuses the holdout cohort outright rather than
relying on an operator to remember.

Run: /home/azureuser/.venv/bin/python -u l1_functional.py
"""
import json
import os
import sys
import time
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
COHORT = "dev"          # not a parameter; see the module docstring
np.random.seed(P["SEED"])


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)


def solve_S1(X, lows, highs):
    """Total L1 slack: one shared lower/upper slack per interval."""
    m, n = X.shape
    nv = n + m
    c = np.concatenate([np.zeros(n), np.ones(m)])
    A = np.zeros((2 * m, nv))
    b = np.zeros(2 * m)
    # (Xq)_i - s_i <= u_i
    A[:m, :n] = X
    A[:m, n:] = -np.eye(m)
    b[:m] = np.asarray(highs)
    # -(Xq)_i - s_i <= -l_i
    A[m:, :n] = -X
    A[m:, n:] = -np.eye(m)
    b[m:] = -np.asarray(lows)
    Aeq = np.zeros((1, nv))
    Aeq[0, :n] = 1.0
    try:
        r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=np.array([1.0]),
                    bounds=[(0, 1)] * n + [(0, None)] * m, method="highs")
    except Exception:
        return None
    return float(r.fun) if r.success else None


def run():
    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    matches = sorted(split[COHORT])
    M, _ = load_manifest()
    log(f"L1 FUNCTIONAL VARIANT — cohort {COHORT} ({len(matches)} matches)")
    log("holdout is not reachable from this script by construction")

    d_inf, d_one = defaultdict(list), defaultdict(list)
    lvl = {"inf": {"pre": [], "post": []}, "one": {"pre": [], "post": []}}
    paired = []
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
            acc = {"inf": {"pre": [], "post": []}, "one": {"pre": [], "post": []}}
            for tag, lo, hi in (("pre", P["PRE_LO"], P["PRE_HI"]),
                                ("post", P["POST_LO"], P["POST_HI"])):
                i0 = int(np.searchsorted(gt, w + lo))
                i1 = int(np.searchsorted(gt, w + hi))
                for i in range(i0, i1 + 1):
                    if not (0 <= i < len(gt)):
                        continue
                    t = gt[i]
                    lows, highs, avail = [], [], 0
                    for c in contracts:
                        cs = ser.get((c["slug"], "canon_token"))
                        ps = ser.get((c["slug"], "comp_token"))
                        can = ES.token_quote_env(cs[0], cs[1], t) if cs else None
                        comp = ES.token_quote_env(ps[0], ps[1], t) if ps else None
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
                    si = ES.solve_S(X, lows, highs)
                    s1 = solve_S1(X, lows, highs)
                    if si is None or s1 is None:
                        continue
                    acc["inf"][tag].append(si)
                    acc["one"][tag].append(s1)
            if not acc["inf"]["pre"] or not acc["inf"]["post"]:
                continue
            di = float(np.mean(acc["inf"]["post"]) - np.mean(acc["inf"]["pre"]))
            do = float(np.mean(acc["one"]["post"]) - np.mean(acc["one"]["pre"]))
            d_inf[match].append(di)
            d_one[match].append(do)
            paired.append((di, do))
            for k in ("inf", "one"):
                for tag in ("pre", "post"):
                    lvl[k][tag].extend(acc[k][tag])
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

    ri, ro = boot(d_inf), boot(d_one)
    pa = np.array(paired)
    agree = float(np.mean(np.sign(pa[:, 0]) == np.sign(pa[:, 1]))) if len(pa) else float("nan")
    rho = float(np.corrcoef(pa[:, 0], pa[:, 1])[0, 1]) if len(pa) > 2 else float("nan")

    log("")
    log("=" * 96)
    log(f"INCOHERENCE FUNCTIONAL — L-infinity vs L1, {COHORT} cohort")
    log("=" * 96)
    log(f"  {'functional':<26}{'pre':>10}{'post':>10}{'D':>10}"
        f"{'95% CI':>22}{'pos/mat':>10}")
    for key, lab, r in (("inf", "L-inf (reported S)", ri), ("one", "L1 (total slack)", ro)):
        pre = C * float(np.mean(lvl[key]["pre"]))
        post = C * float(np.mean(lvl[key]["post"]))
        ci = f"[{C*r['ci'][0]:+.3f}, {C*r['ci'][1]:+.3f}]"
        pos_txt = f"{r['n_pos']}/{r['n_matches']}"
        log(f"  {lab:<26}{pre:>9.3f}c{post:>9.3f}c{C*r['D']:>+10.3f}{ci:>22}"
            f"{pos_txt:>10}")
    log("")
    log(f"  per-goal sign agreement : {100*agree:.1f}%  (n={len(pa)})")
    log(f"  per-goal correlation    : {rho:+.3f}")

    eng = json.load(open(f"{OUT}/results/{COHORT}/event_study.json"))
    want = eng["isolated"]["D"]
    log("")
    log(f"  baseline check: harness {C*ri['D']:+.4f}c vs frozen engine {C*want:+.4f}c")
    if abs(ri["D"] - want) > 2e-5:
        log("  BASELINE MISMATCH — harness has drifted from the engine; results void")
        sys.exit(1)
    log("  baseline reproduces the frozen engine")

    payload = {"cohort": COHORT, "generated_at": datetime.now(timezone.utc).isoformat(),
               "l_inf": ri, "l_one": ro,
               "levels": {k: {"pre_mean": float(np.mean(v["pre"])),
                              "post_mean": float(np.mean(v["post"])),
                              "n": len(v["pre"]) + len(v["post"])}
                          for k, v in lvl.items()},
               "per_goal_sign_agreement": agree, "per_goal_correlation": rho,
               "n_goals": len(pa)}
    dst = f"{OUT}/results/l1_functional_{COHORT}.json"
    with open(dst, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {dst}")


if __name__ == "__main__":
    run()
