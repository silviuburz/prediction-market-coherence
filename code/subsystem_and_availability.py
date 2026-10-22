#!/usr/bin/env python3
"""Contract-subsystem coherence + quoted-contract census.

Answers two referee questions that the existing outputs cannot.

(A) QUOTED-CONTRACT CENSUS.  The manuscript advertises "approximately 46 linked
    contracts per match" but analyses an instant whenever >= MIN_AVAIL (10) are
    genuinely quoted. Nothing reports where in that 4.6x range the typical
    instant actually sits. This reports the distribution of the number of
    contracts carrying at least one real bound, pre-goal vs post-goal, overall
    and by family.

(B) SUBSYSTEM S.  family_attribution.py answers "which relations break" by
    DELETING a family (leave-one-family-out) and by dual mass. Neither is the
    decisive test, because both are consistent with the deflationary story that
    the effect is carried entirely by thin, slow exact-score contracts that lag
    the liquid ones after a goal. The decisive test is the POSITIVE one: keep
    ONLY a liquid subsystem and ask whether S still rises.

    The moneyline triple is the strongest such test. Home/draw/away are the
    three most liquid contracts on the venue, they partition the outcome space,
    and they cannot be dismissed as stale thin markets. For a partition the LP
    reduces to the classic overround/underround check,
        S = max(0, (sum(low) - 1)/3, (1 - sum(high))/3),
    so a positive moneyline-only effect means the three headline prices alone
    stop admitting a common distribution after a goal.

    If S rises on the moneyline alone, the asynchronous-thin-market explanation
    is dead. If S is flat there and the effect lives only in the exact-score
    grid, the manuscript's claim has to be narrowed. Both outcomes are worth
    knowing before a referee finds out first.

Subsystems are KEEP-sets, the complement of the existing LOFO analysis. The
`full` row must reproduce the frozen engine estimate; that is asserted, so a
harness that has drifted from the engine fails loudly rather than reporting a
plausible wrong number.

Only pre/post window grid points of isolated goals are evaluated, matching
family_attribution.py, which is ~400x less LP work than the engine's full grid.

Run: /home/azureuser/.venv/bin/python -u subsystem_and_availability.py --cohort holdout
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402
import event_study as ES                               # noqa: E402
from compiler import build_contracts, load_manifest, state_classes  # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
P = ES.P
G, WIN = P["GRID"], P["WIN"]
FAMS = ["exact-score", "moneyline", "btts", "total-goals", "team-total", "spread"]
C = 100.0

# Keep-sets. `full` is the baseline and must reproduce the frozen engine.
SUBSYS = {
    "full": FAMS,
    "moneyline": ["moneyline"],
    "moneyline+total-goals": ["moneyline", "total-goals"],
    "no-exact-score": ["moneyline", "btts", "total-goals", "team-total", "spread"],
    "exact-score": ["exact-score"],
}
# A subsystem is only evaluated at an instant when at least this many of ITS OWN
# contracts carry a real bound. Without this an unquoted subsystem is handed
# [0,1] everywhere, which is unconstraining, and S collapses to 0 for a reason
# that has nothing to do with coherence.
MIN_SUB = 3

np.random.seed(P["SEED"])


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cohort):
    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    matches = sorted(split[cohort])
    M, _ = load_manifest()
    log(f"SUBSYSTEM + AVAILABILITY — cohort {cohort} ({len(matches)} matches)")

    d_sub = {k: defaultdict(list) for k in SUBSYS}
    # availability census
    av_all = {"pre": [], "post": []}
    av_fam = {"pre": defaultdict(list), "post": defaultdict(list)}
    # how often each subsystem is even evaluable, and how often it certifies S>0
    sub_eval = {k: {"pre": 0, "post": 0} for k in SUBSYS}
    sub_pos = {k: {"pre": 0, "post": 0} for k in SUBSYS}
    n_inst = {"pre": 0, "post": 0}
    n_ev = 0
    t0 = time.time()

    for mi, match in enumerate(matches, 1):
        contracts, info = build_contracts(M, match)
        if contracts is None:
            continue
        reps = state_classes(contracts)
        X = np.array([[c["rule"](h, a) for (h, a) in reps] for c in contracts],
                     dtype=float)
        fam_of = np.array([c["family"] for c in contracts])
        keep_mask = {k: np.isin(fam_of, v) for k, v in SUBSYS.items()}

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

            vals = {"pre": {k: [] for k in SUBSYS}, "post": {k: [] for k in SUBSYS}}
            for tag in ("pre", "post"):
                for i in idx[tag]:
                    t = gt[i]
                    lows, highs = [], []
                    bounded = np.zeros(len(contracts), dtype=bool)
                    for ci_, c in enumerate(contracts):
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
                            bounded[ci_] = True
                    avail = int(bounded.sum())
                    if avail < P["MIN_AVAIL"]:
                        continue
                    n_inst[tag] += 1
                    av_all[tag].append(avail)
                    for f in FAMS:
                        av_fam[tag][f].append(int((bounded & (fam_of == f)).sum()))
                    lows_a, highs_a = np.array(lows), np.array(highs)
                    for k, km in keep_mask.items():
                        # require the subsystem's OWN contracts to be quoted
                        if int((bounded & km).sum()) < MIN_SUB:
                            continue
                        s = ES.solve_S(X[km], list(lows_a[km]), list(highs_a[km]))
                        if s is None:
                            continue
                        sub_eval[k][tag] += 1
                        if s > P["TOL"]:
                            sub_pos[k][tag] += 1
                        vals[tag][k].append(s)

            if not vals["pre"]["full"] or not vals["post"]["full"]:
                continue
            n_ev += 1
            for k in SUBSYS:
                if vals["pre"][k] and vals["post"][k]:
                    d_sub[k][match].append(float(np.mean(vals["post"][k])
                                                 - np.mean(vals["pre"][k])))
        if mi % 10 == 0:
            log(f"  {mi}/{len(matches)} matches, {n_ev} isolated events, "
                f"{time.time()-t0:.0f}s")

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

    res = {k: boot(d_sub[k]) for k in SUBSYS}

    log("")
    log("=" * 100)
    log("(A) QUOTED-CONTRACT CENSUS — contracts carrying at least one real bound")
    log("=" * 100)
    log(f"  {'window':<8}{'n instants':>12}{'median':>9}{'IQR':>14}{'min':>7}"
        f"{'p05':>7}{'max':>7}")
    census = {}
    for tag in ("pre", "post"):
        a = np.array(av_all[tag], dtype=float)
        if not len(a):
            continue
        q = np.percentile(a, [5, 25, 50, 75])
        census[tag] = {"n": len(a), "median": float(q[2]),
                       "iqr": [float(q[1]), float(q[3])], "min": float(a.min()),
                       "p05": float(q[0]), "max": float(a.max()),
                       "mean": float(a.mean())}
        log(f"  {tag:<8}{len(a):>12,}{q[2]:>9.0f}"
            f"{f'[{q[1]:.0f}, {q[3]:.0f}]':>14}{a.min():>7.0f}{q[0]:>7.0f}"
            f"{a.max():>7.0f}")
    log("")
    log(f"  {'family':<16}{'pre median':>12}{'post median':>13}{'pre max':>10}")
    census_fam = {}
    for f in FAMS:
        pr = np.array(av_fam["pre"][f], dtype=float)
        po = np.array(av_fam["post"][f], dtype=float)
        if not len(pr):
            continue
        census_fam[f] = {"pre_median": float(np.median(pr)),
                         "post_median": float(np.median(po)),
                         "pre_max": float(pr.max())}
        log(f"  {f:<16}{np.median(pr):>12.0f}{np.median(po):>13.0f}{pr.max():>10.0f}")

    log("")
    log("=" * 100)
    log("(B) SUBSYSTEM S — keep only these families, recompute the event effect")
    log("=" * 100)
    log(f"  {'subsystem':<24}{'n_ev':>6}{'D (cents)':>12}{'95% CI (cents)':>24}"
        f"{'pos/mat':>10}{'%S>0 pre':>10}{'%S>0 post':>11}")
    for k in SUBSYS:
        r = res[k]
        if not r:
            log(f"  {k:<24}{'insufficient events':>40}")
            continue
        ci = f"[{C*r['ci'][0]:+.3f}, {C*r['ci'][1]:+.3f}]"
        ep, eo = sub_eval[k]["pre"], sub_eval[k]["post"]
        pp = 100.0 * sub_pos[k]["pre"] / ep if ep else float("nan")
        po_ = 100.0 * sub_pos[k]["post"] / eo if eo else float("nan")
        pos_txt = f"{r['n_pos']}/{r['n_matches']}"
        log(f"  {k:<24}{r['n_events']:>6}{C*r['D']:>+12.3f}{ci:>24}"
            f"{pos_txt:>10}{pp:>9.1f}%{po_:>10.1f}%")

    # The baseline must reproduce the frozen engine, or this harness has drifted.
    eng = json.load(open(f"{OUT}/results/{cohort}/event_study.json"))
    want = eng["isolated"]["D"]
    got = res["full"]["D"]
    log("")
    log(f"  baseline check: harness {C*got:+.4f}c vs frozen engine {C*want:+.4f}c "
        f"(delta {C*abs(got-want):.5f}c)")
    if abs(got - want) > 2e-5:
        log("  BASELINE MISMATCH — harness has drifted from the engine; results void")
        sys.exit(1)
    log("  baseline reproduces the frozen engine")

    payload = {"cohort": cohort, "generated_at": datetime.now(timezone.utc).isoformat(),
               "params": P, "min_subsystem_quoted": MIN_SUB,
               "census": census, "census_by_family": census_fam,
               "subsystems": {k: res[k] for k in SUBSYS},
               "subsystem_eval": sub_eval, "subsystem_positive": sub_pos,
               "n_instants": n_inst, "n_events": n_ev}
    dst = f"{OUT}/results/subsystem_availability_{cohort}.json"
    with open(dst, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["dev", "holdout"])
    a = ap.parse_args()
    run(a.cohort)
