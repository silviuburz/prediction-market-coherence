#!/usr/bin/env python3
"""Placebo / dose-response suite for the coherence event study.

The event study says coherence degrades after a goal. The obvious hostile reading
is that it degrades after ANY salient moment, because market makers widen or pull
quotes when attention spikes -- in which case the estimator measures liquidity
withdrawal, not information processing.

ESPN labels every key event, which gives a dose-response ladder ordered by how
much each event revises beliefs about the FINAL SCORE:

    Halftime (id 81)         zero information, scheduled, high attention
    Start 2nd Half (id 82)   zero information, scheduled
    Kickoff (id 80)          zero information, scheduled
    Substitution (id 76)     ~zero
    Yellow card (id 94)      negligible
    Delay / drinks (id 129)  ~zero
    Penalty missed/saved     high attention, NO score change  (113, 114)
    Red card (id 93)         substantial
    GOAL                     large

Prediction under the information account: the effect tracks information content,
so halftime/substitutions/cards are flat and goals are large. Prediction under
the liquidity account: halftime -- a scheduled, universally-known attention
discontinuity -- should degrade coherence as much as a goal.

Everything is computed from the persisted per-match coherence series using the
IDENTICAL window semantics as the frozen engine (searchsorted on absolute time,
inclusive upper slice), so the goal row must reproduce the frozen estimate
exactly. That reproduction is asserted as a self-check.

Run: /home/azureuser/.venv/bin/python -u placebo_suite.py
"""
import glob
import gzip
import json
import lzma
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
LIVE = "/home/azureuser/polymarket/analysis/data/world_cup/live"
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
WIN, ISO_GAP, N_BOOT = 300, 300, 5000
MIN_MATCHES = 5          # below this a match-clustered bootstrap CI is meaningless
C = 100.0                # S is carried in probability units; report cents
np.random.seed(29)

CLASSES = [
    ("goal",           None,           "large        (validation row)"),
    ("red_card",       {"93"},         "substantial"),
    ("penalty_nogoal", {"113", "114"}, "high attention, NO score change"),
    ("yellow_card",    {"94"},         "negligible"),
    ("substitution",   {"76"},         "~zero"),
    ("delay",          {"129"},        "~zero (drinks/injury break)"),
    ("halftime",       {"81"},         "ZERO  (scheduled, score known)"),
    ("start_2nd_half", {"82"},         "ZERO  (scheduled)"),
]


def iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def key_events_by_game():
    """All regulation key events, typed, keyed by ESPN game_id."""
    out = {}
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
        gid = str(d.get("espn_game_id"))
        if not gid or gid == "None":
            continue
        rows, seen = [], set()
        for e in (d.get("keyEvents") or []):
            try:
                per = int((e.get("period") or {}).get("number") or 0)
                if per < 1 or per > 2:
                    continue
                w = iso(e.get("wallclock"))
                eid = str(e.get("id"))
                if not w or eid in seen:
                    continue
                seen.add(eid)
                rows.append({"id": eid, "w": w,
                             "type": str((e.get("type") or {}).get("id")),
                             "scoring": bool(e.get("scoringPlay"))})
            except Exception:
                continue
        # a game may appear in several day-folders; keep the richest copy
        if gid not in out or len(rows) > len(out[gid]):
            out[gid] = sorted(rows, key=lambda z: z["w"])
    return out


def win_mean(arr, gt, centre, lo, hi):
    """Byte-identical to the frozen engine's window semantics."""
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = int(np.searchsorted(gt, centre + hi))
    seg = arr[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if len(seg) else None


def cluster_boot(per_match):
    ms = sorted(per_match)
    vals = np.array([float(np.mean(per_match[m])) for m in ms])
    if len(vals) == 0:
        return None
    D = float(vals.mean())
    res = {"D": D, "n_matches": len(ms),
           "n_events": sum(len(v) for v in per_match.values()),
           "median": float(np.median(vals)),
           "n_pos": int((vals > 0).sum())}
    if len(vals) < MIN_MATCHES:
        res.update({"ci": None, "sig": None, "underpowered": True})
        return res
    boot = np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res.update({"ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0),
                "underpowered": False})
    return res


KE = key_events_by_game()
print(f"key-event games loaded: {len(KE)}")
report = {}

for cohort in ("dev", "holdout"):
    path = f"{OUT}/results/{cohort}/match_stats.json.gz"
    if not os.path.exists(path):
        print(f"\n!! {path} missing -- re-run event_study.py to persist series")
        continue
    with gzip.open(path, "rt") as f:
        MS = json.load(f)
    print("\n" + "=" * 104)
    print(f"COHORT {cohort.upper()}   matches with persisted series: {len(MS)}")
    print("=" * 104)

    rows = {}
    for cname, tset, _note in CLASSES:
        per_match = defaultdict(list)
        n_seen = n_iso = 0
        for match, st in MS.items():
            Sc = np.array(st["Sc"], dtype=float)
            gt = st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"])
            t_start, t_end = st["t_start"], st["t_end"]
            gw = st["goal_w"]
            gid = st.get("game_id")
            evs = KE.get(str(gid), [])
            if cname == "goal":
                anchors = [{"w": w, "type": "goal"} for w in gw]
            else:
                anchors = [e for e in evs if e["type"] in tset and not e["scoring"]]
            same = [a["w"] for a in anchors]
            for a in anchors:
                w = a["w"]
                n_seen += 1
                if w - WIN < t_start or w + WIN > t_end:
                    continue
                # isolation: no goal within ISO_GAP, and no other event of the
                # same class within ISO_GAP (goals compare against other goals)
                if any(abs(w - o) <= ISO_GAP for o in gw if abs(o - w) > 1e-9):
                    continue
                if any(abs(w - o) <= ISO_GAP for o in same if abs(o - w) > 1e-9):
                    continue
                pre = win_mean(Sc, gt, w, PRE_LO, PRE_HI)
                post = win_mean(Sc, gt, w, POST_LO, POST_HI)
                if pre is None or post is None:
                    continue
                n_iso += 1
                per_match[match].append(post - pre)
        r = cluster_boot(per_match)
        rows[cname] = {"result": r, "n_anchors": n_seen, "n_used": n_iso}

    print(f"  {'class':<16}{'info content':<34}{'n_ev':>6}{'n_mt':>6}"
          f"{'D (cents)':>11}{'95% CI (cents)':>24}{'sig':>5}")
    print("  " + "-" * 102)
    for cname, _t, note in CLASSES:
        e = rows[cname]
        r = e["result"]
        if not r:
            print(f"  {cname:<16}{note:<34}{e['n_used']:>6}{'-':>6}"
                  f"{'no events':>11}")
            continue
        if r["ci"] is None:
            print(f"  {cname:<16}{note:<34}{r['n_events']:>6}{r['n_matches']:>6}"
                  f"{C*r['D']:>+11.3f}{'n<5, descriptive only':>24}{'-':>5}")
            continue
        ci = f"[{C*r['ci'][0]:+.3f}, {C*r['ci'][1]:+.3f}]"
        print(f"  {cname:<16}{note:<34}{r['n_events']:>6}{r['n_matches']:>6}"
              f"{C*r['D']:>+11.3f}{ci:>24}{'YES' if r['sig'] else '.':>5}")
    report[cohort] = rows

    # ── self-check: the goal row must reproduce the frozen engine estimate ──
    try:
        eng = json.load(open(f"{OUT}/results/{cohort}/event_study.json"))
        want = C * eng["isolated"]["D"]
        got = C * rows["goal"]["result"]["D"]
        ok = abs(want - got) < 5e-3
        print(f"\n  self-check vs frozen engine: engine {want:+.4f}c, "
              f"offline {got:+.4f}c, diff {got-want:+.5f}c -> "
              f"{'MATCH' if ok else '*** MISMATCH ***'}")
        print(f"  events: engine {eng['isolated']['n_events']}, "
              f"offline {rows['goal']['result']['n_events']}; "
              f"matches: engine {eng['isolated']['n_matches']}, "
              f"offline {rows['goal']['result']['n_matches']}")
    except Exception as ex:
        print(f"\n  self-check unavailable: {ex}")

# ── contrast: goal vs the pure-placebo classes, same matches ───────────────
print("\n" + "=" * 104)
print("CONTRAST — zero-information placebos against the goal effect")
print("A placebo whose CI excludes the goal effect rules out the pure liquidity/")
print("attention account for that class.")
print("=" * 104)
for cohort, rows in report.items():
    g = rows["goal"]["result"]
    if not g:
        continue
    print(f"\n  {cohort.upper()}  goal effect {C*g['D']:+.3f}c")
    for cname in ("halftime", "start_2nd_half", "substitution", "yellow_card",
                  "delay"):
        r = rows[cname]["result"]
        if not r or r["ci"] is None:
            continue
        share = 100.0 * r["D"] / g["D"] if g["D"] else float("nan")
        hi_share = 100.0 * r["ci"][1] / g["D"] if g["D"] else float("nan")
        excl = r["ci"][1] < g["D"]
        print(f"    {cname:<16}{C*r['D']:>+8.3f}c  "
              f"CI [{C*r['ci'][0]:+.3f}, {C*r['ci'][1]:+.3f}]c  "
              f"= {share:>5.1f}% of goal, CI upper {hi_share:>5.1f}%  "
              f"{'goal effect EXCLUDED' if excl else '*** overlaps goal ***'}")

json.dump(report, open(f"{OUT}/results/placebo_suite.json", "w"), indent=2)
print(f"\nresults -> {OUT}/results/placebo_suite.json")
