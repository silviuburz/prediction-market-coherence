#!/usr/bin/env python3
"""Generate the three figure inputs that are not already persisted.

Everything here is computed offline from match_stats.json.gz using the same
searchsorted window semantics as the frozen engine, so no LP is re-solved.

(a) EVENT-TIME PROFILES  -> figs/event_time_profiles.json
    Mean S at every 5 s offset in [-300, +300] for goals and for each placebo
    class, match-clustered with a bootstrap ribbon. Feeds Figure 2.

    IMPORTANT. A profile is indexed by grid LABEL: the value plotted at offset
    -60 is the grid point returned by searchsorted(gt, w - 60), which lies in
    [-60, -55) because the match grid is not aligned to goal times. Averaging
    profile columns over a window therefore reproduces the estimator only
    approximately -- this is exactly the artifact that was removed from the
    specification curve in FINDINGS 12.4. Profiles are for SHAPE only. Figures
    must annotate the effect size with the engine's reported D, never with a
    value recomputed from these columns. The cross-check at the bottom prints
    both so the residual stays visible.

    Event inclusion matches the estimator exactly (isolated, fully inside the
    match window, and both pre- and post-window means computable), so profile
    event and match counts equal the reported n.

(b) THRESHOLD CCDF       -> figs/threshold_ccdf.json
    The pooled distribution of S over pre-window and post-window instants of
    isolated goals, as a complementary CDF over a dense threshold grid rather
    than the five fixed thresholds the engine reports. Feeds Figure 3.

(c) RANDOMIZATION NULL   -> figs/randomization_null.json
    All 5000 null draws per cohort, not just their moments. Feeds Figure 8.

Run: /home/azureuser/.venv/bin/python -u figure_inputs.py
"""
import glob
import gzip
import json
import lzma
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                     # noqa: E402

OUT = "/home/azureuser/projects/wc_coherence"
LIVE = "/home/azureuser/polymarket/analysis/data/world_cup/live"
FIGS = f"{OUT}/figs"
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
WIN, GRID, ISO_GAP = 300, 5, 300
N_BOOT, N_RAND, SEED = 2000, 5000, 29
C = 100.0
OFFSETS = list(range(-WIN, WIN + 1, GRID))
rng = np.random.default_rng(SEED)

CLASSES = [("goal", None), ("red_card", {"93"}), ("yellow_card", {"94"}),
           ("substitution", {"76"}), ("delay", {"129"}), ("halftime", {"81"}),
           ("start_2nd_half", {"82"})]


def iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def key_events_by_game():
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
                rows.append({"w": w, "type": str((e.get("type") or {}).get("id")),
                             "scoring": bool(e.get("scoringPlay"))})
            except Exception:
                continue
        if gid not in out or len(rows) > len(out[gid]):
            out[gid] = rows
    return out


def win_mean(arr, gt, centre, lo, hi):
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = int(np.searchsorted(gt, centre + hi))
    seg = arr[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if len(seg) else None


def boot_band(per_match_rows):
    """per_match_rows: {match: (n_off,) array of that match's mean profile}."""
    ms = sorted(per_match_rows)
    if len(ms) < 3:
        return None
    M = np.vstack([per_match_rows[m] for m in ms])           # (n_match, n_off)
    mean = np.nanmean(M, axis=0)
    idx = rng.integers(0, len(ms), size=(N_BOOT, len(ms)))
    draws = np.nanmean(M[idx], axis=1)                        # (N_BOOT, n_off)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return mean, lo, hi, len(ms)


os.makedirs(FIGS, exist_ok=True)
KE = key_events_by_game()
profiles, ccdf_out, null_out = {}, {}, {}

for cohort in ("dev", "holdout"):
    p = f"{OUT}/results/{cohort}/match_stats.json.gz"
    if not os.path.exists(p):
        print(f"!! missing {p}")
        continue
    with gzip.open(p, "rt") as f:
        MS = json.load(f)
    prepped = {}
    for m, st in MS.items():
        prepped[m] = (np.array(st["Sc"], dtype=float),
                      st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"]),
                      st["t_start"], st["t_end"], sorted(st["goal_w"]),
                      st.get("game_id"), st["elig"])

    # ── (a) event-time profiles ────────────────────────────────────────────
    profiles[cohort] = {}
    for cname, tset in CLASSES:
        per_match, n_ev = {}, 0
        for m, (Sc, gt, t0, t1, gw, gid, _elig) in prepped.items():
            if cname == "goal":
                anchors = list(gw)
            else:
                anchors = [e["w"] for e in KE.get(str(gid), [])
                           if e["type"] in tset and not e["scoring"]]
            rows = []
            for w in anchors:
                if w - WIN < t0 or w + WIN > t1:
                    continue
                if any(abs(w - o) <= ISO_GAP for o in gw if abs(o - w) > 1e-9):
                    continue
                if any(abs(w - o) <= ISO_GAP for o in anchors
                       if abs(o - w) > 1e-9):
                    continue
                # identical inclusion rule to the estimator
                if (win_mean(Sc, gt, w, PRE_LO, PRE_HI) is None
                        or win_mean(Sc, gt, w, POST_LO, POST_HI) is None):
                    continue
                prof = np.full(len(OFFSETS), np.nan)
                for j, off in enumerate(OFFSETS):
                    i = int(np.searchsorted(gt, w + off))
                    if 0 <= i < len(gt):
                        prof[j] = Sc[i]
                rows.append(prof)
                n_ev += 1
            if rows:
                stack = np.vstack(rows)
                with np.errstate(invalid="ignore"):
                    col_ok = np.isfinite(stack).any(axis=0)
                    prof_m = np.full(stack.shape[1], np.nan)
                    prof_m[col_ok] = np.nanmean(stack[:, col_ok], axis=0)
                per_match[m] = prof_m
        band = boot_band(per_match)
        if band is None:
            continue
        mean, lo, hi, nm = band
        profiles[cohort][cname] = {
            "offsets": OFFSETS,
            "mean_cents": [None if not np.isfinite(v) else C * float(v) for v in mean],
            "lo_cents": [None if not np.isfinite(v) else C * float(v) for v in lo],
            "hi_cents": [None if not np.isfinite(v) else C * float(v) for v in hi],
            "n_matches": nm, "n_events": n_ev}
        print(f"  [{cohort}] profile {cname:<16} matches={nm:>3} events={n_ev:>4}")

    # ── (b) threshold CCDF ────────────────────────────────────────────────
    # THREE units, because they are NOT interchangeable and conflating them was
    # a real defect in the first version of this file:
    #   event   -- one observation per isolated goal, the WINDOW MEAN of S.
    #              This is the unit the engine's threshold table uses, so the
    #              figure and the table agree only under this definition.
    #   match   -- event indicators averaged within match, then across matches,
    #              with a match bootstrap. This matches the PRIMARY estimand,
    #              which is match-weighted; the event-pooled table is not.
    #   instant -- raw grid points inside the windows. Descriptive only; it
    #              answers a different question and gives different numbers
    #              (~10.9% -> 53.8% near 0.25c versus 12.8% -> 59.2% by event).
    ev_pre, ev_post, ev_match = [], [], []
    inst_pre, inst_post = [], []
    for m, (Sc, gt, t0, t1, gw, gid, _elig) in prepped.items():
        for w in gw:
            if w - WIN < t0 or w + WIN > t1:
                continue
            if not all(abs(w - o) > ISO_GAP for o in gw if abs(o - w) > 1e-9):
                continue
            a = win_mean(Sc, gt, w, PRE_LO, PRE_HI)
            b = win_mean(Sc, gt, w, POST_LO, POST_HI)
            if a is None or b is None:
                continue
            ev_pre.append(a)
            ev_post.append(b)
            ev_match.append(m)
            for lo_, hi_, sink in ((PRE_LO, PRE_HI, inst_pre),
                                   (POST_LO, POST_HI, inst_post)):
                i0 = int(np.searchsorted(gt, w + lo_))
                i1 = int(np.searchsorted(gt, w + hi_))
                sink.extend([float(v) for v in Sc[i0:i1 + 1] if np.isfinite(v)])

    grid = np.concatenate([[0.0], np.geomspace(1e-4, 0.12, 220)])
    ep, eq = np.array(ev_pre), np.array(ev_post)
    ip, iq = np.array(inst_pre), np.array(inst_post)
    mkeys = sorted(set(ev_match))
    midx = {m: i for i, m in enumerate(mkeys)}
    marr = np.array([midx[m] for m in ev_match])

    def match_band(vals):
        """Match-weighted exceedance with a match-block bootstrap ribbon."""
        per_match = np.full((len(mkeys), len(grid)), np.nan)
        for i in range(len(mkeys)):
            sel = marr == i
            if sel.any():
                per_match[i] = (vals[sel][:, None] > grid[None, :]).mean(axis=0)
        point = np.nanmean(per_match, axis=0)
        idx = rng.integers(0, len(mkeys), size=(N_BOOT, len(mkeys)))
        draws = np.nanmean(per_match[idx], axis=1)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        return point, lo, hi

    mp_pre, lo_pre, hi_pre = match_band(ep)
    mp_post, lo_post, hi_post = match_band(eq)

    ccdf_out[cohort] = {
        "threshold_cents": [C * float(t) for t in grid],
        "event": {"pre": [float((ep > t).mean()) for t in grid],
                  "post": [float((eq > t).mean()) for t in grid],
                  "n_events": int(ep.size)},
        "match_weighted": {
            "pre": [float(v) for v in mp_pre],
            "pre_lo": [float(v) for v in lo_pre],
            "pre_hi": [float(v) for v in hi_pre],
            "post": [float(v) for v in mp_post],
            "post_lo": [float(v) for v in lo_post],
            "post_hi": [float(v) for v in hi_post],
            "n_matches": len(mkeys)},
        "instant": {"pre": [float((ip > t).mean()) for t in grid],
                    "post": [float((iq > t).mean()) for t in grid],
                    "n_pre_instants": int(ip.size),
                    "n_post_instants": int(iq.size)},
        "median_tick_cents": 1.0}
    for lab, t in ((" 0.25c", 0.0025), (" 1.00c", 0.01)):
        j = int(np.argmin(np.abs(grid - t)))
        print(f"  [{cohort}] CCDF at{lab}: event {100*(ep > t).mean():.1f}% -> "
              f"{100*(eq > t).mean():.1f}%  |  match-wtd "
              f"{100*mp_pre[j]:.1f}% -> {100*mp_post[j]:.1f}%  |  instant "
              f"{100*(ip > t).mean():.1f}% -> {100*(iq > t).mean():.1f}%")
    print(f"  [{cohort}] CCDF units: {ep.size} events, {len(mkeys)} matches, "
          f"{ip.size:,} pre / {iq.size:,} post instants")

    # ── (c) full randomization null ────────────────────────────────────────
    obs_per_match = defaultdict(list)
    for m, (Sc, gt, t0, t1, gw, gid, _elig) in prepped.items():
        for w in gw:
            if w - WIN < t0 or w + WIN > t1:
                continue
            if not all(abs(w - o) > ISO_GAP for o in gw if abs(o - w) > 1e-9):
                continue
            a = win_mean(Sc, gt, w, PRE_LO, PRE_HI)
            b = win_mean(Sc, gt, w, POST_LO, POST_HI)
            if a is not None and b is not None:
                obs_per_match[m].append(b - a)
    obs_k = {m: len(v) for m, v in obs_per_match.items() if v}
    obs = float(np.mean([np.mean(obs_per_match[m]) for m in sorted(obs_k)]))
    usable = [m for m in sorted(obs_k) if len(prepped[m][6]) >= obs_k[m]]
    random.seed(SEED)
    null = []
    for _ in range(N_RAND):
        pm = []
        for m in usable:
            Sc, gt = prepped[m][0], prepped[m][1]
            ds = []
            for i in random.sample(prepped[m][6], obs_k[m]):
                t = gt[i]
                a = win_mean(Sc, gt, t, PRE_LO, PRE_HI)
                b = win_mean(Sc, gt, t, POST_LO, POST_HI)
                if a is not None and b is not None:
                    ds.append(b - a)
            if ds:
                pm.append(np.mean(ds))
        if pm:
            null.append(float(np.mean(pm)))
    null = np.array(null)
    null_out[cohort] = {
        "observed_cents": C * obs, "draws_cents": [C * float(v) for v in null],
        "n_draws": int(null.size), "n_exceed": int((null >= obs).sum()),
        "p": (1.0 + int((null >= obs).sum())) / (1.0 + null.size),
        "n_matches": len(obs_k), "n_events": int(sum(obs_k.values())),
        "seed": SEED}
    print(f"  [{cohort}] null draws={null.size} obs={C*obs:+.4f}c "
          f"max={C*null.max():+.4f}c")

json.dump(profiles, open(f"{FIGS}/event_time_profiles.json", "w"))
json.dump(ccdf_out, open(f"{FIGS}/threshold_ccdf.json", "w"))
json.dump(null_out, open(f"{FIGS}/randomization_null.json", "w"))

# ── cross-checks against the engine's own artifacts ────────────────────────
print("\nCROSS-CHECKS")
for cohort in profiles:
    try:
        eng = json.load(open(f"{OUT}/results/{cohort}/event_study.json"))
        g = profiles[cohort]["goal"]
        i_pre = [j for j, o in enumerate(OFFSETS) if PRE_LO <= o <= PRE_HI]
        i_post = [j for j, o in enumerate(OFFSETS) if POST_LO <= o <= POST_HI]
        mp = np.array([v for v in g["mean_cents"]], dtype=float)
        approx = np.nanmean(mp[i_post]) - np.nanmean(mp[i_pre])
        ok = (eng["isolated"]["n_events"] == g["n_events"]
              and eng["isolated"]["n_matches"] == g["n_matches"])
        print(f"  {cohort}: REPORTED D {C*eng['isolated']['D']:+.4f}c  |  "
              f"profile-column arithmetic {approx:+.4f}c "
              f"(offset-grid approximation, do NOT annotate this)")
        print(f"    events engine {eng['isolated']['n_events']} vs "
              f"profile {g['n_events']}; matches "
              f"{eng['isolated']['n_matches']} vs {g['n_matches']}  -> "
              f"{'COUNTS MATCH' if ok else '*** COUNTS DIFFER ***'}")
    except Exception as e:
        print(f"  {cohort}: cross-check unavailable ({e})")

print(f"\n-> {FIGS}/event_time_profiles.json")
print(f"-> {FIGS}/threshold_ccdf.json")
print(f"-> {FIGS}/randomization_null.json")
