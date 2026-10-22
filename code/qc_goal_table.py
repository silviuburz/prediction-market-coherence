#!/usr/bin/env python3
"""GOAL-TABLE QC — cross-validate two independent goal sources for all 104 matches.

Source A: `_espn_summary` -> keyEvents with scoringPlay=True, period<=2, wallclock
Source B: `_espn` scoreboard -> score INCREMENTS while period<=2

Decisive check: count(A) == count(B) == (H + A) of the regulation-time score.
Any disagreement is a data-quality defect that must be resolved before the goal
table is used for a confirmatory event study.

Also reports the timing offset (scoreboard detection time - keyEvents wallclock),
which quantifies our capture lag against ESPN's own event clock.

Known ground truth (user): fra-eng really did have 10 regulation goals; esp-arg's
only goal was in EXTRA TIME and must therefore be excluded by the period<=2 rule.

Read-only. Run: /home/azureuser/.venv/bin/python -u qc_goal_table.py
"""
import glob
import gzip
import json
import lzma
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                        # noqa: E402
from compiler import (LIVE, _canon, _norm, _sim, _tokens,  # noqa: E402
                      load_manifest)


def iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


print("=" * 104)
print("GOAL-TABLE QC — keyEvents vs scoreboard, all matches")
print("=" * 104)

# ── index every scoreboard row by game_id (once) ────────────────────────────
games = defaultdict(list)
for p in glob.glob(f"{LIVE}/*/*/_espn.jsonl.gz"):
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
                gid = str(d.get("game_id") or "")
                if gid:
                    games[gid].append(d)
    except Exception:
        continue
for gid in games:
    games[gid].sort(key=lambda d: iso(d.get("captured_at")) or 0)
print(f"scoreboard game_ids indexed: {len(games)}")

# ── index keyEvents by espn_game_id (once) ──────────────────────────────────
summ = {}
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
    gid = str(d.get("espn_game_id") or "")
    ke = d.get("keyEvents") or []
    if gid and (gid not in summ or len(ke) > len(summ[gid])):
        summ[gid] = ke
print(f"summary game_ids indexed: {len(summ)}")

# ── manifest matches + code->name ───────────────────────────────────────────
M, _ = load_manifest()
by_match = defaultdict(list)
for lst in M.get("markets_by_event", {}).values():
    for mk in lst:
        s = (mk.get("slug") or "").lower()
        m = re.match(r"fifwc-([a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2})", s)
        if m:
            by_match[m.group(1)].append(mk)
print(f"manifest matches: {len(by_match)}\n")


def resolve_gid(match):
    t1, t2 = match.split("-")[0], match.split("-")[1]
    mdate = match[-10:]
    cn = {}
    for mk in by_match[match]:
        s = (mk.get("slug") or "").lower()
        for c in (t1, t2):
            if s == f"fifwc-{match}-{c}":
                cn[c] = mk.get("group_item_title")
    n1, n2 = _norm(cn.get(t1)), _norm(cn.get(t2))
    k1, k2 = _canon(n1), _canon(n2)
    tk1, tk2 = _tokens(cn.get(t1)), _tokens(cn.get(t2))
    for gid, rows in games.items():
        meta = rows[-1]
        hn, an = _norm(meta.get("home_team")), _norm(meta.get("away_team"))
        kh, ka = _canon(hn), _canon(an)
        th, ta = _tokens(meta.get("home_team")), _tokens(meta.get("away_team"))
        ha, aa = _norm(meta.get("home_abbrev")), _norm(meta.get("away_abbrev"))
        kd = {str(r.get("kickoff_iso") or r.get("captured_at") or "")[:10] for r in rows}
        kd.discard("")
        try:
            md = datetime.strptime(mdate, "%Y-%m-%d")
            if kd and not any(abs((datetime.strptime(k, "%Y-%m-%d") - md).days) <= 1
                              for k in kd):
                continue
        except Exception:
            pass

        def side(code, nm, kn, tk):
            if nm and nm == hn:
                return "home"
            if nm and nm == an:
                return "away"
            if kn and kn == kh:
                return "home"
            if kn and kn == ka:
                return "away"
            if tk and tk == th:
                return "home"
            if tk and tk == ta:
                return "away"
            if code == ha:
                return "home"
            if code == aa:
                return "away"
            if nm and hn and (nm in hn or hn in nm):
                return "home"
            if nm and an and (nm in an or an in nm):
                return "away"
            sh, sa = _sim(nm, hn), _sim(nm, an)
            if max(sh, sa) >= 0.75 and abs(sh - sa) > 0.10:
                return "home" if sh > sa else "away"
            return None

        s1, s2 = side(t1, n1, k1, tk1), side(t2, n2, k2, tk2)
        if s1 and s2 and s1 != s2:
            return gid
    return None


rows_out, offsets = [], []
for match in sorted(by_match):
    gid = resolve_gid(match)
    rec = {"match": match, "game_id": gid, "status": "OK"}
    if not gid:
        rec["status"] = "NO_GAME_ID"
        rows_out.append(rec)
        continue
    grows = games[gid]

    # regulation score + score-increment times (source B)
    prev, H, A, incr = None, None, None, []
    for d in grows:
        try:
            per = int(d.get("period") or 0)
        except Exception:
            continue
        hs, as_ = d.get("home_score"), d.get("away_score")
        if not (per and per <= 2 and hs is not None and as_ is not None):
            continue
        cur = (int(hs), int(as_))
        if prev is not None and (cur[0] > prev[0] or cur[1] > prev[1]):
            n_new = (cur[0] - prev[0]) + (cur[1] - prev[1])
            for _ in range(max(0, n_new)):
                incr.append(iso(d.get("captured_at")))
        prev = cur
        H, A = cur
    rec.update({"reg_H": H, "reg_A": A,
                "reg_total": (H + A) if H is not None else None,
                "n_scoreboard_incr": len(incr)})

    # keyEvents goals (source A)
    ke = summ.get(gid)
    if ke is None:
        rec["status"] = "NO_SUMMARY"
        rec["n_keyevents"] = None
        rows_out.append(rec)
        continue
    ga = []
    for e in ke:
        try:
            if not e.get("scoringPlay"):
                continue
            per = int((e.get("period") or {}).get("number") or 0)
            if per < 1 or per > 2:
                continue
            w = iso(e.get("wallclock"))
            if w:
                ga.append(w)
        except Exception:
            continue
    ga.sort()
    dedup = [t for i, t in enumerate(ga) if i == 0 or t - ga[i - 1] > 30]
    rec["n_keyevents"] = len(dedup)
    rec["n_keyevents_raw"] = len(ga)

    if rec["reg_total"] is None:
        rec["status"] = "NO_REG_SCORE"
    elif len(dedup) != rec["reg_total"] or len(incr) != rec["reg_total"]:
        rec["status"] = "MISMATCH"

    for w in dedup:
        if incr:
            near = min((t for t in incr if t), key=lambda t: abs(t - w), default=None)
            if near:
                offsets.append(near - w)
    rows_out.append(rec)

ok = [r for r in rows_out if r["status"] == "OK"]
bad = [r for r in rows_out if r["status"] != "OK"]
print(f"{'match':<24}{'gid':>9}{'regHA':>8}{'tot':>5}{'keyEv':>7}{'sbIncr':>8}  status")
for r in rows_out:
    ha = f"{r.get('reg_H')}-{r.get('reg_A')}" if r.get("reg_H") is not None else "-"
    print(f"{r['match']:<24}{str(r.get('game_id') or '-'):>9}{ha:>8}"
          f"{str(r.get('reg_total') or '-'):>5}{str(r.get('n_keyevents') if r.get('n_keyevents') is not None else '-'):>7}"
          f"{str(r.get('n_scoreboard_incr', '-')):>8}  {r['status']}")

print("\n" + "=" * 104)
print(f"SUMMARY: {len(ok)}/{len(rows_out)} matches fully consistent; {len(bad)} flagged")
print("=" * 104)
cnt = defaultdict(int)
for r in bad:
    cnt[r["status"]] += 1
for k, v in sorted(cnt.items()):
    print(f"  {k:<16}{v}")

tot_goals = sum(r.get("n_keyevents") or 0 for r in rows_out)
tot_reg = sum(r.get("reg_total") or 0 for r in rows_out)
print(f"\n  total regulation goals: keyEvents={tot_goals}  scoreboard-implied={tot_reg}")

if offsets:
    o = np.array([x for x in offsets if x is not None])
    print(f"\n  CAPTURE LAG (scoreboard detection - keyEvents wallclock), n={len(o)}")
    print(f"    median {np.median(o):+.1f}s  p10 {np.percentile(o,10):+.1f}s  "
          f"p90 {np.percentile(o,90):+.1f}s  min {o.min():+.1f}s  max {o.max():+.1f}s")

out = "/home/azureuser/projects/wc_coherence/results/qc_goal_table.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"rows": rows_out,
           "lag_median": float(np.median(offsets)) if offsets else None},
          open(out, "w"), indent=2)
print(f"\nresults -> {out}")
