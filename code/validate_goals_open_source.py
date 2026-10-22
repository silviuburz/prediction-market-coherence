#!/usr/bin/env python3
"""Validate the ESPN-derived goal table against an openly licensed source.

The paper reports that key events reproduce the regulation score in 104 of 104
matches. Both streams come from ESPN, so that is internal consistency, not
external validation -- a point raised in the deep review. This script closes the
gap by checking the same table against English Wikipedia (CC BY-SA), which is a
different source with a different provenance chain, and which we are free to
redistribute.

Wikipedia reports full-match scores including extra time. The estimator uses
regulation only, so the reference table strips extra-time goals using the
per-goal minutes published in the knockout match reports.

Matching is by fixture date plus team names, with an alias table for the naming
differences between ESPN and Wikipedia.

Run: python analysis/studies/world_cup/coherence/validate_goals_open_source.py
"""
import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "figures" / "wc_coherence" / "data"

wiki = json.loads((DATA / "wikipedia_wc2026_results.json").read_text(encoding="utf-8"))
ours = json.loads((DATA / "regulation_scores.json").read_text(encoding="utf-8"))

# ESPN and Wikipedia disagree on several country names; normalise both sides.
ALIAS = {
    "southkorea": "southkorea", "korearepublic": "southkorea", "korea": "southkorea",
    "czechia": "czechrepublic", "czechrepublic": "czechrepublic",
    "capeverde": "capeverde", "caboverde": "capeverde",
    "drcongo": "drcongo", "congodr": "drcongo", "dccongo": "drcongo",
    "ivorycoast": "ivorycoast", "cotedivoire": "ivorycoast",
    "turkiye": "turkey", "turkey": "turkey",
    "usa": "unitedstates", "unitedstates": "unitedstates",
    "iran": "iran", "irisiran": "iran",
    "bosniaandherzegovina": "bosnia", "bosniaherzegovina": "bosnia", "bosnia": "bosnia",
    "curacao": "curacao", "curaao": "curacao",
}


def norm(s):
    # ESPN writes "T\u00fcrkiye" and "Cura\u00e7ao"; strip diacritics before folding
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    k = re.sub(r"[^a-z]", "", s.lower())
    return ALIAS.get(k, k)


def date_candidates(d):
    """Our slugs are dated in UTC; Wikipedia uses the local kickoff date.

    A late kickoff on the US west coast (UTC-7) falls on the next UTC day, so
    three fixtures differ by one. Accept a one-day window in either direction --
    no two fixtures in this tournament share both a date-window and a team pair,
    so this cannot create a false match.
    """
    dt = datetime.strptime(d, "%Y-%m-%d")
    return [(dt + timedelta(days=k)).strftime("%Y-%m-%d") for k in (0, -1, 1)]


wiki_by_key = {}
for m in wiki["matches"]:
    key = (m["date"], frozenset({norm(m["home"]), norm(m["away"])}))
    wiki_by_key[key] = m

print("=" * 74)
print("GOAL TABLE vs OPENLY LICENSED SOURCE (Wikipedia, CC BY-SA)")
print("=" * 74)
print(f"  our matches            : {ours['n_matches']}")
print(f"  our regulation goals   : {ours['n_regulation_goals']}")
print(f"  wiki matches           : {len(wiki['matches'])}")
print(f"  wiki all goals         : {wiki['_totals']['wikipedia_all_goals']}")
print(f"  wiki extra-time goals  : {wiki['_totals']['extra_time_goals_excluded']}")
print(f"  wiki regulation goals  : {wiki['_totals']['regulation_goals']}")

wiki_reg_sum = sum(m["h"] + m["a"] for m in wiki["matches"])
assert wiki_reg_sum == wiki["_totals"]["regulation_goals"], \
    f"reference table sums to {wiki_reg_sum}, header claims " \
    f"{wiki['_totals']['regulation_goals']}"
print(f"  reference table sums   : {wiki_reg_sum}  (self-consistent)")

unmatched, score_ok, score_bad, count_ok, count_bad = [], 0, [], 0, []
offset_days = []
for r in ours["matches"]:
    slug_date = r["match"][-10:]
    pair = frozenset({norm(r["home_team"]), norm(r["away_team"])})
    w, used = None, None
    for cand in date_candidates(slug_date):
        if (cand, pair) in wiki_by_key:
            w, used = wiki_by_key[(cand, pair)], cand
            break
    if w is None:
        unmatched.append((r["match"], r["home_team"], r["away_team"]))
        continue
    if used != slug_date:
        offset_days.append((r["match"], used))
    # orientation: our H/A are ESPN's home/away, which need not match Wikipedia's
    same = norm(r["home_team"]) == norm(w["home"])
    wh, wa = (w["h"], w["a"]) if same else (w["a"], w["h"])
    if (r["H"], r["A"]) == (wh, wa):
        score_ok += 1
    else:
        score_bad.append((r["match"], f"ours {r['H']}-{r['A']}", f"wiki {wh}-{wa}"))
    if r["n_regulation_goals"] == w["h"] + w["a"]:
        count_ok += 1
    else:
        count_bad.append((r["match"], r["n_regulation_goals"], w["h"] + w["a"]))

print(f"\n  fixtures matched       : {len(ours['matches']) - len(unmatched)}"
      f" / {len(ours['matches'])}")
print(f"  oriented score agrees  : {score_ok}")
print(f"  key-event count agrees : {count_ok}")
if offset_days:
    print(f"  matched at UTC offset  : {len(offset_days)}"
          "  (slug dated UTC, Wikipedia dated local kickoff)")
    for m, d in offset_days:
        print(f"    {m}  ->  wiki {d}")

for label, bad in (("UNMATCHED FIXTURES", unmatched),
                   ("SCORE MISMATCHES", score_bad),
                   ("GOAL-COUNT MISMATCHES", count_bad)):
    if bad:
        print(f"\n  {label} ({len(bad)}):")
        for b in bad:
            print(f"    {b}")

ok = (not unmatched) and (not score_bad) and (not count_bad)
print("\n" + ("PASS - the ESPN-derived goal table is independently corroborated"
              if ok else "FAIL - see mismatches above"))
sys.exit(0 if ok else 1)
