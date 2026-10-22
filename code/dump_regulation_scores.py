#!/usr/bin/env python3
"""Dump the ESPN-derived regulation-time score for every match in the study.

Written so the goal table can be checked against an independently licensed
source (Wikipedia). Reads only the ESPN scoreboard and key-event streams, never
the order books, so it is cheap.

Emits, per match: the slug, the ESPN game id, both team names as ESPN reports
them, the regulation-time score, and the number of regulation key-event goals.

Run (from /home/azureuser/projects/wc_coherence):
  /home/azureuser/.venv/bin/python -u code/dump_regulation_scores.py
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_study import OUT, goals_of, load_manifest  # noqa: E402
from compiler import build_contracts  # noqa: E402


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
matches = sorted(set(split["dev"]) | set(split["holdout"]))
log(f"{len(matches)} matches")

M, _ = load_manifest()
rows = []
for i, match in enumerate(matches, 1):
    contracts, info = build_contracts(M, match)
    if contracts is None:
        log(f"  {match}: no contracts, skipped")
        continue
    gid = info.get("game_id")
    goals = goals_of(gid)
    # keyEvents goals are already restricted to periods 1-2 by goals_of
    rows.append({
        "match": match,
        "game_id": str(gid),
        "home_team": info.get("home_team"),
        "away_team": info.get("away_team"),
        "H": info.get("H"),
        "A": info.get("A"),
        "n_regulation_goals": len(goals),
        "goal_wallclocks": [g["w"] for g in goals],
    })
    if i % 20 == 0:
        log(f"  {i}/{len(matches)}")

total = sum(r["n_regulation_goals"] for r in rows)
log(f"matches with contracts : {len(rows)}")
log(f"regulation goals total : {total}")

dest = f"{OUT}/results/regulation_scores.json"
with open(dest, "w") as f:
    json.dump({"n_matches": len(rows), "n_regulation_goals": total,
               "matches": rows}, f, indent=2)
log(f"wrote {dest}")
