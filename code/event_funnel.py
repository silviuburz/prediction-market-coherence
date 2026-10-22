#!/usr/bin/env python3
"""Event-selection funnel: how 300 regulation goals become the analysed sample.

Answers IEEE Access reviewer 2 comment 6 on Access-2026-41761, which asks for a
concise flow table showing how many events each eligibility, isolation and
data-availability criterion removes.

The counts are not recomputed from a reimplementation of the engine. They are
derived from `match_stats.json.gz`, which `event_study.run_cohort` writes for
every match it processes, so each stage below is the engine's own bookkeeping
read back. The two terminal counts are cross-checked against
`event_study_<cohort>.json`, and the script fails if they disagree.

Stages, in the order `run_cohort` applies them:

  1. source population       every regulation goal in the match universe
  2. cohort assignment       by the registered development/holdout split
  3. match processed         match compiles contracts and has at least 20
                             in-play scoreboard rows
  4. goal in capture window  goal falls inside the captured in-play span,
                             widened by 60 s at each end
  5. event eligibility       300 s of capture on each side of the goal
  6. window data present     both the pre-goal and post-goal windows contain at
                             least one instant that cleared the availability
                             gate and solved
  7. isolation               no other goal within 300 s

Usage:
    python event_funnel.py --match-stats <dir> --out <path>

`<dir>` holds `match_stats_dev.json.gz` and `match_stats_holdout.json.gz`.
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

WIN = 300           # capture required each side of an event, seconds
ISO_GAP = 300       # isolation gap, seconds
COHORTS = ("dev", "holdout")


def load_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build(match_stats_dir, data_dir):
    reg = load_json(data_dir / "regulation_scores.json")
    split = load_json(data_dir / "dev_holdout_split.json")
    goals_by_match = {r["match"]: r["n_regulation_goals"] for r in reg["matches"]}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("Event-selection funnel for IEEE Access reviewer 2 comment 6. "
                    "Stage counts are read from the engine's own match_stats, not "
                    "recomputed independently."),
        "criteria": {"capture_each_side_s": WIN, "isolation_gap_s": ISO_GAP},
        "source_population": {"matches": reg["n_matches"],
                              "regulation_goals": reg["n_regulation_goals"]},
        "cohorts": {},
    }

    for cohort in COHORTS:
        matches = split[cohort]
        ms = load_gz(match_stats_dir / f"match_stats_{cohort}.json.gz")

        cohort_goals = sum(goals_by_match.get(m, 0) for m in matches)
        processed = [m for m in matches if m in ms]
        in_window = 0
        eligible = 0
        with_data = 0
        isolated = 0
        matches_with_isolated = set()
        matches_with_any = set()

        for m in processed:
            st = ms[m]
            gw = st["goal_w"]
            in_window += len(gw)
            t_start, t_end = st["t_start"], st["t_end"]
            eligible += sum(1 for w in gw
                            if w - WIN >= t_start and w + WIN <= t_end)
            evs = st["events"]
            with_data += len(evs)
            if evs:
                matches_with_any.add(m)
            iso = [e for e in evs if e["isolated"]]
            isolated += len(iso)
            if iso:
                matches_with_isolated.add(m)

        out["cohorts"][cohort] = {
            "matches_assigned": len(matches),
            "regulation_goals": cohort_goals,
            "matches_processed": len(processed),
            "goals_in_capture_window": in_window,
            "dropped_outside_capture_window": cohort_goals - in_window,
            "eligible_events": eligible,
            "dropped_by_event_eligibility": in_window - eligible,
            "events_with_window_data": with_data,
            "dropped_for_missing_window_data": eligible - with_data,
            "isolated_events": isolated,
            "dropped_by_isolation": with_data - isolated,
            "matches_contributing_any_event": len(matches_with_any),
            "matches_contributing_isolated_event": len(matches_with_isolated),
        }

    return out


def crosscheck(funnel, data_dir):
    """Terminal counts must equal what the frozen engine reported."""
    problems = []
    for cohort in COHORTS:
        eng = load_json(data_dir / f"event_study_{cohort}.json")
        got = funnel["cohorts"][cohort]
        pairs = [
            ("events_with_window_data", got["events_with_window_data"],
             eng["all"]["n_events"]),
            ("matches_contributing_any_event", got["matches_contributing_any_event"],
             eng["all"]["n_matches"]),
            ("isolated_events", got["isolated_events"], eng["isolated"]["n_events"]),
            ("matches_contributing_isolated_event",
             got["matches_contributing_isolated_event"], eng["isolated"]["n_matches"]),
        ]
        for name, ours, theirs in pairs:
            if ours != theirs:
                problems.append(f"{cohort}.{name}: funnel {ours} != engine {theirs}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-stats", required=True,
                    help="directory holding match_stats_<cohort>.json.gz")
    ap.add_argument("--data", default=None,
                    help="artifact directory (default: repo figures/wc_coherence/data)")
    ap.add_argument("--out", default=None, help="output JSON path")
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[4]
    data_dir = Path(a.data) if a.data else repo / "figures" / "wc_coherence" / "data"
    out_path = Path(a.out) if a.out else data_dir / "event_funnel.json"

    funnel = build(Path(a.match_stats), data_dir)
    problems = crosscheck(funnel, data_dir)
    funnel["crosscheck"] = ("terminal counts agree with event_study_<cohort>.json"
                            if not problems else problems)

    total = {k: sum(funnel["cohorts"][c][k] for c in COHORTS)
             for k in funnel["cohorts"]["dev"]}
    funnel["total"] = total

    hdr = f"{'stage':<38}{'dev':>8}{'holdout':>10}{'total':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("matches assigned", "matches_assigned"),
        ("regulation goals", "regulation_goals"),
        ("  less: outside capture window", "dropped_outside_capture_window"),
        ("goals in capture window", "goals_in_capture_window"),
        ("  less: event eligibility (300 s)", "dropped_by_event_eligibility"),
        ("eligible events", "eligible_events"),
        ("  less: missing window data", "dropped_for_missing_window_data"),
        ("events with window data", "events_with_window_data"),
        ("  less: isolation (300 s)", "dropped_by_isolation"),
        ("isolated events analysed", "isolated_events"),
        ("matches contributing", "matches_contributing_isolated_event"),
    ]
    for label, key in rows:
        d = funnel["cohorts"]["dev"][key]
        h = funnel["cohorts"]["holdout"][key]
        print(f"{label:<38}{d:>8}{h:>10}{d + h:>8}")

    print()
    if problems:
        print("CROSSCHECK FAILED")
        for p in problems:
            print("  ", p)
    else:
        print("crosscheck: terminal counts agree with the frozen engine")

    out_path.write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
