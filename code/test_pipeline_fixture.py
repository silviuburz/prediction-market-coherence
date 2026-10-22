#!/usr/bin/env python3
"""End-to-end pipeline check on a synthetic corpus, with no raw data required.

The 23.70 GB corpus cannot be redistributed, so nothing in the package otherwise
exercises the path from captured files to an estimate. This writes a small
synthetic corpus in the layout the collectors produce, points the frozen modules
at it, and runs the real code over it: manifest ingestion, payoff compilation,
quote-interval construction, the linear program, and event aggregation.

The fixture is built so the answer is known in advance. Before the goal the
quoted intervals admit a common distribution, so S is zero. After it, the
moneyline legs are moved so their lower bounds sum above one, which no
distribution can price, and S must rise to a value the test derives in closed
form.

The frozen producers are not modified; only their corpus roots are redirected,
exactly as the sensitivities in the supplement do.

    python code/test_pipeline_fixture.py
"""
import gzip
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MATCH = "aaa-bbb-2026-01-01"
GAME_ID = "9000001"
T1, T2 = "aaa", "bbb"
KICKOFF = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
GOAL_OFFSET = 1800          # seconds after kickoff
SPAN = 900                  # seconds either side of the goal that we quote
STEP = 10                   # collector cadence

# slug suffix -> (token id, pre-goal (bid, ask), post-goal (bid, ask))
# Post-goal moneyline lower bounds sum to 1.05, which is infeasible; the rest
# are left wide so they add no restriction.
CONTRACTS = {
    T1:        ("tok_home", (0.30, 0.40), (0.55, 0.65)),
    T2:        ("tok_away", (0.25, 0.35), (0.25, 0.35)),
    "draw":    ("tok_draw", (0.30, 0.40), (0.25, 0.35)),
    "btts":    ("tok_btts", (0.10, 0.90), (0.10, 0.90)),
    "total-0pt5": ("tok_t05", (0.10, 0.90), (0.10, 0.90)),
    "total-1pt5": ("tok_t15", (0.10, 0.90), (0.10, 0.90)),
    "total-2pt5": ("tok_t25", (0.10, 0.90), (0.10, 0.90)),
    "exact-score-0-0": ("tok_e00", (0.02, 0.98), (0.02, 0.98)),
    "exact-score-1-0": ("tok_e10", (0.02, 0.98), (0.02, 0.98)),
    "exact-score-1-1": ("tok_e11", (0.02, 0.98), (0.02, 0.98)),
    "exact-score-any-other": ("tok_eao", (0.02, 0.98), (0.02, 0.98)),
    "team-total-home-0pt5": ("tok_th0", (0.10, 0.90), (0.10, 0.90)),
    "team-total-away-0pt5": ("tok_ta0", (0.10, 0.90), (0.10, 0.90)),
    "spread-home-1pt5": ("tok_sh1", (0.10, 0.90), (0.10, 0.90)),
}


def iso(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_jsonl_gz(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def build_corpus(root):
    live = f"{root}/live/2026-01-01/window"

    markets = []
    for suffix, (token, _pre, _post) in CONTRACTS.items():
        markets.append({
            "slug": f"fifwc-{MATCH}-{suffix}",
            "condition_id": f"0x{token}",
            "clob_token_ids": [token, f"{token}_no"],
            "group_item_title": suffix,
            "outcomes": ["Yes", "No"],
            "outcome_prices": ["0", "1"],
        })
    manifest = {"markets_by_event": {"evt": markets}}
    os.makedirs(f"{root}/daily/2026-01-01", exist_ok=True)
    with gzip.open(f"{root}/daily/2026-01-01/manifest.json.gz", "wt") as fh:
        json.dump(manifest, fh)

    # scoreboard: one game, in progress, final regulation score 1-0
    rows = []
    for k in range(0, 3):
        t = KICKOFF.timestamp() + GOAL_OFFSET + (k - 1) * SPAN
        rows.append({
            "game_id": GAME_ID,
            "captured_at": iso(datetime.fromtimestamp(t, tz=timezone.utc)),
            "kickoff_iso": iso(KICKOFF),
            "home_team": "Team Aaa", "away_team": "Team Bbb",
            "home_abbrev": T1, "away_abbrev": T2,
            "period": 1, "status": "in",
            "home_score": 0 if k == 0 else 1, "away_score": 0,
        })
    write_jsonl_gz(f"{live}/_espn.jsonl.gz", rows)

    goal_w = KICKOFF.timestamp() + GOAL_OFFSET
    summary = {
        "espn_game_id": GAME_ID,
        "keyEvents": [{
            "id": "evt-goal-1", "scoringPlay": True,
            "period": {"number": 1},
            "clock": {"displayValue": "30'"},
            "wallclock": iso(datetime.fromtimestamp(goal_w, tz=timezone.utc)),
        }],
    }
    write_jsonl_gz(f"{live}/_espn_summary.jsonl.gz", [summary])

    for suffix, (token, pre, post) in CONTRACTS.items():
        slug = f"fifwc-{MATCH}-{suffix}"
        books = []
        t = goal_w - SPAN
        while t <= goal_w + SPAN:
            bid, ask = pre if t < goal_w else post
            books.append({
                "captured_at": iso(datetime.fromtimestamp(t, tz=timezone.utc)),
                "book": {
                    "asset_id": token,
                    "bids": [{"price": f"{bid:.4f}", "size": "500"}],
                    "asks": [{"price": f"{ask:.4f}", "size": "500"}],
                    "tick_size": "0.001",
                },
            })
            t += STEP
        write_jsonl_gz(f"{live}/{slug}/books.jsonl.gz", books)
    return goal_w


def main():
    root = tempfile.mkdtemp(prefix="wc_fixture_")
    try:
        goal_w = build_corpus(root)

        import compiler as cp
        cp.ROOT, cp.LIVE = root, f"{root}/live"
        cp._ESPN_INDEX = None
        import event_study as es
        es.LIVE = f"{root}/live"

        M, mpath = cp.load_manifest()
        print(f"ingestion      manifest read from {os.path.basename(mpath)}, "
              f"{len(cp.match_markets(M, MATCH))} markets for the fixture match")

        contracts, info = cp.build_contracts(M, MATCH)
        assert contracts, "compilation produced no contracts"
        print(f"compilation    {len(contracts)} contracts compiled, "
              f"{info['n_failed_closed']} refused, "
              f"regulation score {info['H']}-{info['A']}")

        reps = cp.state_classes(contracts)
        print(f"state space    {len(reps)} payoff-equivalence classes")

        goals = es.goals_of(info["game_id"])
        assert goals, "no goal recovered from the synthetic summary feed"
        print(f"event feed     {len(goals)} goal at "
              f"{goals[0]['clock']}, offset {goals[0]['w'] - goal_w:+.0f}s "
              f"from the planted wallclock")

        # quote construction and the LP, before and after the goal
        import numpy as np
        X = np.array([[c["rule"](h, a) for (h, a) in reps] for c in contracts],
                     dtype=float)
        ser, _ = {}, None
        for c in contracts:
            books = es.load_slug_books(c["slug"])
            got = books.get(c["canon_token"])
            if got:
                ser[c["slug"]] = got

        def solve_at(t):
            lows, highs = [], []
            for c in contracts:
                got = ser.get(c["slug"])
                env = es.token_quote_env(got[0], got[1], t) if got else None
                if env is None or env[0] is None or env[1] is None:
                    lows.append(0.0)
                    highs.append(1.0)
                else:
                    lows.append(env[0])
                    highs.append(env[1])
            return es.solve_S(X, lows, highs)

        pre = solve_at(goal_w - 60)
        post = solve_at(goal_w + 20)
        print(f"quotes and LP  S before the goal {pre:.4f}, after {post:.4f}")

        # closed form: post-goal moneyline lower bounds are 0.55 + 0.25 + 0.25
        # = 1.05 over three exclusive states, so every bound must move out by
        # (1.05 - 1) / 3.
        expected = (1.05 - 1.0) / 3.0
        assert abs(pre) < 1e-9, f"pre-goal quotes should be coherent, got {pre}"
        assert abs(post - expected) < 1e-6, \
            f"post-goal S is {post}, expected {expected}"
        print(f"aggregation    change {100 * (post - pre):+.3f} cents, "
              f"matching the closed form {100 * expected:+.3f}")
        print("\nthe pipeline runs from captured files to an estimate on data "
              "shipped with this package")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
