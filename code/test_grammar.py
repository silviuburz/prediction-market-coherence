#!/usr/bin/env python3
"""Table-driven tests for the payoff grammar and its threshold boundaries.

The settlement check in §3.5 evaluates each contract at the one score its match
produced, and the description check in the same section reads the venue's stated
rule. Neither exercises a rule at both sides of its own threshold. These do:
every supported family is evaluated at the scores that bracket its boundary, and
every template the compiler is expected to refuse is asserted to compile to
nothing.

Runs standalone, needs no corpus:

    python code/test_grammar.py

and is also collected by pytest if that is available.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compiler import compile_rule  # noqa: E402

MATCH = "arg-che-2026-07-11"
HOME, AWAY = "arg", "che"
PREFIX = f"fifwc-{MATCH}-"

# (slug suffix, expected family, [(h, a, payoff), ...])
CASES = [
    # exact score: exactly one cell pays, including the 0-0 corner
    ("exact-score-0-0", "exact-score",
     [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0)]),
    ("exact-score-2-1", "exact-score",
     [(2, 1, 1), (1, 2, 0), (2, 0, 0), (3, 1, 0)]),
    ("exact-score-3-3", "exact-score",
     [(3, 3, 1), (2, 2, 0), (4, 4, 0)]),

    # moneyline: home, away, draw partition the outcome
    (HOME, "moneyline", [(1, 0, 1), (0, 0, 0), (0, 1, 0), (5, 4, 1)]),
    (AWAY, "moneyline", [(0, 1, 1), (0, 0, 0), (1, 0, 0), (4, 5, 1)]),
    ("draw", "moneyline", [(0, 0, 1), (2, 2, 1), (1, 0, 0), (0, 1, 0)]),

    # both teams to score
    ("btts", "btts", [(1, 1, 1), (2, 3, 1), (0, 1, 0), (1, 0, 0), (0, 0, 0)]),

    # total goals: the threshold is N + 0.5, so N goals is under, N+1 is over
    ("total-0pt5", "total-goals", [(0, 0, 0), (1, 0, 1), (0, 1, 1)]),
    ("total-2pt5", "total-goals",
     [(1, 1, 0), (2, 0, 0), (2, 1, 1), (1, 2, 1), (3, 3, 1)]),
    ("total-9pt5", "total-goals", [(5, 4, 0), (5, 5, 1), (9, 0, 0), (9, 1, 1)]),

    # team total: only the named side's goals count
    ("team-total-home-0pt5", "team-total",
     [(0, 5, 0), (1, 0, 1), (0, 0, 0)]),
    ("team-total-away-1pt5", "team-total",
     [(9, 1, 0), (0, 2, 1), (0, 1, 0)]),

    # spread: margin strictly greater than the threshold
    ("spread-home-1pt5", "spread",
     [(2, 0, 1), (1, 0, 0), (3, 1, 1), (2, 1, 0), (0, 2, 0)]),
    ("spread-away-2pt5", "spread",
     [(0, 3, 1), (0, 2, 0), (1, 4, 1), (1, 3, 0)]),
]

# templates the compiler must refuse
REJECT = [
    "first-half-total-1pt5",          # half-time market, excluded by design
    "second-half-btts",
    "halftime-draw",
    "corners-total-8pt5",             # not a function of the score
    "shots-lionel-messi-gte2",
    "goals-lionel-messi-gte1",
    "first-to-score-home",            # which side scores first is not the score
    "team-to-advance",
    "exact-score-2",                  # malformed
    "total-2pt5x",
    "spread-middle-1pt5",
]


def run():
    failures = []
    checked = 0

    for suffix, family, table in CASES:
        got = compile_rule(PREFIX + suffix, MATCH, HOME, AWAY)
        if got is None:
            failures.append(f"{suffix}: did not compile")
            continue
        fam, rule = got
        if fam != family:
            failures.append(f"{suffix}: family {fam!r}, expected {family!r}")
        for h, a, want in table:
            checked += 1
            have = rule(h, a)
            if have != want:
                failures.append(
                    f"{suffix}: payoff at ({h},{a}) is {have}, expected {want}")

    for suffix in REJECT:
        checked += 1
        if compile_rule(PREFIX + suffix, MATCH, HOME, AWAY) is not None:
            failures.append(f"{suffix}: compiled, but should have been refused")

    # a slug belonging to a different match must never compile into this one
    checked += 1
    if compile_rule("fifwc-bra-fra-2026-07-01-btts", MATCH, HOME, AWAY) is not None:
        failures.append("slug from another match compiled into this fixture")

    return checked, failures


def test_grammar():
    _, failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    n, bad = run()
    print(f"{n} grammar assertions over {len(CASES)} templates "
          f"and {len(REJECT)} refusals")
    if bad:
        print(f"\n{len(bad)} FAILURES:")
        for f in bad:
            print(f"  {f}")
        sys.exit(1)
    print("all payoff rules agree with the truth table at their boundaries")
