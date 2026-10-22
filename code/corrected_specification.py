#!/usr/bin/env python3
"""Corrected-specification re-estimate: complete state space AND complete compiler.

Answers IEEE Access reviewer 2, comments 1 and 2 on Access-2026-41761, which ask
that the main analysis use the complete settlement state space and the corrected
compiler rather than reporting both only as supplementary sensitivities.

Two corrections are applied together, which no previous run did:

  1. The match-cancelled settlement state is appended to the payoff matrix
     (`event_study.run_cohort(..., cancel_state=True)`), as in S2.2.
  2. `first-to-score-neither` is compiled to 1{h == 0 and a == 0}, as in S2.6.
     Checked against recorded settlement across all 103 markets carrying it,
     with no disagreement.

The cancelled-state payoff of the newly compiled family is NOT left to the
`cancel_payoff` default of 0.0. The venue's own resolution text was read for all
104 `first-to-score-neither` markets in the frozen baseline and is identical in
every one:

    "If the game is canceled entirely, with no make-up game, this market will
     resolve to \"Neither\"."

So the contract pays 1.0 in the cancelled state, and `cancel_payoff` is patched
accordingly. Taking the default would have priced the contract at 0 in a state
where the venue says it pays 1.

Neither frozen producer is edited. `compiler.compile_rule` and
`compiler.cancel_payoff` are patched at run time and restored in a `finally`
block, exactly as the two sensitivities this run combines already do. The frozen
confirmatory estimate is untouched and remains the registered result.

Writes results/<cohort>_corrected/ and results/corrected_specification.json.

Run detached:
    cd /home/azureuser/projects/wc_coherence/code
    setsid nohup /home/azureuser/.venv/bin/python -u corrected_specification.py \
        > ../logs/corrected.out 2>&1 < /dev/null &
    tail -f ../logs/corrected_specification.log
"""
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compiler as cp                                   # noqa: E402
import event_study as es                                # noqa: E402

OUT = es.OUT
COHORTS = ["dev", "holdout"]

CANCEL_CLAUSE = ('If the game is canceled entirely, with no make-up game, this '
                 'market will resolve to "Neither".')
CANCEL_CLAUSE_SUPPORT = "104 of 104 first-to-score-neither markets, verbatim identical"

os.makedirs(f"{OUT}/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(f"{OUT}/logs/corrected_specification.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("corrected")

_ORIG_COMPILE = cp.compile_rule
_ORIG_CANCEL = cp.cancel_payoff
NEITHER = re.compile(r"first-to-score-neither$")


def patched_compile_rule(slug, match, home_code, away_code):
    r = _ORIG_COMPILE(slug, match, home_code, away_code)
    if r is not None:
        return r
    s = (slug or "").lower()
    pref = f"fifwc-{match}-"
    if s.startswith(pref) and NEITHER.search(s[len(pref):]):
        return ("first-to-score", lambda h, a: 1.0 if (h == 0 and a == 0) else 0.0)
    return None


def patched_cancel_payoff(c):
    """Cancelled-state payoff, with the newly compiled family read from the venue."""
    if c["family"] == "first-to-score":
        return 1.0
    return _ORIG_CANCEL(c)


def main():
    t0 = time.time()
    log.info("corrected specification: cancel state + first-to-score-neither")
    log.info("cancel clause verified on %s", CANCEL_CLAUSE_SUPPORT)
    log.info("clause: %s", CANCEL_CLAUSE)

    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("IEEE Access Access-2026-41761 reviewer 2 comments 1 and 2: "
                    "complete settlement state space and corrected compiler, "
                    "applied together in the main specification."),
        "corrections": [
            {"id": "cancel_state",
             "detail": "match-cancelled settlement state appended to the payoff matrix",
             "reference": "Supplement S2.2"},
            {"id": "first_to_score_neither",
             "detail": "first-to-score-neither compiled to 1{h == 0 and a == 0}",
             "settlement_check": "103 of 103 markets agree, 0 disagreements",
             "reference": "Supplement S2.6"},
            {"id": "first_to_score_cancel_payoff",
             "detail": "cancelled-state payoff of the newly compiled family set to 1.0",
             "evidence": CANCEL_CLAUSE,
             "support": CANCEL_CLAUSE_SUPPORT,
             "note": ("Not the cancel_payoff default of 0.0. Read from the frozen "
                      "baseline market descriptions, not assumed.")},
        ],
        "frozen_reference": {
            "holdout_D_cents": 0.4361387491631849,
            "dev_D_cents": 0.6674,
            "note": "frozen confirmatory estimate, unrevised, retained as the registered result",
        },
        "runs": [],
    }

    cp.compile_rule = patched_compile_rule
    cp.cancel_payoff = patched_cancel_payoff
    if hasattr(es, "compile_rule"):
        es.compile_rule = patched_compile_rule
    if hasattr(es, "cancel_payoff"):
        es.cancel_payoff = patched_cancel_payoff

    try:
        for cohort in COHORTS:
            matches = split[cohort]
            label = f"{cohort}_corrected"
            log.info("=== %s: %d matches, cancel state + neither contract ===",
                     label, len(matches))
            started = time.time()
            out = es.run_cohort(matches, label, cancel_state=True)
            iso = out.get("isolated", {})
            alle = out.get("all", {})
            row = {
                "cohort": cohort,
                "label": label,
                "D_cents": iso["D"] * 100 if iso.get("D") is not None else None,
                "ci_cents": [v * 100 for v in iso["ci"]] if iso.get("ci") else None,
                "n_events": iso.get("n_events"),
                "n_matches": iso.get("n_matches"),
                "n_pos": iso.get("n_pos"),
                "loo_min_cents": iso["loo_min"] * 100 if iso.get("loo_min") is not None else None,
                "loo_max_cents": iso["loo_max"] * 100 if iso.get("loo_max") is not None else None,
                "all_D_cents": alle["D"] * 100 if alle.get("D") is not None else None,
                "all_n_events": alle.get("n_events"),
                "rand_p": out.get("rand_p"),
                "seconds": round(time.time() - started, 1),
            }
            summary["runs"].append(row)
            log.info("  D=%.4fc ci=[%.4f,%.4f] events=%s matches=%s pos=%s p=%s (%.0fs)",
                     row["D_cents"], row["ci_cents"][0], row["ci_cents"][1],
                     row["n_events"], row["n_matches"], row["n_pos"],
                     row["rand_p"], row["seconds"])
            with open(f"{OUT}/results/corrected_specification.json", "w") as fh:
                json.dump(summary, fh, indent=2)
            log.info("  wrote results/corrected_specification.json")
    finally:
        cp.compile_rule = _ORIG_COMPILE
        cp.cancel_payoff = _ORIG_CANCEL
        if hasattr(es, "compile_rule"):
            es.compile_rule = _ORIG_COMPILE
        if hasattr(es, "cancel_payoff"):
            es.cancel_payoff = _ORIG_CANCEL
        log.info("restored frozen compiler entry points")

    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
