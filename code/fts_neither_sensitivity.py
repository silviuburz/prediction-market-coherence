#!/usr/bin/env python3
"""Add the first-to-score `neither` contract and re-estimate.

A reviewer found that `first-to-score-neither` is dropped as an unsupported
template even though "neither team to score first" is decided entirely by the
regulation-time score: it pays when the match ends 0-0. Tested against recorded
settlement across all 103 markets carrying it, including nine 0-0 matches and the
0-0 final, the rule 1{h=0 and a=0} reproduces every settlement with no
disagreement.

The compiler that produced the frozen results is a recorded producer, so it is
not edited. This patches `compile_rule` at run time and re-estimates, exactly as
the cancellation-state sensitivity does. The frozen primary is untouched.

Adding a contract can only raise S at a fixed instant, since the system gains a
constraint. Its effect on the change around a goal is not signed in advance and
has to be measured, which is what this does.

The sibling `home` and `away` legs stay out: which team scores first is not a
function of the final score.

Writes results/<cohort>_ftsn/ and results/first_to_score_neither.json.

Run detached:
    cd /home/azureuser/projects/wc_coherence/code
    setsid nohup /home/azureuser/.venv/bin/python -u fts_neither_sensitivity.py \
        > ../logs/ftsn.out 2>&1 < /dev/null &
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

os.makedirs(f"{OUT}/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(f"{OUT}/logs/fts_neither.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("ftsn")

_ORIGINAL = cp.compile_rule
NEITHER = re.compile(r"first-to-score-neither$")


def patched_compile_rule(slug, match, home_code, away_code):
    r = _ORIGINAL(slug, match, home_code, away_code)
    if r is not None:
        return r
    s = (slug or "").lower()
    pref = f"fifwc-{match}-"
    if s.startswith(pref) and NEITHER.search(s[len(pref):]):
        return ("first-to-score", lambda h, a: 1.0 if (h == 0 and a == 0) else 0.0)
    return None


def main():
    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "added_family": "first-to-score",
        "added_rule": "1{h == 0 and a == 0}",
        "settlement_check": "103 of 103 markets agree, 0 disagreements",
        "note": ("Post-holdout sensitivity. The frozen compiler is a recorded "
                 "producer and is not modified; the rule is added at run time. "
                 "Adding a contract can only raise S at a fixed instant."),
        "runs": [],
    }

    cp.compile_rule = patched_compile_rule
    # event_study imported the symbol directly, so rebind there too if present
    if hasattr(es, "compile_rule"):
        es.compile_rule = patched_compile_rule
    try:
        for cohort in COHORTS:
            matches = split[cohort]
            label = f"{cohort}_ftsn"
            log.info("=== %s: %d matches, with first-to-score-neither ===",
                     label, len(matches))
            started = time.time()
            out = es.run_cohort(matches, label)
            iso = out.get("isolated", {})
            row = {
                "cohort": cohort,
                "label": label,
                "D_cents": iso.get("D") * 100 if iso.get("D") is not None else None,
                "ci_cents": [v * 100 for v in iso["ci"]] if iso.get("ci") else None,
                "n_events": iso.get("n_events"),
                "n_matches": iso.get("n_matches"),
                "n_pos": iso.get("n_pos"),
                "seconds": round(time.time() - started, 1),
            }
            summary["runs"].append(row)
            log.info("  D=%.4fc ci=%s events=%s matches=%s pos=%s (%.0fs)",
                     row["D_cents"], [round(v, 4) for v in row["ci_cents"]],
                     row["n_events"], row["n_matches"], row["n_pos"],
                     row["seconds"])
            with open(f"{OUT}/results/first_to_score_neither.json", "w") as fh:
                json.dump(summary, fh, indent=2)
    finally:
        cp.compile_rule = _ORIGINAL
        if hasattr(es, "compile_rule"):
            es.compile_rule = _ORIGINAL

    log.info("done; wrote results/first_to_score_neither.json")


if __name__ == "__main__":
    main()
