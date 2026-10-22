#!/usr/bin/env python3
"""Two-sided freshness sensitivity, answering the reviewer's central objection.

The frozen estimator bounds only the PRECEDING observation:

    if t - pt > P["FRESH"]:   # 30 s
        return None
    ... then uses series[i + 1] with no bound on how far ahead it is

so a contract whose next quote arrives minutes later still contributes, and its
interval spans a quote from the future. That is exactly the asynchronous
reconstruction the reviewer says the design cannot rule out.

This re-estimates with a STRICT two-sided bound: a contract enters an instant
only if both bracketing observations lie within `lead` seconds of it. Contracts
that fail are dropped, which shrinks the system, and by the monotonicity lemma a
smaller system can only lower S. So the contract count is reported alongside
every estimate: without it, tightening and shrinking are not separable.

Writes to results/<cohort>_lead<N>/ and never to the frozen labels.

Run detached:
    cd /home/azureuser/projects/wc_coherence/code
    nohup python3 -u freshness_twosided.py > ../logs/freshness.out 2>&1 &
"""
import json
import logging
import os
import sys
import time
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import event_study as es  # noqa: E402

OUT = es.OUT
LEADS = [5, 10, 15, 30]          # seconds; 30 matches the existing FRESH
COHORTS = ["dev", "holdout"]

os.makedirs(f"{OUT}/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(f"{OUT}/logs/freshness_twosided.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("freshness")

_ORIGINAL = es.token_quote_env


def strict_env(lead):
    """Both bracketing observations must be within `lead` seconds of t."""
    def token_quote_env(series, times, t):
        i = bisect_right(times, t) - 1
        if i < 0:
            return None
        pt, pb, pa = series[i]
        if t - pt > lead:
            return None
        if i + 1 >= len(series):
            return None                      # no forward observation: drop
        nt, nb, na = series[i + 1]
        if nt - t > lead:
            return None                      # forward observation too distant
        bid = min([v for v in (pb, nb) if v is not None], default=None)
        ask = max([v for v in (pa, na) if v is not None], default=None)
        return bid, ask
    return token_quote_env


def main():
    split = json.load(open(f"{OUT}/registrations/dev_holdout_split.json"))
    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "frozen_FRESH": es.P["FRESH"],
               "note": ("Strict two-sided bound. Contracts whose bracketing "
                        "observations are not both within `lead` seconds are "
                        "dropped, so system size falls with the bound and must "
                        "be read alongside the estimate."),
               "runs": []}

    for cohort in COHORTS:
        matches = split[cohort]
        for lead in LEADS:
            label = f"{cohort}_lead{lead}"
            log.info("=== %s: %d matches, both sides within %ds ===",
                     label, len(matches), lead)
            started = time.time()
            es.token_quote_env = strict_env(lead)
            try:
                out = es.run_cohort(matches, label)
            except Exception as exc:
                log.error("  FAILED %s: %s", label, exc)
                es.token_quote_env = _ORIGINAL
                continue
            es.token_quote_env = _ORIGINAL
            iso = out.get("isolated", {})
            row = {
                "cohort": cohort,
                "lead_seconds": lead,
                "label": label,
                "D_cents": (iso.get("D") * 100) if iso.get("D") is not None else None,
                "ci_cents": [v * 100 for v in iso["ci"]] if iso.get("ci") else None,
                "n_events": iso.get("n_events"),
                "n_matches": iso.get("n_matches"),
                "n_pos": iso.get("n_pos"),
                "seconds": round(time.time() - started, 1),
            }
            summary["runs"].append(row)
            log.info("  D=%.4fc ci=%s events=%s matches=%s (%.0fs)",
                     row["D_cents"] if row["D_cents"] is not None else float("nan"),
                     [round(v, 4) for v in row["ci_cents"]] if row["ci_cents"] else None,
                     row["n_events"], row["n_matches"], row["seconds"])
            with open(f"{OUT}/results/freshness_twosided.json", "w") as fh:
                json.dump(summary, fh, indent=2)

    log.info("done; wrote results/freshness_twosided.json")


if __name__ == "__main__":
    main()
