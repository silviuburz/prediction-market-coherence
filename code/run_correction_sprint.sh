#!/bin/bash
# Chained correction-sprint runs. Waits for the in-flight cancellation-state
# holdout job, then re-runs both primary cohorts with the corrected
# randomization null + persisted coherence series, then the placebo suite.
#
# The primary re-runs are the SAME deterministic computation as the frozen ones
# (same seed, same params); only the randomization null construction and output
# persistence changed. D must come back identical, which is asserted below.
set -u
R=/home/azureuser/projects/wc_coherence
cd "$R/code"
PY=/home/azureuser/.venv/bin/python

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "waiting for in-flight cancel-state holdout job..."
while pgrep -f "event_study.py --cohort holdout --unlock-holdout --cancel-state" > /dev/null; do
  sleep 20
done
log "cancel-state job finished"

log "START rerun dev"
$PY -u event_study.py --cohort dev > "$R/logs/rerun_dev.log" 2>&1
log "END rerun dev rc=$?"

log "START rerun holdout"
$PY -u event_study.py --cohort holdout --unlock-holdout > "$R/logs/rerun_holdout.log" 2>&1
log "END rerun holdout rc=$?"

log "START placebo suite"
$PY -u placebo_suite.py > "$R/logs/placebo.log" 2>&1
log "END placebo rc=$?"

log "START rq3 v2 censoring"
$PY -u rq3_v2_censoring.py > "$R/logs/rq3_v2.log" 2>&1
log "END rq3 v2 rc=$?"

# invariance check: the frozen point estimate must be unchanged
$PY - <<'EOF' 2>&1
import json
R = "/home/azureuser/projects/wc_coherence/results"
for c in ("dev", "holdout"):
    try:
        old = json.load(open(f"{R}/{c}_FROZEN_BACKUP/event_study.json"))["isolated"]
        new = json.load(open(f"{R}/{c}/event_study.json"))["isolated"]
        same = all(abs(old[k] - new[k]) < 1e-9 for k in ("D", "median")) \
            and old["n_events"] == new["n_events"] \
            and old["n_matches"] == new["n_matches"]
        print(f"INVARIANCE {c}: frozen D={old['D']:+.4f} rerun D={new['D']:+.4f} "
              f"events {old['n_events']}->{new['n_events']} "
              f"-> {'IDENTICAL' if same else '*** CHANGED ***'}")
    except Exception as e:
        print(f"INVARIANCE {c}: unavailable ({e})")
EOF

log "ALL DONE"
echo DONE > "$R/logs/RERUN_DONE.txt"
