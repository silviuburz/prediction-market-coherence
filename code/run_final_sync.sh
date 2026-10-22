#!/bin/bash
# Final synchronisation run.
#
# The engine that produced the sprint results (sha 8e91b46d) differs from the
# committed engine only in a display statement (the rand_diag print was reporting
# probability units with a cents label). No stored value changes. This run exists
# so that the deployed code hash recorded in PROVENANCE.json corresponds exactly
# to the code that produced the artifacts it pins, rather than to a revision that
# differs by a print. The invariance check must again report IDENTICAL.
set -u
R=/home/azureuser/projects/wc_coherence
cd "$R/code"
PY=/home/azureuser/.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "engine sha256 $(sha256sum event_study.py | cut -c1-16)"

log "START dev"
$PY -u event_study.py --cohort dev > "$R/logs/final_dev.log" 2>&1
log "END dev rc=$?"

log "START holdout"
$PY -u event_study.py --cohort holdout --unlock-holdout > "$R/logs/final_holdout.log" 2>&1
log "END holdout rc=$?"

log "START randomization test"
$PY -u randomization_test.py > "$R/logs/randtest.log" 2>&1
log "END randomization rc=$?"

log "START provenance"
WC_GIT_COMMIT="${WC_GIT_COMMIT:-unset}" WC_GIT_BRANCH="${WC_GIT_BRANCH:-unset}" \
  $PY -u provenance_manifest.py > "$R/logs/provenance.log" 2>&1
log "END provenance rc=$?"

$PY - <<'EOF' 2>&1
import json
R = "/home/azureuser/projects/wc_coherence/results"
for c in ("dev", "holdout"):
    old = json.load(open(f"{R}/{c}_FROZEN_BACKUP/event_study.json"))["isolated"]
    new = json.load(open(f"{R}/{c}/event_study.json"))["isolated"]
    same = (abs(old["D"] - new["D"]) < 1e-12
            and old["n_events"] == new["n_events"]
            and old["n_matches"] == new["n_matches"]
            and old["n_pos"] == new["n_pos"]
            and abs(old["ci"][0] - new["ci"][0]) < 1e-12
            and abs(old["ci"][1] - new["ci"][1]) < 1e-12)
    print(f"INVARIANCE {c}: frozen {old['D']:+.9f} rerun {new['D']:+.9f} "
          f"ev {old['n_events']} mt {old['n_matches']} pos {old['n_pos']} "
          f"-> {'IDENTICAL' if same else '*** CHANGED ***'}")
EOF

log "ALL DONE"
echo DONE > "$R/logs/FINAL_DONE.txt"
