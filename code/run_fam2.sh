#!/bin/bash
# Family attribution with the size-matched random-drop control, both cohorts.
# Writes to family_attribution_v2_{cohort}.json so it cannot race the first-pass
# job that may still be running.
set -u
R=/home/azureuser/projects/wc_coherence
cd "$R/code"
PY=/home/azureuser/.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "START fam v2 dev"
$PY -u family_attribution.py --cohort dev > "$R/logs/fam2_dev.log" 2>&1
log "END dev rc=$?"

log "START fam v2 holdout"
$PY -u family_attribution.py --cohort holdout > "$R/logs/fam2_holdout.log" 2>&1
log "END holdout rc=$?"

log "ALL DONE"
echo DONE > "$R/logs/FAM2_DONE.txt"
