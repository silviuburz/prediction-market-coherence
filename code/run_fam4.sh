#!/bin/bash
# Re-run family attribution with the size-matched random-drop control averaged
# over N_RAND_DROP independent subsets. The single-draw version was seed
# sensitive (the exact-score control moved 34% -> 27% when the non-deterministic
# hash seed was replaced with a SHA-256 seed), which is too fragile to publish.
# Provenance is regenerated afterwards so it pins the final artifacts.
set -u
R=/home/azureuser/projects/wc_coherence
cd "$R/code"
PY=/home/azureuser/.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if grep -q "hash(match)" family_attribution.py; then
  log "FATAL: non-deterministic seed present"; exit 1
fi
log "attribution sha $(sha256sum family_attribution.py | cut -c1-16)"
log "N_RAND_DROP=$(grep -m1 'N_RAND_DROP =' family_attribution.py | tr -dc 0-9)"

log "START dev"
$PY -u family_attribution.py --cohort dev > "$R/logs/fam4_dev.log" 2>&1
log "END dev rc=$?"

log "START holdout"
$PY -u family_attribution.py --cohort holdout > "$R/logs/fam4_holdout.log" 2>&1
log "END holdout rc=$?"

log "START provenance (final)"
WC_GIT_COMMIT="${WC_GIT_COMMIT:-unset}" WC_GIT_BRANCH="${WC_GIT_BRANCH:-unset}" \
  $PY -u provenance_manifest.py > "$R/logs/provenance.log" 2>&1
log "END provenance rc=$?"

log "ALL DONE"
echo DONE > "$R/logs/FAM4_DONE.txt"
