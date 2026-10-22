#!/bin/bash
# Post-audit rerun: deterministic attribution, corrected saturation, event-level
# figure inputs, then a regenerated provenance manifest LAST so it pins the final
# state of every producer and artifact.
set -u
R=/home/azureuser/projects/wc_coherence
cd "$R/code"
PY=/home/azureuser/.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# refuse to run the non-deterministic attribution seed
if grep -q "hash(match)" family_attribution.py; then
  log "FATAL: family_attribution.py still uses process-randomised hash(match)"
  exit 1
fi
log "attribution sha $(sha256sum family_attribution.py | cut -c1-16)"

log "START saturation"
$PY -u saturation_lemma.py > "$R/logs/saturation.log" 2>&1
log "END saturation rc=$?"

log "START figure inputs"
$PY -u figure_inputs.py > "$R/logs/figure_inputs.log" 2>&1
log "END figure inputs rc=$?"

log "START attribution dev"
$PY -u family_attribution.py --cohort dev > "$R/logs/fam3_dev.log" 2>&1
log "END attribution dev rc=$?"

log "START attribution holdout"
$PY -u family_attribution.py --cohort holdout > "$R/logs/fam3_holdout.log" 2>&1
log "END attribution holdout rc=$?"

# determinism proof: rerun dev attribution and require identical random-drop values
log "START determinism recheck (dev)"
$PY -u family_attribution.py --cohort dev > "$R/logs/fam3_dev_recheck.log" 2>&1
log "END determinism recheck rc=$?"

$PY - <<'EOF' 2>&1
import json, re
a = open("/home/azureuser/projects/wc_coherence/logs/fam3_dev.log").read()
b = open("/home/azureuser/projects/wc_coherence/logs/fam3_dev_recheck.log").read()
pat = re.compile(r"^\s{2}(\S+)\s+\d+\s+([+-]\d+\.\d+).*?(\d+)%\s*$", re.M)
ra, rb = pat.findall(a), pat.findall(b)
same = ra == rb and len(ra) > 0
print(f"DETERMINISM: {len(ra)} attribution rows, "
      f"{'IDENTICAL across reruns' if same else '*** DIFFER ***'}")
if not same:
    for x, y in zip(ra, rb):
        if x != y:
            print("   ", x, "vs", y)
EOF

log "START provenance (final)"
WC_GIT_COMMIT="${WC_GIT_COMMIT:-unset}" WC_GIT_BRANCH="${WC_GIT_BRANCH:-unset}" \
  $PY -u provenance_manifest.py > "$R/logs/provenance.log" 2>&1
log "END provenance rc=$?"

log "ALL DONE"
echo DONE > "$R/logs/AUDIT_RERUN_DONE.txt"
