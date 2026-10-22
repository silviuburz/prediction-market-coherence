#!/bin/bash
# Correction-sprint runs, take 2.
#
# Take 1 ran with a stale event_study.py that persisted match_stats.json.gz
# WITHOUT the absolute-time anchors (grid_t0, t_start, t_end, goal_w, game_id),
# so the downstream offline analyses could not reconstruct the engine's grid.
# The headline numbers from take 1 were correct and reproduced the frozen
# estimates exactly; only the persisted series was unusable. Corrected engine
# deployed, both cohorts re-run from scratch.
set -u
R=/home/azureuser/projects/wc_coherence
cd "$R/code"
PY=/home/azureuser/.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if ! grep -q grid_t0 event_study.py; then
  log "FATAL: deployed event_study.py lacks grid_t0 — refusing to run"
  exit 1
fi
log "engine sha256 $(sha256sum event_study.py | cut -c1-16)"

log "START rerun dev"
$PY -u event_study.py --cohort dev > "$R/logs/rerun_dev.log" 2>&1
log "END rerun dev rc=$?"

log "START rerun holdout"
$PY -u event_study.py --cohort holdout --unlock-holdout > "$R/logs/rerun_holdout.log" 2>&1
log "END rerun holdout rc=$?"

log "START qc null tightness"
$PY -u qc_null_tightness.py > "$R/logs/qc_null.log" 2>&1
log "END qc null rc=$?"

log "START placebo suite"
$PY -u placebo_suite.py > "$R/logs/placebo.log" 2>&1
log "END placebo rc=$?"

log "START spec curve v3"
$PY -u spec_curve_v3.py > "$R/logs/spec_v3.log" 2>&1
log "END spec curve rc=$?"

log "START family attribution dev"
$PY -u family_attribution.py --cohort dev > "$R/logs/fam_dev.log" 2>&1
log "END family dev rc=$?"

log "START family attribution holdout"
$PY -u family_attribution.py --cohort holdout > "$R/logs/fam_holdout.log" 2>&1
log "END family holdout rc=$?"

log "START rq3 v2 censoring"
$PY -u rq3_v2_censoring.py > "$R/logs/rq3_v2.log" 2>&1
log "END rq3 v2 rc=$?"

log "START provenance manifest"
$PY -u provenance_manifest.py > "$R/logs/provenance.log" 2>&1
log "END provenance rc=$?"

# invariance: the frozen point estimate must be bit-identical after the fixes
$PY - <<'EOF' 2>&1
import json
R = "/home/azureuser/projects/wc_coherence/results"
for c in ("dev", "holdout"):
    try:
        old = json.load(open(f"{R}/{c}_FROZEN_BACKUP/event_study.json"))["isolated"]
        new = json.load(open(f"{R}/{c}/event_study.json"))["isolated"]
        same = (abs(old["D"] - new["D"]) < 1e-9
                and old["n_events"] == new["n_events"]
                and old["n_matches"] == new["n_matches"]
                and old["n_pos"] == new["n_pos"])
        print(f"INVARIANCE {c}: frozen D={old['D']:+.6f} rerun D={new['D']:+.6f} "
              f"ev {old['n_events']}->{new['n_events']} "
              f"pos {old['n_pos']}->{new['n_pos']} "
              f"-> {'IDENTICAL' if same else '*** CHANGED ***'}")
    except Exception as e:
        print(f"INVARIANCE {c}: unavailable ({e})")
EOF

log "ALL DONE"
echo DONE > "$R/logs/RERUN_DONE.txt"
