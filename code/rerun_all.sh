#!/usr/bin/env bash
# Full re-run of the coherence pipeline, for A9 Phase 3.
#
# Snapshots results/ and figs/ FIRST, then regenerates everything in dependency
# order and diffs every output against the frozen hashes in PROVENANCE.json.
#
# Safety:
#   - the corpus is read-only at the filesystem level; nothing here can touch it
#   - results/ and figs/ are copied to *_PRERUN_<stamp>/ before anything runs
#   - a failing step is logged and the run continues, so one break does not
#     hide the state of everything else
#
# Run detached:
#   cd /home/azureuser/projects/wc_coherence
#   setsid --fork ./rerun_all.sh < /dev/null > /tmp/rerun_launch.log 2>&1
#   tail -f logs/rerun_*.log

set -u
PROJ=/home/azureuser/projects/wc_coherence
PY=/home/azureuser/.venv/bin/python
CODE=$PROJ/code
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$PROJ/logs/rerun_${STAMP}.log
FAILED=0
STEP=0

mkdir -p "$PROJ/logs"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "=== coherence full re-run ${STAMP} ==="
say "python: $($PY -V 2>&1)"
say "corpus: $(ls -ld /home/azureuser/polymarket/analysis/data/world_cup | awk '{print $1}') (expect dr-xr-xr-x)"

# ---------------------------------------------------------------- snapshot ---
say "snapshotting results/ and figs/ before touching anything"
cp -a "$PROJ/results" "$PROJ/results_PRERUN_${STAMP}" || { say "SNAPSHOT FAILED - aborting"; exit 1; }
cp -a "$PROJ/figs"    "$PROJ/figs_PRERUN_${STAMP}"    || { say "SNAPSHOT FAILED - aborting"; exit 1; }
say "snapshot: results_PRERUN_${STAMP} ($(du -sh "$PROJ/results_PRERUN_${STAMP}" | cut -f1))"
say "snapshot: figs_PRERUN_${STAMP} ($(du -sh "$PROJ/figs_PRERUN_${STAMP}" | cut -f1))"

run() {
    STEP=$((STEP + 1))
    local label="$1"; shift
    local start=$(date +%s)
    say "--- [${STEP}] ${label} ---"
    say "    \$ $*"
    if ( cd "$CODE" && "$@" ); then
        say "    ok  (${label}, $(( ($(date +%s) - start) / 60 ))m $(( ($(date +%s) - start) % 60 ))s)"
    else
        say "    FAILED (${label}) exit=$?"
        FAILED=$((FAILED + 1))
    fi
}

# ------------------------------------------------- recorded producing order ---
run "validate all matches"        $PY -u f1_validate_all_matches.py
run "event study dev"             $PY -u event_study.py --cohort dev
run "event study holdout"         $PY -u event_study.py --cohort holdout --unlock-holdout
run "cancel sensitivity dev"      $PY -u event_study.py --cohort dev --cancel-state
run "cancel sensitivity holdout"  $PY -u event_study.py --cohort holdout --unlock-holdout --cancel-state
run "crossed interval qc dev"     $PY -u crossed_interval_qc.py --cohort dev
run "crossed interval qc holdout" $PY -u crossed_interval_qc.py --cohort holdout
run "spec curve v2"               $PY -u spec_curve_v2.py
run "spec curve v3"               $PY -u spec_curve_v3.py
run "placebo suite"               $PY -u placebo_suite.py
run "matched placebo"             $PY -u matched_placebo.py
run "randomization test"          $PY -u randomization_test.py
run "family attribution dev"      $PY -u family_attribution.py --cohort dev
run "family attribution holdout"  $PY -u family_attribution.py --cohort holdout
# fam_attr_v2.py is a recorded producer but NOT the one that produced the frozen
# artifact: it writes the same results/family_attribution_v2_<cohort>.json, and
# the recorded command list names only family_attribution.py. Running it here
# overwrote the correct output and dropped size_matched_random_pcts. Left out
# deliberately; it is shipped as superseded code, not run.
run "tick null dev full"          $PY -u tick_null.py --cohort dev
run "tick null holdout full"      $PY -u tick_null.py --cohort holdout
run "tick null dev no-exact"      $PY -u tick_null.py --cohort dev --subsystem no-exact-score
run "tick null holdout no-exact"  $PY -u tick_null.py --cohort holdout --subsystem no-exact-score
run "l1 functional dev"           $PY -u l1_functional.py
run "split label sensitivity"     $PY -u split_lottery.py
run "subsystem dev"               $PY -u subsystem_and_availability.py --cohort dev
run "subsystem holdout"           $PY -u subsystem_and_availability.py --cohort holdout
run "availability census"         $PY -u availability_census.py
run "freshness dev"               $PY -u freshness_and_straddle.py --cohort dev
run "freshness holdout"           $PY -u freshness_and_straddle.py --cohort holdout
run "anchor precision"            $PY -u anchor_precision.py
run "rq3 censoring"               $PY -u rq3_v2_censoring.py
run "saturation lemma"            $PY -u saturation_lemma.py
run "figure inputs"               $PY -u figure_inputs.py

# ------------------------------------------------------------------ verify ---
say "=== verifying regenerated outputs against PROVENANCE.json ==="
$PY -u "$PROJ/verify_rerun.py" --snapshot "$PROJ/results_PRERUN_${STAMP}" | tee -a "$LOG"
VERIFY=$?

say "=== finished: ${STEP} steps, ${FAILED} failed, verifier exit ${VERIFY} ==="
say "log:      $LOG"
say "snapshot: $PROJ/results_PRERUN_${STAMP}"
say "restore with:  rm -rf $PROJ/results && mv $PROJ/results_PRERUN_${STAMP} $PROJ/results"
