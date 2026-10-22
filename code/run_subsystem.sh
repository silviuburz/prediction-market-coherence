#!/usr/bin/env bash
# Subsystem coherence run. Launched detached because the LP work over both
# cohorts runs well past any interactive timeout; the log carries UTC timestamps
# every 10 matches so progress is auditable without attaching to the process.
set -u
cd /home/azureuser/projects/wc_coherence/code
PY=/home/azureuser/.venv/bin/python
LOG=/home/azureuser/projects/wc_coherence/logs/subsystem_$(date -u +%Y%m%dT%H%M%SZ).log
{
  echo "=== START $(date -u +%FT%TZ) ==="
  for CO in holdout dev; do
    echo "--- cohort ${CO} $(date -u +%FT%TZ) ---"
    ${PY} -u subsystem_and_availability.py --cohort "${CO}"
    echo "--- exit ${CO} rc=$? $(date -u +%FT%TZ) ---"
  done
  echo "=== DONE $(date -u +%FT%TZ) ==="
} > "${LOG}" 2>&1
echo "${LOG}"
