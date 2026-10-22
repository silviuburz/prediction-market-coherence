#!/usr/bin/env bash
# Freshness + straddle run. Five variants per cohort, so roughly five times the
# LP work of the subsystem run; launched detached with timestamped progress.
set -u
cd /home/azureuser/projects/wc_coherence/code
PY=/home/azureuser/.venv/bin/python
LOG=/home/azureuser/projects/wc_coherence/logs/freshness_$(date -u +%Y%m%dT%H%M%SZ).log
{
  echo "=== START $(date -u +%FT%TZ) ==="
  for CO in holdout dev; do
    echo "--- cohort ${CO} $(date -u +%FT%TZ) ---"
    ${PY} -u freshness_and_straddle.py --cohort "${CO}"
    echo "--- exit ${CO} rc=$? $(date -u +%FT%TZ) ---"
  done
  echo "=== DONE $(date -u +%FT%TZ) ==="
} > "${LOG}" 2>&1
echo "${LOG}"
