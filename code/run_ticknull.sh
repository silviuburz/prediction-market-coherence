#!/usr/bin/env bash
# Tick-discretization null. Three LPs per instant (observed, projection, and two
# rounding rules), so heavier than the freshness sweep; timestamp every stage.
set -uo pipefail
cd /home/azureuser/projects/wc_coherence/code
PY=/home/azureuser/.venv/bin/python
SUBSYSTEM=${1:-full}
case "${SUBSYSTEM}" in
  full|no-exact-score) ;;
  *) echo "usage: $0 [full|no-exact-score]" >&2; exit 2 ;;
esac
TAG=${SUBSYSTEM//-/_}
LOG=/home/azureuser/projects/wc_coherence/logs/ticknull_${TAG}_$(date -u +%Y%m%dT%H%M%SZ).log
echo "${LOG}"
{
  echo "=== START $(date -u +%FT%TZ) ==="
  echo "--- subsystem ${SUBSYSTEM} ---"
  for CO in dev holdout; do
    echo "--- cohort ${CO} START $(date -u +%FT%TZ) ---"
    ${PY} -u tick_null.py --cohort "${CO}" --subsystem "${SUBSYSTEM}"
    RC=$?
    echo "--- cohort ${CO} EXIT rc=${RC} $(date -u +%FT%TZ) ---"
    if [ "${RC}" -ne 0 ]; then
      exit "${RC}"
    fi
  done
  echo "=== DONE $(date -u +%FT%TZ) ==="
} 2>&1 | tee "${LOG}"
