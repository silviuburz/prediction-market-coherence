#!/usr/bin/env bash
# L1 functional variant, development cohort only. Two LPs per instant, so
# comparable cost to the tick null; detached with timestamped progress.
set -u
cd /home/azureuser/projects/wc_coherence/code
PY=/home/azureuser/.venv/bin/python
LOG=/home/azureuser/projects/wc_coherence/logs/l1func_$(date -u +%Y%m%dT%H%M%SZ).log
{
  echo "=== START $(date -u +%FT%TZ) ==="
  ${PY} -u l1_functional.py
  echo "--- exit rc=$? $(date -u +%FT%TZ) ---"
  echo "=== DONE $(date -u +%FT%TZ) ==="
} > "${LOG}" 2>&1
echo "${LOG}"
