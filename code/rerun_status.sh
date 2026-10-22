#!/usr/bin/env bash
# Progress of the most recent coherence re-run.
cd /home/azureuser/projects/wc_coherence || exit 1
LOG=$(ls -t logs/rerun_*.log 2>/dev/null | head -1)
[ -z "$LOG" ] && { echo "no re-run log found"; exit 1; }

if pgrep -f rerun_all.sh > /dev/null; then STATE=RUNNING; else STATE=FINISHED; fi
DONE=$(grep -c '     ok  (' "$LOG" 2>/dev/null)
FAIL=$(grep -c '    FAILED' "$LOG" 2>/dev/null)
START=$(head -1 "$LOG" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
NOW=$(date -u +%H:%M:%S)

echo "state    : $STATE"
echo "log      : $LOG"
echo "steps ok : ${DONE:-0} / 31     failed: ${FAIL:-0}"
echo "started  : $START UTC     now: $NOW UTC"
echo "--- last 18 lines ---"
tail -n 18 "$LOG"
