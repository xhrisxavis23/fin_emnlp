#!/bin/bash
# Quick status of a running/completed sweep.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts"
RESULTS_CSV="$LOG_DIR/sweep_results.csv"

echo "=== running pipelines ==="
pgrep -af run_pipeline_parallel_per_combo_parallel | grep -v pgrep \
    | awk '{print $1}' | head
echo
echo "=== sweep_master.log tail ==="
tail -20 "$LOG_DIR/sweep_master.log" 2>/dev/null
echo
echo "=== completed jobs (with metrics) ==="
if [ -s "$RESULTS_CSV" ]; then
    column -ts, "$RESULTS_CSV" | head -50
else
    echo "(no results yet)"
fi
echo
echo "=== STOP file? ==="
[ -f "$LOG_DIR/STOP" ] && echo "STOP file present (graceful halt requested)" || echo "no STOP file"
echo
echo "=== mem/cpu ==="
free -h | head -2
top -b -n 1 -p $(pgrep -f run_pipeline_parallel_per_combo_parallel | grep -v $$ | tr '\n' ',' | sed 's/,$//') 2>/dev/null \
    | tail -n +6 | head -10
