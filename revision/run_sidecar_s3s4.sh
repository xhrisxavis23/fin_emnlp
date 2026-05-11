#!/bin/bash
# Sidecar3: S3 + S4 — covers main queue tail.
set -u
PARALLEL=4
SEEDS=8
OUTER_LOOP=3
CONCEPT="Mean Reversion after Panic Selling"
ROOT=/home/dgu_wj92/fin_emnlp/revision
WRAPPER=$ROOT/run_split_experiment.py
PYTHON=/home/dgu_wj92/miniconda3/envs/favor/bin/python
TS=$(date +%Y%m%d_%H%M%S)
BATCH_LOG_DIR=$ROOT/favor/runs/_run_logs/_batch_sidecar3_s3s4_${TS}
: "${OPENAI_API_KEY:?need OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:=https://api.openai.com/v1}"
export OPENAI_API_KEY OPENAI_BASE_URL
SPLITS=(
  "S3|2020-01-01|2021-12-31|2022-01-01|2022-12-31|2023-01-01|2025-12-31"
  "S4|2018-01-01|2020-12-31|2021-01-01|2021-12-31|2022-01-01|2025-12-31"
)
mkdir -p "$BATCH_LOG_DIR"
JOBS=()
for sd in "${SPLITS[@]}"; do
  for s in $(seq 1 $SEEDS); do JOBS+=("${sd}|${s}"); done
done
echo "[$(date +%H:%M:%S)] SIDECAR3 START" > "$BATCH_LOG_DIR/_progress.log"
launch_one() {
  IFS='|' read -r tag ts te vs ve tes tee seed <<< "$1"
  log="$BATCH_LOG_DIR/${tag}_seed${seed}.log"
  sleep "$(awk -v s="${seed}" 'BEGIN{print s*0.5}')"
  echo "[$(date +%H:%M:%S)] start ${tag}_seed${seed}" >> "$BATCH_LOG_DIR/_progress.log"
  "$PYTHON" "$WRAPPER" "$CONCEPT" \
    --train-start "$ts" --train-end "$te" --val-start "$vs" --val-end "$ve" \
    --test-start "$tes" --test-end "$tee" --outer-loop "$OUTER_LOOP" --tag "${tag}_seed${seed}" \
    > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] end   ${tag}_seed${seed} (exit $?)" >> "$BATCH_LOG_DIR/_progress.log"
}
export -f launch_one
export PYTHON WRAPPER CONCEPT BATCH_LOG_DIR OUTER_LOOP
printf '%s\n' "${JOBS[@]}" | xargs -I{} -n1 -P "$PARALLEL" bash -c 'launch_one "$@"' _ {}
echo "[$(date +%H:%M:%S)] SIDECAR3 DONE" >> "$BATCH_LOG_DIR/_progress.log"
