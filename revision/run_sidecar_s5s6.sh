#!/bin/bash
# Sidecar: paper-aligned splits S5+S6 only, 4 parallel.
# Runs alongside main batch to ensure coverage before 10h deadline.

set -u

PARALLEL=4
SEEDS=8
OUTER_LOOP=3
CONCEPT="Mean Reversion after Panic Selling"

ROOT=/home/dgu_wj92/fin_emnlp/revision
WRAPPER=$ROOT/run_split_experiment.py
PYTHON=/home/dgu_wj92/miniconda3/envs/favor/bin/python
TS=$(date +%Y%m%d_%H%M%S)
BATCH_TAG="batch_sidecar_${TS}"
BATCH_LOG_DIR=$ROOT/favor/runs/_run_logs/_${BATCH_TAG}

: "${OPENAI_API_KEY:?need OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:=https://api.openai.com/v1}"
export OPENAI_API_KEY OPENAI_BASE_URL

SPLITS=(
  "S5|2015-01-01|2019-12-31|2020-01-01|2020-12-31|2021-01-01|2025-12-31"
  "S6|2017-01-01|2019-12-31|2020-01-01|2020-12-31|2021-01-01|2025-12-31"
)

mkdir -p "$BATCH_LOG_DIR"
JOBS=()
for sd in "${SPLITS[@]}"; do
  for s in $(seq 1 $SEEDS); do
    JOBS+=("${sd}|${s}")
  done
done
echo "[sidecar] launching ${#JOBS[@]} jobs, parallel=$PARALLEL"
echo "[$(date +%H:%M:%S)] SIDECAR START" > "$BATCH_LOG_DIR/_progress.log"

launch_one() {
  IFS='|' read -r tag ts te vs ve tes tee seed <<< "$1"
  log="$BATCH_LOG_DIR/${tag}_seed${seed}.log"
  sleep "$(awk -v s="${seed}" 'BEGIN{print s*0.5}')"
  echo "[$(date +%H:%M:%S)] start ${tag}_seed${seed}" >> "$BATCH_LOG_DIR/_progress.log"
  "$PYTHON" "$WRAPPER" "$CONCEPT" \
    --train-start "$ts" --train-end "$te" \
    --val-start "$vs" --val-end "$ve" \
    --test-start "$tes" --test-end "$tee" \
    --outer-loop "$OUTER_LOOP" --tag "${tag}_seed${seed}" \
    > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] end   ${tag}_seed${seed} (exit $?)" >> "$BATCH_LOG_DIR/_progress.log"
}
export -f launch_one
export PYTHON WRAPPER CONCEPT BATCH_LOG_DIR OUTER_LOOP

printf '%s\n' "${JOBS[@]}" | xargs -I{} -n1 -P "$PARALLEL" bash -c 'launch_one "$@"' _ {}

echo "[$(date +%H:%M:%S)] SIDECAR DONE" >> "$BATCH_LOG_DIR/_progress.log"
