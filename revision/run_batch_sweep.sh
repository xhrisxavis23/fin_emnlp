#!/bin/bash
# Batch sweep — 6 splits × 8 seeds = 48 runs, 4 parallel.
#
# Usage:
#   bash /home/dgu_wj92/fin_emnlp/revision/run_batch_sweep.sh
#
# Override defaults via env:
#   PARALLEL=6 SEEDS=10 bash run_batch_sweep.sh
#   DRY_RUN=1 bash run_batch_sweep.sh   # prints jobs only, no launch
#
# Each run uses revision/favor (paper-aligned, no v2025 patches) with custom data_split.
# Output: revision/favor/runs/<run_id>/, logs to revision/favor_v2025/runs/_sanity_logs/_batch_*/

set -u

# ============================================================
# Settings
# ============================================================
PARALLEL="${PARALLEL:-4}"
SEEDS="${SEEDS:-8}"
OUTER_LOOP="${OUTER_LOOP:-3}"
CONCEPT="${CONCEPT:-Mean Reversion after Panic Selling}"
DRY_RUN="${DRY_RUN:-0}"

ROOT=/home/dgu_wj92/fin_emnlp/revision
WRAPPER=$ROOT/run_split_experiment.py
PYTHON=/home/dgu_wj92/miniconda3/envs/favor/bin/python
TS=$(date +%Y%m%d_%H%M%S)
BATCH_TAG="batch_${TS}"
BATCH_LOG_DIR=$ROOT/favor/runs/_run_logs/_${BATCH_TAG}
MANIFEST=$BATCH_LOG_DIR/_manifest.tsv

# OpenAI key (set in env from caller)
: "${OPENAI_API_KEY:?ERROR: export OPENAI_API_KEY=sk-... before running}"
: "${OPENAI_BASE_URL:=https://api.openai.com/v1}"
export OPENAI_API_KEY OPENAI_BASE_URL

# ============================================================
# Splits — Tag | Train start | Train end | Val start | Val end | Test start | Test end
# ============================================================
SPLITS=(
  "S1|2022-01-01|2023-12-31|2024-01-01|2024-12-31|2025-01-01|2025-12-31"  # 2/1/1 recent-short
  "S2|2020-01-01|2023-12-31|2024-01-01|2024-12-31|2025-01-01|2025-12-31"  # 4/1/1 long-train recent-test
  "S3|2020-01-01|2021-12-31|2022-01-01|2022-12-31|2023-01-01|2025-12-31"  # 2/1/3 short-train long-test
  "S4|2018-01-01|2020-12-31|2021-01-01|2021-12-31|2022-01-01|2025-12-31"  # 3/1/4 balanced
  "S5|2015-01-01|2019-12-31|2020-01-01|2020-12-31|2021-01-01|2025-12-31"  # 5/1/5 paper-aligned
  "S6|2017-01-01|2019-12-31|2020-01-01|2020-12-31|2021-01-01|2025-12-31"  # 3/1/5 paper-like shorter-train
)

mkdir -p "$BATCH_LOG_DIR"
echo "Batch tag: $BATCH_TAG"
echo "Log dir  : $BATCH_LOG_DIR"
echo "Splits   : ${#SPLITS[@]}, Seeds/split: $SEEDS, Outer-loop: $OUTER_LOOP, Parallel: $PARALLEL"
echo "Total    : $((${#SPLITS[@]} * SEEDS)) runs"
echo "Concept  : $CONCEPT"
echo

# ============================================================
# Build job list
# ============================================================
JOBS=()
for split_def in "${SPLITS[@]}"; do
  for seed in $(seq 1 "$SEEDS"); do
    JOBS+=("${split_def}|${seed}")
  done
done

# Manifest header
{
  echo -e "tag\tseed\ttrain_start\ttrain_end\tval_start\tval_end\ttest_start\ttest_end\tlog_path"
} > "$MANIFEST"

for job in "${JOBS[@]}"; do
  IFS='|' read -r tag ts te vs ve tes tee seed <<< "$job"
  log="$BATCH_LOG_DIR/${tag}_seed${seed}.log"
  echo -e "${tag}\t${seed}\t${ts}\t${te}\t${vs}\t${ve}\t${tes}\t${tee}\t${log}" >> "$MANIFEST"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "=== DRY RUN — first 6 jobs ==="
  head -7 "$MANIFEST" | column -t
  echo "..."
  echo "Total jobs: ${#JOBS[@]}"
  echo "Manifest: $MANIFEST"
  exit 0
fi

# ============================================================
# Launch
# ============================================================
launch_one() {
  IFS='|' read -r tag ts te vs ve tes tee seed <<< "$1"
  log="$BATCH_LOG_DIR/${tag}_seed${seed}.log"
  # Jitter to avoid run_id timestamp collisions across siblings
  sleep "$(awk -v s="${seed}" 'BEGIN{print s*0.5}')"
  echo "[$(date +%H:%M:%S)] start ${tag}_seed${seed}" >> "$BATCH_LOG_DIR/_progress.log"
  "$PYTHON" "$WRAPPER" "$CONCEPT" \
    --train-start "$ts" --train-end "$te" \
    --val-start "$vs" --val-end "$ve" \
    --test-start "$tes" --test-end "$tee" \
    --outer-loop "$OUTER_LOOP" \
    --tag "${tag}_seed${seed}" \
    > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] end   ${tag}_seed${seed} (exit $?)" >> "$BATCH_LOG_DIR/_progress.log"
}
export -f launch_one
export PYTHON WRAPPER CONCEPT BATCH_LOG_DIR OUTER_LOOP

echo "[$(date +%H:%M:%S)] launching ${#JOBS[@]} runs, parallel=${PARALLEL}..."
echo "[$(date +%H:%M:%S)] BATCH START" > "$BATCH_LOG_DIR/_progress.log"

printf '%s\n' "${JOBS[@]}" | xargs -I{} -n1 -P "$PARALLEL" bash -c 'launch_one "$@"' _ {}

echo "[$(date +%H:%M:%S)] BATCH DONE" >> "$BATCH_LOG_DIR/_progress.log"
echo
echo "=== batch done ==="
echo "manifest: $MANIFEST"
echo "progress: $BATCH_LOG_DIR/_progress.log"
echo "logs    : $BATCH_LOG_DIR/*.log"
echo
echo "Aggregate metrics with the dashboard:"
echo "  python $ROOT/build_run_dashboard.py"
