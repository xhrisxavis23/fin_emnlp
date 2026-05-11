#!/bin/bash
# Reproduction launcher for CSI500 FaVOR (paper Table 1) - revision/favor module test
# Usage: ./launch_run.sh <run_label>
# Concept is hard-coded to match runs/20260207_051736/run_config.json
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <run_label>" >&2; exit 1
fi
LABEL="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/repro_logs/${LABEL}.log"

CONCEPT="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."

# Resource limits matching run_cn_limited.sh
CORES=20
export POLARS_MAX_THREADS=$CORES
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export NUMEXPR_MAX_THREADS=$CORES
ulimit -v $((150*1024*1024)) 2>/dev/null || true

export MARKET=cn
export FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"
export STAGE4_ENABLE_OPTUNA=True
export PYTHONWARNINGS=ignore
export STAGE4_FIXED_QUANTILES=None
export STAGE4_COMBO_WORKERS=4
export STAGE4_OPTUNA_N_JOBS=1

source /opt/conda/etc/profile.d/conda.sh
conda activate quant

cd "$ROOT"
echo "[$(date +%F_%T)] [$LABEL] starting; python=$(which python)" | tee -a "$LOG"
# IMPORTANT: concept is the FIRST positional arg (the script reads args[0] as concept iff it doesn't start with --)
nice -n 10 python run_pipeline_parallel_per_combo_parallel.py "$CONCEPT" --combo-workers 4 --optuna-jobs 1 --outer-loop 1 2>&1 | tee -a "$LOG"
echo "[$(date +%F_%T)] [$LABEL] finished" | tee -a "$LOG"
