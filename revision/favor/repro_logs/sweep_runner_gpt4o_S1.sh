#!/bin/bash
# ====================================================================
# FaVOR gpt-4o re-run sweep — 4 best settings × S1 split
# ====================================================================
# Goal: 5/11 splits sweep 의 핵심 finding (LLM stochasticity 가 dominant) 을
# 다른 backbone (gpt-4o, paper 표준) 에서 재확인 + 직접 비교.
#
#   4 settings: B01_stoploss_005, B04_thr_07_095, A13_volcomp_h5, A15_volcomp_h20
#   1 split   : S1 (train 2022-01-01 ~ 2023-12-31, val 2024, test 2025)
#   model     : gpt-4o (paper 표준; 5/11 sweep 의 gpt-5.4-mini 와 비교)
#   = 4 jobs total
#
# 비교 대상 (5/11 sweep 동일 split S1):
#   gpt-5.4-mini IS-best OOS IR:
#     B01_S1 −0.55  B04_S1 −1.32  A13_S1 −1.69  A15_S1 −0.44
#
# Resources: 3 parallel × 4 combo_workers each, outer_loop=3.
# Expected wall ~1.5 h, API cost ~$15 (gpt-4o ≈ 17× gpt-5.4-mini).
# ====================================================================

set -u
# Use KST timezone for run_id timestamps + all log lines
export TZ='Asia/Seoul'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts_gpt4o_S1"
MASTER_LOG="$LOG_DIR/sweep_gpt4o_S1_master.log"
RESULTS_CSV="$LOG_DIR/sweep_gpt4o_S1_results.csv"
mkdir -p "$ART_DIR"

# ─── conda + python ─────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate quant
PY_BIN="${PY_BIN:-/home/dgu/.conda/envs/quant/bin/python}"

# load OPENAI_API_KEY from .env
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

# qlib data location
export FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"

# parallelism
CORES=4
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export NUMEXPR_MAX_THREADS=$CORES
export STAGE4_OPTUNA_N_JOBS=1

# ─── concept text ───────────────────────────────────────────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."
C_VOLCOMP="In a strong uptrend, pullbacks toward the 20-day moving average accompanied by compressed volatility increase the probability of an upside continuation."

# ─── baseline (only model differs from sweep_runner_splits.sh) ──────
declare -A BASE=(
    [FAVOR_LLM_MODEL]="gpt-4o"                  # ← changed from gpt-5.4-mini
    [FAVOR_LLM_TEMPERATURE]="0.7"
    [FAVOR_HORIZON_DAYS]="5"
    [FAVOR_STOP_LOSS_THRESHOLD]="-0.10"
    [STAGE4_N_TRIALS]="50"
    [FAVOR_THRESHOLD_MIN]="0.55"
    [FAVOR_THRESHOLD_MAX]="0.95"
    [FAVOR_ENTRY_CONFIRM_RULE]="none"
    [FAVOR_NATIVE_STRATEGY]="trigger_exit"
    [FAVOR_COMBO_PASS_RATE]="0.5"
)

# ─── single split (S1 only) ─────────────────────────────────────────
SPLIT_LABEL="S1"
TRAIN_START="2022-01-01"
TRAIN_END="2023-12-31"
VAL_START="2024-01-01"
VAL_END="2024-12-31"
TEST_START="2025-01-01"
TEST_END="2025-12-31"

# ─── 4 winning settings ─────────────────────────────────────────────
SETTINGS=(
  "B01_stoploss_005|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=-0.05"
  "B04_thr_07_095|$C_PAPER|FAVOR_THRESHOLD_MIN=0.7"
  "A13_volcomp_h5|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5"
  "A15_volcomp_h20|$C_VOLCOMP|FAVOR_HORIZON_DAYS=20"
)

# ─── build JOBS ──────────────────────────────────────────────────────
JOBS=()
for setting_def in "${SETTINGS[@]}"; do
    IFS='|' read -r st_label concept overrides <<< "$setting_def"
    label="${st_label}_${SPLIT_LABEL}_gpt4o"
    split_envs="FAVOR_TRAIN_START=$TRAIN_START FAVOR_TRAIN_END=$TRAIN_END FAVOR_VAL_START=$VAL_START FAVOR_VAL_END=$VAL_END FAVOR_TEST_START=$TEST_START FAVOR_TEST_END=$TEST_END"
    JOBS+=("${label}|${concept}|${overrides} ${split_envs}")
done

PARALLEL=${PARALLEL_JOBS:-3}
OUTER_LOOP=${OUTER_LOOP:-3}
echo "[$(date +%F\ %T)] gpt-4o S1 sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL, outer_loop=$OUTER_LOOP" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] log dir: $LOG_DIR" | tee -a "$MASTER_LOG"

if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,n_combos,is_best_oos_ir,is_best_oos_ar,oracle_oos_ir,oracle_oos_ar" > "$RESULTS_CSV"
fi

# ─── single-job runner (identical pattern to sweep_runner_splits.sh) ─
run_one_job() {
    local label="$1" concept="$2" overrides="$3"
    local job_log="$LOG_DIR/sweep_gpt4o_S1_${label}.log"
    local marker="$ART_DIR/${label}.run_id"

    local env_args=()
    for k in "${!BASE[@]}"; do env_args+=("$k=${BASE[$k]}"); done
    for kv in $overrides; do env_args+=("$kv"); done

    local run_id="$(date +%Y%m%d_%H%M%S)_${label}"
    env_args+=("FAVOR_RUN_ID=$run_id")

    if [ -f "$ROOT/runs/$run_id/specs/stage4_summary.json" ]; then
        echo "[$(date +%F\ %T)] [$label] SKIP" | tee -a "$MASTER_LOG"
        echo "$run_id" > "$marker"
        return 0
    fi
    if [ -f "$marker" ]; then
        local prev_run_id
        prev_run_id=$(cat "$marker")
        if [ "$prev_run_id" != "$run_id" ] && [ -f "$ROOT/runs/$prev_run_id/specs/stage4_summary.json" ]; then
            echo "[$(date +%F\ %T)] [$label] SKIP (older run at $prev_run_id)" | tee -a "$MASTER_LOG"
            return 0
        fi
    fi

    local start_ts=$(date +%s) start_str=$(date +%F\ %T)
    {
        echo "============================================================"
        echo "[$start_str] [$label] starting (gpt-4o)"
        echo "concept: $concept"
        echo "env: ${env_args[*]}"
        echo "outer_loop: $OUTER_LOOP"
        echo "============================================================"
    } >> "$job_log"
    echo "[$start_str] [$label] starting" >> "$MASTER_LOG"

    cd "$ROOT"
    (
        for kv in "${env_args[@]}"; do export "$kv"; done
        nice -n 10 python run_pipeline_parallel_per_combo_parallel.py "$concept" \
            --combo-workers 4 \
            --optuna-jobs "${STAGE4_OPTUNA_N_JOBS:-1}" \
            --outer-loop "$OUTER_LOOP" 2>&1
    ) >> "$job_log" 2>&1
    local exit_code=$?
    local end_ts=$(date +%s) end_str=$(date +%F\ %T)
    local wall=$((end_ts - start_ts))

    echo "$run_id" > "$marker"

    local metrics=$("$PY_BIN" -c "
import json, sys
try:
    s = json.load(open('$ROOT/runs/$run_id/specs/stage4_summary.json'))
    iters = sorted([k for k in s if k.startswith('outer_iter_')])
    best = None; best_score = -1e18
    for k in iters:
        combos = s[k].get('all_combinations', [])
        if not combos: continue
        bis = max(combos, key=lambda c: (c.get('insample', {}).get('excess_return_with_cost', {}) or {}).get('information_ratio', -1e9))
        score = (bis.get('insample', {}).get('excess_return_with_cost', {}) or {}).get('information_ratio', -1e9)
        if score > best_score:
            best_score = score; best = (k, combos, bis)
    if best:
        ik, combos, bis = best
        bos = max(combos, key=lambda c: (c.get('outsample', {}).get('excess_return_with_cost', {}) or {}).get('information_ratio', -1e9))
        b_oos = bis.get('outsample', {}).get('excess_return_with_cost', {}) or {}
        o_oos = bos.get('outsample', {}).get('excess_return_with_cost', {}) or {}
        print(f\"{len(combos)},{b_oos.get('information_ratio', '')},{b_oos.get('annualized_return', '')},{o_oos.get('information_ratio', '')},{o_oos.get('annualized_return', '')}\")
    else:
        print(',,,,')
except Exception:
    print(',,,,')
")
    [ -z "$metrics" ] && metrics=",,,,"

    echo "$label,$run_id,$exit_code,$start_str,$end_str,$wall,$metrics" >> "$RESULTS_CSV"
    echo "[$end_str] [$label] finished (exit=$exit_code, wall=${wall}s, run_id=$run_id, metrics=$metrics)" \
        | tee -a "$MASTER_LOG" >> "$job_log"
}

# ─── parallel scheduler ──────────────────────────────────────────────
i=0
for job in "${JOBS[@]}"; do
    if [ -f "$LOG_DIR/STOP" ]; then break; fi
    IFS='|' read -r label concept overrides <<< "$job"
    i=$((i+1))
    echo "[$(date +%F\ %T)] queue: ($i/${#JOBS[@]}) $label" | tee -a "$MASTER_LOG"

    run_one_job "$label" "$concept" "$overrides" &

    while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do sleep 10; done
done

wait
echo "[$(date +%F\ %T)] gpt-4o S1 sweep complete; ${#JOBS[@]} jobs processed" | tee -a "$MASTER_LOG"

# ─── rebuild dashboard ──────────────────────────────────────────────
echo "[$(date +%F\ %T)] rebuilding favor_dashboard.html ..." | tee -a "$MASTER_LOG"
if "$PY_BIN" "$LOG_DIR/build_report.py" 2>&1 | tee -a "$MASTER_LOG"; then
    echo "[$(date +%F\ %T)] dashboard rebuilt: $ROOT/favor_dashboard.html" | tee -a "$MASTER_LOG"
else
    echo "[$(date +%F\ %T)] dashboard rebuild failed" | tee -a "$MASTER_LOG"
fi
