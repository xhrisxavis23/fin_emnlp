#!/bin/bash
# ====================================================================
# FaVOR MDD-targeted sweep — gpt-5.4-mini × S5 (paper-aligned) × n_trials=20
# ====================================================================
# Goal: 최대낙폭(MDD) sweet spot 탐색. 손절 -10% 와 -5% 사이의 중간값
# (-7%, -8%) + 가설을 vol-compression / compressed 로 바꿔 동일 lever
# 효과 비교. S1 (테스트 1년) 대신 S5 (테스트 5년, paper-aligned) 로 검증.
#
#   model       : gpt-5.4-mini
#   split       : S5 (train 2015-2019, val 2020, test 2021-2025)
#   n_trials    : 20
#   outer_loop  : 3
#   parallel    : 2 (M-runner 마무리와 겹쳐도 안전)
#
# 구성: 6 jobs
#   N01: paper concept × stop -0.07
#   N02: paper concept × stop -0.08
#   N03: volcomp concept × stop -0.10
#   N04: volcomp concept × stop -0.07
#   N05: compressed concept × stop -0.07
#   N06: paper concept × stop -0.10  (M01 의 S5 검증)
# ====================================================================

set -u
export TZ='Asia/Seoul'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts_mini_S5_mdd"
MASTER_LOG="$LOG_DIR/sweep_mini_S5_mdd_master.log"
RESULTS_CSV="$LOG_DIR/sweep_mini_S5_mdd_results.csv"
mkdir -p "$ART_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate quant
PY_BIN="${PY_BIN:-/home/dgu/.conda/envs/quant/bin/python}"

[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

export FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"

CORES=4
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export NUMEXPR_MAX_THREADS=$CORES
export STAGE4_OPTUNA_N_JOBS=1

# ─── concept text ────────────────────────────────────────────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."
C_COMPRESSED="Following a prior price decline, some stocks enter a compressed trading state characterized by reduced intraday range and the absence of further downside follow-through. When selling pressure becomes exhausted while latent demand continues to absorb supply, prices may remain temporarily stable despite elevated participation. This imbalance can resolve abruptly, leading to a sharp upward price movement over the subsequent few trading days."
C_VOLCOMP="In a strong uptrend, pullbacks toward the 20-day moving average accompanied by compressed volatility increase the probability of an upside continuation."

# ─── baseline ────────────────────────────────────────────────────────
declare -A BASE=(
    [FAVOR_LLM_MODEL]="gpt-5.4-mini"
    [FAVOR_LLM_TEMPERATURE]="0.7"
    [FAVOR_HORIZON_DAYS]="5"
    [FAVOR_STOP_LOSS_THRESHOLD]="-0.10"
    [STAGE4_N_TRIALS]="20"
    [FAVOR_THRESHOLD_MIN]="0.55"
    [FAVOR_THRESHOLD_MAX]="0.95"
    [FAVOR_ENTRY_CONFIRM_RULE]="none"
    [FAVOR_NATIVE_STRATEGY]="trigger_exit"
    [FAVOR_COMBO_PASS_RATE]="0.5"
)

# ─── split (S5 — paper-aligned) ──────────────────────────────────────
SPLIT_LABEL="S5"
TRAIN_START="2015-01-01"
TRAIN_END="2019-12-31"
VAL_START="2020-01-01"
VAL_END="2020-12-31"
TEST_START="2021-01-01"
TEST_END="2025-12-31"

# ─── 6 jobs ──────────────────────────────────────────────────────────
JOBS=(
    "N01_paper_stop007|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=-0.07"
    "N02_paper_stop008|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=-0.08"
    "N03_volcomp_stop010|$C_VOLCOMP|FAVOR_STOP_LOSS_THRESHOLD=-0.10"
    "N04_volcomp_stop007|$C_VOLCOMP|FAVOR_STOP_LOSS_THRESHOLD=-0.07"
    "N05_compressed_stop007|$C_COMPRESSED|FAVOR_STOP_LOSS_THRESHOLD=-0.07"
    "N06_paper_stop010|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=-0.10"
)

SPLIT_ENVS="FAVOR_TRAIN_START=$TRAIN_START FAVOR_TRAIN_END=$TRAIN_END FAVOR_VAL_START=$VAL_START FAVOR_VAL_END=$VAL_END FAVOR_TEST_START=$TEST_START FAVOR_TEST_END=$TEST_END"

PARALLEL=${PARALLEL_JOBS:-2}
OUTER_LOOP=${OUTER_LOOP:-3}
echo "[$(date +%F\ %T)] mini × S5 × MDD sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL, outer_loop=$OUTER_LOOP" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] log dir: $LOG_DIR" | tee -a "$MASTER_LOG"

if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,n_combos,is_best_oos_ir,is_best_oos_ar,is_best_oos_mdd,is_best_oos_cr,oracle_oos_ir,oracle_oos_ar,oracle_oos_mdd,oracle_oos_cr" > "$RESULTS_CSV"
fi

run_one_job() {
    local label="$1" concept="$2" overrides="$3"
    local job_log="$LOG_DIR/sweep_mini_S5_mdd_${label}.log"
    local marker="$ART_DIR/${label}.run_id"

    local env_args=()
    for k in "${!BASE[@]}"; do env_args+=("$k=${BASE[$k]}"); done
    for kv in $overrides; do env_args+=("$kv"); done
    for kv in $SPLIT_ENVS; do env_args+=("$kv"); done

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
        echo "[$start_str] [$label] starting (mini × S5 × n_trials=20)"
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
import json
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
        b = bis.get('outsample', {}).get('excess_return_with_cost', {}) or {}
        o = bos.get('outsample', {}).get('excess_return_with_cost', {}) or {}
        print(f\"{len(combos)},{b.get('information_ratio','')},{b.get('annualized_return','')},{b.get('max_drawdown','')},{b.get('cumulative_return','')},{o.get('information_ratio','')},{o.get('annualized_return','')},{o.get('max_drawdown','')},{o.get('cumulative_return','')}\")
    else:
        print(',,,,,,,,')
except Exception:
    print(',,,,,,,,')
")
    [ -z "$metrics" ] && metrics=",,,,,,,,"

    echo "$label,$run_id,$exit_code,$start_str,$end_str,$wall,$metrics" >> "$RESULTS_CSV"
    echo "[$end_str] [$label] finished (exit=$exit_code, wall=${wall}s, run_id=$run_id, metrics=$metrics)" \
        | tee -a "$MASTER_LOG" >> "$job_log"
}

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
echo "[$(date +%F\ %T)] mini × S5 × MDD sweep complete; ${#JOBS[@]} jobs processed" | tee -a "$MASTER_LOG"

echo "[$(date +%F\ %T)] rebuilding favor_dashboard.html ..." | tee -a "$MASTER_LOG"
if "$PY_BIN" "$LOG_DIR/build_report.py" 2>&1 | tee -a "$MASTER_LOG"; then
    echo "[$(date +%F\ %T)] dashboard rebuilt: $ROOT/favor_dashboard.html" | tee -a "$MASTER_LOG"
else
    echo "[$(date +%F\ %T)] dashboard rebuild failed" | tee -a "$MASTER_LOG"
fi
