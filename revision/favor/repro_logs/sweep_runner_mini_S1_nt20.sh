#!/bin/bash
# ====================================================================
# FaVOR Phase 4 sweep — gpt-5.4-mini × S1 × n_trials=20 × multi-axis
# ====================================================================
# Goal: 지금까지 한 번도 시도 안 한 mini × n_trials=20 (paper frozen 정합)
# 환경에서 concept + hyperparameter 축을 함께 흔들어 honest IS-best 기준
# IR 양수 + MDD > -0.30 인 Pareto winner setting 식별.
#
#   model       : gpt-5.4-mini
#   split       : S1 (train 2022~23, val 2024, test 2025)
#   n_trials    : 20 (KDD frozen 정합, paper baseline 동일)
#   outer_loop  : 3
#   parallel    : 3 (cpu 사용률 확인 전엔 보수적으로 유지)
#
# 구성: 5 + 12 + 3 = 20 jobs
#   A. 5 concept × baseline hp  (5)
#   B. paper + compressed × 6 hp lever  (12)
#   C. volcomp 변형  (3)
# ====================================================================

set -u
# Use KST timezone for run_id timestamps + all log lines
export TZ='Asia/Seoul'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts_mini_S1_nt20"
MASTER_LOG="$LOG_DIR/sweep_mini_S1_nt20_master.log"
RESULTS_CSV="$LOG_DIR/sweep_mini_S1_nt20_results.csv"
mkdir -p "$ART_DIR"

# ─── conda + python ─────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate quant
PY_BIN="${PY_BIN:-/home/dgu/.conda/envs/quant/bin/python}"

# load OPENAI_API_KEY from .env
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

# qlib data location
export FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"

# parallelism (per-job)
CORES=4
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export NUMEXPR_MAX_THREADS=$CORES
export STAGE4_OPTUNA_N_JOBS=1

# ─── concept text (Phase 0 sweep_runner.sh 와 동일) ──────────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."
C_UPTREND="In a strong uptrend, when price pulls back to the 20-day moving average, buying near that level increases the probability of price retesting the previous high within the next 5-10 trading days."
C_PANIC="After a sharp sell-off, stocks that close near the day's high, indicating strong intraday recovery, are more likely to rebound over the next 3 trading days."
C_COMPRESSED="Following a prior price decline, some stocks enter a compressed trading state characterized by reduced intraday range and the absence of further downside follow-through. When selling pressure becomes exhausted while latent demand continues to absorb supply, prices may remain temporarily stable despite elevated participation. This imbalance can resolve abruptly, leading to a sharp upward price movement over the subsequent few trading days."
C_VOLCOMP="In a strong uptrend, pullbacks toward the 20-day moving average accompanied by compressed volatility increase the probability of an upside continuation."

# ─── baseline ───────────────────────────────────────────────────────
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

# ─── single split (S1) ──────────────────────────────────────────────
SPLIT_LABEL="S1"
TRAIN_START="2022-01-01"
TRAIN_END="2023-12-31"
VAL_START="2024-01-01"
VAL_END="2024-12-31"
TEST_START="2025-01-01"
TEST_END="2025-12-31"

# ─── 20 jobs ────────────────────────────────────────────────────────
# Format: "label|concept|var1=val1 var2=val2 ..."
JOBS=(
  # ── A. 5 concept × baseline (5) ───────────────────────────────────
  "M01_paper_base|$C_PAPER|FAVOR_HORIZON_DAYS=5"
  "M02_uptrend_base|$C_UPTREND|FAVOR_HORIZON_DAYS=5"
  "M03_panic_base|$C_PANIC|FAVOR_HORIZON_DAYS=3"
  "M04_compressed_base|$C_COMPRESSED|FAVOR_HORIZON_DAYS=5"
  "M05_volcomp_base|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5"

  # ── B-paper. paper × 6 hp 레버 (6) ────────────────────────────────
  "M06_paper_stop005|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=-0.05"
  "M07_paper_stopNone|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=None"
  "M08_paper_thr0795|$C_PAPER|FAVOR_THRESHOLD_MIN=0.7"
  "M09_paper_h10|$C_PAPER|FAVOR_HORIZON_DAYS=10"
  "M10_paper_h20|$C_PAPER|FAVOR_HORIZON_DAYS=20"
  "M11_paper_entry_up|$C_PAPER|FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos"

  # ── B-compressed. compressed × 6 hp 레버 (6) ──────────────────────
  "M12_compressed_stop005|$C_COMPRESSED|FAVOR_STOP_LOSS_THRESHOLD=-0.05"
  "M13_compressed_stopNone|$C_COMPRESSED|FAVOR_STOP_LOSS_THRESHOLD=None"
  "M14_compressed_thr0795|$C_COMPRESSED|FAVOR_THRESHOLD_MIN=0.7"
  "M15_compressed_h10|$C_COMPRESSED|FAVOR_HORIZON_DAYS=10"
  "M16_compressed_h20|$C_COMPRESSED|FAVOR_HORIZON_DAYS=20"
  "M17_compressed_entry_up|$C_COMPRESSED|FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos"

  # ── C. volcomp 변형 (3) ───────────────────────────────────────────
  "M18_volcomp_h10|$C_VOLCOMP|FAVOR_HORIZON_DAYS=10"
  "M19_volcomp_h20|$C_VOLCOMP|FAVOR_HORIZON_DAYS=20"
  "M20_volcomp_stop005|$C_VOLCOMP|FAVOR_STOP_LOSS_THRESHOLD=-0.05"
)

# attach split env to every job
SPLIT_ENVS="FAVOR_TRAIN_START=$TRAIN_START FAVOR_TRAIN_END=$TRAIN_END FAVOR_VAL_START=$VAL_START FAVOR_VAL_END=$VAL_END FAVOR_TEST_START=$TEST_START FAVOR_TEST_END=$TEST_END"

PARALLEL=${PARALLEL_JOBS:-3}
OUTER_LOOP=${OUTER_LOOP:-3}
echo "[$(date +%F\ %T)] Phase 4 mini × S1 × n_trials=20 sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL, outer_loop=$OUTER_LOOP" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] log dir: $LOG_DIR" | tee -a "$MASTER_LOG"

if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,n_combos,is_best_oos_ir,is_best_oos_ar,is_best_oos_mdd,is_best_oos_cr,oracle_oos_ir,oracle_oos_ar,oracle_oos_mdd,oracle_oos_cr" > "$RESULTS_CSV"
fi

# ─── single-job runner ──────────────────────────────────────────────
run_one_job() {
    local label="$1" concept="$2" overrides="$3"
    local job_log="$LOG_DIR/sweep_mini_S1_nt20_${label}.log"
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
        echo "[$start_str] [$label] starting (mini × S1 × n_trials=20)"
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

    # extract honest IS-best + oracle OOS metrics (IR, AR, MDD, CR)
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

# ─── parallel scheduler ─────────────────────────────────────────────
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
echo "[$(date +%F\ %T)] Phase 4 mini × S1 × n_trials=20 sweep complete; ${#JOBS[@]} jobs processed" | tee -a "$MASTER_LOG"

# ─── rebuild dashboard ──────────────────────────────────────────────
echo "[$(date +%F\ %T)] rebuilding favor_dashboard.html ..." | tee -a "$MASTER_LOG"
if "$PY_BIN" "$LOG_DIR/build_report.py" 2>&1 | tee -a "$MASTER_LOG"; then
    echo "[$(date +%F\ %T)] dashboard rebuilt: $ROOT/favor_dashboard.html" | tee -a "$MASTER_LOG"
else
    echo "[$(date +%F\ %T)] dashboard rebuild failed" | tee -a "$MASTER_LOG"
fi
