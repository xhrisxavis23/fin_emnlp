#!/bin/bash
# ====================================================================
# SP500 × gpt-5.4-mini × A02 setting × 4 concept (single seed)
# ====================================================================
# Goal:
#   CSI500 Phase 6 의 winner setting A02 (paper × Calmar × stop=None ×
#   entry=up_day × pr=0.4 × t=0.55) 를 SP500 시장으로 옮겨, concept 4종
#   (paper / compressed / volcomp / panic) 의 ablation 을 검증.
#   mini 는 reasoning model 이라 multi-seed 무의미 → single seed.
#
# Plumbing:
#   MARKET=us → config.py 에서 qlib_market=sp500 + REG_US 자동 라우팅.
#   FAVOR_QLIB_PROVIDER_URI_US 로 ~/.qlib_full/qlib_data/sh_sp500_qlib 지정.
#   FaVOR frozen 코드 무수정 — env hook 만으로 시장 swap.
#
# A02 setting (byte-identical with CSI500 Phase 6):
#   FAVOR_STAGE4_OBJECTIVE=calmar
#   FAVOR_STOP_LOSS_THRESHOLD=None
#   FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos
#   FAVOR_COMBO_PASS_RATE=0.4
#   FAVOR_THRESHOLD_MIN=0.55
#   FAVOR_THRESHOLD_MAX=0.95
#   STAGE4_N_TRIALS=20
#   FAVOR_NATIVE_STRATEGY=trigger_exit
#   split S1 (22-23 / 24 / 25), outer_loop=3
#
# Layout (4 jobs):
#   SP_M01_paper_A02      — paper concept
#   SP_M02_compressed_A02 — compressed concept
#   SP_M03_volcomp_A02    — volcomp concept
#   SP_M04_panic_A02      — panic concept
#
# Total: 4 jobs, parallel=4. Expected wall ~1-1.5h. Cost ~$0.20.
# ====================================================================

set -u
export TZ='Asia/Seoul'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts_sp500_mini_A02_concepts"
MASTER_LOG="$LOG_DIR/sweep_sp500_mini_A02_concepts_master.log"
RESULTS_CSV="$LOG_DIR/sweep_sp500_mini_A02_concepts_results.csv"
mkdir -p "$ART_DIR"

# ─── conda + python ─────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate quant
PY_BIN="${PY_BIN:-/home/dgu/.conda/envs/quant/bin/python}"

# load OPENAI_API_KEY from .env
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

# ─── SP500 market routing ───────────────────────────────────────────
export MARKET=us
export FAVOR_QLIB_PROVIDER_URI_US="$HOME/.qlib_full/qlib_data/sh_sp500_qlib"

# ─── cost override (baseline 매칭: SP500 min_cost = 5, paper-run 관례) ────
#   config.py 자동값은 SP500 min_cost=0 이지만 revision baseline
#   (split_2y_rerun/configs/*_sp500.yaml) 및 paper-run convention 은 5.
#   run_phase6/cost_patch.py 가 load_rd_config 를 wrap 하여 강제 override.
#   open/close_cost (0 / 0.0005), deal_price (open), limit_threshold (None)
#   은 config.py 자동값 그대로 사용.
export FAVOR_MIN_COST_OVERRIDE=5

# parallelism (per-job)
CORES=4
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export NUMEXPR_MAX_THREADS=$CORES
export STAGE4_OPTUNA_N_JOBS=1

# ─── concept text (byte-identical with Phase 6 CSI500) ───────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."
C_COMPRESSED="Following a prior price decline, some stocks enter a compressed trading state characterized by reduced intraday range and the absence of further downside follow-through. When selling pressure becomes exhausted while latent demand continues to absorb supply, prices may remain temporarily stable despite elevated participation. This imbalance can resolve abruptly, leading to a sharp upward price movement over the subsequent few trading days."
C_VOLCOMP="In a strong uptrend, pullbacks toward the 20-day moving average accompanied by compressed volatility increase the probability of an upside continuation."
C_PANIC="After a sharp sell-off, stocks that close near the day's high, indicating strong intraday recovery, are more likely to rebound over the next 3 trading days."

# ─── BASE env (A02 setting, mini) ────────────────────────────────────
declare -A BASE=(
    [FAVOR_LLM_MODEL]="gpt-5.4-mini"
    [FAVOR_LLM_TEMPERATURE]="0.7"
    [FAVOR_STOP_LOSS_THRESHOLD]="None"
    [FAVOR_ENTRY_CONFIRM_RULE]="up_day_and_close_pos"
    [FAVOR_COMBO_PASS_RATE]="0.4"
    [FAVOR_THRESHOLD_MIN]="0.55"
    [FAVOR_THRESHOLD_MAX]="0.95"
    [STAGE4_N_TRIALS]="20"
    [FAVOR_NATIVE_STRATEGY]="trigger_exit"
    [FAVOR_STAGE4_OBJECTIVE]="calmar"
)

# ─── split S1 (2y/1y/1y) ────────────────────────────────────────────
TRAIN_START="2022-01-01"
TRAIN_END="2023-12-31"
VAL_START="2024-01-01"
VAL_END="2024-12-31"
TEST_START="2025-01-01"
TEST_END="2025-12-31"
SPLIT_ENVS="FAVOR_TRAIN_START=$TRAIN_START FAVOR_TRAIN_END=$TRAIN_END FAVOR_VAL_START=$VAL_START FAVOR_VAL_END=$VAL_END FAVOR_TEST_START=$TEST_START FAVOR_TEST_END=$TEST_END"

# ─── JOBS — "label|concept" (A02 setting fixed in BASE) ──────────────
JOBS=()
JOBS+=("SP_M01_paper_A02|$C_PAPER")
JOBS+=("SP_M02_compressed_A02|$C_COMPRESSED")
JOBS+=("SP_M03_volcomp_A02|$C_VOLCOMP")
JOBS+=("SP_M04_panic_A02|$C_PANIC")

PARALLEL=${PARALLEL_JOBS:-4}
OUTER_LOOP=${OUTER_LOOP:-3}
echo "[$(date +%F\ %T)] SP500 mini A02 × 4 concept sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL, outer_loop=$OUTER_LOOP" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] MARKET=$MARKET, provider_uri=$FAVOR_QLIB_PROVIDER_URI_US" | tee -a "$MASTER_LOG"

if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,objective_mode,actual_horizon_days,n_combos,is_best_oos_ir,is_best_oos_ar,is_best_oos_mdd,is_best_oos_cr,oracle_oos_ir,oracle_oos_ar,oracle_oos_mdd,oracle_oos_cr" > "$RESULTS_CSV"
fi

# ─── single-job runner ──────────────────────────────────────────────
run_one_job() {
    local label="$1" concept="$2"
    local job_log="$LOG_DIR/sweep_sp500_mini_A02_concepts_${label}.log"
    local marker="$ART_DIR/${label}.run_id"

    local env_args=()
    for k in "${!BASE[@]}"; do env_args+=("$k=${BASE[$k]}"); done
    for kv in $SPLIT_ENVS; do env_args+=("$kv"); done

    local run_id="$(date +%Y%m%d_%H%M%S)_${label}"
    env_args+=("FAVOR_RUN_ID=$run_id")

    if [ -f "$ROOT/runs/$run_id/specs/stage4_summary.json" ]; then
        echo "[$(date +%F\ %T)] [$label] SKIP" | tee -a "$MASTER_LOG"
        echo "$run_id" > "$marker"
        return 0
    fi

    local start_ts=$(date +%s) start_str=$(date +%F\ %T)
    {
        echo "============================================================"
        echo "[$start_str] [$label] starting (SP500, mini, A02, obj=calmar)"
        echo "concept: $concept"
        echo "env: ${env_args[*]}"
        echo "outer_loop: $OUTER_LOOP"
        echo "============================================================"
    } >> "$job_log"
    echo "[$start_str] [$label] starting" >> "$MASTER_LOG"

    cd "$ROOT"
    (
        for kv in "${env_args[@]}"; do export "$kv"; done
        export MARKET=us
        export FAVOR_QLIB_PROVIDER_URI_US="$HOME/.qlib_full/qlib_data/sh_sp500_qlib"
        export FAVOR_MIN_COST_OVERRIDE=5
        nice -n 10 python run_phase6/run_pipeline_v3.py "$concept" \
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
    actual_h = ''
    for k in iters:
        hh = s[k].get('horizon_days') or s[k].get('hypothesis_horizon_days')
        if hh and not actual_h: actual_h = str(hh)
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
        print(f\"{actual_h},{len(combos)},{b.get('information_ratio','')},{b.get('annualized_return','')},{b.get('max_drawdown','')},{b.get('cumulative_return','')},{o.get('information_ratio','')},{o.get('annualized_return','')},{o.get('max_drawdown','')},{o.get('cumulative_return','')}\")
    else:
        print(f'{actual_h},,,,,,,,,')
except Exception:
    print(',,,,,,,,,')
")
    [ -z "$metrics" ] && metrics=",,,,,,,,,"

    echo "$label,$run_id,$exit_code,$start_str,$end_str,$wall,calmar,$metrics" >> "$RESULTS_CSV"
    echo "[$end_str] [$label] finished (exit=$exit_code, wall=${wall}s, run_id=$run_id, metrics=$metrics)" \
        | tee -a "$MASTER_LOG" >> "$job_log"
}

# ─── parallel scheduler ─────────────────────────────────────────────
i=0
for job in "${JOBS[@]}"; do
    if [ -f "$LOG_DIR/STOP" ]; then break; fi
    IFS='|' read -r label concept <<< "$job"
    i=$((i+1))
    echo "[$(date +%F\ %T)] queue: ($i/${#JOBS[@]}) $label" | tee -a "$MASTER_LOG"

    run_one_job "$label" "$concept" &

    while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do sleep 10; done
done

wait
echo "[$(date +%F\ %T)] SP500 mini A02 × 4 concept sweep complete; ${#JOBS[@]} jobs processed" | tee -a "$MASTER_LOG"
