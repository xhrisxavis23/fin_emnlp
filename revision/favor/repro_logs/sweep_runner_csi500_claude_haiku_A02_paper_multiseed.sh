#!/bin/bash
# ====================================================================
# CSI500 × claude-haiku-4-5 × A02 (paper concept) × 5 seed
# ====================================================================
# Goal:
#   기존 A02 paper × CSI500 multi-seed 결과 (mini single +1.92, gpt-4o 5seed
#   mean=-0.667, flash 5seed mean=-1.212) 와 동일 lever 로 claude-haiku-4-5
#   분포를 추가하여 4 backbone 비교 table 의 paper × csi500 row 완성.
#
# Plumbing (Anthropic OpenAI-호환 endpoint, llm_client.py 무수정):
#   OPENAI_API_KEY=$ANTHROPIC_API_KEY
#   OPENAI_BASE_URL=$ANTHROPIC_BASE_URL
#   (sanity test 통과: chat.completions.create + beta.chat.completions.parse
#    + temperature=0.9 5/5 unique 확인)
#
# A02 lever (byte-identical with phase6/sp500 sweep):
#   FAVOR_STAGE4_OBJECTIVE=calmar
#   FAVOR_STOP_LOSS_THRESHOLD=None
#   FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos
#   FAVOR_COMBO_PASS_RATE=0.4
#   FAVOR_THRESHOLD_MIN=0.55 / MAX=0.95
#   STAGE4_N_TRIALS=20
#   FAVOR_NATIVE_STRATEGY=trigger_exit
#   split S1 (22-23 / 24 / 25), outer_loop=3
#   concept: paper ("breakout pullback")
#
# Total: 5 jobs, parallel=5. Expected wall ~1-2h. Cost ~$1-1.5
# (haiku-4-5 ~$0.80/M input).
# ====================================================================

set -u
export TZ='Asia/Seoul'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts_csi500_claude_haiku_A02_paper_multiseed"
MASTER_LOG="$LOG_DIR/sweep_csi500_claude_haiku_A02_paper_multiseed_master.log"
RESULTS_CSV="$LOG_DIR/sweep_csi500_claude_haiku_A02_paper_multiseed_results.csv"
mkdir -p "$ART_DIR"

# ─── conda + python ─────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate quant
PY_BIN="${PY_BIN:-/home/dgu/.conda/envs/quant/bin/python}"

# load .env (ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL)
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

# ─── Claude 라우팅: OpenAI SDK 가 OPENAI_BASE_URL 자동 인식 ───────────
export OPENAI_API_KEY="$ANTHROPIC_API_KEY"
export OPENAI_BASE_URL="$ANTHROPIC_BASE_URL"

# CSI500 (CN) — MARKET unset → config.py default 'cn'
unset MARKET 2>/dev/null
export FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"

# parallelism (per-job)
CORES=4
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export NUMEXPR_MAX_THREADS=$CORES
export STAGE4_OPTUNA_N_JOBS=1

# ─── concept (paper, byte-identical with Phase 6) ─────────────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."

# ─── BASE env (A02 lever, claude-haiku-4-5) ──────────────────────────
declare -A BASE=(
    [FAVOR_LLM_MODEL]="claude-haiku-4-5"
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
TRAIN_START="2022-01-01"; TRAIN_END="2023-12-31"
VAL_START="2024-01-01";   VAL_END="2024-12-31"
TEST_START="2025-01-01";  TEST_END="2025-12-31"
SPLIT_ENVS="FAVOR_TRAIN_START=$TRAIN_START FAVOR_TRAIN_END=$TRAIN_END FAVOR_VAL_START=$VAL_START FAVOR_VAL_END=$VAL_END FAVOR_TEST_START=$TEST_START FAVOR_TEST_END=$TEST_END"

# ─── JOBS — paper × 5 seed ──────────────────────────────────────────
JOBS=()
JOBS+=("CSI_claude_paper_A02_seed01|$C_PAPER")
JOBS+=("CSI_claude_paper_A02_seed02|$C_PAPER")
JOBS+=("CSI_claude_paper_A02_seed03|$C_PAPER")
JOBS+=("CSI_claude_paper_A02_seed04|$C_PAPER")
JOBS+=("CSI_claude_paper_A02_seed05|$C_PAPER")

PARALLEL=${PARALLEL_JOBS:-1}   # Anthropic 10k output tok/min rate limit 회피 — 순차 실행
OUTER_LOOP=${OUTER_LOOP:-3}
echo "[$(date +%F\ %T)] CSI500 claude-haiku-4-5 A02 × paper multi-seed sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL, outer_loop=$OUTER_LOOP" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] MARKET=cn(default), OPENAI_BASE_URL=$OPENAI_BASE_URL" | tee -a "$MASTER_LOG"

if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,objective_mode,actual_horizon_days,n_combos,is_best_oos_ir,is_best_oos_ar,is_best_oos_mdd,is_best_oos_cr,oracle_oos_ir,oracle_oos_ar,oracle_oos_mdd,oracle_oos_cr" > "$RESULTS_CSV"
fi

# ─── single-job runner ──────────────────────────────────────────────
run_one_job() {
    local label="$1" concept="$2"
    local job_log="$LOG_DIR/sweep_csi500_claude_haiku_A02_paper_multiseed_${label}.log"
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
        echo "[$start_str] [$label] starting (CSI500, claude-haiku-4-5, A02, obj=calmar)"
        echo "concept: $concept"
        echo "env: ${env_args[*]}"
        echo "outer_loop: $OUTER_LOOP"
        echo "============================================================"
    } >> "$job_log"
    echo "[$start_str] [$label] starting" >> "$MASTER_LOG"

    cd "$ROOT"
    (
        for kv in "${env_args[@]}"; do export "$kv"; done
        export OPENAI_API_KEY="$ANTHROPIC_API_KEY"
        export OPENAI_BASE_URL="$ANTHROPIC_BASE_URL"
        unset MARKET 2>/dev/null
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
echo "[$(date +%F\ %T)] CSI500 claude-haiku-4-5 A02 × paper multi-seed sweep complete; ${#JOBS[@]} jobs processed" | tee -a "$MASTER_LOG"
