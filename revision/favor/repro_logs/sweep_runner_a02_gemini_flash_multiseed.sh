#!/bin/bash
# ====================================================================
# A02 multi-seed verification — gemini-2.5-flash × A02 exact × N seeds
# ====================================================================
# Goal:
#   A02 정확 셋팅을 backbone 만 gemini-2.5-flash 로 바꿔 multi-seed 검증.
#   gpt-5.4-mini (원본) vs gpt-4o (1차 검증) 결과와 3-way 비교.
#
# Plumbing:
#   Gemini 의 OpenAI-호환 endpoint 사용 (코드 수정 0).
#   GEMINI_API_KEY / GEMINI_BASE_URL → OPENAI_API_KEY / OPENAI_BASE_URL 로
#   복사해 OpenAI SDK 가 자동 라우팅. favor 의 util/llm_client.py 무수정.
#   Sanity 검증 (chat.completions + beta.chat.completions.parse 둘 다 OK) 통과.
#
# Replication setting (A02 와 동일, backbone 만 교체):
#   FAVOR_LLM_MODEL=gemini-2.5-flash  (← 유일한 변경)
#   FAVOR_STAGE4_OBJECTIVE=calmar
#   FAVOR_STOP_LOSS_THRESHOLD=None
#   FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos
#   FAVOR_COMBO_PASS_RATE=0.4
#   FAVOR_THRESHOLD_MIN=0.55
#   FAVOR_THRESHOLD_MAX=0.95
#   STAGE4_N_TRIALS=20
#   FAVOR_NATIVE_STRATEGY=trigger_exit
#   split S1 (train 22-23 / val 24 / test 25)
#   concept: paper ("breakout pullback")
#   outer_loop=3, combo-workers=4, optuna-jobs=1
#
# Note on FaVOR multi-temperature 구조 (모두 agent literal, env 무시):
#   hypothesis_agent.py:185      temperature=0.9  ← multi-seed variation 주범
#   observation_agent.py:82      temperature=0.7
#   formula_agent.py (5 sites)   temperature=0.7
#   validation_agent.py:754      temperature=0.1  ← deterministic
#   gemini-2.5-flash 는 reasoning model 이 아니어서 temperature 진짜 동작 →
#   gpt-4o 와 동일 원리로 5 seed 호출 시 자연스러운 variation 발생.
#
# Expected cost (gemini-2.5-flash pricing: input $0.075/M, output $0.30/M):
#   gpt-5.4-mini A02 = $0.05 / job → flash 추정 ~$0.02 / job × 5 = ~$0.10
#   gpt-4o per-seed = $1.76 → flash 의 ~88× 저렴
#
# Expected wall: gemini-flash 는 매우 빠른 모델 → ~30-60 min 추정
# ====================================================================

set -u
export TZ='Asia/Seoul'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts_a02_gemini_flash_multiseed"
MASTER_LOG="$LOG_DIR/sweep_a02_gemini_flash_multiseed_master.log"
RESULTS_CSV="$LOG_DIR/sweep_a02_gemini_flash_multiseed_results.csv"
mkdir -p "$ART_DIR"

# ─── conda + python ─────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate quant
PY_BIN="${PY_BIN:-/home/dgu/.conda/envs/quant/bin/python}"

# load env vars from .env (GEMINI_API_KEY / GEMINI_BASE_URL)
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

# ─── Gemini 라우팅: OpenAI SDK 가 OPENAI_BASE_URL env 자동 인식 ──────
export OPENAI_API_KEY="$GEMINI_API_KEY"
export OPENAI_BASE_URL="$GEMINI_BASE_URL"

# qlib data location
export FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"

# parallelism (per-job)
CORES=4
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export NUMEXPR_MAX_THREADS=$CORES
export STAGE4_OPTUNA_N_JOBS=1

# ─── concept (A02 의 paper, byte-identical with Phase 6) ─────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."

# ─── BASE env (A02 exact + gemini-2.5-flash) ─────────────────────────
declare -A BASE=(
    [FAVOR_LLM_MODEL]="gemini-2.5-flash"
    [FAVOR_STOP_LOSS_THRESHOLD]="None"
    [FAVOR_ENTRY_CONFIRM_RULE]="up_day_and_close_pos"
    [FAVOR_COMBO_PASS_RATE]="0.4"
    [FAVOR_THRESHOLD_MIN]="0.55"
    [FAVOR_THRESHOLD_MAX]="0.95"
    [STAGE4_N_TRIALS]="20"
    [FAVOR_NATIVE_STRATEGY]="trigger_exit"
    [FAVOR_STAGE4_OBJECTIVE]="calmar"
)

# ─── split S1 (2y / 1y / 1y) ────────────────────────────────────────
TRAIN_START="2022-01-01"
TRAIN_END="2023-12-31"
VAL_START="2024-01-01"
VAL_END="2024-12-31"
TEST_START="2025-01-01"
TEST_END="2025-12-31"
SPLIT_ENVS="FAVOR_TRAIN_START=$TRAIN_START FAVOR_TRAIN_END=$TRAIN_END FAVOR_VAL_START=$VAL_START FAVOR_VAL_END=$VAL_END FAVOR_TEST_START=$TEST_START FAVOR_TEST_END=$TEST_END"

# ─── JOBS array — N seeds ───────────────────────────────────────────
N_SEEDS=${N_SEEDS:-5}
JOBS=()
for i in $(seq 1 "$N_SEEDS"); do
    label=$(printf "A02_flash_seed%02d" "$i")
    JOBS+=("$label")
done

PARALLEL=${PARALLEL_JOBS:-5}
OUTER_LOOP=${OUTER_LOOP:-3}
echo "[$(date +%F\ %T)] A02 × gemini-2.5-flash multi-seed sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL, outer_loop=$OUTER_LOOP" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] OPENAI_BASE_URL=$OPENAI_BASE_URL (Gemini OpenAI-compat)" | tee -a "$MASTER_LOG"

if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,objective_mode,actual_horizon_days,n_combos,is_best_oos_ir,is_best_oos_ar,is_best_oos_mdd,is_best_oos_cr,oracle_oos_ir,oracle_oos_ar,oracle_oos_mdd,oracle_oos_cr" > "$RESULTS_CSV"
fi

# ─── single-job runner ──────────────────────────────────────────────
run_one_job() {
    local label="$1"
    local concept="$C_PAPER"
    local job_log="$LOG_DIR/sweep_a02_gemini_flash_multiseed_${label}.log"
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
        echo "[$start_str] [$label] starting (A02 × gemini-2.5-flash)"
        echo "env: ${env_args[*]}"
        echo "outer_loop: $OUTER_LOOP"
        echo "============================================================"
    } >> "$job_log"
    echo "[$start_str] [$label] starting" >> "$MASTER_LOG"

    cd "$ROOT"
    (
        for kv in "${env_args[@]}"; do export "$kv"; done
        # Gemini 라우팅 env 도 sub-shell 에 inherit
        export OPENAI_API_KEY="$GEMINI_API_KEY"
        export OPENAI_BASE_URL="$GEMINI_BASE_URL"
        nice -n 10 python run_phase6/run_pipeline_v2.py "$concept" \
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
for label in "${JOBS[@]}"; do
    if [ -f "$LOG_DIR/STOP" ]; then break; fi
    i=$((i+1))
    echo "[$(date +%F\ %T)] queue: ($i/${#JOBS[@]}) $label" | tee -a "$MASTER_LOG"

    run_one_job "$label" &

    while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
done

wait
echo "[$(date +%F\ %T)] A02 × gemini-2.5-flash multi-seed sweep complete; ${#JOBS[@]} jobs processed" | tee -a "$MASTER_LOG"
