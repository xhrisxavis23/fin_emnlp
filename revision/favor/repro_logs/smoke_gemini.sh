#!/usr/bin/env bash
# Gemini smoke test — FaVOR 코드 0 변경, env override 만으로 라우팅
# 목적: gemini-2.5-flash 가 Stage 1-4 전체를 통과하는지 확인 (코드 호환성)

set -u
ROOT=/home/dgu/fin/revision/revision/favor
cd "$ROOT"

# .env 로드
set -a
source "$ROOT/.env"
set +a

# Gemini 로 OpenAI client redirect
export OPENAI_API_KEY="$GEMINI_API_KEY"
export OPENAI_BASE_URL="$GEMINI_BASE_URL"

# Minimal smoke 세팅
export FAVOR_LLM_MODEL="gemini-2.5-flash"
export FAVOR_LLM_TEMPERATURE="0.7"
export FAVOR_HORIZON_DAYS="5"
export FAVOR_STOP_LOSS_THRESHOLD="-0.10"
export STAGE4_N_TRIALS="2"
export FAVOR_THRESHOLD_MIN="0.55"
export FAVOR_THRESHOLD_MAX="0.95"
export FAVOR_ENTRY_CONFIRM_RULE="none"
export FAVOR_NATIVE_STRATEGY="trigger_exit"
export FAVOR_COMBO_PASS_RATE="0.5"
export FAVOR_TRAIN_START="2022-01-01"
export FAVOR_TRAIN_END="2023-12-31"
export FAVOR_VAL_START="2024-01-01"
export FAVOR_VAL_END="2024-12-31"
export FAVOR_TEST_START="2025-01-01"
export FAVOR_TEST_END="2025-12-31"

# KST 타임스탬프
export TZ='Asia/Seoul'
RUN_ID="$(date +%Y%m%d_%H%M%S)_smoke_gemini25flash"
export FAVOR_RUN_ID="$RUN_ID"

LOG="$ROOT/repro_logs/smoke_gemini_${RUN_ID}.log"

echo "=== Gemini smoke test ===" | tee "$LOG"
echo "run_id      : $RUN_ID" | tee -a "$LOG"
echo "model       : $FAVOR_LLM_MODEL" | tee -a "$LOG"
echo "base_url    : $OPENAI_BASE_URL" | tee -a "$LOG"
echo "n_trials    : $STAGE4_N_TRIALS  outer_loop=1  combo_workers=1" | tee -a "$LOG"
echo "concept     : 'Continuation breakout pattern with pullback'" | tee -a "$LOG"
echo "started     : $(date -Iseconds)" | tee -a "$LOG"
echo "=========================" | tee -a "$LOG"

PY_BIN="${PY_BIN:-/home/dgu/.conda/envs/quant/bin/python}"
"$PY_BIN" run_pipeline_parallel_per_combo_parallel.py \
    "Continuation breakout pattern with pullback to the 20-day moving average increases the probability of upside continuation in the next 5 days." \
    --combo-workers 1 \
    --optuna-jobs 1 \
    --outer-loop 1 2>&1 | tee -a "$LOG"

EXIT=${PIPESTATUS[0]}
echo "===" | tee -a "$LOG"
echo "exit_code: $EXIT" | tee -a "$LOG"
echo "ended    : $(date -Iseconds)" | tee -a "$LOG"
echo "log      : $LOG" | tee -a "$LOG"
exit $EXIT
