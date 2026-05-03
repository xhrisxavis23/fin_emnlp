#!/bin/bash
################################################################################
# CN Market (CSI500) Pipeline Runner (LOCAL OpenAI-compatible LLM)
#
# Model: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
#
# Suggested vLLM server:
#   python3 -m vllm.entrypoints.openai.api_server \
#     --model /path/to/DeepSeek-R1-0528-Qwen3-8B \
#     --served-model-name deepseek-r1-0528-qwen3-8b \
#     --port 8002 \
#     --dtype bfloat16 \
#     --trust-remote-code \
#     --enable-auto-tool-choice \
#     --tool-call-parser qwen3_xml
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PY_BIN="${PY_BIN:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python"
fi

export MARKET=cn
export PYTHONWARNINGS=ignore

export FINAGENT_OPENAI_BASE_URL="${FINAGENT_OPENAI_BASE_URL:-http://127.0.0.1:8002/v1}"
export OPENAI_BASE_URL="$FINAGENT_OPENAI_BASE_URL"
export FINAGENT_OPENAI_API_KEY="EMPTY"
export OPENAI_API_KEY="EMPTY"

export LLM_MODEL="${LLM_MODEL:-deepseek-r1-0528-qwen3-8b}"

# Where to store run artifacts (in case the repo is not writable).
export FINAGENT_RUNS_DIR="${FINAGENT_RUNS_DIR:-/tmp/finagent_runs}"

# Local models often struggle to reliably generate 2–3 formulas per observation.
# Relax the guard to require at least 1 formula per observation by default.
export FINAGENT_FORMULAS_PER_OBS_MIN="${FINAGENT_FORMULAS_PER_OBS_MIN:-1}"
export FINAGENT_FORMULAS_PER_OBS_MAX="${FINAGENT_FORMULAS_PER_OBS_MAX:-3}"

# Stage3: make strictness monotonicity checks less brittle with local models.
export FINAGENT_STAGE3_RANDOM_GRID_STEPS="${FINAGENT_STAGE3_RANDOM_GRID_STEPS:-10}"
export FINAGENT_STAGE3_MONOTONICITY_THRESHOLD="${FINAGENT_STAGE3_MONOTONICITY_THRESHOLD:-0.6}"

# Force sequential execution to avoid multiprocessing overhead/issues.
export STAGE4_N_PROCESSES=1
export STAGE3_N_PROCESSES=1

echo "=================================================="
echo "🇨🇳 CN Market (CSI500) + Local LLM"
echo "=================================================="
echo "BASE_URL: $FINAGENT_OPENAI_BASE_URL"
echo "MODEL:    $LLM_MODEL"
echo "STAGE3_N_PROCESSES: $STAGE3_N_PROCESSES"
echo "STAGE4_N_PROCESSES: $STAGE4_N_PROCESSES"
echo "STAGE3_RANDOM_GRID_STEPS: $FINAGENT_STAGE3_RANDOM_GRID_STEPS"
echo "STAGE3_MONOTONICITY_THRESHOLD: $FINAGENT_STAGE3_MONOTONICITY_THRESHOLD"
echo ""

if [ "${SKIP_SMOKE_TEST:-0}" != "1" ]; then
  echo "🔎 Smoke testing tool calling..."
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY_BIN" scripts/llm_smoke_test.py --model "$LLM_MODEL" --base-url "$FINAGENT_OPENAI_BASE_URL" --api-key "EMPTY"
  echo ""
fi

echo "🚀 Running pipeline..."
"$PY_BIN" run_pipeline_new.py "$@"
