#!/bin/bash
################################################################################
# CN Market Pipeline Runner - Resource Limited (FIXED!)
# 환경변수 이름 수정: Python 파일이 실제로 읽는 변수명 사용
#
# 수정 사항:
#   ❌ STAGE4_MAX_COMBINATIONS_TO_EVALUATE (Python이 안 읽음)
#   ✅ STAGE4_COMBO_WORKERS (Python이 읽음)
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ═══════════════════════════════════════════════════════════════════════════
# 리소스 제한 설정
# ═══════════════════════════════════════════════════════════════════════════

# CPU 코어 제한 (각 실험당)
CORES_PER_EXPERIMENT=20

# 메모리 제한 (GB 단위)
MEMORY_LIMIT_GB=150

# ⚠️ 중요: 올바른 환경변수 이름 사용!
# 동시 처리할 조합(combination) 워커 수
COMBO_WORKERS=4

# 각 조합당 Optuna trial 병렬 수
# combo_workers > 1이면 optuna_jobs=1 권장 (총 병렬도 제어)
OPTUNA_JOBS=1

# ═══════════════════════════════════════════════════════════════════════════
# CPU 제한 (환경변수)
# ═══════════════════════════════════════════════════════════════════════════

export POLARS_MAX_THREADS=$CORES_PER_EXPERIMENT
export OMP_NUM_THREADS=$CORES_PER_EXPERIMENT
export MKL_NUM_THREADS=$CORES_PER_EXPERIMENT
export OPENBLAS_NUM_THREADS=$CORES_PER_EXPERIMENT
export NUMEXPR_MAX_THREADS=$CORES_PER_EXPERIMENT

# ═══════════════════════════════════════════════════════════════════════════
# 메모리 제한 (ulimit)
# ═══════════════════════════════════════════════════════════════════════════

MEMORY_LIMIT_KB=$((MEMORY_LIMIT_GB * 1024 * 1024))
ulimit -v $MEMORY_LIMIT_KB 2>/dev/null || echo "Warning: Could not set memory limit with ulimit"

# ═══════════════════════════════════════════════════════════════════════════
# Stage4 설정 (올바른 변수명!)
# ═══════════════════════════════════════════════════════════════════════════

export MARKET=cn
export STAGE4_ENABLE_OPTUNA=True
export PYTHONWARNINGS=ignore
export STAGE4_FIXED_QUANTILES=None

# ✅ 올바른 환경변수 이름
export STAGE4_COMBO_WORKERS=$COMBO_WORKERS
export STAGE4_OPTUNA_N_JOBS=$OPTUNA_JOBS

# ═══════════════════════════════════════════════════════════════════════════
# 정보 출력
# ═══════════════════════════════════════════════════════════════════════════

echo "=================================================="
echo "🇨🇳 CN Market - Resource Limited (FIXED)"
echo "=================================================="
echo "✅ CN market configuration loaded"
echo ""
echo "⚙️  Resource Limits (per experiment):"
echo "   - CPU cores: $CORES_PER_EXPERIMENT"
echo "   - Memory limit: ${MEMORY_LIMIT_GB}GB"
echo "   - Combo workers: $COMBO_WORKERS (parallel combinations)"
echo "   - Optuna jobs: $OPTUNA_JOBS (trials per combo)"
echo "   - Total parallel: ~$((COMBO_WORKERS * OPTUNA_JOBS)) processes"
echo ""
echo "📊 Safe to run 2-3 experiments concurrently"
echo "=================================================="
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════════════════════════

echo "🚀 Running pipeline with resource limits..."

# CLI 인자로도 전달 (이중 보험)
nice -n 10 python run_pipeline_parallel_per_combo_parallel.py \
    --combo-workers $COMBO_WORKERS \
    --optuna-jobs $OPTUNA_JOBS \
    "$@"