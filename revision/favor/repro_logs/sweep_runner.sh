#!/bin/bash
# ====================================================================
# FaVOR sweep runner — 10-hour budget
# ====================================================================
# Sweeps over the levers most likely to affect OOS performance:
#   1. concept (5 variants)
#   2. horizon_days {5, 10, 20}
#   3. stop_loss_threshold {-0.05, -0.10, None}
#   4. n_trials {20, 50, 100}
#   5. threshold range [0.55, 0.95] vs [0.7, 0.95]
#   6. entry_confirm_rule {none, up_day_and_close_pos}
#   7. native_strategy {trigger_exit, topk_dropout}
#   8. combination_pass_rate {0.4, 0.5, 0.6}
#
# Phase A: concept × horizon grid (5 × 3 = 15 runs)
# Phase B: paper-concept micro-tune (8 runs)
# Total : 23 runs
#
# Resources: 3 runs in parallel × 4 combo_workers each = 12 active workers,
# n_trials=50 baseline, ~30-90 min per run, expected wall time 6-9 hours.
#
# Usage:
#   nohup ./repro_logs/sweep_runner.sh > /dev/null 2>&1 &
#
# Graceful halt:
#   touch repro_logs/STOP   # stops launching new jobs; in-flight jobs continue
#
# Re-runnable: jobs whose label dir already has stage4_summary.json are skipped.
# ====================================================================

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts"
MASTER_LOG="$LOG_DIR/sweep_master.log"
RESULTS_CSV="$LOG_DIR/sweep_results.csv"
mkdir -p "$ART_DIR"

# ─── conda + python ─────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate quant

# ─── shared env ─────────────────────────────────────────────────────
export MARKET=cn
export FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"
export STAGE4_ENABLE_OPTUNA=True
export STAGE4_FIXED_QUANTILES=None
export STAGE4_COMBO_WORKERS=4
export STAGE4_OPTUNA_N_JOBS=1
export PYTHONWARNINGS=ignore
# CPU caps so 3 parallel runs fit in 64 cores comfortably
CORES=20
export POLARS_MAX_THREADS=$CORES
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export NUMEXPR_MAX_THREADS=$CORES

# ─── concept text ───────────────────────────────────────────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."
C_UPTREND="In a strong uptrend, when price pulls back to the 20-day moving average, buying near that level increases the probability of price retesting the previous high within the next 5-10 trading days."
C_PANIC="After a sharp sell-off, stocks that close near the day's high, indicating strong intraday recovery, are more likely to rebound over the next 3 trading days."
C_COMPRESSED="Following a prior price decline, some stocks enter a compressed trading state characterized by reduced intraday range and the absence of further downside follow-through. When selling pressure becomes exhausted while latent demand continues to absorb supply, prices may remain temporarily stable despite elevated participation. This imbalance can resolve abruptly, leading to a sharp upward price movement over the subsequent few trading days."
C_VOLCOMP="In a strong uptrend, pullbacks toward the 20-day moving average accompanied by compressed volatility increase the probability of an upside continuation."

# Baseline used when a sweep dimension is not explicitly varied.
# (matches paper Table 1 settings)
declare -A BASE=(
    [FAVOR_LLM_MODEL]="gpt-5.4-mini"
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

# ─── job list ───────────────────────────────────────────────────────
# Format: "label|concept|var1=val1 var2=val2 ..."
# Each job inherits BASE then overlays its own var=val pairs.
JOBS=(
  # ─── Phase A: 5 concepts × 3 horizons (15) ─────────────────────────
  "A01_paper_h5|$C_PAPER|FAVOR_HORIZON_DAYS=5"
  "A02_paper_h10|$C_PAPER|FAVOR_HORIZON_DAYS=10"
  "A03_paper_h20|$C_PAPER|FAVOR_HORIZON_DAYS=20"
  "A04_uptrend_h5|$C_UPTREND|FAVOR_HORIZON_DAYS=5"
  "A05_uptrend_h10|$C_UPTREND|FAVOR_HORIZON_DAYS=10"
  "A06_uptrend_h20|$C_UPTREND|FAVOR_HORIZON_DAYS=20"
  "A07_panic_h5|$C_PANIC|FAVOR_HORIZON_DAYS=5"
  "A08_panic_h10|$C_PANIC|FAVOR_HORIZON_DAYS=10"
  "A09_panic_h20|$C_PANIC|FAVOR_HORIZON_DAYS=20"
  "A10_compressed_h5|$C_COMPRESSED|FAVOR_HORIZON_DAYS=5"
  "A11_compressed_h10|$C_COMPRESSED|FAVOR_HORIZON_DAYS=10"
  "A12_compressed_h20|$C_COMPRESSED|FAVOR_HORIZON_DAYS=20"
  "A13_volcomp_h5|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5"
  "A14_volcomp_h10|$C_VOLCOMP|FAVOR_HORIZON_DAYS=10"
  "A15_volcomp_h20|$C_VOLCOMP|FAVOR_HORIZON_DAYS=20"

  # ─── Phase B: paper concept × micro-tune (8) ───────────────────────
  "B01_stoploss_005|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=-0.05"
  "B02_stoploss_none|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=None"
  "B03_n_trials_100|$C_PAPER|STAGE4_N_TRIALS=100"
  "B04_thr_07_095|$C_PAPER|FAVOR_THRESHOLD_MIN=0.7"
  "B05_entry_uday|$C_PAPER|FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos"
  "B06_strat_topk|$C_PAPER|FAVOR_NATIVE_STRATEGY=topk_dropout"
  "B07_combopass_04|$C_PAPER|FAVOR_COMBO_PASS_RATE=0.4"
  "B08_combopass_06|$C_PAPER|FAVOR_COMBO_PASS_RATE=0.6"
)

PARALLEL=${PARALLEL_JOBS:-3}
echo "[$(date +%F\ %T)] sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] log dir: $LOG_DIR" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] artifact dir: $ART_DIR" | tee -a "$MASTER_LOG"

# results.csv header
if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,n_combos,best_is_ir,best_is_oos_ir,best_is_oos_ar,oracle_oos_ir,oracle_oos_ar" > "$RESULTS_CSV"
fi

# ─── single-job runner ───────────────────────────────────────────────
run_one_job() {
    local label="$1"
    local concept="$2"
    local overrides="$3"
    local job_log="$LOG_DIR/sweep_${label}.log"
    local marker="$ART_DIR/${label}.run_id"

    # Build env: base + overrides
    local env_args=()
    for k in "${!BASE[@]}"; do env_args+=("$k=${BASE[$k]}"); done
    # Override with job-specific
    for kv in $overrides; do env_args+=("$kv"); done

    # Deterministic run_id: <timestamp>_<label>
    local run_id="$(date +%Y%m%d_%H%M%S)_${label}"
    env_args+=("FAVOR_RUN_ID=$run_id")

    # Skip if this exact run already finished
    if [ -f "$ROOT/runs/$run_id/specs/stage4_summary.json" ]; then
        echo "[$(date +%F\ %T)] [$label] SKIP (already has stage4_summary.json at runs/$run_id)" | tee -a "$MASTER_LOG"
        echo "$run_id" > "$marker"
        return 0
    fi
    # Also accept any prior run_id captured in the marker (back-compat with older sweeps)
    if [ -f "$marker" ]; then
        local prev_run_id
        prev_run_id=$(cat "$marker")
        if [ "$prev_run_id" != "$run_id" ] && [ -f "$ROOT/runs/$prev_run_id/specs/stage4_summary.json" ]; then
            echo "[$(date +%F\ %T)] [$label] SKIP (older successful run at runs/$prev_run_id)" | tee -a "$MASTER_LOG"
            return 0
        fi
    fi

    local start_ts=$(date +%s)
    local start_str=$(date +%F\ %T)
    {
        echo "============================================================"
        echo "[$start_str] [$label] starting"
        echo "concept: $concept"
        echo "env overrides: ${env_args[*]}"
        echo "============================================================"
    } >> "$job_log"
    echo "[$start_str] [$label] starting" >> "$MASTER_LOG"

    cd "$ROOT"
    # Run in subshell so env doesn't leak
    (
        for kv in "${env_args[@]}"; do export "$kv"; done
        nice -n 10 python run_pipeline_parallel_per_combo_parallel.py "$concept" \
            --combo-workers "${STAGE4_COMBO_WORKERS:-4}" \
            --optuna-jobs "${STAGE4_OPTUNA_N_JOBS:-1}" \
            --outer-loop 1
    ) >> "$job_log" 2>&1
    local exit_code=$?

    # run_id is deterministic (set above via FAVOR_RUN_ID); just record it
    echo "$run_id" > "$marker"

    local end_str=$(date +%F\ %T)
    local end_ts=$(date +%s)
    local wall=$((end_ts - start_ts))

    # Try to extract metrics from stage4_summary.json
    local metrics
    metrics=$(python3 - "$ROOT" "$run_id" 2>/dev/null <<'PY'
import json, sys
from pathlib import Path
root, rid = sys.argv[1], sys.argv[2]
p = Path(root)/"runs"/rid/"specs"/"stage4_summary.json"
if not p.exists():
    print(",,,,,")
    sys.exit(0)
j = json.load(open(p))
o = j.get("outer_iter_1") or next(iter(j.values()))
combos = o.get("all_combinations", [])
if not combos:
    print(f"{0},,,,,")
    sys.exit(0)
def ir_is(c): return c["insample"]["excess_return_with_cost"]["information_ratio"]
def ir_oos(c): return c["outsample"]["excess_return_with_cost"]["information_ratio"]
def ar_oos(c): return c["outsample"]["excess_return_with_cost"]["annualized_return"]
bis = max(combos, key=ir_is)
bos = max(combos, key=ir_oos)
print(f'{len(combos)},{ir_is(bis):.4f},{ir_oos(bis):.4f},{ar_oos(bis):.4f},{ir_oos(bos):.4f},{ar_oos(bos):.4f}')
PY
)
    [ -z "$metrics" ] && metrics=",,,,,"

    echo "$label,${run_id:-na},$exit_code,$start_str,$end_str,$wall,$metrics" >> "$RESULTS_CSV"
    echo "[$end_str] [$label] finished (exit=$exit_code, wall=${wall}s, run_id=${run_id:-na}, metrics=$metrics)" \
        | tee -a "$MASTER_LOG" >> "$job_log"
}

# ─── parallel scheduler ──────────────────────────────────────────────
i=0
for job in "${JOBS[@]}"; do
    if [ -f "$LOG_DIR/STOP" ]; then
        echo "[$(date +%F\ %T)] STOP file present — halting new launches" | tee -a "$MASTER_LOG"
        break
    fi
    IFS='|' read -r label concept overrides <<< "$job"
    i=$((i+1))
    echo "[$(date +%F\ %T)] queue: ($i/${#JOBS[@]}) $label" | tee -a "$MASTER_LOG"

    run_one_job "$label" "$concept" "$overrides" &

    # bound parallelism
    while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do
        sleep 10
    done
done

# wait for any remaining
wait
echo "[$(date +%F\ %T)] sweep complete; ${#JOBS[@]} jobs processed" | tee -a "$MASTER_LOG"
