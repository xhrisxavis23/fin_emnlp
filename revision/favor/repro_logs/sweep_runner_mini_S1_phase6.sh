#!/bin/bash
# ====================================================================
# FaVOR Phase 6 sweep — mini × S1 × Stage 4 objective (Calmar/IR/IR-λMDD)
# ====================================================================
# Goal:
#   Stage 4 Optuna objective 를 IR → {Calmar, IR-λMDD} 로 교체했을 때
#   honest (is-best) 모드에서 baseline transformer (IR=0.30 / AR=0.027 /
#   MDD=-0.20 / CR=0.125) 4지표 동시 능가 셋팅이 1→다수로 늘어나는지 검증.
#
# Layout (50 jobs):
#   Block A (10) — winner anchor × {IR, Calmar}
#   Block B (10) — compressed neighborhood × {IR, Calmar}
#   Block C (10) — volcomp    neighborhood × {IR, Calmar}
#   Block D (12) — 신규 concept 3종 (gap_fill / volume_spike / oversold_bounce) × Calmar
#   Block F (4)  — top anchor × IR-λMDD (λ=2)
#   Block G (4)  — Calmar × horizon dependency (panic h=3 vs paper h=5+)
#
# Notes:
#   - FAVOR_HORIZON_DAYS env 는 일부러 unset — LLM hypothesis 의 horizon_days
#     필드가 stage4.py:1819 에서 cfg.stage4.horizon_days 를 덮어쓴다.
#   - 새 launcher: run_phase6/run_pipeline_v2.py  (stage4 _create_objective 를
#     run_phase6.objective_patch 로 monkey-patch 후 동일 파이프라인 실행).
#   - 기존 sweep_runner_mini_S1_phase5.sh 의 골격을 그대로 따른다; 결과 CSV /
#     로그 / artifact 경로만 phase6 로 분리.
#
# Total: 50 jobs, parallel=5. Expected wall ~4-5h. Cost ~$5-15.
# ====================================================================

set -u
export TZ='Asia/Seoul'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/repro_logs"
ART_DIR="$LOG_DIR/sweep_artifacts_mini_S1_phase6"
MASTER_LOG="$LOG_DIR/sweep_mini_S1_phase6_master.log"
RESULTS_CSV="$LOG_DIR/sweep_mini_S1_phase6_results.csv"
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

# ─── concept text (existing 4 + new 3) ──────────────────────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."
C_PANIC="After a sharp sell-off, stocks that close near the day's high, indicating strong intraday recovery, are more likely to rebound over the next 3 trading days."
C_COMPRESSED="Following a prior price decline, some stocks enter a compressed trading state characterized by reduced intraday range and the absence of further downside follow-through. When selling pressure becomes exhausted while latent demand continues to absorb supply, prices may remain temporarily stable despite elevated participation. This imbalance can resolve abruptly, leading to a sharp upward price movement over the subsequent few trading days."
C_VOLCOMP="In a strong uptrend, pullbacks toward the 20-day moving average accompanied by compressed volatility increase the probability of an upside continuation."

C_GAP_FILL="After an overnight price gap up or down, the price often retraces back to fill at least half of the gap within 2 trading days."
C_VOLUME_SPIKE="Stocks with abnormally high volume relative to their 20-day average tend to continue moving in the day's direction over the next 5 trading days."
C_OVERSOLD_BOUNCE="Stocks that have declined for 5 or more consecutive sessions with progressively smaller losses tend to rebound over the next 5-7 trading days."

# ─── baseline (FAVOR_HORIZON_DAYS intentionally absent) ──────────────
declare -A BASE=(
    [FAVOR_LLM_MODEL]="gpt-5.4-mini"
    [FAVOR_LLM_TEMPERATURE]="0.7"
    [FAVOR_STOP_LOSS_THRESHOLD]="-0.05"
    [STAGE4_N_TRIALS]="20"
    [FAVOR_THRESHOLD_MIN]="0.55"
    [FAVOR_THRESHOLD_MAX]="0.95"
    [FAVOR_ENTRY_CONFIRM_RULE]="none"
    [FAVOR_NATIVE_STRATEGY]="trigger_exit"
    [FAVOR_COMBO_PASS_RATE]="0.5"
    [FAVOR_STAGE4_OBJECTIVE]="ir"
    [FAVOR_STAGE4_MDD_LAMBDA]="2.0"
)

# ─── split S1 (2y/1y/1y) ────────────────────────────────────────────
SPLIT_LABEL="S1"
TRAIN_START="2022-01-01"
TRAIN_END="2023-12-31"
VAL_START="2024-01-01"
VAL_END="2024-12-31"
TEST_START="2025-01-01"
TEST_END="2025-12-31"
SPLIT_ENVS="FAVOR_TRAIN_START=$TRAIN_START FAVOR_TRAIN_END=$TRAIN_END FAVOR_VAL_START=$VAL_START FAVOR_VAL_END=$VAL_END FAVOR_TEST_START=$TEST_START FAVOR_TEST_END=$TEST_END"

# ─── helper: build "K=V K=V ..." override string ────────────────────
ov() {
    # ov stop entry pr thr objective
    # stop: -0.05 / -0.07 / None
    # entry: none / up_day_and_close_pos
    # pr: 0.4 / 0.5 / 0.6
    # thr: 0.55 / 0.70
    # objective: ir / calmar / ir_minus_mdd
    local stop="$1" entry="$2" pr="$3" thr="$4" obj="$5"
    echo "FAVOR_STOP_LOSS_THRESHOLD=$stop FAVOR_ENTRY_CONFIRM_RULE=$entry FAVOR_COMBO_PASS_RATE=$pr FAVOR_THRESHOLD_MIN=$thr FAVOR_STAGE4_OBJECTIVE=$obj"
}

# ─── JOBS array — 50 entries: "label|concept|overrides" ──────────────
JOBS=()

# Block A — winner anchor × {IR, Calmar} (10 jobs)
JOBS+=("A01_paper_anchor_ir|$C_PAPER|$(ov None up_day_and_close_pos 0.4 0.55 ir)")
JOBS+=("A02_paper_anchor_calmar|$C_PAPER|$(ov None up_day_and_close_pos 0.4 0.55 calmar)")
JOBS+=("A03_compressed_anchor_ir|$C_COMPRESSED|$(ov None none 0.6 0.55 ir)")
JOBS+=("A04_compressed_anchor_calmar|$C_COMPRESSED|$(ov None none 0.6 0.55 calmar)")
JOBS+=("A05_volcomp_anchor_ir|$C_VOLCOMP|$(ov -0.05 up_day_and_close_pos 0.6 0.55 ir)")
JOBS+=("A06_volcomp_anchor_calmar|$C_VOLCOMP|$(ov -0.05 up_day_and_close_pos 0.6 0.55 calmar)")
JOBS+=("A07_volcomp2_anchor_ir|$C_VOLCOMP|$(ov -0.05 none 0.4 0.55 ir)")
JOBS+=("A08_volcomp2_anchor_calmar|$C_VOLCOMP|$(ov -0.05 none 0.4 0.55 calmar)")
JOBS+=("A09_panic_anchor_ir|$C_PANIC|$(ov -0.05 none 0.6 0.70 ir)")
JOBS+=("A10_panic_anchor_calmar|$C_PANIC|$(ov -0.05 none 0.6 0.70 calmar)")

# Block B — compressed neighborhood × {IR, Calmar} (10 jobs)
# anchor: compressed, sN, e0, pr06, t55 → 5 variants × 2 obj
JOBS+=("B01_compressed_s05_ir|$C_COMPRESSED|$(ov -0.05 none 0.6 0.55 ir)")
JOBS+=("B02_compressed_s05_calmar|$C_COMPRESSED|$(ov -0.05 none 0.6 0.55 calmar)")
JOBS+=("B03_compressed_s07_ir|$C_COMPRESSED|$(ov -0.07 none 0.6 0.55 ir)")
JOBS+=("B04_compressed_s07_calmar|$C_COMPRESSED|$(ov -0.07 none 0.6 0.55 calmar)")
JOBS+=("B05_compressed_e1_ir|$C_COMPRESSED|$(ov None up_day_and_close_pos 0.6 0.55 ir)")
JOBS+=("B06_compressed_e1_calmar|$C_COMPRESSED|$(ov None up_day_and_close_pos 0.6 0.55 calmar)")
JOBS+=("B07_compressed_pr05_ir|$C_COMPRESSED|$(ov None none 0.5 0.55 ir)")
JOBS+=("B08_compressed_pr05_calmar|$C_COMPRESSED|$(ov None none 0.5 0.55 calmar)")
JOBS+=("B09_compressed_t70_ir|$C_COMPRESSED|$(ov None none 0.6 0.70 ir)")
JOBS+=("B10_compressed_t70_calmar|$C_COMPRESSED|$(ov None none 0.6 0.70 calmar)")

# Block C — volcomp neighborhood × {IR, Calmar} (10 jobs)
# anchor: volcomp, s05, e1, pr06, t55 → 5 variants × 2 obj
JOBS+=("C01_volcomp_sN_ir|$C_VOLCOMP|$(ov None up_day_and_close_pos 0.6 0.55 ir)")
JOBS+=("C02_volcomp_sN_calmar|$C_VOLCOMP|$(ov None up_day_and_close_pos 0.6 0.55 calmar)")
JOBS+=("C03_volcomp_s07_ir|$C_VOLCOMP|$(ov -0.07 up_day_and_close_pos 0.6 0.55 ir)")
JOBS+=("C04_volcomp_s07_calmar|$C_VOLCOMP|$(ov -0.07 up_day_and_close_pos 0.6 0.55 calmar)")
JOBS+=("C05_volcomp_e0_ir|$C_VOLCOMP|$(ov -0.05 none 0.6 0.55 ir)")
JOBS+=("C06_volcomp_e0_calmar|$C_VOLCOMP|$(ov -0.05 none 0.6 0.55 calmar)")
JOBS+=("C07_volcomp_pr05_ir|$C_VOLCOMP|$(ov -0.05 up_day_and_close_pos 0.5 0.55 ir)")
JOBS+=("C08_volcomp_pr05_calmar|$C_VOLCOMP|$(ov -0.05 up_day_and_close_pos 0.5 0.55 calmar)")
JOBS+=("C09_volcomp_t70_ir|$C_VOLCOMP|$(ov -0.05 up_day_and_close_pos 0.6 0.70 ir)")
JOBS+=("C10_volcomp_t70_calmar|$C_VOLCOMP|$(ov -0.05 up_day_and_close_pos 0.6 0.70 calmar)")

# Block D — 신규 concept × Calmar only (12 jobs)
JOBS+=("D01_gapfill_default_calmar|$C_GAP_FILL|$(ov -0.05 none 0.5 0.55 calmar)")
JOBS+=("D02_gapfill_s07_calmar|$C_GAP_FILL|$(ov -0.07 none 0.5 0.55 calmar)")
JOBS+=("D03_gapfill_e1_calmar|$C_GAP_FILL|$(ov -0.05 up_day_and_close_pos 0.5 0.55 calmar)")
JOBS+=("D04_gapfill_pr06_calmar|$C_GAP_FILL|$(ov -0.05 none 0.6 0.55 calmar)")
JOBS+=("D05_volumespike_default_calmar|$C_VOLUME_SPIKE|$(ov -0.05 none 0.5 0.55 calmar)")
JOBS+=("D06_volumespike_s07_calmar|$C_VOLUME_SPIKE|$(ov -0.07 none 0.5 0.55 calmar)")
JOBS+=("D07_volumespike_e1_calmar|$C_VOLUME_SPIKE|$(ov -0.05 up_day_and_close_pos 0.5 0.55 calmar)")
JOBS+=("D08_volumespike_pr06_calmar|$C_VOLUME_SPIKE|$(ov -0.05 none 0.6 0.55 calmar)")
JOBS+=("D09_oversold_default_calmar|$C_OVERSOLD_BOUNCE|$(ov -0.05 none 0.5 0.55 calmar)")
JOBS+=("D10_oversold_s07_calmar|$C_OVERSOLD_BOUNCE|$(ov -0.07 none 0.5 0.55 calmar)")
JOBS+=("D11_oversold_e1_calmar|$C_OVERSOLD_BOUNCE|$(ov -0.05 up_day_and_close_pos 0.5 0.55 calmar)")
JOBS+=("D12_oversold_pr06_calmar|$C_OVERSOLD_BOUNCE|$(ov -0.05 none 0.6 0.55 calmar)")

# Block F — IR-λMDD penalty (4 jobs, λ=2 via BASE)
JOBS+=("F01_compressed_anchor_irmddpen|$C_COMPRESSED|$(ov None none 0.6 0.55 ir_minus_mdd)")
JOBS+=("F02_volcomp_anchor_irmddpen|$C_VOLCOMP|$(ov -0.05 up_day_and_close_pos 0.6 0.55 ir_minus_mdd)")
JOBS+=("F03_panic_anchor_irmddpen|$C_PANIC|$(ov -0.05 none 0.6 0.70 ir_minus_mdd)")
JOBS+=("F04_paper_anchor_irmddpen|$C_PAPER|$(ov None up_day_and_close_pos 0.4 0.55 ir_minus_mdd)")

# Block G — Calmar × horizon dependency (4 jobs)
JOBS+=("G01_panic_s07_calmar|$C_PANIC|$(ov -0.07 none 0.6 0.70 calmar)")
JOBS+=("G02_panic_sN_calmar|$C_PANIC|$(ov None none 0.6 0.70 calmar)")
JOBS+=("G03_paper_s05_calmar|$C_PAPER|$(ov -0.05 up_day_and_close_pos 0.4 0.55 calmar)")
JOBS+=("G04_paper_s07_calmar|$C_PAPER|$(ov -0.07 up_day_and_close_pos 0.4 0.55 calmar)")

PARALLEL=${PARALLEL_JOBS:-5}
OUTER_LOOP=${OUTER_LOOP:-3}

# ─── optional smoke filter ──────────────────────────────────────────
# SMOKE_JOB=A04_compressed_anchor_calmar  → run only that one job (for smoke test)
if [ -n "${SMOKE_JOB:-}" ]; then
    FILTERED=()
    for j in "${JOBS[@]}"; do
        IFS='|' read -r lbl _rest <<< "$j"
        if [ "$lbl" = "$SMOKE_JOB" ]; then
            FILTERED+=("$j")
        fi
    done
    if [ "${#FILTERED[@]}" -eq 0 ]; then
        echo "[ERROR] SMOKE_JOB='$SMOKE_JOB' did not match any of the ${#JOBS[@]} jobs" >&2
        exit 2
    fi
    JOBS=("${FILTERED[@]}")
    echo "[$(date +%F\ %T)] SMOKE mode: 1 job ($SMOKE_JOB)" | tee -a "$MASTER_LOG"
fi

echo "[$(date +%F\ %T)] Phase 6 mini × S1 × Stage 4 objective sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL, outer_loop=$OUTER_LOOP" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] log dir: $LOG_DIR" | tee -a "$MASTER_LOG"

if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,objective_mode,actual_horizon_days,n_combos,is_best_oos_ir,is_best_oos_ar,is_best_oos_mdd,is_best_oos_cr,oracle_oos_ir,oracle_oos_ar,oracle_oos_mdd,oracle_oos_cr" > "$RESULTS_CSV"
fi

# ─── single-job runner ──────────────────────────────────────────────
run_one_job() {
    local label="$1" concept="$2" overrides="$3"
    local job_log="$LOG_DIR/sweep_mini_S1_phase6_${label}.log"
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

    # objective mode (for CSV reporting)
    local obj_mode="ir"
    for kv in $overrides; do
        if [[ "$kv" == FAVOR_STAGE4_OBJECTIVE=* ]]; then
            obj_mode="${kv#FAVOR_STAGE4_OBJECTIVE=}"
        fi
    done

    local start_ts=$(date +%s) start_str=$(date +%F\ %T)
    {
        echo "============================================================"
        echo "[$start_str] [$label] starting (Phase 6, obj=$obj_mode)"
        echo "concept: $concept"
        echo "env: ${env_args[*]}"
        echo "outer_loop: $OUTER_LOOP"
        echo "============================================================"
    } >> "$job_log"
    echo "[$start_str] [$label] starting (obj=$obj_mode)" >> "$MASTER_LOG"

    cd "$ROOT"
    (
        for kv in "${env_args[@]}"; do export "$kv"; done
        nice -n 10 python run_phase6/run_pipeline_v2.py "$concept" \
            --combo-workers 4 \
            --optuna-jobs "${STAGE4_OPTUNA_N_JOBS:-1}" \
            --outer-loop "$OUTER_LOOP" 2>&1
    ) >> "$job_log" 2>&1
    local exit_code=$?
    local end_ts=$(date +%s) end_str=$(date +%F\ %T)
    local wall=$((end_ts - start_ts))

    echo "$run_id" > "$marker"

    # extract honest IS-best + oracle OOS metrics (IR, AR, MDD, CR) and actual horizon used
    local metrics=$("$PY_BIN" -c "
import json
try:
    s = json.load(open('$ROOT/runs/$run_id/specs/stage4_summary.json'))
    iters = sorted([k for k in s if k.startswith('outer_iter_')])
    best = None; best_score = -1e18
    actual_h = ''
    for k in iters:
        # try to capture the horizon used (best-effort; varies by schema version)
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

    echo "$label,$run_id,$exit_code,$start_str,$end_str,$wall,$obj_mode,$metrics" >> "$RESULTS_CSV"
    echo "[$end_str] [$label] finished (exit=$exit_code, wall=${wall}s, obj=$obj_mode, run_id=$run_id, metrics=$metrics)" \
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
echo "[$(date +%F\ %T)] Phase 6 sweep complete; ${#JOBS[@]} jobs processed" | tee -a "$MASTER_LOG"
