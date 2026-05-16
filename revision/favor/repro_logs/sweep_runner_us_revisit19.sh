#!/usr/bin/env bash
################################################################################
# US (S&P500) re-test of Top 19 settings from CSI500 sweeps
#
# 출처: favor_dashboard 의 130 runs 중
#   - IS-best lens (honest selection): IR/AR/NR > 0, MDD 우선 정렬 → Top 10
#   - Oracle lens                     : IR/AR/NR > 0, MDD 우선 정렬 → Top 10
#   - Union = 19 unique settings (1 overlap: B3_volcomp_h5_sN_e1_pr06_t55)
#
# 정규화: 모든 setting 을 n_trials=20, outer_loop=3 (Phase 5 spec) 으로 통일.
#   원본 5/10 sweep (A13, B01_stoploss_005, N01) 는 n_trials=50, outer_loop=1
#   이었으나 US 비교의 공정성을 위해 통일.
#
# Split: S1 (2022-23 / 24 / 25), 동일.
# Market: MARKET=us → qlib_data sh_sp500_qlib, benchmark ^GSPC.
################################################################################
set -u

ROOT="/home/dgu/fin/revision/revision/favor"
REPRO="$ROOT/repro_logs"
ART_DIR="$REPRO/sweep_us_revisit19_artifacts"
LOG_DIR="$REPRO/sweep_us_revisit19_logs"
MASTER_LOG="$LOG_DIR/master.log"
RESULTS_CSV="$REPRO/sweep_us_revisit19_results.csv"

mkdir -p "$ART_DIR" "$LOG_DIR"
cd "$ROOT"

# .env 로드 (OPENAI_API_KEY)
set -a
source "$ROOT/.env"
set +a

PY_BIN="${PY_BIN:-/home/dgu/.conda/envs/quant/bin/python}"
PARALLEL=${PARALLEL_JOBS:-5}
OUTER_LOOP=${OUTER_LOOP:-3}

# ─── concept text (Phase 5 와 동일) ─────────────────────────────────
C_PAPER="After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it."
C_PANIC="After a sharp sell-off, stocks that close near the day's high, indicating strong intraday recovery, are more likely to rebound over the next 3 trading days."
C_COMPRESSED="Following a prior price decline, some stocks enter a compressed trading state characterized by reduced intraday range and the absence of further downside follow-through. When selling pressure becomes exhausted while latent demand continues to absorb supply, prices may remain temporarily stable despite elevated participation. This imbalance can resolve abruptly, leading to a sharp upward price movement over the subsequent few trading days."
C_VOLCOMP="In a strong uptrend, pullbacks toward the 20-day moving average accompanied by compressed volatility increase the probability of an upside continuation."

# ─── BASE (Phase 5 spec + US market) ────────────────────────────────
declare -A BASE=(
    [MARKET]="us"
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

# ─── S1 split (2022-23 train / 24 val / 25 test) ────────────────────
SPLIT_ENVS="FAVOR_TRAIN_START=2022-01-01 FAVOR_TRAIN_END=2023-12-31 FAVOR_VAL_START=2024-01-01 FAVOR_VAL_END=2024-12-31 FAVOR_TEST_START=2025-01-01 FAVOR_TEST_END=2025-12-31"

# ─── 19 SETTINGS (label | concept | overrides) ─────────────────────
# IS-best lens Top 10:
SETTINGS=(
  # IS Top10 — MDD 우선
  "U01_B3_volcomp_h5_s05_e0_pr06_t70|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5 FAVOR_STOP_LOSS_THRESHOLD=-0.05 FAVOR_ENTRY_CONFIRM_RULE=none FAVOR_COMBO_PASS_RATE=0.6 FAVOR_THRESHOLD_MIN=0.70"
  "U02_B2_compressed_h5_sN_e0_pr06_t55|$C_COMPRESSED|FAVOR_HORIZON_DAYS=5 FAVOR_STOP_LOSS_THRESHOLD=None FAVOR_ENTRY_CONFIRM_RULE=none FAVOR_COMBO_PASS_RATE=0.6 FAVOR_THRESHOLD_MIN=0.55"
  "U03_B4_panic_h3_s05_e0_pr06_t70|$C_PANIC|FAVOR_HORIZON_DAYS=3 FAVOR_STOP_LOSS_THRESHOLD=-0.05 FAVOR_ENTRY_CONFIRM_RULE=none FAVOR_COMBO_PASS_RATE=0.6 FAVOR_THRESHOLD_MIN=0.70"
  "U04_B1_paper_h10_sN_e0_pr04_t55|$C_PAPER|FAVOR_HORIZON_DAYS=10 FAVOR_STOP_LOSS_THRESHOLD=None FAVOR_ENTRY_CONFIRM_RULE=none FAVOR_COMBO_PASS_RATE=0.4 FAVOR_THRESHOLD_MIN=0.55"
  "U05_N01_paper_stop007|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=-0.07"
  "U06_B3_volcomp_h5_s05_e0_pr04_t70|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5 FAVOR_STOP_LOSS_THRESHOLD=-0.05 FAVOR_ENTRY_CONFIRM_RULE=none FAVOR_COMBO_PASS_RATE=0.4 FAVOR_THRESHOLD_MIN=0.70"
  "U07_B2_compressed_h5_sN_e1_pr04_t55|$C_COMPRESSED|FAVOR_HORIZON_DAYS=5 FAVOR_STOP_LOSS_THRESHOLD=None FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos FAVOR_COMBO_PASS_RATE=0.4 FAVOR_THRESHOLD_MIN=0.55"
  "U08_B3_volcomp_h5_sN_e1_pr06_t55|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5 FAVOR_STOP_LOSS_THRESHOLD=None FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos FAVOR_COMBO_PASS_RATE=0.6 FAVOR_THRESHOLD_MIN=0.55"
  "U09_B4_panic_h3_sN_e0_pr04_t55|$C_PANIC|FAVOR_HORIZON_DAYS=3 FAVOR_STOP_LOSS_THRESHOLD=None FAVOR_ENTRY_CONFIRM_RULE=none FAVOR_COMBO_PASS_RATE=0.4 FAVOR_THRESHOLD_MIN=0.55"
  "U10_A13_volcomp_h5|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5"
  # Oracle Top10 — 9 new + 1 overlap (B3_volcomp_h5_sN_e1_pr06_t55 = U08, 제외됨)
  "U11_B3_volcomp_h5_s05_e1_pr06_t55|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5 FAVOR_STOP_LOSS_THRESHOLD=-0.05 FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos FAVOR_COMBO_PASS_RATE=0.6 FAVOR_THRESHOLD_MIN=0.55"
  "U12_M13_compressed_stopNone|$C_COMPRESSED|FAVOR_STOP_LOSS_THRESHOLD=None"
  "U13_B1_paper_h10_sN_e1_pr04_t55|$C_PAPER|FAVOR_HORIZON_DAYS=10 FAVOR_STOP_LOSS_THRESHOLD=None FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos FAVOR_COMBO_PASS_RATE=0.4 FAVOR_THRESHOLD_MIN=0.55"
  "U14_B1_paper_h10_s05_e1_pr04_t55|$C_PAPER|FAVOR_HORIZON_DAYS=10 FAVOR_STOP_LOSS_THRESHOLD=-0.05 FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos FAVOR_COMBO_PASS_RATE=0.4 FAVOR_THRESHOLD_MIN=0.55"
  "U15_M18_volcomp_h10|$C_VOLCOMP|FAVOR_HORIZON_DAYS=10"
  "U16_M08_paper_thr0795|$C_PAPER|FAVOR_THRESHOLD_MIN=0.70"
  "U17_B3_volcomp_h5_s05_e0_pr04_t55|$C_VOLCOMP|FAVOR_HORIZON_DAYS=5 FAVOR_STOP_LOSS_THRESHOLD=-0.05 FAVOR_ENTRY_CONFIRM_RULE=none FAVOR_COMBO_PASS_RATE=0.4 FAVOR_THRESHOLD_MIN=0.55"
  "U18_B01_stoploss_005|$C_PAPER|FAVOR_STOP_LOSS_THRESHOLD=-0.05"
  "U19_B1_paper_h10_s05_e1_pr04_t70|$C_PAPER|FAVOR_HORIZON_DAYS=10 FAVOR_STOP_LOSS_THRESHOLD=-0.05 FAVOR_ENTRY_CONFIRM_RULE=up_day_and_close_pos FAVOR_COMBO_PASS_RATE=0.4 FAVOR_THRESHOLD_MIN=0.70"
)

# ─── Build JOBS ──────────────────────────────────────────────────────
JOBS=()
for s in "${SETTINGS[@]}"; do
    JOBS+=("$s")
done

echo "[$(date +%F\ %T)] US revisit-19 sweep starting; ${#JOBS[@]} jobs, parallel=$PARALLEL, outer_loop=$OUTER_LOOP, market=us" | tee -a "$MASTER_LOG"
echo "[$(date +%F\ %T)] log dir: $LOG_DIR" | tee -a "$MASTER_LOG"

if [ ! -s "$RESULTS_CSV" ]; then
    echo "label,run_id,exit_code,start,end,wall_seconds,n_combos,is_best_oos_ir,is_best_oos_ar,is_best_oos_mdd,is_best_oos_nr,oracle_oos_ir,oracle_oos_ar,oracle_oos_mdd,oracle_oos_nr" > "$RESULTS_CSV"
fi

# ─── single-job runner ──────────────────────────────────────────────
run_one_job() {
    local label="$1" concept="$2" overrides="$3"
    local job_log="$LOG_DIR/${label}.log"
    local marker="$ART_DIR/${label}.run_id"

    local env_args=()
    for k in "${!BASE[@]}"; do env_args+=("$k=${BASE[$k]}"); done
    for kv in $overrides; do env_args+=("$kv"); done
    for kv in $SPLIT_ENVS; do env_args+=("$kv"); done

    local run_id="$(date +%Y%m%d_%H%M%S)_${label}"
    env_args+=("FAVOR_RUN_ID=$run_id")

    if [ -f "$ROOT/runs/$run_id/specs/stage4_summary.json" ]; then
        echo "[$(date +%F\ %T)] [$label] SKIP (already done)" | tee -a "$MASTER_LOG"
        echo "$run_id" > "$marker"
        return 0
    fi

    local start_ts=$(date +%s) start_str=$(date +%F\ %T)
    echo "[$start_str] [$label] START run_id=$run_id" | tee -a "$MASTER_LOG" >> "$job_log"

    (
        for kv in "${env_args[@]}"; do export "$kv"; done
        nice -n 10 "$PY_BIN" run_pipeline_parallel_per_combo_parallel.py "$concept" \
            --combo-workers 4 \
            --optuna-jobs "${STAGE4_OPTUNA_N_JOBS:-1}" \
            --outer-loop "$OUTER_LOOP" 2>&1
    ) >> "$job_log" 2>&1
    local exit_code=$?
    local end_ts=$(date +%s) end_str=$(date +%F\ %T)
    local wall=$((end_ts - start_ts))

    echo "$run_id" > "$marker"

    local metrics=$("$PY_BIN" -c "
import json, sys
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
            best_score = score; best = (k, bis)
    if best is None: print(',,,,,,,,'); sys.exit(0)
    k, bis = best
    combos = s[k].get('all_combinations', [])
    bos = max(combos, key=lambda c: (c.get('outsample', {}).get('excess_return_with_cost', {}) or {}).get('information_ratio', -1e9))
    b = bis.get('outsample', {}).get('excess_return_with_cost', {}) or {}
    o = bos.get('outsample', {}).get('excess_return_with_cost', {}) or {}
    print(f\"{len(combos)},{b.get('information_ratio','')},{b.get('annualized_return','')},{b.get('max_drawdown','')},{b.get('net_return','')},{o.get('information_ratio','')},{o.get('annualized_return','')},{o.get('max_drawdown','')},{o.get('net_return','')}\")
except Exception as e:
    print(',,,,,,,,')
")
    [ -z "$metrics" ] && metrics=",,,,,,,,"

    echo "$label,$run_id,$exit_code,$start_str,$end_str,$wall,$metrics" >> "$RESULTS_CSV"
    echo "[$end_str] [$label] DONE exit=$exit_code wall=${wall}s metrics=$metrics" | tee -a "$MASTER_LOG" >> "$job_log"
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
echo "[$(date +%F\ %T)] US revisit-19 sweep finished — all jobs done" | tee -a "$MASTER_LOG"
