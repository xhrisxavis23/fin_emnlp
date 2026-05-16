# A02 셋팅 — Phase 6 sweep 의 유일한 honest 4지표 통과

## 최종 결과 (gpt-5.4-mini × S1 × CSI500)

| metric | honest (is-best) | oracle (oos-best) |
|---|---:|---:|
| IR | **+1.92** | +1.92 |
| AR | **+0.733** | +0.733 |
| MDD | **−0.143** | −0.143 |
| n_combos | 19 | 19 |
| wall | 2205 s (36.7 min) | — |

- run_id: `20260514_165138_A02_paper_anchor_calmar`
- run dir: `revision/revision/favor/runs/20260514_165138_A02_paper_anchor_calmar/`
- 특이점: **honest = oracle 동일값** → IS-best 와 OOS-best 가 같은 trial 로 수렴. cherry-pick gap 0.

## 정확한 env / config

```bash
# === LLM ===
FAVOR_LLM_MODEL="gpt-5.4-mini"              # ← 다음 실험에서 "gpt-4o" 로 교체
# FAVOR_LLM_TEMPERATURE 는 dead env var — config.py 가 읽지만 agents 가 무시.
# 실제 사용 temperature 는 각 agent 의 literal 값:
#   hypothesis_agent.py:185      temperature=0.9   (가장 큰 variance source)
#   observation_agent.py:82      temperature=0.7
#   formula_agent.py (5 sites)   temperature=0.7
#   validation_agent.py:754      temperature=0.1   (의도된 deterministic)

# === Stage 4 objective (Phase 6 신규 lever) ===
FAVOR_STAGE4_OBJECTIVE="calmar"             # = AR / |MDD| 최대화

# === Stage 4 backtest ===
FAVOR_STOP_LOSS_THRESHOLD="None"            # 손절 없음 — A02 fragility 의 핵심 셋팅
FAVOR_ENTRY_CONFIRM_RULE="up_day_and_close_pos"
FAVOR_NATIVE_STRATEGY="trigger_exit"
FAVOR_THRESHOLD_MIN="0.55"
FAVOR_THRESHOLD_MAX="0.95"
STAGE4_N_TRIALS="20"

# === Stage 3 ===
FAVOR_COMBO_PASS_RATE="0.4"

# === horizon ===
# FAVOR_HORIZON_DAYS unset — LLM hypothesis 가 결정
# 실측 actual_horizon_days = 5 (A02 의 경우)

# === Split S1 (2y / 1y / 1y) ===
FAVOR_TRAIN_START="2022-01-01"
FAVOR_TRAIN_END="2023-12-31"
FAVOR_VAL_START="2024-01-01"
FAVOR_VAL_END="2024-12-31"
FAVOR_TEST_START="2025-01-01"
FAVOR_TEST_END="2025-12-31"

# === Qlib data ===
FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"

# === Outer loop / parallelism ===
OUTER_LOOP=3
STAGE4_COMBO_WORKERS=4
STAGE4_OPTUNA_N_JOBS=1
```

## Concept 텍스트 (Stage 1 hypothesis_agent 의 seed)

```
After a breakout to a new high, a pullback toward the 20-day moving average
often serves as support, increasing the probability of price revisiting
the breakout level or exceeding it.
```

(label: `paper`. Phase 5 launcher 의 `C_PAPER` 와 동일.)

## CLI 호출 형태

```bash
python run_phase6/run_pipeline_v2.py "<C_PAPER text 위 그대로>" \
    --combo-workers 4 \
    --optuna-jobs 1 \
    --outer-loop 3
```

`run_pipeline_v2.py` 가 import 시점에 `objective_patch.apply()` 호출 →
3 모듈 (`stage4`, `stage4_parallel`, `stage4_parallel_per_combo`) 의
`_create_objective` 가 `_create_objective_v2` 로 rebind → `FAVOR_STAGE4_OBJECTIVE=calmar`
환경변수 인식.

## Phase 6 sweep 의 같은 anchor 변형 (fragility 비교)

| label | stop_loss | honest IR | honest AR | honest MDD | KPI |
|---|---|---:|---:|---:|---|
| **A02** | **None** | **+1.92** | **+0.733** | **−0.143** | ✅ |
| G03 paper_s05_calmar | −0.05 | −0.73 | −0.192 | −0.411 | ✗ |
| G04 paper_s07_calmar | −0.07 | −0.82 | −0.224 | −0.404 | ✗ |
| F04 paper_anchor_irmddpen | None (obj=ir−2·\|MDD\|) | FAIL | FAIL | FAIL | ✗ |

→ stop_loss 단일 변경으로 honest IR 이 +1.92 → −0.73 / −0.82 로 붕괴.
→ objective 만 Calmar → IR−λMDD 로 바꿔도 (F04) FAIL — Calmar 가 결정적 lever.

## gpt-4o 로 옮길 때 변경할 부분

**유일하게 바뀌는 것**:
```bash
FAVOR_LLM_MODEL="gpt-4o"                    # gpt-5.4-mini → gpt-4o
```

**유의 사항**:
- gpt-4o 는 agent 별 literal temperature 모두 진짜로 동작:
  - hypothesis@0.9 → multi-seed variation 의 *주된 source* (창의적 가설 다양성)
  - formula@0.7 → 적당한 paraphrase variation
  - validation@0.1 → near-deterministic (의도된 strict judgment)
- 따라서 multi-seed sweep 의 결과 variation 은 주로 hypothesis 단계에서 발생.
  실제로 5 회 호출 → 5 가지 hypothesis → 5 가지 formula bundle → 5 가지 backtest 결과 기대.
- gpt-4o pricing: input $2.50/M, output $10/M (gpt-5.4-mini 의 ~10×).
- A02 1 회 token 비용 추정 $0.50-1.00 (mini 의 $0.05 대비 ~10×) → seed 5 회 ≈ $2.5-5.
- wall time 은 비슷하거나 더 빠를 수도 (gpt-4o reasoning 없음).
- 검증 목표: A02 의 honest IR=+1.92 가 backbone 교체 후에도 재현되는가
  - mean(honest IR) > +0.30 + std 작음 → A02 = robust + backbone 무관 → publishing-grade
  - mean ≈ 0 → A02 = gpt-5.4-mini 의 단일 lucky basin → paper limitation 으로 정직 보고
  - mean > 0 + std 큼 → robust 하지만 backbone 의존성 큼 → 추가 backbone 검증 필요

## 산출물 위치

- 원본 launcher: `revision/revision/favor/repro_logs/sweep_runner_mini_S1_phase6.sh` (A02 line)
- 패치 module: `revision/revision/favor/run_phase6/objective_patch.py`
- 새 entrypoint: `revision/revision/favor/run_phase6/run_pipeline_v2.py`
- Phase 6 sweep CSV: `revision/revision/favor/repro_logs/sweep_mini_S1_phase6_results.csv`
- Phase 6 dashboard: `revision/revision/favor/favor_dashboard_phase6.html`
