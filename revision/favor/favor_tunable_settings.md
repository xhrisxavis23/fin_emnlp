# FaVOR pipeline tunable settings — 종합 정리

작성일: 2026-05-12
대상 코드베이스: `/home/dgu/fin/revision/revision/favor/`
대상 frozen 비교군: `/home/dgu/fin/01_15_new_qlib/` (KDD 2026 frozen, paper Table 1 출처)

본 문서는 FaVOR pipeline 의 모든 단계 (split, model, Stage 1~4) 에서 **바꿀 수 있는 모든 세팅값** 을 정리한다.
각 세팅마다:

- **현재 default** 값
- **env override** 가능 여부 (가능하면 env 변수 이름)
- **우리가 Phase 0~4 에서 흔든 적 있는지** (해당 Phase + 값)
- **paper 가 어떤 값을 썼는지** (확인된 경우)
- **변경 시 예상 영향**

---

## 표기 규약

- ✓ env: 환경변수로 sweep 가능
- ✗ env: 코드/config 수정 필요
- ★ 흔듦: Phase 0~4 에서 이미 ablation 한 dimension
- ☆ 미흔듦: 아직 한 번도 안 바꿔봄
- ⚠ dead: 코드에 있지만 실제로는 효과 없음 (deprecated / legacy)

---

# 0. 데이터 split (실험 기간)

## 0.1 train / val / test 기간

| param | env | 현재 default | paper Table 1 | 우리가 흔든 값 |
|---|---|---|---|---|
| `FAVOR_TRAIN_START` | ✓ | (config) | 2015-01-01 | 2015/2018/2020/2022 |
| `FAVOR_TRAIN_END` | ✓ | (config) | 2019-12-31 | 2019/2020/2021/2023 |
| `FAVOR_VAL_START` | ✓ | (config) | 2020-01-01 | 2020/2021/2022/2024 |
| `FAVOR_VAL_END` | ✓ | (config) | 2020-12-31 | 2020/2021/2022/2024 |
| `FAVOR_TEST_START` | ✓ | (config) | 2021-01-01 | 2021/2022/2023/2025 |
| `FAVOR_TEST_END` | ✓ | (config) | 2025-12-31 | 2023/2025 |

### ★ 흔든 split

| label | train | val | test | 비율 | 사용 Phase |
|---|---|---|---|---|---|
| **S5 (paper)** | 2015~19 | 2020 | 2021~25 | 5/1/5 | Phase 0, Phase 1, Phase 2 |
| S1 | 2022~23 | 2024 | 2025 | 2/1/1 | Phase 1, Phase 2, Phase 3, Phase 4 |
| S2 | 2020~23 | 2024 | 2025 | 4/1/1 | Phase 1, Phase 2 |
| S3 | 2020~21 | 2022 | 2023~25 | 2/1/3 | Phase 1 |
| S4 | 2018~20 | 2021 | 2022~25 | 3/1/4 | Phase 1 |
| S6 | 2017~19 | 2020 | 2021~25 | 3/1/5 | Phase 1 |

### ☆ 안 흔든 split

- ICU 형식 (train+val 합쳐 walk-forward)
- 1y/1y/3y, 7y/1y/2y 등 극단 비율
- US 시장 (S&P 500) — 코드 path 존재 (`MARKET=us`) 만 unused

---

# 1. 모델 (LLM backbone)

## 1.1 LLM 모델 선택

| param | env | 현재 default | paper Table 1 | 우리가 흔든 값 |
|---|---|---|---|---|
| `FAVOR_LLM_MODEL` | ✓ | gpt-4o | **gpt-4o** | ★ {gpt-4o, gpt-5.4-mini} |

### ★ 흔든 모델

| model | Phase | 효과 |
|---|---|---|
| **gpt-4o** (paper 표준) | Phase 1 (b200), Phase 3 | IS-fit 강함, OOS 정직 능가 어려움 |
| **gpt-5.4-mini** | Phase 0, Phase 2, Phase 4 | 저렴, 빠름. honest OOS 결과 4o 와 큰 차이 없음 |

### ☆ 안 흔든 모델

- gpt-4o-mini (4o 와 mini 사이 가성비)
- gpt-4-turbo (legacy)
- Claude 4.6 Sonnet / 4.6 Opus
- Gemini 2.5 Pro
- Local LLM: Qwen3-30B, Llama-3.x, DeepSeek (Track 2-2 로 진행 예정)

---

# 2. Stage 1 — Hypothesis / Observation / Formula 생성

## 2.1 env-overridable

| param | env | default | 효과 |
|---|---|---|---|
| `FAVOR_LLM_MODEL` | ✓ | gpt-4o | Stage 1 전체 LLM backbone |
| `FAVOR_LLM_TEMPERATURE` | ⚠ dead | 0.7 | **agents 가 literal 박아 호출 → env 무시됨** |

## 2.2 hardcoded in code (code edit 필요)

### LLM temperature (Stage 1 의 핵심 lever)

| 위치 | step | 현재 | ★/☆ |
|---|---|---:|---|
| `agent/hypothesis_agent.py:185` | step 0 (hypothesis 생성) | **0.9** | ☆ |
| `agent/observation_agent.py:82` | step 1 (observation 분해) | 0.7 | ☆ |
| `agent/formula_agent.py:638` | step 2 formula gen (main) | 0.7 | ☆ |
| `agent/formula_agent.py:764` | step 2 formula gen (alt) | 0.7 | ☆ |
| `agent/formula_agent.py:846` | step 2 cross-iter refinement | 0.7 | ☆ |
| `agent/formula_agent.py:344, 916` | self-correction | 0.7 | ☆ |

→ ☆ **모두 미흔듦**. paper 도 0.7 추정 (코드 default). 낮추면 reproducibility ↑, 다양성 ↓.

### Formula 개수 범위

| 위치 | param | default | ★/☆ |
|---|---|---:|---|
| `agent/formula_agent.py:137-138` | `min_formulas_per_observation` | 2 | ☆ |
| 동일 | `max_formulas_per_observation` | 3 | ☆ |
| `agent/formula_agent.py:664-665` | guard 호출 시 actual | 2, 3 | ☆ |

→ 실측 Phase 4: 5 observations × 평균 2 formulas = ~10 formulas/iter (max=3 까지 잘 안 씀)

### Self-correction iteration

| 위치 | param | default | ★/☆ |
|---|---|---:|---|
| `run/config.py` `Stage1Config.refine_rounds` | config | **10 (dead)** | ☆ |
| `agent/formula_agent.py:533, 702` | 실제 호출시 | **1** | ☆ |

→ ⚠ config 의 10 은 caller 가 1 로 덮어쓰여 dead. 진짜 값은 1.

### Stage 1 step 3 evaluation retry

| 위치 | param | default | ★/☆ |
|---|---|---:|---|
| `run/pipeline/stage1.py:294` | `max_eval_retries` | 2 | ☆ |

### OHLCV column 제한

| 위치 | param | default | ★/☆ |
|---|---|---|---|
| `run/config.py` `Stage1Config.allowed_ohlcv_columns` | config | `["open", "high", "low", "close", "volume"]` | ☆ |

→ ☆ `tradingvalue`, `marketcap`, `sharesoutstanding` 등 추가 가능

### Caller 가 빈 채로 호출하는 hook (caller code 만 바꿔도 됨)

| param | 함수 | 현재 호출값 | ★/☆ |
|---|---|---|---|
| `knowledge` | `purpose_hypothesis`, `purpose_formula` | `""` | ☆ |
| `feedback` | `purpose_hypothesis` | `""` (outer_loop>1 일 때 채워짐) | 부분 ★ |
| `formula_memory` | `purpose_formula` | `None` | ☆ |
| `hypothesis_memory` | `purpose_hypothesis` | outer_loop>1 시 자동 | 부분 ★ |

→ ☆ **`knowledge` 와 `formula_memory` 는 코드 path 살아있는데 미사용**. Alpha101, past best formulas 주입 가능한 미사용 lever.

---

# 3. Stage 2 — Distribution validation

Stage 2 의 작동: formula value → quantile partition → 각 quantile 에서 raw OHLCV 통계 → LLM 이 PASS/FAIL 판정

## 3.1 env-overridable

| param | env | default | 효과 |
|---|---|---|---|
| `FAVOR_LLM_MODEL` | ✓ | gpt-4o | step 3 LLM judgment backbone |

→ Stage 2 만의 env 변수는 **0 개**.

## 3.2 `Stage2Config` (config.py)

| param | default | 상태 | ★/☆ |
|---|---|---|---|
| `n_quantiles` | 5 | ✓ ACTIVE | ☆ |
| `monotonicity_threshold` | 0.8 | ⚠ dead (Legacy, 미사용) | n/a |

### `n_quantiles` 영향

- 5 (현재): 5 분위 분할 — formula 의 monotonicity 검사
- 10: 더 정교, sample 50+ 필요
- 3: 단순 high/mid/low, PASS 쉬워짐

## 3.3 hardcoded in agent (`agent/validation_agent.py`)

| 위치 | param | default | ★/☆ |
|---|---|---:|---|
| `:754` | LLM judge temperature | **0.1** (매우 낮음, deterministic 의도) | ☆ |
| `:386` | `MIN_BINS_FOR_VALIDATION` | 3 | ☆ |
| `:71-77` | `RAW_ELEMENTS` (MAG, DIR, VOL, POS) | 4 element | ☆ |

### `RAW_ELEMENTS` (LLM 에 보여주는 통계 vocabulary)

```python
"MAG": "H - L (range)"
"DIR": "C - O (direction)"
"VOL": "V (volume)"
"POS": "(C - L) / (H - L) (close position)"
```

→ ☆ 추가 가능 element: `AMOUNT = V * C`, `GAP = O[t] - C[t-1]`, `RET = log(C/C[-1])` 등.

### Per-quantile 에서 LLM 에 보여주는 metrics

`_format_distribution_summary` 안:
- mean path
- skewness path
- kurtosis path
- q90 (high tail)

→ ☆ median, IQR, t-stat 등 추가 가능

## 3.4 구조적 선택

| 항목 | 현재 | 대안 | ★/☆ |
|---|---|---|---|
| Pooling 방식 | panel pooling (전 ticker 합쳐) | per-ticker validation + majority vote | ☆ |
| Prompt 톤 | 표준 | 엄격 / 느슨 (PASS 비율 직접 통제) | ☆ |

### 실측 (Phase 4 첫 4 runs Stage 2 통과율)

| run | pass_rate (iter 1~3) |
|---|---:|
| M01_paper, M02_uptrend, M03_panic, M10_paper_h20 | **모두 1.0** |

→ Stage 2 현재 **너무 관대 (filter 역할 거의 없음)**. ☆ 엄격하게 만들 lever 다수 미사용.

---

# 4. Stage 3 — Hypothesis instance validation

Stage 3 의 작동: Stage 2 PASS formula 들의 Cartesian combo → strictness grid 별 hypothesis instance 평가 → monotonicity 검사 → 2-tier filter

## 4.1 env-overridable

| param | env | default | ★/☆ |
|---|---|---|---|
| `FAVOR_COMBO_PASS_RATE` | ✓ | 0.5 | ★ {0.4, 0.5, 0.6} (Phase 0 의 B07, B08) |

→ Stage 3 의 nontrivial env 는 이거 **1 개 뿐**.

## 4.2 `Stage3Config` (config.py)

| param | default | 상태 | ★/☆ |
|---|---|---|---|
| `horizon_days` | 5 | ✓ ACTIVE (fallback) | 부분 ★ (LLM hypothesis 가 결정) |
| `monotonicity_threshold` | 0.7 | ✓ ACTIVE | ☆ |
| `strictness_grid` | 5-level dict | ⚠ dead (use_random_grid=True 라 unused) | n/a |
| `use_random_grid` | True | ✓ ACTIVE | ☆ |
| **`random_grid_steps`** | **3** | ✓ ACTIVE → q50/q70/q90 3 level | ☆ |
| `combination_pass_rate_threshold` | 0.5 | ✓ ACTIVE (env 로 override) | ★ (env) |
| `combination_s2_improvement_threshold` | 0.01 | ⚠ DEPRECATED (주석 명시) | n/a |
| `n_processes` | 8 | 속도만 영향 | n/a |

### `monotonicity_threshold` 영향

- 0.7 (현재): 인접 strictness level 비교 시 70 % 이상이 같은 방향이면 monotonic 판정
- 0.5: 통과 combo ↑
- 0.9: 엄격

### `random_grid_steps = 3` (특수 케이스)

- n_steps=3: thresholds = [0.5, 0.7, 0.9] (q50/q70/q90)
- n_steps=5: top10/30/50/70/90 percentile (linspace)
- n_steps=10: 더 dense

→ ☆ 한 번도 안 바꿈. 5 또는 10 으로 늘리면 monotonicity 판정 더 정교.

## 4.3 hardcoded in agent (`agent/hypothesis_validation_agent.py`)

| 위치 | param | default | ★/☆ |
|---|---|---:|---|
| `:76` | `DEFAULT_STRICTNESS_GRID` (class const) | 5 level | n/a (unused) |
| `:471` | outcome variable = `forward_return = close.shift(-h) / close - 1` | 고정 | ☆ |
| `:657-665` | monotonicity metrics | 4 종 (precision, mean_return, sharpe, s2_ratio) | ☆ |
| `:680~` | adjacent-level comparison | Spearman 안 씀 | ☆ |

### `_aggregate_stage3_ticker_results` (stage3.py:267)

```python
if pass_rate >= 0.7:  verdict="PASS", generalizability="HIGH"
elif pass_rate >= 0.5: verdict="PASS", generalizability="MEDIUM"
else:                  verdict="FAIL"
```

→ ☆ hardcoded 0.7 / 0.5. `combination_pass_rate_threshold` 와 다른 layer 의 cutoff.

## 4.4 ☆ 안 흔든 dimension

- `monotonicity_threshold` (0.7 → {0.5, 0.9})
- `random_grid_steps` (3 → {5, 10})
- outcome variable (forward return → log return, vol-adjusted)
- monotonicity metrics (4 → +MDD, +win_rate)
- aggregated verdict thresholds (0.7 / 0.5)

---

# 5. Stage 4 — Optuna threshold + qlib backtest

Stage 4 의 작동: Stage 3 PASS combo → Optuna TPE 로 per-formula threshold 탐색 (IS excess IR maximize) → OOS qlib backtest

## 5.1 env-overridable (Phase 0~4 에서 가장 많이 흔든 stage)

| param | env | default | paper Table 1 | 우리가 흔든 값 |
|---|---|---|---|---|
| `FAVOR_HORIZON_DAYS` | ✓ | 5 | 5 | ★ {3, 5, 10, 20} |
| `FAVOR_STOP_LOSS_THRESHOLD` | ✓ | −0.05 | −0.10 | ★ {−0.05, −0.10, None} |
| `FAVOR_ENTRY_CONFIRM_RULE` | ✓ | "none" | "none" | ★ {none, up_day_and_close_pos} |
| `FAVOR_NATIVE_STRATEGY` | ✓ | "trigger_exit" | trigger_exit | ★ {trigger_exit, topk_dropout} |
| `FAVOR_THRESHOLD_MIN` | ✓ | 0.55 | 0.55 | ★ {0.55, 0.7} |
| `FAVOR_THRESHOLD_MAX` | ✓ | 0.95 | 0.95 | ★ {0.95} |
| **`STAGE4_N_TRIALS`** | ✓ | 20 (Phase 4 default) | **20** | ★ {3, 20, 50, 100} |

## 5.2 `Stage4Config` (config.py, env 없음)

### Optuna 관련

| param | default | ★/☆ |
|---|---|---|
| `enable_optuna` | True | ☆ |
| `fixed_quantiles` | [0.9] | ☆ |
| `threshold_step` | **0.05** (config 에 없음, fallback) | ☆ |

### Combo selection

| param | default | ★/☆ |
|---|---|---|
| `max_combinations_to_evaluate` | −1 (all) | ☆ |
| `combination_selection_criterion` | "s2_improvement" | ☆ |

### Position 관리

| param | default | ★/☆ |
|---|---|---|
| `lookback_window` | 20 | ☆ |
| `entry_confirm_lag_days` | 0 | ☆ |
| `entry_close_pos_min` | 0.7 | ☆ |
| `entry_daily_return_min` | 0.0 | ☆ |

### Trigger-exit 세부

| param | default | ★/☆ |
|---|---|---|
| `ref_price_fn` | "max_high" | ☆ |
| `trigger_price_field` | "high" | ☆ |
| `trigger_op` | "gte" | ☆ |
| `trigger_kmin` | 1 | ☆ |
| `trigger_kmax` | None | ☆ |

### TopkDropout 세부 (native_strategy="topk_dropout" 일 때만)

| param | default | ★/☆ |
|---|---|---|
| `topk` | 50 | ☆ |
| `n_drop` | 5 | ☆ |
| `hold_thresh` | 1 | ☆ |

### 기타

| param | default | ★/☆ |
|---|---|---|
| `n_processes` | 8 | 속도만 |
| `optuna_log_every` | 0 | log 만 |
| `combined_signal_q` | 0.9 | ☆ (combined strategy 만) |

## 5.3 hardcoded in code (`run/pipeline/stage4*.py`)

| 위치 | param | default | ★/☆ |
|---|---|---|---|
| `stage4_parallel_per_combo.py:110` | Optuna `TPESampler(seed=42 + combo_idx)` | 고정 | ☆ |
| `stage4_parallel_per_combo.py:109` | Optuna `direction="maximize"` | maximize | n/a |
| `stage4.py:1784` | objective return value | **excess IR vs benchmark (IS)** | ☆ ← **핵심 lever** |
| `_simulate_positions_fast` | stop_loss + time_stop 동시 적용 | 변경 불가 (priority 없음) | n/a |

### ☆ 가장 큰 미사용 lever: **Optuna objective = IR only**

현재 IR 만 maximize → OOS MDD 통제 불가 → publishing gate 못 넘는 직접 원인.

대안:

1. **Multi-objective**: `directions=["maximize", "maximize"]` 로 (IR, −MDD) 동시 optimize. Pareto front 탐색.
2. **Penalty**: `return ir - lambda * abs(mdd)` (lambda 튜닝 필요)
3. **Calmar 대체**: `return annualized_return / max_drawdown` (단일 objective)
4. **Sortino / Sharpe**: 다운사이드 vol 만 penalize

## 5.4 Qlib 거래 비용 (`QlibConfig`, config.py)

| param | CN default | US default | paper | ★/☆ |
|---|---|---|---|---|
| `open_cost` | 0.0005 (5 bp) | 0.0 | 0.0005 (추정) | ☆ |
| `close_cost` | 0.0015 (15 bp) | 0.0005 | 0.0015 (추정) | ☆ |
| `min_cost` | 5.0 (CNY) | 0 | 5.0 (추정) | ☆ |
| `init_cash` | 1e8 | 1e8 | 1e8 | n/a |
| `limit_threshold` | 0.095 | None | 0.095 | ☆ |
| `deal_price` | "open" (T+1) | "open" | "open" | ☆ |

→ ☆ 거래비용 0 으로 설정 시 IR 평균 ↑↑ (단 paper 와 비교 불공정).

---

# 6. Phase 0~4 의 우리 sweep history

| Phase | 날짜 | model | split | n_trials | ol | 흔든 dim | 결과 |
|---|---|---|---|---:|---:|---|---|
| 0 | 5/10 | mini | S5 | 50 | 1 | concept × hp lever (23 jobs) | honest IR 4건 paper 능가, MDD 0건 |
| 1 | b200 | gpt-4o | 6 splits | (config) | 3 | split (S1~S6) × 4 setting | oracle 9건, honest 1건 paper-aligned 능가 |
| 2 | 5/11 | mini | S1, S2, S5 | 50 | 3 | 4 setting × 3 split | honest 0/12 |
| 3 | 5/12 | gpt-4o | S1 | 50 | 3 | backbone 비교 (4 setting) | honest 0/4 |
| **4** | 5/12 | **mini** | **S1** | **20** | 3 | concept × hp (20 jobs) | honest IR 0/20, honest MDD 0/20 (closest: −0.227) |

## 누적: 어떤 dimension 까지 손댔는가

| dimension | 흔든 횟수 | 사용 값 |
|---|---:|---|
| split | 6 종 | S1, S2, S3, S4, S5, S6 |
| model | 2 종 | gpt-4o, gpt-5.4-mini |
| outer_loop | 2 종 | 1, 3 |
| n_trials | 4 종 | 3, 20, 50, 100 |
| concept | 5 종 | paper, uptrend, panic, compressed, volcomp |
| horizon_days | 4 종 | 3, 5, 10, 20 |
| stop_loss | 3 종 | −0.05, −0.10, None |
| threshold_min | 2 종 | 0.55, 0.7 |
| threshold_max | 1 종 | 0.95 |
| entry_confirm | 2 종 | none, up_day_and_close_pos |
| native_strategy | 2 종 | trigger_exit, topk_dropout |
| combo_pass_rate | 3 종 | 0.4, 0.5, 0.6 |

---

# 7. ☆ 아직 안 흔든 lever 우선순위 (impact 추정)

| 순위 | lever | stage | 변경 방법 | 예상 impact |
|---|---|---|---|---|
| **1** | **Optuna objective metric (IR → IR+MDD or Calmar)** | 4 | code edit 1 줄 | **MDD 통제, publishing gate 해결 가능성** |
| 2 | LLM temperature (hypothesis 0.9, others 0.7 → 0.3) | 1 | code edit 7 곳 | LLM stochasticity 통제, 단일-seed reproducibility ↑ |
| 3 | `monotonicity_threshold` Stage 3 (0.7 → {0.5, 0.9}) | 3 | config | combo 통과 비율 직접 조절 |
| 4 | `random_grid_steps` (3 → 5 또는 10) | 3 | config | strictness ladder 정교화, monotonicity 판정 신뢰도 |
| 5 | `n_quantiles` Stage 2 (5 → {3, 10}) | 2 | config | Stage 2 filter 엄격 / 느슨 |
| 6 | Stage 2 LLM judge temperature (0.1 → 0.0 or 0.3) | 2 | code edit | PASS/FAIL deterministic vs diverse |
| 7 | `max_formulas_per_observation` (3 → {2, 4}) | 1 | code edit | combo 수 직접 통제 |
| 8 | knowledge 주입 (Alpha101 등) | 1 | caller edit | formula 다양성 + 품질 |
| 9 | `threshold_step` (0.05 → {0.025, 0.1}) | 4 | config 추가 | Optuna 탐색 해상도 |
| 10 | Optuna sampler (TPE → CMA-ES / Random) | 4 | code edit | global optimum 발견 능력 |
| 11 | outcome variable (forward return → log/vol-adj) | 3 | code edit | Stage 3 의 metric basis 변경 |
| 12 | RAW_ELEMENTS 추가 (MAG/DIR/VOL/POS → +AMOUNT, GAP) | 2 | code edit + prompt | LLM judge 의 vocabulary 확장 |
| 13 | Pooling vs per-ticker validation | 2 | 구조 변경 (큰 수정) | distribution 통계 의 ticker-aware 화 |
| 14 | 거래비용 (5/15 bp → 0) | 4 | config | IR 평균 boost, 비교 불공정 |
| 15 | OHLCV columns (5 → +tradingvalue/marketcap) | 1 | config | formula vocabulary 확장 |

---

# 8. 결정 가이드 (어디부터 흔들지)

## A. MDD 가 publishing gate 인 현재 상황

→ **1순위: Stage 4 objective metric 변경 (IR + MDD penalty / Calmar)**. 단일 변경으로 가장 큰 impact 기대.

## B. LLM stochasticity 가 single-seed reproducibility 의 원인이면

→ **2순위: Stage 1 temperature 전체 0.3 으로 통일**. Phase 4 의 결과 노이즈 줄어들지 검증.

## C. Stage 2~3 의 filter 가 너무 관대 (현재 pass_rate 1.0)

→ **3, 5순위: monotonicity_threshold 0.9, n_quantiles 10** 으로 strict 화. Stage 4 로 가는 combo 수 줄어 over-selection bias ↓.

## D. 더 작은 (cheaper) test 면

→ **2~3 단계의 짧은 sweep**: 1 setting × 3 temperature × 3 n_quantiles = 9 jobs. mini × n_trials=20 으로 1~2 시간 + $0.5 이내.

---

# 9. paper Table 1 의 hyperparameter (확인된 / 추정)

| param | paper 본문 | 실제 frozen run (`01_15_new_qlib`) | 우리 sweep 의 정합 여부 |
|---|---|---|---|
| model | gpt-4o | gpt-4o (126 runs 확인) | ✓ Phase 1, 3 |
| n_trials | 50 (§3) | **20** (실제 frozen) | ✓ Phase 4 (20) |
| horizon_days | 5 | 5 | ✓ |
| stop_loss | −0.10 | −0.10 | ✓ |
| outer_loop_max | 5 (§4.4) | **1** (frozen 의 outer_loop_used=False) | △ Phase 0 만 ol=1 |
| combination_pass_rate | 명시 안 됨 | 0.5 | ✓ |
| threshold range | [0.55, 0.95] | [0.55, 0.95] | ✓ |
| MDD baseline bar | −0.2224 | −0.4181 (Alpha158 실측) | n/a |

→ paper 본문 vs frozen run 의 **7 건 mismatch** 중 outer_loop_max (5 vs 1), n_trials (50 vs 20), MDD baseline 가 결과에 직접 영향.

---

작성: 2026-05-12 / Phase 4 완료 직후
