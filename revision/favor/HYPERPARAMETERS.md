# FaVOR — Hyperparameters & Run Layout (revision/favor)

이 문서는 `revision/revision/favor/` 모듈에서 **건드릴 수 있는 모든 knob**과 **결과 산출 경로**를 한곳에 모은 참조 문서다. config 정의는 `run/config.py`. 마지막 업데이트: 2026-05-10.

---

## 0. Quick reference

- **CLI 진입점**: `python run_pipeline_parallel_per_combo_parallel.py "<concept>" --combo-workers N --optuna-jobs M --outer-loop K`
- **Wrapper**: `repro_logs/launch_run.sh <run_label>` (concept을 첫 위치 인자로 강제하는 안전 wrapper)
- **Conda env**: `quant` (polars + qlib + optuna + openai 포함)
- **API key**: `revision/revision/favor/.env` 의 `OPENAI_API_KEY`
- **데이터**: `~/.qlib_full/qlib_data/cn_data` (zip에서 푼 풀 커버리지 2005-2026), `FAVOR_QLIB_PROVIDER_URI_CN` env로 라우팅
- **결과**: `revision/revision/favor/runs/<YYYYMMDD_HHMMSS>/` (per-run dir, 절대 덮어쓰지 않음)

---

## 1. 기간 — `DataSplitConfig` (`config.py:195`)

| field | 기본값 | 의미 | override |
|---|---|---|---|
| `train_start` / `train_end` | `2015-01-01` ~ `2019-12-31` | Stage 2/3 formula·combo 검증, Optuna IS | config.py |
| `val_start` / `val_end` | `2020-01-01` ~ `2020-12-31` | Optuna validation 명목, 현재는 사실상 IS와 합쳐 사용 | config.py |
| `test_start` / `test_end` | `2021-01-01` ~ `2025-12-31` | OOS 최종 평가 | config.py |

> Note: `data_split` 객체는 매 run의 `runs/<ts>/run_config.json`에 그대로 박혀 reproducibility 용도로 보존됨.

---

## 2. LLM — `LLMConfig` (`config.py:22`)

| field | 기본값 | 의미 | override |
|---|---|---|---|
| `model_name` | `gpt-4o` | 가설/관측/formula/검증 LLM | config.py / `OPENAI_BASE_URL` proxy |
| `temperature` | `0.7` | 다양성 (낮추면 결정적, 높이면 다양) | config.py |
| `max_tokens` | `2048` | 응답 길이 한도 | config.py |

**대체 모델**: `gpt-4o-mini`, `gpt-4-turbo`, `o1-*`, `claude-sonnet-4-6`/`claude-opus-4-7`, `gemini-2.5-pro` 등 — LiteLLM proxy via `OPENAI_BASE_URL`.
**관련 paper artifact**: Table 4 (LLM backbone ablation).

---

## 3. 시장·거래 — `QlibConfig` (`config.py:28`)

| field | CN 기본값 | US 기본값 | 의미 | override |
|---|---|---|---|---|
| `qlib_market` | `csi500` | `sp500` | 종목 풀 (csi500/csi300/all/sp500) | `MARKET=cn`/`us` |
| `provider_uri` | `~/.qlib/qlib_data/cn_data` | `~/.qlib/sh_sp500_qlib` | qlib 데이터 dir | `FAVOR_QLIB_PROVIDER_URI_{CN,US}` env (revision에 추가됨) |
| `region` | `cn` | `us` | qlib region | `MARKET` |
| `open_cost` | `0.0005` | `0` | 매수 수수료 | config.py |
| `close_cost` | `0.0015` | `0.0005` | 매도 수수료 | config.py |
| `min_cost` | `5.0` | `0.0` | 최소 거래비용 | config.py |
| `init_cash` | `1e8` | `1e8` | 초기 자본 | config.py |
| `limit_threshold` | `0.095` | `None` | 일일 가격 limit | config.py |
| `deal_price` | `open` | `open` | 체결 가격 hint (T+1 open) | config.py |
| `benchmark` | `SH000905` (CSI500) | `^GSPC` | 비교 벤치마크 | config.py |

---

## 4. 데이터 컬럼 매핑 — `DataConfig` (`config.py:52`)

기본은 영문 OHLCV. 보통 안 건드림.

| field | 기본값 | 의미 |
|---|---|---|
| `date_col` | `timestamp` | 날짜 컬럼 |
| `asset_col` | `ticker` | 종목 컬럼 |
| `price_col` | `close` | 종가 |
| `open_col` / `high_col` / `low_col` / `volume_col` | `open` / `high` / `low` / `volume` | OHLCV |
| `trading_value_col` | `tradingvalue` | 거래대금 |
| `market_cap_col`, `shares_outstanding_col` | placeholder | 펀더멘털 (미사용) |

---

## 5. Stage 1 (formula 생성) — `Stage1Config` (`config.py:73`)

| field | 기본값 | 의미 |
|---|---|---|
| `allowed_ohlcv_columns` | `[open, high, low, close, volume]` | LLM이 사용 가능한 raw column. `tradingvalue` 추가 등 가능 |
| `refine_rounds` | `10` | Stage1 내부 self-correction 최대 라운드 |

---

## 6. Stage 2 (formula validation) — `Stage2Config` (`config.py:78`)

| field | 기본값 | 의미 | OOS 영향 |
|---|---|---|---|
| `n_quantiles` | `5` | formula 값 분위수 분할 수 | 중 |
| `monotonicity_threshold` | `0.8` | quantile bin 단조성 기준. 낮을수록 PASS↑ | 중 |

---

## 7. Stage 3 (combination validation) — `Stage3Config` (`config.py:83`)

| field | 기본값 | 의미 | OOS 영향 |
|---|---|---|---|
| `horizon_days` | `5` | 가설이 horizon 미명시 시 fallback | 중 |
| `monotonicity_threshold` | `0.7` | strictness 단조성 기준 | 중 |
| `strictness_grid` | `very_loose=0.1` ~ `very_strict=0.9` | strictness level dict (`use_random_grid=False`일 때만) | 저 |
| `use_random_grid` | `True` | 랜덤 grid 사용 여부 | 저 |
| `random_grid_steps` | `3` | 랜덤 grid 단계 수 | 저 |
| `combination_pass_rate_threshold` | `0.5` | combo PASS 받으려면 ticker의 50% 이상 통과 | **고** |
| `combination_s2_improvement_threshold` | `0.01` | (deprecated) ΔS2 magnitude — 거의 미사용 | — |
| `n_processes` | `8` | Stage3 worker 수 | 속도만 |

> `combination_pass_rate_threshold`를 낮추면 약한 combo도 통과 → Stage4 후보 풀↑ → 좋은 combo 발견 확률↑ but 노이즈↑.

---

## 8. Stage 4 (Optuna + backtest) — `Stage4Config` (`config.py:118`)

### Optuna / 평가 모드

| field | 기본값 | 의미 | env override | OOS 영향 |
|---|---|---|---|---|
| `enable_optuna` | `True` | 끄면 fixed_quantiles만 평가 | `STAGE4_ENABLE_OPTUNA` | 고 |
| `n_trials` | `20` | Optuna trial 수 (paper는 50) | `STAGE4_N_TRIALS` | 고 |
| `threshold_min` | `0.55` | per-formula threshold 검색 하한 | config.py | 고 |
| `threshold_max` | `0.95` | 검색 상한 | config.py | 고 |
| `fixed_quantiles` | `[0.9]` | fixed-q 평가 (paper: `[0.8, 0.9]`) | `STAGE4_FIXED_QUANTILES` | 중 |
| `combined_signal_q` | `0.9` | OR-aggregated 단일 strategy의 글로벌 q | config.py | 저 |

### Combo 선택

| field | 기본값 | 의미 | env override |
|---|---|---|---|
| `max_combinations_to_evaluate` | `-1` (전부) | top-N만 평가 | `STAGE4_MAX_COMBINATIONS_TO_EVALUATE` |
| `combination_selection_criterion` | `s2_improvement` | top-N 정렬 기준: `s2_improvement` / `mean_return` / `pass_rate` | config.py |

### 포지션 / 진입 / 청산

| field | 기본값 | 의미 | OOS 영향 |
|---|---|---|---|
| `horizon_days` | `5` | **포지션 보유 기간** (paper Table 1: 5) | **고** |
| `lookback_window` | `20` | trigger/entry 기준 lookback | 중 |
| `stop_loss_threshold` | `-0.05` | 누적 수익 stop-loss (**paper: −0.10**). `0.0`=break-even, `None`=disable | **고** |
| `entry_confirm_rule` | `none` | `none` / `close_pos` / `up_day` / `up_day_and_close_pos` | 고 |
| `entry_confirm_lag_days` | `0` | 0=동일일 confirm, 1=다음날 confirm | 중 |
| `entry_close_pos_min` | `0.7` | close가 day-range의 상위 70% 이상 (close_pos 룰일 때) | 저 |
| `entry_daily_return_min` | `0.0` | 일일 수익 양수 요구 (up_day 룰일 때) | 저 |

### Trigger-exit 전략 (`native_strategy = "trigger_exit"`)

| field | 기본값 | 의미 |
|---|---|---|
| `ref_price_fn` | `max_high` | 기준가 함수: `max_high` / `min_low` / `q50_close` |
| `trigger_price_field` | `high` | trigger 비교용 일중 가격 필드 (`high`/`low`/`close`) |
| `trigger_op` | `gte` | 비교 연산 (`gte`/`lte`) |
| `trigger_kmin` | `1` | trigger 체크 시작 day offset |
| `trigger_kmax` | `None` (=horizon_days) | trigger 체크 끝 day offset |

### 백테스트 전략 선택

| field | 기본값 | 의미 |
|---|---|---|
| `native_strategy` | `trigger_exit` | `trigger_exit` (custom) / `topk_dropout` (Qlib 기본) |
| `topk` | `50` | TopkDropout일 때 포트폴리오 크기 |
| `n_drop` | `5` | TopkDropout rebalance마다 drop 수 |
| `hold_thresh` | `1` | 최소 보유 임계 |

### 병렬·로깅

| field | 기본값 | 의미 | env override |
|---|---|---|---|
| `n_processes` | `8` | Stage4 worker 수 (legacy) | — |
| (combo workers) | 1 | combo 단위 병렬 | `STAGE4_COMBO_WORKERS` (CLI: `--combo-workers`) |
| (optuna jobs) | 8 (auto-1 if combo>1) | combo당 Optuna 병렬 | `STAGE4_OPTUNA_N_JOBS` (CLI: `--optuna-jobs`) |
| `optuna_log_every` | `0` (off) | N trial마다 한 줄 로그 | config.py |

> **OOS 성능 직격탄**: `horizon_days`, `stop_loss_threshold`, `n_trials`, `entry_confirm_rule`, `native_strategy`, `threshold_{min,max}`.

---

## 9. Refinement loop — `RefinementConfig` (`config.py:246`)

| field | 기본값 | 의미 |
|---|---|---|
| `enable_inner_loop` | `True` | Stage1 ⇄ Stage2 inner refinement (formula 재생성) |
| `max_inner_iterations` | `3` | inner loop 최대 iter |
| `enable_outer_loop` | `True` | Stage4 → Stage1 outer refinement (가설 자체 refine) |
| `max_outer_iterations` | `5` | outer loop 최대 iter (CLI `--outer-loop N`로 강제) |

> Paper 원본 run은 `max_outer_iterations=5`였으나 outer_iter_1에서 84-combo 가설이 한 번에 통과해 추가 iter 안 함 (`outer_loop_used: false`).

---

## 10. Pipeline control — `PipelineControlConfig` (`config.py:235`)

| field | 기본값 | 의미 |
|---|---|---|
| `enable_stage2` | `True` | False면 모든 formula PASS로 간주 (validation 없이) |
| `enable_stage3` | `True` | False면 모든 combo PASS |

> Paper Fig 5 / Table 3 ablation에 사용. OOS 비교용으로만 끄고 켜기.

---

## 11. Concept 자체 (CLI 첫 인자)

Hyperparameter는 아니지만 **OOS 결과에 가장 큰 영향**.

```bash
python run_pipeline_parallel_per_combo_parallel.py \
  "<자연어 가설 텍스트>" \
  --combo-workers 4 --optuna-jobs 1 --outer-loop 5
```

다양한 종류 (paper의 `run_cn.sh` 주석에서 발췌):
- mean reversion / panic selling
- breakout + pullback (paper Table 1)
- compressed volatility + uptrend
- short-term sell-off rebound
- short-term momentum
- selloff exhaustion + intraday recovery

> ⚠️ `run_cn_limited.sh`의 wrapper는 `--combo-workers` 등을 concept 앞에 붙여서, 스크립트의 `args[0]`이 `--`로 시작 → **concept이 default("Mean Reversion after Panic Selling")로 떨어짐**. **반드시** `repro_logs/launch_run.sh` 또는 `python run_pipeline_parallel_per_combo_parallel.py "<concept>" ...` (concept을 첫 위치 인자로) 형태로 호출.

---

## 12. 환경 변수 — 체크리스트

| env var | 필수? | 의미 |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI 인증 (`.env`로도 OK) |
| `OPENAI_BASE_URL` | 선택 | LiteLLM/proxy 사용 시 |
| `MARKET` | ✅ | `cn` 또는 `us` |
| `FAVOR_QLIB_PROVIDER_URI_CN` | 선택 | CN qlib dir 오버라이드 (revision에서 추가됨) |
| `FAVOR_QLIB_PROVIDER_URI_US` | 선택 | US qlib dir 오버라이드 |
| `STAGE4_ENABLE_OPTUNA` | 선택 | `True`/`False` |
| `STAGE4_N_TRIALS` | 선택 | 정수 |
| `STAGE4_MAX_COMBINATIONS_TO_EVALUATE` | 선택 | 정수 (-1 = 전부) |
| `STAGE4_FIXED_QUANTILES` | 선택 | `0.8,0.9` 콤마 구분 또는 `None` |
| `STAGE4_COMBO_WORKERS` | 선택 | 병렬 combo worker 수 |
| `STAGE4_OPTUNA_N_JOBS` | 선택 | combo당 Optuna n_jobs |
| `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_MAX_THREADS` | 선택 | 라이브러리 스레드 제어 |
| `PYTHONWARNINGS=ignore` | 선택 | warning 노이즈 억제 |

---

## 13. 결과·로그 경로 (어디 뭐가 저장되나)

매 run마다 `revision/revision/favor/runs/<YYYYMMDD_HHMMSS>/` 디렉터리 생성. **타임스탬프는 launch 시각, 따라서 같은 시각에 여러 개 띄우면 stagger(`sleep 2-3`)** 필수.

```
revision/revision/favor/
├── .env                                    # API key (gitignore 관례)
├── run_pipeline.py                         # 단일 프로세스 entry
├── run_pipeline_parallel_per_combo_parallel.py  # 병렬 entry (combo-level)
├── run_cn.sh / run_cn_limited.sh / run_us.sh / run_us_limited.sh
├── HYPERPARAMETERS.md                      # 본 문서
├── favor_모듈_정리.md                      # 모듈 구조 설명
│
├── repro_logs/                             # 사용자가 만든 재현 실험용 작업 dir
│   ├── launch_run.sh                       # concept 안전 wrapper (권장)
│   ├── compare.py                          # 결과 비교 스크립트
│   ├── REPORT.md                           # 직전 분석 보고서
│   ├── run1.log / run2.log / run3.log      # 직전 3-run의 stdout (tee로 캡처)
│   └── ...                                 # 새 run마다 추가됨
│
├── runs/                                   # 모든 pipeline run의 산출물 루트
│   └── <YYYYMMDD_HHMMSS>/                  # 한 run의 root
│       ├── run_config.json                 # ⭐ 이 run의 모든 config 스냅샷
│       │
│       ├── specs/                          # ⭐ Stage별 산출물 JSON (가장 중요)
│       │   ├── hypothesis.json             # Stage1: outer_iter_*별 가설
│       │   ├── observation_plan.json       # Stage1: 관측 분해
│       │   ├── formula_bundle.json         # Stage1: formula 정의
│       │   ├── stage2_summary.json         # Stage2: formula PASS/FAIL
│       │   ├── stage3_result.json          # Stage3: combo PASS/FAIL (집계)
│       │   ├── stage3_ticker_details.json  # Stage3: ticker별 상세
│       │   ├── stage4_summary.json         # ⭐ Stage4: 모든 combo의 IS/OOS 메트릭 (Table 1 source)
│       │   ├── refinement_history.json     # inner loop 흐름
│       │   ├── outer_loop_history.json     # outer loop 흐름
│       │   ├── llm_usage.json              # 모델 호출 횟수·토큰
│       │   └── llm_usage_detailed.json
│       │
│       ├── data/                           # 중간 데이터
│       │   └── price_with_formulas_iter_<N>.parquet  # OHLCV+formula 값 패널
│       │
│       ├── reports/                        # 사람이 읽는 마크다운 리포트
│       │   ├── stage2.md
│       │   ├── stage3.md
│       │   └── stage4.md                   # 가장 자주 열어볼 파일
│       │
│       ├── logs/                           # ⭐ 실제 로그
│       │   ├── run.log                     # ⭐ 메인 stdout 로그 (tee 안 거친 깨끗한 버전)
│       │   └── agents/                     # ⭐ LLM 호출 trace (디버깅용)
│       │       ├── llm_calls.jsonl                                # 모든 LLM 호출 누적 (한 줄=한 호출)
│       │       ├── react_hypothesis_agent_<HHMMSS>.json           # 가설 생성 trace
│       │       ├── react_observation_agent_<HHMMSS>.json          # 관측 분해 trace
│       │       ├── react_formula_agent_generation_<HHMMSS>.json   # formula 생성 trace
│       │       ├── react_formula_agent_self_correction_<HHMMSS>.json
│       │       ├── react_validation_agent_<HHMMSS>.json
│       │       └── react_validation_agent_<HHMMSS>_<id>_<formula>.json  # ticker·formula별
│       │
│       ├── agents/                         # (logs/agents의 alias 또는 sub-dir, 비어있을 수도)
│       │
│       └── qlib_artifacts/                 # ⭐ Stage4 백테스트 raw 산출물 (paper Fig 4 source)
│           └── iter_<N>/                   # outer iter별
│               └── combo_<idx>/            # combo별
│                   ├── oos/                # Out-of-sample 결과
│                   │   ├── report_normal_1day.pkl    # ⭐ 일일 수익률 시계열 (DataFrame)
│                   │   ├── port_analysis_1day.pkl    # qlib risk_analysis 결과
│                   │   ├── positions_normal_1day.pkl # 일별 포지션
│                   │   └── trades.pkl                # 거래 내역
│                   ├── is/                 # In-sample (동일 구조)
│                   ├── fixed_q80/{is,oos}/ # fixed-q 평가 결과
│                   └── fixed_q90/{is,oos}/
```

### 어디부터 보면 되나 — 분석 우선순위

| 보고 싶은 것 | 파일 |
|---|---|
| 이 run의 설정 전체 | `runs/<ts>/run_config.json` |
| 이 run에서 LLM이 만든 가설/formula | `runs/<ts>/specs/{hypothesis,formula_bundle}.json` |
| 이 run의 모든 combo 메트릭 (Table 1 source) | `runs/<ts>/specs/stage4_summary.json` |
| 이 run의 사람이 읽는 요약 | `runs/<ts>/reports/stage4.md` |
| 백테스트 daily curve (Figure 4용) | `runs/<ts>/qlib_artifacts/iter_*/combo_*/oos/report_normal_1day.pkl` |
| LLM이 무슨 prompt 받고 무슨 답 했나 (디버깅) | `runs/<ts>/logs/agents/react_*_*.json` 또는 `llm_calls.jsonl` |
| 메인 진행 로그 (실시간) | `runs/<ts>/logs/run.log` 또는 `repro_logs/<label>.log` (launch_run.sh 사용 시) |
| 실패 분석 (Stage2/3 어디서 막혔나) | `runs/<ts>/specs/{stage2_summary,stage3_result,refinement_history}.json` |
| 직전 3-run 종합 비교 | `repro_logs/compare.py` 실행 |

### CLI 호출 → 로그 매핑 요약

```bash
./repro_logs/launch_run.sh myrun
# →
#   stdout/stderr (tee):  repro_logs/myrun.log   ← 사용자가 tail로 모니터링
#   pipeline 내부 로그:   runs/<ts>/logs/run.log ← 더 깨끗한 버전
#   per-LLM trace:        runs/<ts>/logs/agents/*.json
#   결과 JSON:            runs/<ts>/specs/*.json
#   백테스트 산출물:      runs/<ts>/qlib_artifacts/iter_*/combo_*/{is,oos,fixed_q*}/
```

---

## 14. 자주 쓰는 패턴 (실험 시 cheat sheet)

### A. 1회 빠른 smoke test (최소 비용 검증)
```bash
STAGE4_N_TRIALS=5 STAGE4_MAX_COMBINATIONS_TO_EVALUATE=10 \
  ./repro_logs/launch_run.sh smoke
```

### B. Paper 원본 설정 그대로 (CN, full)
```bash
STAGE4_N_TRIALS=50 \
  ./repro_logs/launch_run.sh paper_repro
# 단, launch_run.sh의 stop_loss_threshold는 -0.05 (paper는 -0.10) → config.py 수정 필요 시 추가 편집
```

### C. 다양한 concept 풀 만들기 (best-of-N 탐색)
```bash
for c in concept1 concept2 concept3; do
  RUN_LABEL="${c// /_}"
  # repro_logs/launch_run.sh를 concept별로 변형해 호출
done
```

### D. n_trials sweep (Optuna 깊이별 비교)
```bash
for n in 20 50 100 200; do
  STAGE4_N_TRIALS=$n ./repro_logs/launch_run.sh trials_${n}
done
```

### E. horizon sweep (Stage4Config.horizon_days)
> env var 없음 → `run/config.py`의 `horizon_days` 직접 편집해야 함. (필요하면 env hook 하나 더 만들어 줄 수 있음)

---

## 15. 변경 이력 / 추가된 hook

revision에서 추가된 환경변수 hook (모두 additive — 미설정 시 기본값 유지):

| env var | 매핑 대상 | 비고 |
|---|---|---|
| `FAVOR_QLIB_PROVIDER_URI_CN` | `qlib.provider_uri` (CN 모드) | `~/.qlib_full/qlib_data/cn_data` 같은 대체 경로 |
| `FAVOR_QLIB_PROVIDER_URI_US` | `qlib.provider_uri` (US 모드) | |
| `FAVOR_LLM_MODEL` | `llm.model_name` | 예: `gpt-5.4-mini`, `gpt-4o` |
| `FAVOR_LLM_TEMPERATURE` | `llm.temperature` | float |
| `FAVOR_HORIZON_DAYS` | `stage4.horizon_days` | int |
| `FAVOR_STOP_LOSS_THRESHOLD` | `stage4.stop_loss_threshold` | float; 또는 `none`/`null`/`disable` → Python `None` |
| `FAVOR_ENTRY_CONFIRM_RULE` | `stage4.entry_confirm_rule` | `none` / `close_pos` / `up_day` / `up_day_and_close_pos` |
| `FAVOR_NATIVE_STRATEGY` | `stage4.native_strategy` | `trigger_exit` / `topk_dropout` |
| `FAVOR_THRESHOLD_MIN` | `stage4.threshold_min` | float |
| `FAVOR_THRESHOLD_MAX` | `stage4.threshold_max` | float |
| `FAVOR_COMBO_PASS_RATE` | `stage3.combination_pass_rate_threshold` | float |

추가하면 좋을 후보:
- `FAVOR_HYPOTHESIS_JSON=path` — frozen hypothesis.json을 Stage1에 강제 주입 (재현성↑, 미구현)
