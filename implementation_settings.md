# FaVOR — Implementation Settings & Code Locations

> **Source**: KDD 2026 제출 시점 실험 (paper `../FaVOR_paper.pdf`)
> **조사일**: 2026-04-28
> **본 문서 목적**: revision 작업 시 "원본 실험이 어디서 어떻게 돌아갔는지" 빠르게 참조하기 위함. **모든 경로/설정은 read-only 참조용**이며, CLAUDE.md 규칙에 따라 절대 수정·이동·삭제하지 않는다.

---

## 0. 한 줄 요약

논문의 모든 FaVOR 실험은 **`/home/dgu/fin/01_15_new_qlib/`** 에서 돌아갔다.
각 실험 실행은 `runs/{YYYYMMDD_HHMMSS}/` 디렉토리에 통째로 기록되며, `run_config.json` 한 파일이 그 실행의 완전한 재현 스펙이다.

---

## 1. 메인 코드 베이스 (FaVOR pipeline)

**Root**: `/home/dgu/fin/01_15_new_qlib/`

### 1.1 진입점 (entry points)

| 파일 | 역할 |
|---|---|
| `run_cn.sh` / `run_us.sh` | 단일 실행 launcher (`MARKET=cn` or `us` 환경변수 + `run_pipeline.py` 호출) |
| `run_cn_parallel.sh` / `run_us_parallel.sh` | Stage 4 조합단위 병렬 launcher (`run_pipeline_parallel_per_combo_parallel.py`) |
| `run_cn_parallel_all*.sh` | 전 조합 평가 모드 |
| `run_cn_new.sh`, `run_cn_new_all.sh` | "_new" 계열 (refactored) launcher |
| `run_pipeline.py` | 단일 pipeline runner. CLI: `python run_pipeline.py "<concept>" [--outer-loop N]` |
| `run_pipeline_parallel*.py` | 병렬 변형 |
| `run_pipeline_topkdropout_*.py` | 베이스라인 (TopkDropout + LightGBM/MLP/GAT) |
| `run_stage4_suite.sh/.py` | Stage 4 단독 sweep |
| `run_weighted_backtest.sh/.py` | 가중 백테스트 변형 |
| `generate_us_factor_files.py` | US 데이터셋 factor 생성 헬퍼 |

> **주된 호출 예** (논문 메인 결과 재현):
> ```
> ./run_cn.sh "Short-term mean reversion after panic selling" --outer-loop 5
> ./run_us.sh "<hypothesis>" --outer-loop 5
> ```

### 1.2 파이프라인 모듈 (`run/pipeline/`)

| 파일 | Stage |
|---|---|
| `stage1.py` | Stage 1: Hypothesis Decomposition (Hypothesis Agent → Observation Agent → Factor Agent) |
| `stage2.py` | Stage 2: Factor-Level Validation (Validation Agent + 5-quantile evidence) |
| `stage3.py`, `stage3_new.py` | Stage 3: Factor Integration (Monotonic Strictness 검증) |
| `stage4.py`, `stage4_new.py`, `stage4_new_all.py` | Stage 4: Optimization & Backtesting (Qlib 사용) |
| `stage4_for_ticker.py`, `stage4_parallel*.py` | Stage 4 병렬 변형 |
| `refinement_2to1.py` | Stage 2 → Stage 1 inner loop (formula 재생성) |
| `refinement_4to1.py` | Stage 4 → Stage 1 outer loop (hypothesis 재생성, Figure 5 의 5 round 그대로) |
| `strategy.py` | Qlib `TriggerExitStrategy` 등 백테스트 전략 |

### 1.3 Agent (`agent/`)

| 파일 | 역할 |
|---|---|
| `hypothesis_agent.py` (206 lines) | $\mathcal{A}_H$ — 가설 생성 |
| `observation_agent.py` (100 lines) | $\mathcal{A}_O$ — 가설 → observable conditions 분해 |
| `formula_agent.py` (930 lines) | $\mathcal{A}_F$ — observation → 후보 factor 수식 생성 |
| `validation_agent.py` (1022 lines) | $\mathcal{A}_V$ — Stage 2 distributional validation. **본 논문의 핵심 reasoning agent** |
| `hypothesis_validation_agent.py` (974 lines) | hypothesis 자체에 대한 sanity check |
| `factor_coder_code_agent.py`, `costeer_full_code_agent.py` | factor 수식의 실행 가능 코드화 (CoSTEER 기반) |
| `diagnostics_agent.py`, `diagnostics_tools.py` | 실패 원인 진단 |
| `coder_code_agent.py`, `base_agent.py` | 공통 베이스 |

### 1.4 Prompt (`prompts/`)

| 파일 | 라인 수 | 매핑 stage |
|---|---|---|
| `hypothesis_agent_prompts.py` | 293 | Stage 1 — hypothesis generation, regen |
| `observation_agent_prompts.py` | 143 | Stage 1 — observation decomposition |
| `formula_agent_prompts.py` | 343 | Stage 1 — factor formula generation |
| `validation_agent_prompts.py` | 107 | **Stage 2 — distributional judgment (PASS/FAIL 결정의 핵심)** |
| `hypothesis_validation_prompts.py` | 96 | hypothesis validity check |
| `diagnostics_agent_prompts.py` | 120 | failure diagnosis |
| `raw_idea_prompts.py` | 11 | seed concept |

> **`validation_agent_prompts.py` 의 핵심 시스템 프롬프트** (Table 2 case study 의 reasoning 을 만들어내는 곳):
> - 사용 통계: `DIR = Close - Open`, `MAG = High - Low`, `POS = (Close - Low)/(High - Low)`, `VOL`
> - PASS 조건 4개 (rules A~D): location, tail, multi-stat consistency, no contradiction
> - 출력은 강제 tool call (`distribution_judgment_tool`)
> - paper §3.3.2 의 (i)~(iv) Validation Criteria 가 그대로 박혀 있음

### 1.5 핵심 설정 (`run/config.py`)

**기본 LLM 설정** (`LLMConfig`):
| 필드 | 값 |
|---|---|
| `model_name` | `"gpt-4o"` |
| `temperature` | `0.7` *(다양성을 위해 0.3 → 0.7로 올림)* |
| `max_tokens` | `2048` |

**시장별 자동 분기** (`load_rd_config()`, `MARKET` 환경변수):

| 항목 | CN (CSI500) | US (S&P500) |
|---|---|---|
| `qlib_market` | `"csi500"` | `"sp500"` |
| `region` | `"cn"` | `"us"` |
| `provider_uri` | `~/.qlib/qlib_data/cn_data` | `~/.qlib/sh_sp500_qlib` |
| `open_cost` (buy fee) | `0.0005` | `0` |
| `close_cost` (sell fee) | `0.0015` | `0.0005` |
| `min_cost` | `5.0` | `0.0` |
| `limit_threshold` | `0.095` | `None` |
| `benchmark` | `SH000905` | `^GSPC` |

→ 논문 §4.1 의 거래비용 (CSI: buy 0.05% / sell 0.15%, SP: sell 0.05%) 과 일치.

**Data split** (`DataSplitConfig`) — 모든 실험 공통:
- Train: **2015-01-01 ~ 2019-12-31** (Stage 2/3 factor validation)
- Validation: **2020-01-01 ~ 2020-12-31** (Optuna threshold 최적화)
- Test: **2021-01-01 ~ 2025-12-31** (최종 OOS, Table 1 대상)

**Stage 별 hyperparameter** (`run/config.py` 기본값):

| Stage | 파라미터 | 기본값 | 의미 |
|---|---|---|---|
| 1 | `allowed_ohlcv_columns` | `[open, high, low, close, volume]` | 외부 정보 차단 |
| 1 | `refine_rounds` | `10` | formula 자가수정 최대 횟수 |
| 2 | `n_quantiles` | `5` | 5-bin partition (논문 §3.3.1) |
| 2 | `monotonicity_threshold` | `0.8` | bin-wise mean monotonicity 합격선 |
| 3 | `horizon_days` | `5` | hypothesis 미지정 시 기본 forward window |
| 3 | `monotonicity_threshold` | `0.7` | strictness-level monotonicity (논문에서 보고된 값) |
| 3 | `strictness_grid` | `{0.1, 0.3, 0.5, 0.7, 0.9}` | σ 계단 |
| 3 | `use_random_grid` | `True` | progressive random grid 모드 |
| 3 | `random_grid_steps` | `3` | random grid 생성 수 |
| 3 | `combination_pass_rate_threshold` | `0.5` | 50% ticker 통과 = PASS |
| 3 | `combination_s2_improvement_threshold` | `0.01` | (deprecated, backward-compat) |
| 3 | `n_processes` | `8` | 병렬 워커 수 |
| 4 | `enable_optuna` | `True` | threshold 최적화 |
| 4 | `n_trials` | `20` | Optuna trials |
| 4 | `threshold_min`, `threshold_max` | `0.55`, `0.95` | Optuna 탐색 영역 |
| 4 | `combined_signal_q` | `0.9` | 최종 signal quantile (논문) |
| 4 | `horizon_days` | `5` | forward holding window |
| 4 | `lookback_window` | `20` | rolling window |
| 4 | `stop_loss_threshold` | `-0.05` | **5% 손절** *(rebuttal 의 −10% 와 불일치 — §6 Discrepancy 참조)* |
| 4 | `trigger_kmin`, `trigger_kmax` | `1`, `5` | 진입 trigger 검사 day offset (논문 k=1..5) |
| 4 | `ref_price_fn` | `"max_high"` | trigger 기준가격 함수 |
| 4 | `trigger_price_field` | `"high"` | 비교 가격 필드 |
| 4 | `trigger_op` | `"gte"` | 비교 연산자 |
| 4 | `native_strategy` | `"trigger_exit"` | Qlib 전략 |
| 4 | `topk`, `n_drop`, `hold_thresh` | `50`, `5`, `1` | TopkDropout 베이스라인용 |
| Refinement | `enable_inner_loop`, `max_inner_iterations` | `True`, `3` | Stage1 ⇄ Stage2 재생성 |
| Refinement | `enable_outer_loop`, `max_outer_iterations` | `True`, `5` | **Stage4 → Stage1 재생성. Figure 5 의 5 round** |

---

## 2. 실험 결과 디렉토리 (`runs/`)

**경로**: `/home/dgu/fin/01_15_new_qlib/runs/`
**개수**: 30+ 개 (가장 이른 실행: `20260118_143229`, 가장 최근 — 페이퍼 백테스트 직전: `20260209_073324`)

### 2.1 한 run 디렉토리 구조 (예: `20260209_073324/`)

```
runs/{run_id}/
├── run_config.json              # 이 run 의 완전한 재현 스펙 (config 스냅샷)
├── agents/                      # agent 별 detailed log (해당 run 은 비어있음)
├── data/                        # iteration 별 parquet (factor values 등)
├── logs/                        # raw stdout/stderr 로그
├── qlib_artifacts/
│   ├── iter_1/  iter_2/  iter_3/  iter_4/    # outer iteration 별 Qlib report (.pkl, .csv)
├── reports/
│   ├── stage2.md                # PASS/FAIL 판정 + reasoning + evidence 표 (논문 Table 2 case study 의 원천)
│   ├── stage3.md                # 조합 strictness 검증 결과
│   └── stage4.md                # 백테스트 summary
└── specs/
    ├── hypothesis.json          # outer_iter_1..5 별 가설
    ├── observation_plan.json    # observation decomposition
    ├── formula_bundle.json      # 후보 factor 수식
    ├── stage2_summary.json      # Stage 2 판정 + evidence packet (per-formula)
    ├── stage3_result.json       # Stage 3 조합 결과
    ├── stage3_ticker_details.json
    ├── stage4_summary.json      # 모든 조합 × IS/OOS 백테스트 (n_trades, win_rate, IR, MDD 등 전체)
    ├── outer_loop_history.json  # outer iteration 누적 정보
    ├── refinement_history.json  # inner loop 누적 정보
    ├── llm_usage.json           # 토큰/비용 요약
    └── llm_usage_detailed.json  # call-by-call 상세
```

### 2.2 `run_config.json` 의 필드 (재현 핵심)

위 §1.5 의 모든 값이 그대로 직렬화되어 있음. 추가로:
- `run_id`, `concept`, `timestamp`
- `stage4_skipped` (bool), `stage3_verdict` ("PASS"/"FAIL"), `outer_loop_used` (bool)

### 2.3 가장 최근 (페이퍼 백테스트 직전) run: `20260209_073324`

- Concept: **"Mean Reversion after Panic Selling"** (논문 Table 2 의 hypothesis 와 같은 계열)
- Outer loop: 5 iterations 모두 수행 (`outer_iter_1` ~ `outer_iter_5`)
- Outer iter 1: hypothesis_id `BH_MR_PanicSelling_5D_v1`, 12 formulas → 11 PASS / 1 FAIL → Stage 3 FAIL
- Stage 4 가 한 iteration 에서 21 조합 평가 (`stage4_summary.json` 의 `n_combinations_evaluated: 21`)
- LLM cost (한 run 기준): **$1.75 / 119 calls / 489K tokens** (Stage 2 Distribution Validation 만 72 calls 의 비중 가장 큼)

→ 페이퍼 평균 비용 추정 시 참고 가능. 5 outer × 다수 hypothesis × 2 markets × 다수 backbone → 전체 페이퍼 비용은 수백 달러 단위.

---

## 3. LLM Backbone Switching (Table 4 — GPT-4o / Claude-4.5-sonnet / Gemini-2.5-pro)

### 3.1 메인 디렉토리에서 백본 변경
- 기본: `run/config.py` 의 `LLMConfig.model_name = "gpt-4o"`
- `OPENAI_BASE_URL` / `OPENAI_API_KEY` 환경변수로 OpenAI-호환 endpoint 라우팅 가능

### 3.2 `01_15_new_qlib copy/` 의 백본 라우팅 (실험적)
**위치**: `/home/dgu/fin/01_15_new_qlib copy/`

이 디렉토리는 메인의 사본이지만 **다른 LLM 백본 실험 + 결과 캐시** 가 추가되어 있다:

| 추가 파일 | 역할 |
|---|---|
| `litellm_proxy.yaml` | **LiteLLM proxy 설정 — Claude / Gemini 라우팅 핵심** (아래 인용) |
| `run_cn_qwen3_8b.sh` | Qwen3-8B 로컬 vLLM 서버 사용 (port 8001) |
| `run_cn_llama31_8b_instruct.sh` | Llama-3.1-8B-Instruct 로컬 vLLM (port 8000) |
| `run_cn_deepseek_r1_0528_qwen3_8b.sh`, `run_us_deepseek_r1_0528_qwen3_8b.sh` | DeepSeek-R1-distill-Qwen3-8B |
| `metrics_timeseries.csv` | 백테스트 시계열 metric (캐시) |
| `2026-02-{02,03,08}_*.pkl` | 백테스트 보고서 pickle (Qlib report) — Feb 8 까지 7개 |
| `progress/` | 추가 진행상황 메모 |

`litellm_proxy.yaml` 내용:
```yaml
model_list:
  - model_name: claude-tools
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
  - model_name: gemini-tools
    litellm_params:
      model: gemini/gemini-3-pro-preview
```
→ 논문 Table 4 의 Claude-4.5-sonnet / Gemini-2.5-pro 라벨과 **완전히 동일하지는 않음** — 위 yaml 은 `claude-3-5-sonnet-latest` 와 `gemini-3-pro-preview` 를 가리키므로, 실제 페이퍼 표의 모델 라벨을 어떻게 매핑했는지는 revision 시 재확인 필요.

로컬 LLM 스크립트의 패턴 (예: `run_cn_qwen3_8b.sh`):
```bash
export FINAGENT_OPENAI_BASE_URL="http://127.0.0.1:8001/v1"
export OPENAI_BASE_URL="$FINAGENT_OPENAI_BASE_URL"
export OPENAI_API_KEY="EMPTY"
export LLM_MODEL="qwen3-8b"
export FINAGENT_RUNS_DIR="/tmp/finagent_runs"
```
→ 로컬 모델 실험은 페이퍼에 직접 등장하지 않음. revision E1/E5/E8 (multi-backbone) 에서 재활용 가능.

---

## 4. 베이스라인 코드 위치 (Table 1 비교 대상)

| 베이스라인 | 위치 | 형태 |
|---|---|---|
| **AlphaAgent** | `/home/dgu/fin/AlphaAgent/` | KDD'25 origin. `alphaagent/scenarios/qlib/prompts_alphaagent.yaml`, `prompts_rdagent.yaml` 보유. `stage2.py`, `stage3.py`, `validation_agent.py`, `judge_agent.py` 등 풀 패키지. `log/2026-01-28..2026-02-01` 다수 실행 로그 |
| **RD-Agent (RD-Agent-Quant)** | `/home/dgu/fin/RD-Agent/` | Microsoft 원본 |
| **Alpha158, MACD, LightGBM, MLP, Linear, Transformer, XGBoost** | `/home/dgu/fin/qlib/` (Qlib 기본 제공) | 본 메인 디렉토리의 `run_pipeline_topkdropout_{lgbm,mlp,GATs}.py` 가 직접 호출 |
| **AlphaForge** (논문에는 없음, revision 베이스라인 후보) | `/home/dgu/fin/AlphaForge/` | AAAI'25, `train_AFF.py / combine_AFF.py / exp_AFF_calc_result.ipynb` |

> 메인 코드의 `coder/factor_coder/prompts_alphaagent.yaml` 은 AlphaAgent 의 prompt 형식을 가져와 통합한 흔적. 즉 FaVOR 는 AlphaAgent 의 코더 인터페이스를 일부 재사용.

---

## 5. Figure / Table 별 산출 위치

| 페이퍼 항목 | 산출 위치 |
|---|---|
| Table 1 (CSI500/SP500 main results) | 각 baseline + FaVOR 실행이 `runs/{run_id}/specs/stage4_summary.json` 의 `outsample.excess_return_with_cost` (AR, IR, MDD, CR 환산) 에서 산출. 집계 CSV: `01_15_new_qlib/analysis/alphaagent_perf.csv`, `ours_perf_full.csv`, `alphaforge_exp_perf.csv` |
| Figure 4 (cumulative excess returns) | `01_15_new_qlib/analysis/Excess_Return_{CN,US,CSI 500,S&P 500}.pdf` |
| Table 2 (case study) | `runs/{run_id}/reports/stage2.md` 의 PASS/FAIL reasoning + `specs/stage2_summary.json` 의 `evidence_packet` |
| Figure 5 (5-round IR evolution, 10 runs) | `runs/{run_id}/specs/outer_loop_history.json` 가 outer iter 별 metric 보유. paper "10 independent runs averaged" 의 raw 데이터는 **여러 run 디렉토리에 분산 저장** (run id timestamp 로 구분) |
| Table 3 (stage-wise ablation) | `pipeline_control.enable_stage2/enable_stage3` 토글 후 재실행한 별도 run 들의 결과 |
| Table 4 (LLM backbone) | LiteLLM proxy / 환경변수 변경 후 별도 run. **표의 모델 라벨과 실제 사용 라벨 매핑 확인 필요** (§3.2 참조) |
| Final result PDF | `01_15_new_qlib/analysis/Final_Result_{CSI 500, S&P 500, CN}.pdf`, `paper/aggregated_stage4_results.csv` |

---

## 6. ⚠️ Paper / Rebuttal / 코드 간 불일치 (확인·정정 필요)

다음 항목은 **revision 시 재확인 필수**:

1. **Stop loss**: 코드 default `-0.05` (5%) vs Rebuttal "stop-loss −10%" → 어느 값이 페이퍼 결과인지 확인 후 본문/rebuttal 정정.
2. **Stage 4 Optuna trial 수**: 코드 `n_trials=20` vs Rebuttal "thresholds in [0.55, 0.95] with step 0.05 (50 trials)" → 실제 페이퍼 결과는 어느 설정인지 확인.
3. **Stage 2 monotonicity threshold**: 코드 default `0.8` vs Rebuttal "0.7 at q50, q70, q90". `Stage3` 값(0.7)이 rebuttal 에 잘못 적힌 것일 수 있음.
4. **Table 4 backbone 라벨**: 페이퍼 "Claude-4.5-sonnet, Gemini-2.5-pro" vs `litellm_proxy.yaml` "claude-3-5-sonnet-latest, gemini-3-pro-preview" — 매핑이 모호.
5. **Train/Val/Test split**: rebuttal 일부에서 "fixed split 2015–2019 / 2020 / 2021–2025" 와 코드(`DataSplitConfig`) 일치. ✅
6. **Outer-loop 5 round + 10 independent runs**: 페이퍼 §4.4 와 config (`max_outer_iterations=5`) 는 일치. 단 "10 independent runs" 의 raw data 는 **여러 run id 에 분산** — revision 시 어떤 run id 들이 그 10개에 속했는지 매핑 필요. (`specs/outer_loop_history.json` 묶음 보존 위치 확인)

---

## 7. 환경 / 의존성

- **Python**: 3.10+ 추정 (`__pycache__/*.cpython-310.pyc`, 일부 `.cpython-39.pyc`, `.cpython-312.pyc` 도 존재 → 환경 혼재 가능)
- **주요 라이브러리**: `qlib`, `polars`, `pandas`, `optuna`, `litellm`, `openai`, `pydantic` (BaseModel 기반 config)
- **LD_LIBRARY_PATH 자동 처리**: `run_pipeline.py` 의 `_ensure_conda_lib_in_ld_library_path()` 에서 `$CONDA_PREFIX/lib` 자동 추가 (torch/faiss 호환)
- **OpenAI API key 로딩**: `run/config.py` `_load_env_from_dotenv()` — repo-root `.env` 자동 로드
- **데이터**:
  - CN: Qlib 기본 `~/.qlib/qlib_data/cn_data` (Baostock 출처)
  - US: `~/.qlib/sh_sp500_qlib` (Yahoo Finance 출처) + `01_15_new_qlib/qlib/download_sp500_yahoo.py`
- **Benchmark**: CSI500=`SH000905`, S&P500=`^GSPC`

---

## 8. Revision 시 빠른 재현 레시피 (참고용)

> **CLAUDE.md 규칙대로 기존 디렉토리는 절대 건드리지 않는다.** 실제 재현은 `revision/exp/` 아래에 코드를 새로 복사한 뒤 수행할 것.

**원본을 그대로 다시 돌리고 싶다면** (참고만):
```bash
# CSI500 main result
cd /home/dgu/fin/01_15_new_qlib
./run_cn_parallel.sh \
  "After a sharp sell-off, stocks that close near the day's high, indicating strong intraday recovery, are more likely to rebound over the next 3 trading days." \
  --outer-loop 5

# S&P500 main result
./run_us_parallel.sh "<same hypothesis>" --outer-loop 5

# Backbone swap (Claude / Gemini via LiteLLM proxy)
# 1) Start LiteLLM proxy with 01_15_new_qlib copy/litellm_proxy.yaml
# 2) export OPENAI_BASE_URL=http://localhost:4000
#    export LLM_MODEL=claude-tools  # or gemini-tools
# 3) Re-run the same launcher
```

**run 결과 위치**: `runs/{새 timestamp}/` — `specs/stage4_summary.json` 와 `reports/stage4.md` 가 핵심.

---

## 9. Revision 작업과의 매핑 (revision_todo.md 참조)

| revision_todo.md 항목 | 본 문서에서 참고할 곳 |
|---|---|
| E1 (LLM-validator vs rule-based filter) | §1.3 `validation_agent.py` (1022 lines), §1.4 `validation_agent_prompts.py` 의 Rules A~D 가 rule-based 변환의 청사진 |
| E3 (sweet-spot search) | §1.5 `Stage3Config.strictness_grid`, `Stage4Config.threshold_min/max` 를 더 촘촘하게 grid화 |
| E4 (sensitivity) | §1.5 표의 4개 핵심 파라미터 (`n_quantiles`, monotonicity threshold ×2, `combined_signal_q`) |
| E5 (stability + human agreement) | §2.1 `specs/stage2_summary.json` 의 `evidence_packet` + `reasoning` 필드 — N=10 독립 실행 후 결정 일치율 측정 |
| E6 (iterative refinement, 5→15 round) | §1.5 `RefinementConfig.max_outer_iterations` |
| E7 (DL/RL baselines) | §4 — AlphaForge/AlphaAgent 디렉토리 위치 |
| E8 (multi-backbone Table 1) | §3 — LiteLLM proxy 설정 + 로컬 LLM 스크립트 패턴 |
| 6. 불일치 정정 | §6 의 5개 항목 모두 |

---

## Appendix A. 빠른 인덱스

- 메인 코드: `/home/dgu/fin/01_15_new_qlib/`
- 메인 설정: `01_15_new_qlib/run/config.py`
- 메인 launcher: `01_15_new_qlib/run_{cn,us}{,_parallel,_new,_new_all}.sh`
- 메인 entry: `01_15_new_qlib/run_pipeline*.py`
- 결과: `01_15_new_qlib/runs/{YYYYMMDD_HHMMSS}/`
- 가장 최근 run: `01_15_new_qlib/runs/20260209_073324/`
- LLM 라우팅 (Claude/Gemini): `01_15_new_qlib copy/litellm_proxy.yaml`
- 로컬 LLM 실험: `01_15_new_qlib copy/run_cn_{qwen3_8b,llama31_8b_instruct,deepseek_r1_*}.sh`
- 베이스라인 코드: `AlphaAgent/`, `RD-Agent/`, `AlphaForge/`, `qlib/`
- 페이퍼용 집계: `01_15_new_qlib/analysis/*.csv`, `paper/aggregated_stage4_results.csv`, `01_15_new_qlib/paper/`
