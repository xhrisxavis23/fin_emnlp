# FaVOR — 모듈 정리

> **목적**: KDD 2026 제출본에서 EMNLP 2026 revision으로 넘어가는 시점에 `01_15_new_qlib/` 안의 거대한 실험 트리에서 **"FaVOR 파이프라인이 동작하는 데 꼭 필요한 코드"** 만 추려서 `revision/favor/` 로 옮긴 결과를 정리한 문서.
>
> **기준 시점**: 2026-02-09 마지막 실험 로그 (`runs/20260209_073324/`) 의 세팅에 가장 가까운 형태.
> **추출 방법**: `run_pipeline.py` + `run_pipeline_parallel_per_combo_parallel.py` 를 시작점으로 AST import 그래프를 닫힐 때까지 추적. 거기에 런타임에 로드되는 리소스(yaml/jinja) 4개를 추가.
> **결과**: Python 86개 + 리소스 4개 + launcher .sh 4개 + `.env.example` + 본 문서.
>
> 원본은 손대지 않았다 (CLAUDE.md 절대 규칙). `revision/favor/` 만 사용하면 된다.

---

## 1. 한눈에 보기

```
revision/favor/
├── favor_모듈_정리.md                     ← 이 파일
├── .env.example                           ← API 키 템플릿
├── run_pipeline.py                        ← canonical entry (단일 프로세스)
├── run_pipeline_parallel_per_combo_parallel.py  ← Feb 9 final entry (병렬 stage4)
├── run_cn.sh / run_us.sh                  ← canonical launcher (CN / US)
├── run_cn_limited.sh / run_us_limited.sh  ← Feb 9 final launcher (resource-limited 병렬)
├── run/                ← 파이프라인 코어
│   ├── config.py       ← 모든 설정 dataclass + MARKET 분기 + .env 로더
│   ├── main.py         ← 스테이지 오케스트레이션 (`run_pipeline`, `run_outer_loop`)
│   ├── pipeline/       ← Stage 1~4 + refinement loop + 백테스트 strategy
│   └── util/           ← config_utils / data_utils / pipeline_utils
├── agent/              ← LLM agent (Hypothesis / Observation / Formula / Validation 등)
├── prompts/            ← 위 agent들이 쓰는 system/user 프롬프트
├── schemas/            ← stage 출력 dataclass + LLM tool schema
├── coder/              ← CoSTEER + factor_coder (수식 → 실행 코드)
│   ├── CoSTEER/        ← CoSTEER 프레임워크
│   ├── factor_coder/   ← 실제 사용되는 expression-based factor 코더
│   └── polars_factor_coder/  ← ⚠️ dead code (§9 참조)
├── components/         ← knowledge_management/graph 만 사용
├── core/               ← CoSTEER가 의존하는 base 클래스
├── oai/                ← LLM 설정 헬퍼
└── util/               ← run_context, llm_client, llm_tracker, json_utils
```

원본 모듈 루트 (`run`, `agent`, `prompts`, …) 를 그대로 미러링했기 때문에 **import 경로 수정 없이** 그대로 동작한다.

---

## 2. 모듈 별 역할

### 2.1 entry points (최상위)

| 파일 | 역할 | 비고 |
|---|---|---|
| `run_pipeline.py` | canonical CLI. `run.main.run_pipeline` / `run_outer_loop` 호출 | LD_LIBRARY_PATH 자동 보강 후 self-exec |
| `run_pipeline_parallel_per_combo_parallel.py` | Feb 9 final용. Stage4를 조합 단위 병렬로 monkey-patch | `--combo-workers`, `--optuna-jobs` CLI |
| `run_cn.sh` | `MARKET=cn python run_pipeline.py "$@"` | 단일 프로세스 |
| `run_us.sh` | `MARKET=us python run_pipeline.py "$@"` | 단일 프로세스 |
| `run_cn_limited.sh` | resource-limited 병렬판. CPU/메모리 제한 + `STAGE4_COMBO_WORKERS` 등 환경변수 세팅 후 parallel runner 호출 | **Feb 9 마지막 실험에 가장 가까운 launcher** |
| `run_us_limited.sh` | 위와 같지만 US (`MARKET=us`) | 동일 패턴 |

### 2.2 `run/` — 파이프라인 코어

| 파일 | 역할 |
|---|---|
| `run/config.py` | `LLMConfig`, `QlibConfig`, `Stage{1..4}Config`, `DataSplitConfig`, `RefinementConfig`, `PipelineControlConfig`, `RDConfig` Pydantic 모델. `load_rd_config()` 가 `MARKET=cn|us` 환경변수에 따라 거래비용/벤치마크/provider_uri를 자동 분기. repo-root `.env` 도 여기서 로드 |
| `run/main.py` | **오케스트레이터**. `run_pipeline()` 가 Stage1 → 2 → 3 → 4 를 호출하고, `run_outer_loop()` 이 가설을 5 라운드 재생성 |
| `run/pipeline/stage1.py` | Stage 1: Hypothesis Decomposition (가설 → observation → 후보 수식) |
| `run/pipeline/stage2.py` | Stage 2: Factor-Level Validation (5분위 분포 + LLM 판정) |
| `run/pipeline/stage3.py` | Stage 3: Factor Integration (strictness grid 단조성 검증) |
| `run/pipeline/stage4.py` | Stage 4: Optimization & Backtesting (Optuna + Qlib `TriggerExitStrategy`) |
| `run/pipeline/stage4_parallel.py` | Stage 4 의 multi-process 변형. `parallel_per_combo` 가 의존 |
| `run/pipeline/stage4_parallel_per_combo.py` | 조합 단위로 더 잘게 병렬화. Feb 9 final launcher가 사용 |
| `run/pipeline/refinement_2to1.py` | inner loop: Stage2 FAIL → Stage1 formula 재생성 |
| `run/pipeline/refinement_4to1.py` | outer loop: Stage4 결과 → Stage1 가설 재생성 (논문 Figure 5) |
| `run/pipeline/strategy.py` | Qlib custom `TriggerExitStrategy` 등 |
| `run/util/config_utils.py` | config 파싱 헬퍼 |
| `run/util/data_utils.py` | OHLCV 컬럼 표준화 |
| `run/util/pipeline_utils.py` | `compute_formula_values`, `FormulaComputeResult` 등 stage 공용 함수 |

### 2.3 `agent/` — LLM 에이전트

| 파일 | 페이퍼 표기 | 역할 |
|---|---|---|
| `base_agent.py` | — | 공통 베이스 |
| `hypothesis_agent.py` | $\mathcal{A}_H$ | 가설 생성/재생성 |
| `observation_agent.py` | $\mathcal{A}_O$ | 가설 → observable conditions 분해 |
| `formula_agent.py` | $\mathcal{A}_F$ | observation → 후보 factor 수식 생성 |
| `validation_agent.py` | $\mathcal{A}_V$ | **Stage 2 distribution-based judgment (논문의 핵심 reasoning agent)** |
| `hypothesis_validation_agent.py` | — | 가설 자체에 대한 sanity check (Stage 3 에서 사용) |
| `factor_coder_code_agent.py` | — | 수식 → executable 코드. `coder/factor_coder/` 를 백엔드로 사용 |
| `costeer_full_code_agent.py` | — | CoSTEER 전체 루프를 들고 있는 코더 |
| `coder_code_agent.py` | — | Polars 기반 대안 코더. ⚠️ dead code 경유 (§9) |
| `diagnostics_agent.py` | — | 실패 원인 진단 (`hypothesis_validation_agent` 의 alias wrapper) |
| `diagnostics_tools.py` | — | 위 agent 가 쓰는 진단 도구 |

### 2.4 `prompts/` — system / user prompts

| 파일 | 매핑 stage |
|---|---|
| `hypothesis_agent_prompts.py` | Stage 1 — hypothesis generation / regeneration |
| `observation_agent_prompts.py` | Stage 1 — observation decomposition |
| `formula_agent_prompts.py` | Stage 1 — factor formula generation |
| `validation_agent_prompts.py` | **Stage 2 — PASS/FAIL 결정의 핵심 프롬프트.** 논문 §3.3.2 의 (i)~(iv) Validation Criteria 가 그대로 박혀 있고, Rules A~D (location / tail / multi-stat / no contradiction) 도 여기에 있다 |

> **참고**: 원본 `prompts/` 에는 `diagnostics_agent_prompts.py`, `hypothesis_validation_prompts.py`, `raw_idea_prompts.py` 도 있지만 **현재 파이프라인의 import 그래프 상 어디서도 참조되지 않는다** (legacy). 따라서 이번 추출에서는 빠졌다.

### 2.5 `schemas/` — 스테이지 출력 스키마

| 파일 | 용도 |
|---|---|
| `behavioral_formula.py` | Behavioral formula tool schema (LLM tool call) |
| `formula.py` | `FORMULA_TOOL` (formula_agent 가 강제 호출) |
| `hypothesis.py` | hypothesis dataclass |
| `hypothesis_validation_dataclasses.py` | Stage3 가 쓰는 `HypothesisValidationResult` 등 |
| `observation.py` | observation dataclass |
| `validation.py` | `distribution_judgment_tool` 등 validation_agent tool schema |
| `validation_dataclasses.py` | `FormulaValidationResult` 등 dataclass |

> `schemas/` 에는 `__init__.py` 가 없고 namespace package 로 동작한다. (원본도 동일)

### 2.6 `coder/` — 수식 실행 백엔드

| 파일 | 역할 |
|---|---|
| `CoSTEER/` | RD-Agent 의 CoSTEER 프레임워크. evolutionary code generation의 코어 |
| `CoSTEER/prompts.yaml` | CoSTEER 가 LLM 에 보내는 프롬프트 (런타임 로드) |
| `factor_coder/` | **실제 사용되는** expression-based factor 코더 |
| `factor_coder/prompts.yaml`, `prompts_alphaagent.yaml` | factor 코더 system prompt (런타임 로드) |
| `factor_coder/template.jinjia2` | factor evaluator 가 채우는 jinja 템플릿 |
| `factor_coder/expr_parser.py`, `factor.py`, `factor_ast.py`, `function_lib.py` | 수식 파싱 / AST / 사용 가능 함수 라이브러리 |
| `polars_factor_coder/` | ⚠️ Polars 기반 대안 코더. **현재 파이프라인은 사용하지 않으며 모듈 자체가 broken** (§9) |

### 2.7 그 외

| 모듈 | 역할 |
|---|---|
| `core/` | CoSTEER 가 상속하는 base 클래스 (`Scenario`, `Experiment`, `KnowledgeBase`, `Prompts` 로더 등) |
| `components/knowledge_management/graph.py` | CoSTEER 의 knowledge graph |
| `oai/llm_conf.py`, `oai/llm_utils.py` | OpenAI 호환 client 헬퍼 |
| `util/llm_client.py` | **실제 LLM 호출의 진입점** (`call_llm`). ReAct 루프 + tool-calling |
| `util/llm_tracker.py` | 토큰/비용 트래킹 |
| `util/run_context.py` | `RunContext` — `runs/{run_id}/` 디렉토리 만들고 specs/reports/logs 저장 |
| `util/qlib_data_loader.py` (없음) | **이번 추출 대상 아님**. 원본도 import 그래프에 안 잡힘 |
| `util/json_utils.py` | `strip_code_fence` 등 |
| `log.py` | 최상위 로거 설정 |

---

## 3. 환경 / 의존성

### 3.1 Python

- **3.10 권장** (원본 `__pycache__` 에 `cpython-310.pyc` 가 가장 많음)
- 3.12 도 import smoke test 통과 (단, qlib 일부 deprecation 경고)

### 3.2 외부 패키지

`run/main.py` 와 stage 들이 import 하는 주요 third-party 라이브러리:

| 패키지 | 용도 |
|---|---|
| `polars`, `pandas`, `numpy` | 데이터 |
| `qlib` (microsoft/qlib) | 백테스트 |
| `optuna` | Stage 4 threshold 최적화 |
| `openai` | LLM client (litellm 안 쓰면 직접) |
| `pydantic` | config 모델 |
| `jinja2` | factor template |
| `pyparsing` | factor expression 파서 |
| `dill`, `filelock` | CoSTEER persistence |
| `fuzzywuzzy` | symbol matching (knowledge_management) |
| `pydantic_settings` | 일부 설정 |

> 검증 환경: `/home/dgu/.conda/envs/quant` (Python 3.12, polars 1.35.2, qlib 0.9.7) 에서 84/86 모듈 import 성공. (실패한 4개는 §9 의 dead code)

### 3.3 데이터

데이터 자체는 코드 트리 안에 들어있지 않다. **외부에서 받아 정해진 경로에 깔아두는 구조**이며, 정해진 경로는 `run/config.py` 의 `QlibConfig.provider_uri` (CN) / `load_rd_config()` 의 US 분기에서 하드코딩됨.

#### 3.3.1 코드가 데이터를 찾는 위치

| 시장 | provider_uri (default) | instrument pool 결정 방식 | benchmark | region 코드 |
|---|---|---|---|---|
| CN (`MARKET=cn`) | `~/.qlib/qlib_data/cn_data` | `D.instruments("csi500")` (qlib 자체 instrument set) | `SH000905` | `cn` |
| US (`MARKET=us`) | `~/.qlib/sh_sp500_qlib` | `~/.qlib/sh_sp500_csv/*.csv` 의 stem 을 ticker 로 사용 | `^GSPC` | `us` |

진입점: `run/config.py:load_price_data()` (`L346`) → `qlib.init(provider_uri, region)` → `D.features(instruments_pool, fields=[$open,$high,$low,$close,$volume,$factor], start, end)`.

> 참고: US 의 경우 두 디렉토리가 함께 필요하다.
> - `~/.qlib/sh_sp500_qlib/` — qlib bin 포맷 데이터 (`features/`, `instruments/`, `calendars/` 서브디렉토리)
> - `~/.qlib/sh_sp500_csv/` — ticker 목록 추출용 CSV (코드는 파일명만 읽음)

#### 3.3.2 데이터를 어디서 가져오나

**(A) CSI500 — qlib 공식 dump (간단)**

`01_15_new_qlib/qlib/scripts/get_data.py` (Microsoft Qlib 자체 유틸. 본 추출 대상 아님 — 원본 트리에서 직접 호출):

```bash
cd /home/dgu/fin/01_15_new_qlib/qlib/scripts
python get_data.py qlib_data \
    --target_dir ~/.qlib/qlib_data/cn_data \
    --interval 1d --region cn
```

다운로드 URL (`get_data.py` 내부에서 자동 결정):
```
https://qlibpublic.blob.core.windows.net/data/default/stock_data/v2/qlib_data_cn_1d_latest.zip
```

압축 해제 후 결과:
```
~/.qlib/qlib_data/cn_data/
├── calendars/day.txt
├── instruments/{all.txt, csi300.txt, csi500.txt, ...}
└── features/{ticker}/{open,high,low,close,volume,factor}.day.bin
```

데이터 출처: **Baostock** (페이퍼 §4 와 일치).

**(B) S&P500 — Yahoo Finance 크롤 + dump_bin 변환 (2단계)**

크롤러 위치: `01_15_new_qlib/qlib/scripts/data_collector/yahoo/collector.py`

1. **raw CSV 수집** (Yahoo Finance):
   ```bash
   cd /home/dgu/fin/01_15_new_qlib/qlib/scripts/data_collector/yahoo
   python collector.py download_data \
       --source_dir ~/.qlib/sh_sp500_csv \
       --start 2015-01-01 --end 2025-12-31 \
       --region us --interval 1d
   ```
   → `~/.qlib/sh_sp500_csv/{ticker}.csv` 생성. 코드는 이 파일명만 ticker 추출에 사용.

2. **qlib bin 포맷으로 변환**:
   ```bash
   cd /home/dgu/fin/01_15_new_qlib/qlib/scripts
   python dump_bin.py dump_all \
       --csv_path ~/.qlib/sh_sp500_csv \
       --qlib_dir ~/.qlib/sh_sp500_qlib \
       --include_fields open,high,low,close,volume,factor
   ```
   → `~/.qlib/sh_sp500_qlib/features/`, `instruments/`, `calendars/` 생성.

3. (선택) **factor 파일 보강**: `01_15_new_qlib/generate_us_factor_files.py` 가 일부 누락된 `$factor` 컬럼을 채워주는 헬퍼 (Feb 9 운영 환경에서 사용된 흔적 있음). 본 추출 대상 아님.

⚠️ **Yahoo Finance 는 중국에서 차단**되어 있어 VPN 필요할 수 있다고 collector README 가 경고한다.

#### 3.3.3 현재 머신 상태

```bash
$ ls ~/.qlib/                                  # → No such file or directory
$ find / -maxdepth 6 -iname "*sp500*" -o -iname "*csi500*" 2>/dev/null   # → 없음
```

이전 Feb 9 실행 머신과 다른 환경이거나, 그 사이 `~/.qlib/` 가 정리됨. **revision 실험을 다시 돌리려면 위 (A)/(B) 절차로 데이터를 새로 받아야 한다.**

#### 3.3.4 데이터 옮길 때 (다른 머신으로 이전 시)

전체 통째로 옮기는 게 가장 빠르다 (다시 받지 않고). 옮길 디렉토리:

| 디렉토리 | 크기 추정 | 필수 |
|---|---|---|
| `~/.qlib/qlib_data/cn_data/` | ~2 GB (압축) / ~5 GB (해제) | CN 실험 시 필수 |
| `~/.qlib/sh_sp500_qlib/` | ~500 MB | US 실험 시 필수 |
| `~/.qlib/sh_sp500_csv/` | ~200 MB | US 실험 시 필수 (ticker 목록) |

옮기는 방법 예시:

```bash
# 송신측
tar -czf qlib_data.tar.gz -C ~/ .qlib/qlib_data .qlib/sh_sp500_qlib .qlib/sh_sp500_csv
rsync -av qlib_data.tar.gz dest_host:/path/

# 수신측
mkdir -p ~/.qlib && tar -xzf qlib_data.tar.gz -C ~/
ls ~/.qlib/   # qlib_data, sh_sp500_qlib, sh_sp500_csv 가 보여야 함
```

옮긴 뒤 sanity check:

```bash
# CN
ls ~/.qlib/qlib_data/cn_data/instruments/csi500.txt   # 파일 존재
wc -l ~/.qlib/qlib_data/cn_data/instruments/csi500.txt  # 수백~천 줄

# US
ls ~/.qlib/sh_sp500_qlib/{calendars,instruments,features} | head
ls ~/.qlib/sh_sp500_csv/*.csv | wc -l  # ~500
```

**경로 변경 시**: `~/.qlib/...` 가 아닌 다른 위치에 두려면 `run/config.py:33` 의 `provider_uri` 와 `run/config.py:328,338` (US 분기) 를 같이 바꿔야 한다. 또는 환경별로 분기시키려면 `load_rd_config()` 안에 `os.getenv("QLIB_PROVIDER_URI")` 를 추가하는 hook 을 넣으면 깔끔하다.

### 3.4 `.env`

`run/config.py:_load_env_from_dotenv()` 가 **repo-root (즉 `revision/favor/.env`)** 의 `.env` 를 자동 로드한다. `revision/favor/.env.example` 을 복사해 채우면 된다.

```bash
cp revision/favor/.env.example revision/favor/.env
# 편집: OPENAI_API_KEY=sk-... 입력
```

---

## 4. 실제 실행 방법

### 4.1 Feb 9 final 세팅 (resource-limited 병렬판)

```bash
cd /home/dgu/fin/revision/favor

# CSI500
./run_cn_limited.sh \
  "After a sharp sell-off, stocks that close near the day's high, indicating strong intraday recovery, are more likely to rebound over the next 3 trading days." \
  --outer-loop 5

# S&P500
./run_us_limited.sh "<같은 가설 또는 다른 가설>" --outer-loop 5
```

`run_cn_limited.sh` 가 내부적으로 세팅하는 것:
- CPU: `OMP_NUM_THREADS=20`, `POLARS_MAX_THREADS=20`, … (각 실험당 20 코어)
- 메모리: `ulimit -v 150GB`
- Stage 4: `STAGE4_COMBO_WORKERS=4` × `STAGE4_OPTUNA_N_JOBS=1`
- 진입점: `python run_pipeline_parallel_per_combo_parallel.py "$@"`

### 4.2 단일 프로세스 (debug용)

```bash
./run_cn.sh "<concept>" --outer-loop 5
./run_us.sh "<concept>" --outer-loop 5
```

내부적으로 `python run_pipeline.py "$@"` 호출. Stage 4 도 단일 프로세스에서 실행되며, 작은 실험이나 디버깅에 적합.

### 4.3 직접 호출

```python
# revision/favor/ 가 cwd 또는 PYTHONPATH 에 있어야 함
from run.main import run_pipeline, run_outer_loop
from run.config import load_rd_config

cfg = load_rd_config()  # MARKET 환경변수에 따라 cn/us 자동 분기
result = run_outer_loop(
    concept="...",
    config=cfg,
    max_outer_iterations=5,
)
```

### 4.4 결과물

`runs/{YYYYMMDD_HHMMSS}/` 에 통째로 떨어진다 (페이퍼 figure/table 생성에 그대로 사용).

```
runs/{run_id}/
├── run_config.json              ← 이 run 의 완전한 재현 스펙
├── agents/                      ← agent 별 ReAct trace
├── data/                        ← iteration 별 parquet (factor values 등)
├── logs/                        ← raw stdout/stderr
├── qlib_artifacts/              ← Qlib report (.pkl, .csv)
├── reports/
│   ├── stage2.md                ← PASS/FAIL + reasoning (논문 Table 2 case study)
│   ├── stage3.md
│   └── stage4.md
└── specs/
    ├── hypothesis.json
    ├── observation_plan.json
    ├── formula_bundle.json
    ├── stage2_summary.json      ← evidence_packet + reasoning per formula
    ├── stage3_result.json
    ├── stage4_summary.json      ← outsample.excess_return_with_cost (논문 Table 1 의 원천)
    ├── outer_loop_history.json  ← 논문 Figure 5
    ├── refinement_history.json
    └── llm_usage{,_detailed}.json
```

`util/run_context.py:RunContext` 가 이 구조를 만들고 채운다.

---

## 5. 모델 변경 방법

### 5.1 다른 OpenAI 모델로 교체 (gpt-4o → gpt-4.1, o1, …)

가장 단순한 방법: `run/config.py` 의 `LLMConfig.model_name` 을 바꾼다.

```python
# revision/favor/run/config.py:22-25
class LLMConfig(BaseModel):
    model_name: str = "gpt-4o"     # ← 여기를 "gpt-4.1-2025-04-14" 등으로
    temperature: float = 0.7
    max_tokens: int = 2048
```

저장 후 다시 실행하면 끝. (현재 코드는 환경변수로 모델을 바꾸는 hook 이 없다. 필요시 `LLMConfig.__init__` 에 `os.getenv("LLM_MODEL")` 분기를 추가하면 된다.)

### 5.2 OpenAI 호환 endpoint 라우팅 (Claude / Gemini / 로컬 LLM)

OpenAI Python SDK 는 `OPENAI_BASE_URL` 환경변수를 자동으로 읽는다. 따라서 `.env` 또는 launcher 에 다음을 추가하면 된다:

```bash
export OPENAI_BASE_URL=http://localhost:4000   # LiteLLM proxy
export OPENAI_API_KEY=any-string
```

LiteLLM proxy 예시 (`01_15_new_qlib copy/litellm_proxy.yaml` 참조):

```yaml
model_list:
  - model_name: claude-tools
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
  - model_name: gemini-tools
    litellm_params:
      model: gemini/gemini-3-pro-preview
```

이 경우 `LLMConfig.model_name = "claude-tools"` (또는 `"gemini-tools"`) 로 바꿔야 한다.

### 5.3 로컬 vLLM 백본

`01_15_new_qlib copy/run_cn_qwen3_8b.sh` 패턴:

```bash
export FINAGENT_OPENAI_BASE_URL="http://127.0.0.1:8001/v1"
export OPENAI_BASE_URL="$FINAGENT_OPENAI_BASE_URL"
export OPENAI_API_KEY="EMPTY"
# LLMConfig.model_name 도 "qwen3-8b" 등으로
```

### 5.4 어디를 같이 봐야 하나

- `util/llm_client.py:_get_client` — OpenAI 클라이언트 생성. `OPENAI_API_KEY`, `OPENAI_BASE_URL` 만 사용.
- `util/llm_client.py:call_llm` — tool-calling ReAct 루프. 모든 agent 가 이걸 거친다.
- `oai/llm_utils.py` — 추가 헬퍼 (현 파이프라인에서는 거의 사용 안 함).
- `agent/*.py` — 각 agent 가 `LLMConfig.model_name` 을 받아 `call_llm(model=...)` 으로 호출.

---

## 6. 시장 (CN ↔ US) 변경 방법

`MARKET` 환경변수만 바꾸면 된다.

```bash
export MARKET=us   # or cn
```

`run/config.py:load_rd_config` 가 자동으로 다음을 분기한다:

| 항목 | CN (CSI500) | US (S&P500) |
|---|---|---|
| `qlib_market` | `csi500` | `sp500` |
| `region` | `cn` | `us` |
| `provider_uri` | `~/.qlib/qlib_data/cn_data` | `~/.qlib/sh_sp500_qlib` |
| `open_cost` | `0.0005` | `0.0` |
| `close_cost` | `0.0015` | `0.0005` |
| `min_cost` | `5.0` | `0.0` |
| `limit_threshold` | `0.095` | `None` |
| `benchmark` | `SH000905` | `^GSPC` |

→ 논문 §4.1 의 거래비용 표와 그대로 일치.

---

## 7. 주요 config 노브 (`run/config.py`)

대부분의 실험 변수는 `RDConfig` 한 객체 안에 있다. 변경할 수 있는 주요 노브:

### Stage 1 (`Stage1Config`)
- `allowed_ohlcv_columns`: `[open, high, low, close, volume]` — Stage 1 이 외부 정보를 못 쓰게 함.
- `refine_rounds`: `10` — formula 자가수정 최대 횟수.

### Stage 2 (`Stage2Config`)
- `n_quantiles`: `5` — 5분위 partition (논문 §3.3.1).
- `monotonicity_threshold`: `0.8` — bin-wise mean monotonicity 합격선.

### Stage 3 (`Stage3Config`)
- `horizon_days`: `5` — hypothesis 미지정시 기본 forward window.
- `monotonicity_threshold`: `0.7` — strictness-level monotonicity.
- `strictness_grid`: `{very_loose:0.1, loose:0.3, medium:0.5, strict:0.7, very_strict:0.9}`.
- `use_random_grid`: `True`, `random_grid_steps`: `3` — progressive random grid 모드.
- `combination_pass_rate_threshold`: `0.5` — 50% ticker 통과 = PASS.
- `n_processes`: `8` — 병렬 워커 수.

### Stage 4 (`Stage4Config`)
- `enable_optuna`: `True`, `n_trials`: `20`.
- `threshold_min`, `threshold_max`: `0.55`, `0.95` — Optuna 탐색 영역.
- `combined_signal_q`: `0.9` — 최종 signal quantile (논문).
- `horizon_days`: `5` (Feb 9 final run 은 `9`), `lookback_window`: `20`.
- `stop_loss_threshold`: `-0.05` (5% 손절). ⚠️ rebuttal 의 −10% 와 불일치 — `revision/implementation_settings.md §6` 참조.
- `trigger_kmin`, `trigger_kmax`: `1`, `5` — 진입 trigger 검사 day offset.
- `native_strategy`: `"trigger_exit"` — Qlib 전략.

### Data split (`DataSplitConfig`) — 모든 실험 공통
- Train: **2015-01-01 ~ 2019-12-31** (Stage 2/3 factor validation)
- Validation: **2020-01-01 ~ 2020-12-31** (Optuna threshold 최적화)
- Test: **2021-01-01 ~ 2025-12-31** (최종 OOS, Table 1 대상)

### Refinement (`RefinementConfig`)
- `enable_inner_loop`: `True`, `max_inner_iterations`: `3` — Stage1 ⇄ Stage2 재생성.
- `enable_outer_loop`: `True`, `max_outer_iterations`: `5` — **Stage4 → Stage1 재생성. 논문 Figure 5 의 5 round**.

### Pipeline control (`PipelineControlConfig`)
- `enable_stage2`: `True`, `enable_stage3`: `True` — Table 3 ablation 시 토글.

---

## 8. 검증 (Smoke test)

`/home/dgu/.conda/envs/quant` 환경에서:

```bash
cd /home/dgu/fin/revision/favor
PYTHONPATH=. /home/dgu/.conda/envs/quant/bin/python -c \
  "from run.main import run_pipeline, run_outer_loop; \
   from run.pipeline.stage4_parallel_per_combo import run_stage4; \
   print('OK')"
```

→ `OK` 출력. canonical entry 와 Feb 9 parallel entry 모두 import 성공.

전수 import 결과: **84개 모듈 중 80개 OK, 4개 FAIL**. 실패한 4개는 모두 `coder/polars_factor_coder/` 패키지로 §9 의 dead code 이며, **원본 `01_15_new_qlib/` 에서도 동일하게 fail** (재현됨). 파이프라인은 이 패키지를 사용하지 않으므로 실험에 영향이 없다.

---

## 9. 알려진 이슈 / dead code

### 9.1 `coder/polars_factor_coder/` — broken & 미사용

- 패키지 import 시 `from schemas.code import CODE_TOOL` 가 실패한다 (`schemas/code.py` 가 원본에도 없다).
- 호출 경로: `agent/coder_code_agent.py:CoderCodeAgent.__init__` 안에서 **lazy import** 됨. 즉 `CoderCodeAgent` 를 인스턴스화할 때만 깨진다.
- 현재 `run/main.py` 는 `FactorCoderCodeAgent` 만 쓰고 `CoderCodeAgent` 는 인스턴스화하지 않으므로 실험에 문제가 되지 않는다.
- 이 패키지를 깨끗하게 만들고 싶다면 (revision 에서 polars 코더가 필요해지면) `schemas/code.py` 를 추가로 작성해야 한다. 페이퍼 재현에는 필요 없다.

### 9.2 의도적으로 빠진 것들

원본 `01_15_new_qlib/` 에 있었지만 import 그래프에 안 잡혀서 **이번 추출에 포함되지 않은** 항목:

| 항목 | 빠진 이유 |
|---|---|
| `run_pipeline_parallel*.py` (per_combo 외 4개), `run_pipeline_topkdropout_*.py`, `run_stage4_suite.py`, `run_weighted_backtest.py`, `run_pipeline_new*.py` | Feb 9 final launcher 가 안 부른다. baseline (LightGBM/MLP/GAT) 은 `qlib/` 자체 또는 `AlphaAgent/` 에서 별도 실행 |
| `run_cn_new*.sh`, `run_cn_parallel*.sh`, `run_us_new*.sh`, `run_us_parallel.sh` | 위와 같은 이유 |
| `run/pipeline/stage*_new*.py`, `stage*_parallel*.py` (per_combo 제외), `stage4_for_ticker.py`, `stage4_backup.py`, `stage3_new.py`, `_unused/` | `run/main.py` 가 import 하지 않는 변형/레거시 |
| `runs/`, `logs/`, `mlruns/`, `analysis/`, `paper/`, `docs/`, `__pycache__/` | 결과물·로그·분석·문서. 코드 동작에 불필요 |
| `0126_분석.ipynb`, `수식검증.ipynb` | 분석 노트북 |
| `STAGE2_CHANGES.md` | 메타 문서 (원본 보존됨) |
| `prompts/diagnostics_agent_prompts.py`, `prompts/hypothesis_validation_prompts.py`, `prompts/raw_idea_prompts.py` | import 그래프에서 참조되지 않음 (legacy) |
| `schemas/diagnostics.py`, `schemas/formula_expression.py`, `schemas/hypothesis_validation.py`, `schemas/behavioral_formula.py` 외 | 일부는 dead, 일부는 사용됨 — 사용되는 것만 복사함 |
| `components/document_reader/`, `components/loader/` | 어디에서도 import 안 됨 |
| `coder/factor_coder/template_debug.jinjia2`, `coder/factor_coder/test.py` | 디버그/테스트용, 런타임에 안 씀 |
| `util/qlib_data_loader.py`, `util/_unused/` | 미사용 |
| `oai/__init__.py` 외 `llm_conf.py`/`llm_utils.py` 만 사용 | 다른 oai 파일은 없음 |
| `core/conf.py`, `core/developer.py`, `core/evaluation.py`, `core/evolving_agent.py`, `core/evolving_framework.py`, `core/exception.py`, `core/experiment.py`, `core/knowledge_base.py`, `core/prompts.py`, `core/scenario.py`, `core/template.py`, `core/utils.py` | 전부 import 그래프에 잡혀 포함됨 (CoSTEER 가 의존) |

원본 (01_15_new_qlib/) 에서 동일 코드를 그대로 다시 돌리고 싶다면 그 디렉토리에서 직접 실행하면 된다. **CLAUDE.md 절대 규칙에 따라 원본은 손대지 않는다.**

### 9.3 paper / rebuttal / 코드 간 불일치

`revision/implementation_settings.md §6` 에 정리되어 있다. 핵심만:

1. **Stop loss**: 코드 default `-0.05` (5%) ↔ rebuttal "−10%". 어느 값이 페이퍼 결과인지 확인 후 본문/rebuttal 정정 필요.
2. **Stage 4 Optuna trial 수**: 코드 `n_trials=20` ↔ rebuttal "step 0.05 (50 trials)". 확인 필요.
3. **Stage 2 monotonicity threshold**: 코드 default `0.8` ↔ rebuttal "0.7 at q50,q70,q90". Stage3 값(0.7)을 잘못 적은 것일 수 있음.
4. **Table 4 backbone 라벨**: 페이퍼 "Claude-4.5-sonnet, Gemini-2.5-pro" ↔ `litellm_proxy.yaml` "claude-3-5-sonnet-latest, gemini-3-pro-preview". 매핑 모호.

---

## 10. 다음번 실험 돌리기 전 체크리스트

- [ ] `cp .env.example .env` 후 `OPENAI_API_KEY` 입력 (필요시 `OPENAI_BASE_URL` 도)
- [ ] **데이터 존재 확인**: `~/.qlib/qlib_data/cn_data/` (CN) / `~/.qlib/sh_sp500_qlib/` + `~/.qlib/sh_sp500_csv/` (US). **없으면 §3.3.2 의 다운로드 절차를 먼저 수행**, 또는 다른 머신에서 옮겨오기 (§3.3.4).
- [ ] sanity check: `ls ~/.qlib/qlib_data/cn_data/instruments/csi500.txt` (CN) / `ls ~/.qlib/sh_sp500_csv/*.csv | wc -l` 가 ~500 (US).
- [ ] conda env 활성화: `conda activate quant` (또는 polars/qlib/openai 가 설치된 환경)
- [ ] `cd revision/favor`
- [ ] launcher 실행 — Feb 9 final 재현이면 `./run_cn_limited.sh "<concept>" --outer-loop 5`
- [ ] `runs/{새 timestamp}/specs/stage4_summary.json` 의 `outsample.excess_return_with_cost` 확인
- [ ] LLM 백본 바꾸려면 §5, 시장 바꾸려면 §6, hyperparameter 만지려면 §7 참조
