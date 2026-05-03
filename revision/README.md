# FaVOR Revision Workspace (KDD 2026 → EMNLP 2026)

> **상태**: KDD 2026 제출본의 모든 reproducibility 자료를 추출 완료.
> **원본 보존**: `/home/dgu/fin/01_15_new_qlib/`, `/home/dgu/fin/AlphaAgent/`, `/home/dgu/fin/RD-Agent/`, `/home/dgu/fin/AlphaForge/`, `/home/dgu/fin/01_15_new_qlib copy/` 는 **CLAUDE.md 절대 규칙대로 frozen-snapshot** 으로 보존. 본 디렉토리는 모두 사본.
> **총 크기**: 772MB (원본 ~140GB+ 에서 산출물·캐시 제외)

---

## 디렉토리 맵

| 경로 | 크기 | 내용 | 원본 |
|---|---|---|---|
| **`favor/`** | 1.3MB | FaVOR 파이프라인 코드 (모듈화) — entry, agent, prompts, schemas, coder, run/ | `01_15_new_qlib/` (import 그래프 추출) |
| **`baselines/`** | 670MB | Table 1 비교 대상 (AlphaAgent, RD-Agent, AlphaForge, Qlib benchmarks, TopkDropout runners) | `AlphaAgent/`, `RD-Agent/`, `AlphaForge/`, `01_15_new_qlib/qlib/` |
| **`analysis/`** | 16MB | Table/Figure 생성용 노트북 + 집계 CSV + Excess Return / Final Result PDF | `01_15_new_qlib/analysis/` |
| **`paper_artifacts/`** | 85MB | Alpha158 베이스라인 백테스트 .pkl + MLflow run artifacts (5 hash 디렉토리) + `aggregated_stage4_results.csv` | `01_15_new_qlib/paper/` |
| **`backbone/`** | <1MB | LiteLLM proxy 설정 (Claude/Gemini 라우팅, Table 4) + 로컬 LLM (Qwen/Llama/DeepSeek) launcher | `01_15_new_qlib copy/` |
| **`runs_reference/20260209_073324/`** | 2.2MB | 페이퍼 백테스트 직전 reference run 의 spec JSON + report (1.2GB ticker_details.json 제외) | `01_15_new_qlib/runs/20260209_073324/` |
| `implementation_settings.md` | — | 원본 코드/설정/run 위치 reference 문서 | (revision 자체 문서) |
| `revision_todo.md` | — | EMNLP revision 작업 항목 (E1~E8 + 불일치 정정) | (revision 자체 문서) |

---

## Reproducibility 체크리스트 — 모두 모음 ✓

논문 결과를 처음부터 재현하려면 필요한 모든 자료가 포함되어 있습니다.

| 필요 자료 | 위치 | 상태 |
|---|---|---|
| **FaVOR 코드** (entry, agent, prompts, schemas, pipeline) | `favor/` | ✓ |
| **FaVOR 설정** (LLM, market, data split, stage hyperparam) | `favor/run/config.py` | ✓ |
| **FaVOR 모듈 정리 문서** (각 파일 역할) | `favor/favor_모듈_정리.md` | ✓ |
| **데이터 준비 스크립트** (US factor) | `favor/generate_us_factor_files.py` | ✓ |
| **데이터** (CN: Qlib `~/.qlib/qlib_data/cn_data`, US: `~/.qlib/sh_sp500_qlib`) | 외부 의존 (Qlib 표준 위치) | ⚠ 사용자 시스템에 별도 설치 필요 |
| **Baseline #1 — AlphaAgent (KDD'25)** | `baselines/alphaagent/` | ✓ |
| **Baseline #2 — RD-Agent-Quant (MS)** | `baselines/rdagent/` | ✓ |
| **Baseline #3 — Qlib benchmarks** (Alpha158/MACD/Linear/MLP/LightGBM/XGBoost/Transformer) | `baselines/qlib_benchmarks/` | ✓ |
| **Baseline #4 — TopkDropout runners** (LGBM/MLP/GATs entry script) | `baselines/qlib_topkdropout/` | ✓ |
| **Baseline 추가 후보 — AlphaForge (AAAI'25, T5Q4 리뷰어 요청)** | `baselines/alphaforge/` | ✓ |
| **Backbone 라우팅** (Claude/Gemini Table 4 — LiteLLM proxy) | `backbone/litellm_proxy.yaml` | ✓ |
| **로컬 LLM 실험** (Qwen3-8B / Llama-3.1-8B / DeepSeek-R1) | `backbone/run_*.sh` | ✓ |
| **페이퍼 reference run** (concept, hypothesis, formula, stage2/3/4 결과) | `runs_reference/20260209_073324/` | ✓ |
| **집계 CSV** (Table 1 산출 ground truth) | `analysis/{alphaagent_perf,ours_perf_full,alphaforge_exp_perf,ours_pairwise_compare}.csv`, `paper_artifacts/aggregated_stage4_results.csv` | ✓ |
| **Alpha158 베이스라인 결과 pkl** (검증용) | `paper_artifacts/alpha158_{csi500,sp500}_report_normal_1day.pkl` | ✓ |
| **MLflow run artifacts** | `paper_artifacts/{34f59ac...,912768...,dccaa4...,f5b39c...}/` (5 hash) | ✓ |
| **Table/Figure 생성 노트북** (Table 1, Figure 4, Figure 5) | `analysis/cum_return.ipynb`, `0203.ipynb`, `B_factor_lgbm_topkdropout_analysis.ipynb`, `A_stage4_event_analysis.ipynb`, `ours_full_scan.ipynb` 외 | ✓ |
| **페이퍼 누적곡선 PDF** (Figure 4) | `analysis/Excess_Return_{CN,US,CSI 500,S&P 500}.pdf` | ✓ |
| **페이퍼 최종 결과 PDF** | `analysis/Final_Result_{CN, CSI 500, S&P 500}.pdf` | ✓ |
| **분석 보조 스크립트** | `analysis/{diagnose_mdd_and_ticker_returns,export_qlib_report,run_analyzer}.py` | ✓ |
| **페이퍼 원고 PDF** | `/home/dgu/fin/FaVOR_paper.pdf` (워크스페이스 루트) | ✓ |
| **페이퍼 LaTeX 소스** | `/home/dgu/fin/FaVOR_paper_tex.zip` (워크스페이스 루트) | ✓ |
| **리뷰 요약 / Rebuttal** | `/home/dgu/fin/FaVOR_Reviews_Summary.md`, `FaVOR_Author_Responses.md` | ✓ |

### 의도적으로 제외된 항목 (디스크 절감 + 원본 read-only 참조)

| 제외 항목 | 원본 위치 | 이유 |
|---|---|---|
| 444 개 run 디렉토리 전체 | `01_15_new_qlib/runs/` (100GB) | 원본 보존, 1개 reference run 만 추출 |
| `runs/20260209_073324/specs/stage3_ticker_details.json` | (1.2GB) | 디버깅용 raw, 페이퍼 표·그림에 직접 사용 안 됨 |
| `runs/20260209_073324/data/`, `qlib_artifacts/` | (511MB) | per-iteration parquet/pkl, 재계산 가능 |
| AlphaAgent `log/`, `results/`, `pickle_cache/`, `git_ignore_folder/` | (42.6GB) | 실행 로그·캐시, 원본 참조 |
| top-level `paper/` (36GB) | `/home/dgu/fin/paper/` | 옛 산출물 collection, 핵심 자료는 `01_15_new_qlib/paper/` 에 정리되어 있음 |
| MLflow 추적 로그 | `/home/dgu/fin/mlruns/` | 원본 참조 |
| Top-level `data/` (1.8GB) | `/home/dgu/fin/data/` | WRDS·KRX 원시 데이터, 외부 데이터 소스 (재다운로드 가능) |

→ 위 항목이 필요하면 원본 절대 경로로 read-only 접근.

---

## 환경 / 의존성

> ⚠ 본 워크스페이스에는 통합된 `requirements.txt` 가 없습니다 (원본도 마찬가지). 각 베이스라인은 자기 패키지 내부의 `pyproject.toml` 에 의존성을 명시합니다.

| 의존성 명세 | 위치 |
|---|---|
| Qlib | `baselines/qlib_benchmarks/pyproject.toml`, `setup.py` |
| AlphaAgent | `baselines/alphaagent/pyproject.toml` |
| RD-Agent | `baselines/rdagent/pyproject.toml` |
| AlphaForge | `baselines/alphaforge/requirements.txt` |
| FaVOR | (`favor/` 자체엔 명세 없음 — `qlib`, `polars`, `pandas`, `optuna`, `litellm`, `openai`, `pydantic` 등을 사용. revision 시 `favor/requirements.txt` 작성 권장) |

**Python**: 3.10+ (원본은 3.9/3.10/3.12 혼재)
**OpenAI API key**: `favor/.env.example` 참조하여 `.env` 작성

---

## 빠른 재현 레시피

### 1. FaVOR 메인 파이프라인 (CSI500)
```bash
cd /home/dgu/fin/revision/favor
cp .env.example .env  # API key 입력
MARKET=cn ./run_cn.sh "Short-term mean reversion after panic selling" --outer-loop 5
```

### 2. ML/DL 베이스라인 (LightGBM + TopkDropout)
```bash
cd /home/dgu/fin/revision/favor   # PYTHONPATH 기준 (FaVOR run/ import)
MARKET=cn python ../baselines/qlib_topkdropout/run_pipeline_topkdropout_lgbm.py "<concept>" --outer-loop 5
```

### 3. AlphaAgent 베이스라인
```bash
cd /home/dgu/fin/revision/baselines/alphaagent
pip install -e .
# 시나리오 실행은 alphaagent/app/cli.py 참조
```

### 4. Backbone 변경 (Table 4 — Claude / Gemini)
```bash
# 1) LiteLLM proxy 시작
litellm --config /home/dgu/fin/revision/backbone/litellm_proxy.yaml --port 4000
# 2) FaVOR 환경변수 세팅 후 재실행
export OPENAI_BASE_URL=http://localhost:4000
export LLM_MODEL=claude-tools  # 또는 gemini-tools
cd /home/dgu/fin/revision/favor && ./run_cn.sh "<concept>"
```

### 5. 로컬 LLM (Qwen3-8B / Llama-3.1-8B / DeepSeek-R1)
```bash
# vLLM 서버 먼저 실행 후
bash /home/dgu/fin/revision/backbone/run_cn_qwen3_8b.sh "<concept>"
```

---

## Revision 작업 디렉토리 (TBD)

EMNLP revision 신규 실험 코드는 `revision/exp/{rule_validator,llm_consistency,sweetspot_analysis,...}/` 형태로 추가 예정. `revision_todo.md` 참고.

---

## 참고 문서

- `implementation_settings.md` — 원본 코드 위치 / 설정 / run 디렉토리 reference
- `revision_todo.md` — E1~E8 revision 작업 항목 + 불일치 정정 6개
- `favor/favor_모듈_정리.md` — FaVOR 모듈화 추출 상세
- `baselines/README.md` — 각 베이스라인 진입점·실행법
