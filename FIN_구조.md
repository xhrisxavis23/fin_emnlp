# `/home/dgu/fin/` 디렉토리 구조 분석

## 1. Qlib 계열 (Microsoft Qlib — 알파 팩터 마이닝 프레임워크)

| 항목 | 역할 |
|---|---|
| **`qlib/`** | Microsoft Qlib 원본 저장소. 금융 시계열 모델링·팩터 엔지니어링 기반 라이브러리. RD-Agent 통합용. |
| **`01_15_new_qlib/`** | 사용자 커스텀 포크. `agent/`, `analysis/`, `coder/`, `components/` + 노트북(`0126_분석.ipynb`). `STAGE2_CHANGES.md`로 보아 활발히 확장 중인 실험 브랜치. |
| **`01_15_new_qlib copy/`** | 위 폴더의 작업 사본 (4/27 갱신). 추가된 `.pkl` 캐시 파일이 있어 별도 결과 저장용으로 분기. |

## 2. 에이전트 프레임워크 (LLM 기반 자동 연구)

| 항목 | 역할 |
|---|---|
| **`RD-Agent/`** | Microsoft RD-Agent. LLM 자율 에이전트로 팩터 마이닝/모델 최적화 (NeurIPS 2025). 다른 에이전트 시스템의 코어 의존성. |
| **`AlphaAgent/`** | KDD 2025 논문 구현. Idea/Factor/Eval 3-에이전트로 **decay-resistant alpha factor** 채굴. RD-Agent 위에 구축. |
| **`AlphaForge/`** | 팩터 마이닝 + GA 결합. 3단계 파이프라인(채굴 → 결합 → 평가)을 csi300/csi500에 적용. |
| **`FinAgent/`** | 사용자 통합 실험 워크스페이스. `RD-Agent/`, `11_14/`, `TRADE_MASTER_DATA/`, `old_prompts/`, `workspace/` 등 다수의 에이전트 시스템을 한곳에 결합. |
| **`FinAgent_4090/`** | RTX 4090 GPU 환경에 맞춘 FinAgent 작업 사본 (12/24 최신). |

## 3. LLM 서빙 / 모델

| 항목 | 역할 |
|---|---|
| **`vllm/`** | vLLM 추론 스크립트(`run_*.py`) 모음. 5–8월 배치 추론 실험. |
| **`fastervllm/`** | 경량 vLLM 배치 래퍼 (`run_vllm_batch.py/sh`). 처리 속도 최적화 시도. |
| **`LLM/`** | 로컬 모델 가중치 저장소. Meta-Llama-3-8B-Instruct 풀 체크포인트. |

## 4. 분석·출력·포트폴리오

| 항목 | 역할 |
|---|---|
| **`analysis/`** | 대용량 노트북(`0625.ipynb` ~42MB)과 섹터별(Communications, Consumer, Energy …) PDF 백테스트 결과. |
| **`portfolio/`** | 포트폴리오 구성·ML 실험. `main_all.py`, `main_ml.py`, `utils.py` + `output/` 13개 결과 서브디렉터리. |
| **`paper/`** | 논문/리포트 산출물 수집소. `config.yaml`, `analysis/`, `portfolio/`, `vllm/`, `ml/`. |
| **`prompt/`** | 프롬프트 엔지니어링 라이브러리. `feature.ipynb`(41KB), `functions_b.py`/`functions_d.py` 변형 + 실행 스크립트. |

## 5. 데이터·메타

| 항목 | 역할 |
|---|---|
| **`data/`** | 원시·전처리 금융 데이터(~1.8GB). `stock_data_2012-2023`, `dachxiu_wrds*` (WRDS 팩터), `shrout`, `real2/`. |
| **`Cowork_db/`** | 협업 DB. KRX(한국거래소) 워크플로 노트북, `Quintillion/`, `TRADE_MASTER_DATA/`, `dict_features.json`. |
| **`mlruns/`** | MLflow 실험 추적 로그. |
| **`others/`** | 잡다한 실험 (`01_ml/`, `02_agent/`, `data.ipynb`). |
| **`home/`** | 컨테이너/원격용 홈 디렉터리 미러. |

## 6. 아카이브·기타

| 항목 | 역할 |
|---|---|
| **`AlphaAgent.tar`** (1.1 GB) | AlphaAgent 백업 (1/28). |
| **`RD-Agent.tar`** (482 MB) | RD-Agent 백업 (1/28). |
| **`FinAgent_4090.tar`** (55 GB) | FinAgent_4090 백업 — 학습된 모델 가중치 포함 추정. |
| **`output.png`** | 단일 시각화 결과. |
| **`.vscode/`** | 워크스페이스 설정 (거의 비어 있음). |
| **`columns/`** | 빈 placeholder. |

---

## 전체 그림

이 워크스페이스는 **Microsoft Qlib + RD-Agent**를 기반으로, **AlphaAgent / AlphaForge / FinAgent** 등 LLM 에이전트 기반 알파 팩터 자동 발굴 시스템을 결합한 연구 환경입니다. 사용자는 (a) Qlib 포크를 여러 갈래로 분기해 실험하고, (b) GPU(4090) 최적화 배포본을 별도로 유지하며, (c) WRDS·KRX 데이터, 프롬프트 라이브러리, vLLM 추론 파이프라인을 함께 운용합니다. `paper/`·`portfolio/`·`analysis/`는 산출물 모음, `mlruns/`는 실험 추적, `*.tar`는 대용량 스냅샷 백업입니다.
