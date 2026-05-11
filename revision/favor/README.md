# FaVOR — revision/favor (pro6000 sweep snapshot)

EMNLP 2026 revision 용 FaVOR 파이프라인의 모듈화 코드 + sweep 인프라.

> **Note.** `runs/` (sweep 산출물 13 GB) 와 qlib raw data 는 git 에서 제외했습니다.
> 아래 [재현 방법](#재현-방법) 참고.

---

## 동일 실험을 다른 서버에서 돌리는 데 필요한 것 (4 개)

| # | 항목 | 비고 |
|---|---|---|
| 1 | 이 디렉토리 (`revision/favor/`) | 코드 + sweep_runner + dashboard |
| 2 | qlib CN data (2005-01 ~ 2026-01) | `~/.qlib_full/qlib_data/cn_data` — 별도 배포 (1.2 GB) |
| 3 | OpenAI API key | `.env.example` 복사해서 `.env` 작성 |
| 4 | Python env | `requirements.txt` 의 패키지 (Python 3.12 권장) |

---

## 재현 방법

```bash
# 1. clone (이 repo의 pro6000 브랜치)
git clone -b pro6000 https://github.com/xhrisxavis23/fin_emnlp.git
cd fin_emnlp/revision/favor

# 2. qlib data 배치
#    원본 .qlib.zip (598 MB) 을 받아 압축 해제
unzip /path/to/.qlib.zip -d ~/.qlib_full/
#    → ~/.qlib_full/qlib_data/cn_data/  (instruments/, calendars/, features/)

# 3. python env
conda create -n quant python=3.12 -y
conda activate quant
pip install -r requirements.txt

# 4. .env 작성
cp .env.example .env
#    OPENAI_API_KEY 채우기
echo 'FAVOR_QLIB_PROVIDER_URI_CN="$HOME/.qlib_full/qlib_data/cn_data"' >> .env

# 5. sweep 실행 (23 jobs, ~7-10 h, 3-parallel)
cd repro_logs
bash sweep_runner.sh
#    → ../runs/20YYMMDD_HHMMSS_<label>/ 로 결과 격납

# 6. dashboard 빌드
python build_report.py
#    → favor_dashboard.html (브라우저로 열기)
```

---

## 디렉토리 구조

```
revision/favor/
├── run/                 ← 파이프라인 entry + config + Stage 1-4
├── agent/               ← 가설/관측/공식 LLM agents
├── coder/               ← formula codegen + sandboxing
├── prompts/             ← LLM 프롬프트
├── schemas/             ← JSON schema 정의
├── util/                ← run_context, logger 등 (race-condition fix 포함)
├── oai/                 ← OpenAI client wrapper
├── core/, components/   ← stage 부속 모듈
│
├── run_pipeline.py                         ← 단일 run entry
├── run_pipeline_parallel_per_combo_parallel.py  ← 병렬 combo
├── run_cn.sh / run_cn_limited.sh           ← CN sweep wrapper
├── run_us.sh / run_us_limited.sh           ← US sweep wrapper
│
├── HYPERPARAMETERS.md   ← 15-section hyperparameter catalog
├── requirements.txt
├── .env.example
│
└── repro_logs/          ← sweep 인프라
    ├── sweep_runner.sh           ← 23-job 오케스트레이터
    ├── launch_run.sh             ← 단일 run launcher (concept 위치 인자 fix)
    ├── build_report.py           ← favor_dashboard.html 생성기
    ├── sweep_status.sh           ← 진행 모니터
    ├── compare.py                ← 두 run 비교 유틸
    ├── REPORT.md                 ← 초기 3-run 분석
    ├── sweep_results.csv         ← 23-job 집계 CSV
    └── favor_dashboard.html      ← 브라우저용 dashboard (생성 결과 샘플)
```

---

## sweep 23 jobs 요약 (gpt-5.4-mini · CSI 500 · 2015-2025)

Phase A (15 jobs): 5 concept × 3 horizon
Phase B (8 jobs): paper concept × 단일 lever 변경 (stop_loss, n_trials, threshold range, entry_confirm, native_strategy, combo_pass_rate)

honest IS-best selection 기준 paper Table 1 (+0.6470 OOS IR) 능가하는 4 건:

| label | OOS IR | OOS AR |
|---|---:|---:|
| `B01_stoploss_005` | **+1.0592** | **+0.3879** |
| `B04_thr_07_095` | +0.7846 | +0.2330 |
| `A15_volcomp_h20` | +0.7718 | +0.2343 |
| `A13_volcomp_h5` | +0.7139 | +0.1976 |

자세한 분석은 `repro_logs/REPORT.md` + `favor_dashboard.html` 참고.

---

## 환경 변수 (코드 수정 없이 sweep 조절)

`run/config.py` 에서 인식하는 override 들:

| env var | default | 용도 |
|---|---|---|
| `FAVOR_QLIB_PROVIDER_URI_CN` | `~/.qlib/qlib_data/cn_data` | CN qlib 데이터 경로 |
| `FAVOR_QLIB_PROVIDER_URI_US` | `~/.qlib/sh_sp500_qlib` | US qlib 데이터 경로 |
| `FAVOR_LLM_MODEL` | `gpt-4o-mini` | LLM 모델명 |
| `FAVOR_LLM_TEMPERATURE` | `0.7` | LLM temperature |
| `FAVOR_HORIZON_DAYS` | `5` | 보유 기간 |
| `FAVOR_STOP_LOSS_THRESHOLD` | `-0.10` | 손절 (None 가능) |
| `FAVOR_ENTRY_CONFIRM_RULE` | `tplus1` | T+1 진입 (`uday` 도 가능) |
| `FAVOR_NATIVE_STRATEGY` | `qlib_legacy` | backtest 모드 |
| `FAVOR_THRESHOLD_MIN/MAX` | `0.55 / 0.95` | quantile 검색 범위 |
| `FAVOR_COMBO_PASS_RATE` | (없음) | combo 필터 통과율 |
| `FAVOR_RUN_ID` | (없음) | 명시적 run_id (parallel race fix) |
