## 목표
- AlphaAgent가 생성한 **가설/수식/팩터 코드**를 stage2 → stage3 파이프라인에 연결해서,
  1) 수식 단위의 Pass/Fail 판정이 재현 가능하게 수행되고
  2) Qlib backtest 성능 지표(예: IC/ICIR/Annualized Return/IR/MDD)가 함께 저장/비교되도록 만든다.

## 현재 확인된 저장 위치/산출물(중요)
- **팩터 코드/수식/결과 값(팩터 단위 워크스페이스)**
  - 위치: `git_ignore_folder/RD-Agent_workspace/<uuid>/`
  - 수식/이름: `factor.py` 안의 `expr = "..."`, `name = "..."` (예: `git_ignore_folder/RD-Agent_workspace/.../factor.py`)
  - 실행 산출: `result.h5` (팩터 값)
- **이번 run의 팩터명 ↔ 워크스페이스 경로 매핑**
  - 위치: `log/<run>/d/<pid>/common_logs.log` 안의 `"evolving code workspace: File Factor[...]: <path>"`
- **Qlib backtest 결과**
  - AlphaAgent 로그 텍스트: `log/<run>/ef/<pid>/common_logs.log` 안의 `"Backtesting results:"` 블록
  - 정량 지표 CSV(export): `python export.py --log-dir log/<run> --out log/<run>/metrics.csv`

## 선행 조건(재현성)
- Qlib CN 데이터 경로 일관화
  - 실제 데이터: `~/.qlib/qlib_data/sh_sp500_qlib`
  - 설정 파일에서 `provider_uri`는 위 경로로 통일
- 캘린더 마지막 날짜 이슈 회피
  - CN day calendar 마지막이 `2025-12-29`라서 backtest `end_time`을 마지막 날짜로 두면 IndexError가 날 수 있음
  - 현재 안전값: `end_time = 2025-12-26` (템플릿들에 반영 필요)

## 구체화된 작업 항목
### A. stage2.py / stage3.py “입출력 계약” 정의
1) stage2.py 입력(최소)
   - 팩터 메타: `factor_name`, `factor_expression`, `factor_workspace_path`
   - 팩터 코드: `factor.py` (원본 그대로 보관)
   - 실행 결과: `result.h5` (팩터 값)
2) stage2.py 출력(권장)
   - `stage2_result.json`
     - pass/fail
     - fail 사유(파싱/런타임/결측률/극단치/중복 등)
     - 정규화된 수식(파서 적용 후)
     - 통계 요약(결측률, 분포, 상수 여부, 날짜 커버리지 등)
3) stage3.py 입력
   - stage2_result + (pass인 항목만) backtest 결과 요약(혹은 backtest 재실행 옵션)
4) stage3.py 출력
   - `stage3_result.json`
     - 최종 채택/기각 + 사유
     - 성능 요약(IC/ICIR/RankIC/RankICIR + 수익/리스크)
     - “검증에 걸러졌지만 성능은 좋았다/나빴다” 케이스 분류

### B. AlphaAgent 결과를 stage2/3로 “수집(export)”하는 방법 구현
1) 한 번의 AlphaAgent run(log/<run>)에서 아래를 수집
   - `ef/**/common_logs.log`에서 loop별 `"Backtesting results:"` 블록(지표)
   - `d/**/common_logs.log`에서 팩터명 ↔ `RD-Agent_workspace/<uuid>` 경로 매핑
   - 각 `<uuid>/factor.py`에서 `expr`, `name` 추출
2) 수집 결과를 중간 포맷으로 저장(예: `log/<run>/artifacts/factors.jsonl`)
   - row 예시 키: `loop`, `factor_name`, `expr`, `workspace_path`, `factor_py_path`, `result_h5_path`, `metrics`

### B-1. Stage2/Stage3 공통 실행 원칙(핵심 제약)
- Stage2에서는 **수익/IC/Sharpe 등 예측력 지표를 절대 사용하지 않는다**(PASS/FAIL 판단 금지, 기록도 금지 권장).
- Stage2/Stage3 모두 **IS만 사용하고 OOS(test)는 봉인**한다.
  - 최소 분리: Stage2/Stage3는 `segments.train` + `segments.valid`만 사용
  - `segments.test`는 Stage4/최종 리포팅용으로만 사용(판정에 사용 금지)
- 분위수/strictness 분할은 드리프트/스케일 문제를 피하기 위해 **날짜별 cross-sectional 방식**을 기본으로 한다.
  - 예: 각 날짜 t의 종목 단면에서 score 분위수 산출 → 버킷 라벨 부여
  - (선택) time-series 분위수는 별도 실험으로만 허용

### C. Pass/Fail 기준을 명시(예시 초안)
1) 코드/실행
   - `factor.py` 실행 성공 + `result.h5` 생성
2) 데이터 품질
   - 결측률 상한(예: < 30%), 상수/거의 상수 제거
   - extreme outlier 비율 제한(예: winsorize 후에도 분산 0이면 fail)
3) 경제적/논리적 필터(선택)
   - look-ahead / label leakage 의심 패턴 탐지(예: 미래값 Ref(-k) 사용 등)
4) 중복/유사도(선택)
   - 기존 SOTA factor와 상관이 너무 높으면 drop(현재 코드엔 dedup 비활성화되어 있으므로 기준/정책 결정 필요)

### C-0. Stage2 PASS/FAIL(관측 구현 검증) 정의
- 입력: `factor.py`의 `expr` + `result.h5`(팩터 값) + raw OHLCV(`daily_pv.h5`)
- 절차:
  1) IS 구간(= train/valid)만 필터링
  2) score(팩터 값) 기준으로 분위수 버킷 분할(기본: 날짜별 cross-sectional)
  3) 버킷별 raw 파생치 벡터(mag/dir/vol/pos 등)의 분포를 관측
  4) polarity 방향과 일치하게 “버킷이 강해질수록 분포가 단조 이동”하면 PASS, 아니면 FAIL
- 주의: Stage2는 “알파”가 아니라 “관측 구현(측정기 정상)”만 본다.

### C-1. Stage3에서 “오탐(S2)” 정의를 Stage2와 분리(tautology 방지)
문제: Stage2가 `dir/pos/vol/mag` 등 raw 분포 이동을 보고 PASS를 줬는데, Stage3에서 오탐을 그 중 하나(예: `dir>0`)로 정의하면
결국 “Stage2를 threshold만 바꿔 반복 측정”하는 tautology가 되기 쉽다.

해결(원칙): Stage3 오탐은 **단일 raw 파생치가 아니라, 사전에 고정된 합성 규칙**으로 정의한다.
- Stage2: raw 분포 **벡터 전체**(mag/dir/vol/pos 등)의 “일관된 이동”이 있는지 PASS/FAIL
- Stage3: 이벤트 내부에서 “raw-consistency가 깨진 케이스”를 오탐으로 정의하되, 다음 중 한 방식으로 고정
  1) 합성 규칙(사전 고정): 예) polarity=하방압력 → FAIL(오탐) = `(dir>0) AND (pos>0.7)` 같은 다중 조건
  2) raw-consistency score(사전 고정):
     - Stage2에서 사용한 여러 raw 파생치를 이용해 **일치 점수**(sign一致, rank 합, z-score 합 등)로 스칼라 점수를 만들고
       score가 임계 미만이면 오탐으로 정의
중요: 이 오탐 규칙은 “팩터값을 다시 말하는 규칙”이 아니어야 한다(팩터값/분위수 자체를 재사용 금지).

### D. strictness 단조성 검증의 구간 분리(과적합 방지)
문제: strictness grid(50→30→…→2)는 사실상 “모델 선택”인데, 같은 구간(IS)에서 단조성이 잘 나오도록 ladder를 조정하면 과적합이 가능하다.

해결(최소 비용 3-way split 고정):
1) **train**:
   - Stage2 실행(관측 구현 PASS/FAIL)
   - Stage3 오탐 정의(합성 규칙 / raw-consistency score) 고정
   - strictness ladder(예: top 50/30/20/10/5/2%) 고정
2) **valid**:
   - Stage3 단조성 PASS/FAIL 판정(strictness↑ ⇒ 오탐↓)
3) **test**:
   - 끝까지 봉인(Stage4/최종 리포팅에서만 참고; Stage2/3 판단에 사용 금지)

추가 주의(구현 디테일):
- strictness↑는 보통 “threshold 수치↑”가 아니라 **선택 비율↓(top p%에서 p 감소)**로 구현된다.
  - 단조성 체크 방향(↑/↓)을 p-grid 정의와 함께 고정해야 혼동이 없다.
- Sharpe/IC 등 성과 지표는 **보조로만 기록**하고 Stage3 PASS/FAIL에는 미사용한다.

### D. “성능은 뽑되, 검증 필터에 걸리는지” 실험 설계
1) 동일 run에서:
   - (1) 검증 필터 없이: 성능 지표만 추출(`export.py`)
   - (2) 검증 필터 적용: stage2/stage3로 pass/fail 라벨링
2) 비교 리포트 생성
   - pass vs fail 그룹의 평균 성능 비교
   - “fail인데 성능 좋음” Top-N 케이스 따로 저장

### E. 코드 연결(구현 단위)
1) stage2.py를 “팩터 워크스페이스 경로 입력”을 받는 CLI로 변경/추가
   - 예: `python stage2.py --factor-ws git_ignore_folder/RD-Agent_workspace/<uuid> --out ...`
2) stage3.py는 stage2 결과 + metrics를 입력으로 받아 최종 판단
3) alphaagent 실행 후 자동으로 stage2→stage3를 돌리는 래퍼 스크립트(예: `run_pipeline.py`) 추가

## 완료(acceptance) 조건
- `alphaagent mine ...` 실행 1회 → `log/<run>/metrics.csv` 생성(export.py)
- 같은 run에서 팩터별로 `stage2_result.json` / `stage3_result.json` 생성
- Stage3는 `train/valid/test` 3-way split을 따르고, PASS/FAIL은 **valid에서만** 결정(test 봉인)
- 특정 팩터(예: Downward_Deviation_Volume_Surge_Factor)의
  - 수식/코드 위치(`RD-Agent_workspace/<uuid>/factor.py`)
  - pass/fail 결과
  - backtest 지표
  를 한 파일(리포트/테이블)에서 추적 가능




### Q1. “이건 그냥 feature engineering 아닌가요?”
① 방어 논리
Feature engineering은 예측력을 높이기 위한 표현 변환
우리는 예측을 전혀 보지 않고,
관측 조건이 강화될수록 분포가 단조적으로 변하는지만 본다
즉, 목적 함수가 다르다

핵심 문장
Feature engineering optimizes representation for prediction, whereas our framework evaluates whether a formula implements a hypothesis as a monotonic observational condition.

② 사전 차단 실험 (강력 추천)
같은 수식을 두고

(A) IC/Sharpe 기준 필터
(B) Stage2 monotonicity 기준 필터

선정 결과가 얼마나 다른지 비교 테이블 or figure

👉 “같은 수식, 다른 평가 축”을 보여주면 이 질문은 끝난다.


### Q4. “프롬프트를 잘 쓰면 해결되는 문제 아닌가요?”
① 방어 논리
문제는 생성 능력이 아니라 검증 부재
LLM은 관측 단조성이라는 구조적 제약을 내재하지 않음

핵심 문장
Prompting can improve surface-level alignment, but cannot enforce data-level observational constraints.

② 사전 차단 실험 (있으면 매우 강함)
동일 가설에 대해:

vanilla prompt
chain-of-thought prompt
constraint-heavy prompt

FAIL 비율이 크게 줄지 않음을 보여줌


### Q6. “Overfitting을 줄인다는 증거가 있나요?”
① 방어 논리
Stage2/3은 OOS를 전혀 사용하지 않음
구조적으로 overfitting이 일어나기 어려운 파이프라인

핵심 문장
Our pipeline structurally prevents overfitting by separating observational validation from performance evaluation.

② 사전 차단 실험

Stage2 PASS / FAIL 그룹의
IS-OOS 성능 gap 비교

FAIL 그룹이 gap이 더 큼을 보여주면 매우 설득력 있음