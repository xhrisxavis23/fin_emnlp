# FaVOR Revision Plan — KDD 2026 → EMNLP 2026

> **Source materials**
> - 원고: `../FaVOR_paper.pdf`, `../FaVOR_paper_tex.zip`
> - 리뷰: `../FaVOR_Reviews_Summary.md`
> - Rebuttal: `../FaVOR_Author_Responses.md`
> - 워크스페이스 구조: `../FIN_구조.md`
> - 작성일: 2026-04-28
>
> **목표 학회**: EMNLP 2026
> **현 점수 요약**: Novelty 평균 2.25 / Tech 2.25 / Repro 2.25 / Pres 2.75 (10점 만점 환산 시 reject 영역)

---

## 0. Executive Summary — 무엇이 가장 위험한가

리뷰 4건을 종합하면 reject로 향하는 가장 큰 세 갈래:

1. **Novelty (3/4 리뷰어가 지적)** — "AlphaAgent / RD-Agent / 기존 multi-agent 패턴의 조립"이라는 인식. 이건 점수 1~2점짜리 평가로 직결되며, 단순한 글쓰기 수정으로는 절대 뒤집히지 않음.
2. **Ablation 의 정체성 위기 (QYCP, K743)** — "LLM validator를 단순 statistical filter로 대체했을 때 어떻게 되는가" 라는 가장 핵심적인 질문에 대한 답이 부재. 본 논문의 주장 자체("LLM의 reasoning이 가치를 더한다")가 검증되지 않은 상태.
3. **Monotonic Strictness 가정의 정당화 (K743, QYCP)** — economically naive 라는 강한 표현이 두 번 등장. rebuttal에서 "directional consistency check"라고 재정의했지만, 본문에선 그렇게 읽히지 않음.

→ **Revision 전략**: 위 세 갈래를 **새 실험 + 재구성된 framing**으로 정면돌파. Rebuttal에서 "future work"로 미룬 항목들 중 다수를 **이번 revision에서 실제로 수행**해야 함.

추가로, **KDD → EMNLP 이동은 단순한 venue 변경이 아니라 contribution 재정의**가 필요 (§1 참조).

---

## 1. Venue Shift: KDD → EMNLP의 함의

EMNLP는 NLP/언어모델 학회이다. **finance backtest 성능 자체로는 평가받지 않는다.**
재구성 방향:

| 측면 | KDD 2026 (이전) | EMNLP 2026 (목표) |
|---|---|---|
| 주된 contribution 어필 | 새 finance pipeline | **LLM agent의 structured reasoning + validation** |
| 평가 축 | 백테스트 metric (AR, IR, MDD) | LLM 판단의 신뢰성, consistency, calibration, prompt design |
| 베이스라인 | quant + DL/RL | quant + DL/RL + **LLM agent baselines (single-call, CoT, ReAct, vanilla AlphaAgent 등)** |
| Ablation 의 핵심 | stage 단위 contribution | **LLM-as-validator vs rule-based filter**, prompt 변형, model size 변형 |
| 핵심 질문 | 이 시스템이 돈을 더 잘 버는가 | **LLM이 hypothesis–factor consistency를 인간 수준으로 판단할 수 있는가, 그리고 그 판단이 downstream에 의미 있는 차이를 만드는가** |
| 주된 산출물 | trading strategy | **agent reasoning trace, validation decision dataset, 인간-LLM agreement 분석** |

### 1.1 EMNLP 어필을 위한 새 contribution 후보 (택 1~2)

- **C1. "Hypothesis-Factor Consistency"라는 새 task formulation** — LLM이 경제 가설과 정량 factor 간 의미론적 정합성을 판단하는 작업을 NLP task로 정의. 데이터셋 + 평가 지표 + 인간 라벨 비교 포함.
- **C2. LLM validator 의 reasoning trace 품질 분석** — Stage-2 validation 결정이 단순 numeric threshold check를 넘어서는지를 정성·정량적으로 검증. (이게 사실상 QYCP의 W2를 정면돌파하는 새 contribution이 됨)
- **C3. Multi-agent 분업 구조에서의 책임 분배 연구** — Decomposition / Validation / Integration 각 stage가 LLM에게 요구하는 reasoning 종류가 다름을 보이고, 각 stage별 model size sensitivity를 분석.
- **C4. LLM의 finance domain calibration** — backbone 비교를 단순 성능 표가 아니라, "어떤 모델이 어떤 reasoning에 강한가"의 NLP-스러운 분석으로 재구성.

→ **권장**: C1 + C2 를 메인으로, C3 를 부가 분석으로. C4 는 기존 Table 4 확장.

---

## 2. 리뷰어별 우려사항 → 액션 매핑

각 항목의 **[우선순위]**: 🔴 Critical (revision의 성패 결정) / 🟠 High / 🟡 Medium / 🟢 Nice-to-have

### 2.1 Reviewer m3Gj (가장 우호적, confidence 1)

| ID | 우려 | 액션 | 우선순위 |
|---|---|---|---|
| m3Gj-W1 | validation 정량 threshold 부재 | rebuttal에서 답한 모든 threshold 표를 본문 §3.3/3.4 에 명시. 별도 "Hyperparameter Specification" 표 추가 | 🟠 High |
| m3Gj-W2 | LLM backbone 표/본문 불일치 | 기술 표현 수정 + Table 1 을 best backbone (Gemini-2.5-pro)로 재 report 또는 multi-backbone 평균으로 재구성 | 🔴 Critical |
| m3Gj-W3 | strategy-level statistics 부재 | rebuttal Table (trades=2674, holding=5.6d 등)를 본문에 정식 포함 + 베이스라인 대조 | 🟠 High |
| m3Gj-W4 | year-by-year / regime breakdown | sub-period table 추가 (2021/22/23/24/25, 강세장/약세장 분리) | 🟡 Medium |
| m3Gj-W5 | LLM 판단 stability | rebuttal Variance 표 (Var=0.003~0.015)를 본문 정식 포함 + N≥5 독립 실행으로 확장 | 🟠 High |

### 2.2 Reviewer K743 (가장 적대적, score 2 위주)

| ID | 우려 | 액션 | 우선순위 |
|---|---|---|---|
| K743-W1 | Monotonic Strictness oversimplifies | "global monotonicity 가정이 아닌 directional consistency selection"이라는 rebuttal 의 핵심 메시지를 §3.4 본문에 재서술 + sweet-spot 실험 (§4의 새 실험 E3) | 🔴 Critical |
| K743-W2 | OHLCV 한정성 | "design choice"라는 정당화 + EMNLP 맥락에선 "controlled, reproducible NLP task formulation"으로 재포지셔닝 | 🟠 High |
| K743-W3 | DSL 제약이 reasoning을 가린다 | DSL 의 expressive power를 정량적으로 제시 (operator 수, AST depth, 표현 가능 factor 클래스) + DSL 없이 free-form 생성 시 비교 (실패율 등) | 🟠 High |
| K743-W4 | LLM hallucination 위험 | LLM 판단 vs human label 일치율 (새 실험 E5) | 🔴 Critical |
| K743-W5 | 거래비용 모델 단순화 | turnover, slippage proxy (e.g., 0.1%/0.2%/0.3% 민감도), market impact discussion | 🟡 Medium |
| K743-W6 | LLM vs hard-coded filter ablation 부재 | **새 실험 E1 (가장 중요)** | 🔴 Critical |
| K743-W7 | 프롬프트가 너무 handcraft | prompt sensitivity (E4), 자동 prompt search 결과 비교 | 🟡 Medium |
| K743-W8 | Cartesian product → multicollinearity | factor 간 상관관계 행렬, VIF, 효과적 자유도 (E2) | 🟠 High |

### 2.3 Reviewer QYCP (균형, score 2 위주)

| ID | 우려 | 액션 | 우선순위 |
|---|---|---|---|
| QYCP-W1 | Novelty: 기존 부품 조립 | §1의 contribution 재정의 (C1, C2). "consistency enforcement"를 메인 contribution으로 명문화 | 🔴 Critical |
| QYCP-W2/Q2 | LLM-validator vs rule-based filter ablation | **새 실험 E1** + Stage-2 결정의 "context-dependent" 측면을 정성 분석 (case study) | 🔴 Critical |
| QYCP-W3 | Monotonic Strictness 가정 | K743-W1 과 동일. rebuttal 의 "selection principle" 정의를 본문에 재서술 | 🔴 Critical |
| QYCP-Q1 | Table 1 backbone 선택 | m3Gj-W2 와 동일 | 🔴 Critical |
| QYCP-Q3 | Figure 5 변동성 | 5 round → ≥10 round 또는 다중 seed 평균 (새 실험 E6) | 🟠 High |

### 2.4 Reviewer T5Q4 (적대적, confidence 4)

| ID | 우려 | 액션 | 우선순위 |
|---|---|---|---|
| T5Q4-W1 | Novelty 부족, AlphaAgent 와 유사 | §1 재서술 + AlphaAgent / RD-Agent 와의 차이점 표 (component-wise comparison table) | 🔴 Critical |
| T5Q4-W2 | §3.3 작성 quality 낮음 | 전면 재작성: 정량 기준, 식, 알고리즘 박스 추가 | 🟠 High |
| T5Q4-W3 | DL/RL 베이스라인 부족 | **AlphaForge, AlphaQCM, KDD'23 RL alpha** 등 추가 (새 실험 E7) | 🔴 Critical |
| T5Q4-W4 | 하이퍼파라미터 sensitivity 부재 | sensitivity analysis 표 추가 (E4) | 🟠 High |
| T5Q4-W5 | Cartesian product cost | factor scaling 시 cost (시간/토큰/달러) 곡선 그림 추가 (E2) | 🟡 Medium |

---

## 3. 핵심 weakness 우선순위 (cross-cutting)

리뷰어 다수가 공통으로 지적한 항목을 우선순위 순으로:

1. **🔴 Novelty 재정의** (QYCP-W1, T5Q4-W1, K743 implicit) — Contribution 자체를 EMNLP 어필로 재구성 (§1.1)
2. **🔴 LLM validator vs rule-based filter ablation** (QYCP-W2, K743-W6, K743-W4) — 새 실험 E1
3. **🔴 Monotonic Strictness 재정당화** (K743-W1, QYCP-W3) — sweet-spot 실험 E3 + 본문 재서술
4. **🔴 DL/RL 베이스라인 추가** (T5Q4-W3) — 새 실험 E7
5. **🔴 Table 1 backbone 일관성 / 본문-표 모순** (m3Gj-W2, QYCP-Q1) — 재실험 + 글 수정
6. **🟠 정량 threshold / sensitivity 명시** (m3Gj-W1, T5Q4-W4) — 새 실험 E4
7. **🟠 Strategy-level statistics + sub-period** (m3Gj-W3, m3Gj-W4) — rebuttal 표 + 확장
8. **🟠 LLM 판단 stability + human agreement** (m3Gj-W5, K743-W4) — 새 실험 E5
9. **🟠 Multicollinearity / scaling cost** (K743-W8, T5Q4-W5) — 새 실험 E2
10. **🟡 Iterative refinement evidence 강화** (QYCP-Q3) — 새 실험 E6

---

## 4. 새 실험 목록 (Experiments)

각 실험: **목적 / 설계 / 산출물 / 예상 비용 / 우선순위**.
실험 코드는 모두 `revision/exp/EXX_*` 디렉토리에 신규 작성. 기존 코드는 복사 후 참고만 (CLAUDE.md 규칙).

### E1. LLM-Validator vs Rule-Based Filter (🔴 Critical)
- **목적**: QYCP-W2 / K743-W6 의 핵심 질문에 정면 답변. "LLM의 reasoning이 hard-coded statistical filter 대비 의미있는 가치를 더하는가?"
- **설계**:
  - Variant A (LLM): 기존 Stage-2 LLM validator
  - Variant B (Rule): §3.3.2 의 모든 정량 기준을 hard-coded Python 스크립트로 구현 (quantile bin pass rate, monotonicity τ=0.7, cross-ticker pass-rate ρ=0.5 등)
  - Variant C (Random): 동일 통계 계산하되 결정은 random (sanity check)
  - Variant D (Hybrid): rule 통과한 것만 LLM 이 추가 평가
- **데이터셋**: CSI 500 + S&P 500, 동일한 Stage-1 후보 factor pool (재현성 위해 seed 고정)
- **평가**:
  - Downstream: AR, IR, MDD, sharpe (B vs A 차이가 백테스트 성능에 얼마나 반영되는가)
  - Decision-level: A의 결정 ↔ B의 결정 confusion matrix, 일치율, A만 reject/accept 한 case 의 case study
  - Reasoning quality: A의 reasoning trace에서 단순 threshold check를 넘어선 추론이 등장한 비율 (인간 라벨 또는 LLM-judge)
- **산출물**: `revision/exp/E01_llm_vs_rule/` — 결과 CSV, confusion matrix 그림, case study 노트북
- **예상 비용**: GPU 24h + LLM API ~$200 (factor pool 재평가)
- **EMNLP 어필**: contribution C2 의 핵심 증거

### E2. Cartesian-Product Scalability & Multicollinearity (🟠 High)
- **목적**: K743-W8, T5Q4-W5
- **설계**: Stage-1 candidate factor 수를 K = {5, 10, 20, 40, 80}로 변화시키며
  - (a) Stage-3 까지 통과한 factor 조합 수 m(K)
  - (b) factor 간 pairwise correlation 분포 + VIF
  - (c) 효과적 자유도 (effective rank, 99% variance를 설명하는 PC 수)
  - (d) total cost: time, GPU, API tokens, USD
- **산출물**: scaling 곡선 그림 + multicollinearity 분포 + cost table
- **예상 비용**: 12h GPU + ~$80 API

### E3. Monotonic Strictness — Sweet-Spot Search (🔴 Critical)
- **목적**: K743-W1 / QYCP-W3 의 "sweet spot phenomenon" 우려에 직접 답
- **설계**:
  - threshold σ ∈ {q10, q20, ..., q95} (19개 지점)
  - 각 σ 에서 (i) trade 수, (ii) hit rate, (iii) AR, (iv) IR, (v) MDD 측정
  - U-shape / inverted-U / 단조 증가 어느 패턴인지 검증
  - 만약 sweet spot 존재한다면 "Monotonic Strictness"의 정의를 "directional consistency" 로 정식 재서술 + 그림으로 visualize
- **산출물**: σ-vs-metric 곡선 + 새로운 정의 박스 (Definition 3 재작성)
- **예상 비용**: 8h GPU

### E4. Hyperparameter Sensitivity (🟠 High)
- **목적**: m3Gj-W1, T5Q4-W4
- **설계**: rebuttal 표의 4개 핵심 파라미터 각각에 대해 ±50% 범위에서 grid:
  - Quantile bins ∈ {3, 5, 7, 10}
  - Monotonicity τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
  - Pass-rate ρ ∈ {0.3, 0.5, 0.7}
  - Signal quantile q ∈ {0.7, 0.8, 0.9, 0.95}
- **산출물**: 4-panel heatmap (각 파라미터 vs IR), 안정 영역 박스
- **예상 비용**: 16h GPU + ~$120 API

### E5. LLM Decision Stability + Human Agreement (🟠 High)
- **목적**: m3Gj-W5, K743-W4
- **설계**:
  - **Stability**: 동일 후보 factor 200개에 대해 N=10 독립 실행, 결정 일치율 (Cohen's κ) 측정
  - **Human agreement**: 50개 factor 에 대해 도메인 전문가 (저자 1~2명) 라벨 vs LLM 결정 비교, agreement matrix
  - **Cross-model**: GPT-4o vs Gemini-2.5-pro vs Claude-4.5-sonnet 의 모델간 일치율
- **산출물**: stability matrix, human-LLM agreement table, model-model κ 표
- **예상 비용**: ~$300 API + 사람 라벨링 시간 ~8h
- **EMNLP 어필**: contribution C2 의 핵심 증거

### E6. Iterative Refinement — Robust Evidence (🟠 High)
- **목적**: QYCP-Q3 (Figure 5 5 round → 변동성 큼)
- **설계**: outer-loop iteration 을 5 → **15 round + 5 seed** 로 확장. iteration-vs-best-IC 곡선의 평균±std band 그림
- **산출물**: 새 Figure 5 (with confidence band), monotonic 성격에 대한 hedged 서술
- **예상 비용**: 20h GPU + ~$250 API

### E7. DL/RL Baselines (🔴 Critical)
- **목적**: T5Q4-W3
- **추가 베이스라인**:
  - **AlphaForge** (AAAI'25) — `../AlphaForge/` 디렉토리에 코드 있음 → revision/exp/E07_baselines/AlphaForge_run/ 에 복사 후 동일 train/test split 으로 재실행
  - **AlphaQCM** (ICML'25) — 공개 코드 있을 시 reproduce
  - **Yu et al. KDD'23 RL alpha** — 공개 코드 reproduce
  - **GP / DSO / 단순 RL** — `../AlphaForge/exp_*_calc_result.ipynb` 에 baseline 결과 이미 있을 가능성 → 재활용 (단, 동일 split 인지 확인 필수)
- **주의**: rebuttal에서 "RL은 다른 search formulation"이라 빠뜨렸지만, 이 변명은 EMNLP 리뷰어한텐 안 통할 가능성 높음 → 그냥 비교
- **산출물**: 확장된 Table 1 (CSI 500 + S&P 500 모두에서 +5~7개 베이스라인)
- **예상 비용**: 30h GPU (baseline reproduce 가 가장 시간 잡아먹음)

### E8. Multi-Backbone Re-evaluation of Table 1 (🔴 Critical)
- **목적**: m3Gj-W2, QYCP-Q1
- **설계**: Table 1 을 단일 GPT-4o 결과가 아닌 **3개 backbone 평균±std** 또는 **best-of-3** 로 재실험 (또는 둘 다 보고)
- **산출물**: 새 Table 1 (multi-backbone). 본문 narrative 도 backbone-agnostic 으로 재작성
- **예상 비용**: 24h GPU + ~$400 API (CSI500 + S&P500 × 3 backbone)

### E9. (Optional) AlphaAgent / RD-Agent 와의 component-wise diff (🟡 Medium)
- **목적**: T5Q4-W1 의 "AlphaAgent 와 유사" 인식 정면 반박
- **설계**: 표 하나로 — Hypothesis gen / Validation / Integration / Optimization 4축 × {AlphaAgent, RD-Agent-Quant, FaVOR} 비교, 각 cell 에 한 줄 차이점
- **산출물**: 본문 §2 또는 §3 도입부 표
- **예상 비용**: 글 작업만, 0 compute

---

## 5. 글쓰기·구조 revision 항목

| ID | 위치 | 작업 | 우선순위 |
|---|---|---|---|
| W-01 | §1 Introduction | Contribution 재서술 — C1 (Hypothesis-Factor Consistency task formulation) + C2 (LLM validator의 reasoning trace 분석) 메인으로 | 🔴 Critical |
| W-02 | §2 Related Work | AlphaAgent / RD-Agent / AlphaForge / AlphaQCM 차이점 표 (E9 산출물) | 🔴 Critical |
| W-03 | §3.3 Validation | T5Q4-W2: 정량 기준, 식, 알고리즘 박스 추가. rebuttal threshold 표를 본문에 정식 등재 | 🟠 High |
| W-04 | §3.4 Integration | K743-W1 / QYCP-W3: "Monotonic Strictness" 재정의 — global assumption 이 아닌 selection principle 임을 명시. E3 결과 반영 | 🔴 Critical |
| W-05 | §4 Experiments | E1, E5, E7 결과 통합. Table 1 multi-backbone (E8). | 🔴 Critical |
| W-06 | §4 LLM Backbone Ablation | m3Gj-W2: 본문-표 모순 수정 ("strongest overall" → "best risk-return balance" 등 명확화) | 🟠 High |
| W-07 | §4 Strategy stats | rebuttal Table (trades, holding, profit factor) 본문 등재 | 🟠 High |
| W-08 | §4 Sub-period | year-by-year breakdown 표 추가 | 🟡 Medium |
| W-09 | §5 Discussion / Limitations | OHLCV 한정성, RL 미포함 사유, 거래비용 단순화 등을 솔직히 명시 (limitation 섹션 강화) | 🟠 High |
| W-10 | Appendix | 모든 prompt + 모든 hyperparameter + 모든 split 정확히 기재 (reproducibility 점수가 평균 2.25 임을 기억) | 🟠 High |
| W-11 | Title / Abstract | EMNLP 어필 — finance-only 어조보다 "LLM agent reasoning + structured validation" 어조로 일부 재구성 | 🟠 High |

---

## 6. EMNLP 특화 추가 작업

EMNLP 리뷰어가 finance 베이스라인을 모를 가능성이 높음. 다음을 추가로 고려:

1. **NLP 평가 metric 추가**:
   - Reasoning trace 품질 (LLM-as-judge 또는 expert label)
   - Validation 결정의 calibration (ECE, reliability diagram)
   - Decomposition 결과의 다양성 (semantic distinctness, BERTScore 분산)
2. **Reproducibility checklist** (EMNLP 양식) 모든 항목 응답.
3. **Ethics / broader impact** 절: financial market manipulation 가능성, risk disclosure 등 EMNLP 가 요구하는 형태로 작성.
4. **데이터셋 공개**: validation decision dataset (factor → human label → LLM decision) 을 부속 데이터셋으로 공개 — 이게 자체로 EMNLP-relevant contribution 이 될 수 있음.

---

## 7. 작업 디렉토리 구조 (권고)

```
revision/
├── revision_todo.md              # ← 이 문서
├── notes/                        # 분석/회의 메모
│   ├── reviewer_response_strategy.md
│   ├── novelty_repositioning.md
│   └── emnlp_camera_ready_checklist.md
├── exp/                          # 새 실험 (모두 신규 작성)
│   ├── E01_llm_vs_rule/
│   ├── E02_scalability/
│   ├── E03_strictness_sweetspot/
│   ├── E04_sensitivity/
│   ├── E05_stability_human_agreement/
│   ├── E06_iterative_refinement/
│   ├── E07_baselines/
│   │   ├── alphaforge_run/      # ../AlphaForge/ 복사 후 참고
│   │   ├── alphaqcm_run/
│   │   └── kdd23_rl_run/
│   ├── E08_multi_backbone/
│   └── E09_component_diff_table/
├── data/                         # 새 라벨/스냅샷 (원본은 ../data/ 그대로)
│   └── human_labels/             # E5 사람 라벨링 결과
├── results/                      # 모든 실험 산출물
│   └── (E01 ~ E09 별 하위 폴더)
├── figures/                      # 논문용 그림 신규 생성
└── paper_emnlp/                  # EMNLP 제출용 새 tex 트리
    ├── main.tex                  # ../FaVOR_paper_tex.zip 풀어 복사 후 수정
    ├── sections/
    └── figures/
```

> 위 구조는 권고이며, 실제 폴더 생성은 각 실험 시작 시점에 진행. **기존 디렉토리는 절대 수정 안 함** (CLAUDE.md 규칙).

---

## 8. 실행 순서 (제안)

### Phase 1 — 즉시 시작 (Week 1~2): Critical 만 처리
1. **E1** (LLM vs rule) — 가장 큰 reject 사유 직접 답변
2. **E7** (DL/RL baselines) — AlphaForge 부터 시작 (코드 이미 있음)
3. **E8** (multi-backbone Table 1) — 글 모순 해결
4. **E3** (sweet-spot) — Monotonic Strictness 재서술의 근거
5. **W-01, W-02, W-04** 글쓰기 시작 — contribution 재정의

### Phase 2 — Week 3~4: High priority
6. **E5** (stability + human agreement) — EMNLP 어필 핵심
7. **E4** (hyperparameter sensitivity)
8. **E2** (scalability)
9. **E6** (iterative refinement 확장)
10. **W-03, W-05, W-06, W-09, W-10, W-11**

### Phase 3 — Week 5~6: 마감 전 마무리
11. EMNLP 양식으로 paper_emnlp/ 빌드
12. Reproducibility checklist + ethics
13. 모든 표/그림 최종화
14. 내부 review 1~2회

---

## 9. 위험 요소 / 가설

- **시간 위험**: E1 + E5 + E7 만 해도 GPU 60h+, API $1000+ 예상. EMNLP 마감 전 끝낼 수 있는지 별도 계산 필요.
- **결과 위험**:
  - E1 에서 LLM ≈ rule 결과가 나오면 contribution 자체가 흔들림 → 그 경우 contribution 을 "task formulation + dataset" (C1 메인) 으로 더 옮겨야 함.
  - E3 에서 명백한 sweet-spot 이 발견되면 본문 §3.4 의 정의를 새로 써야 함.
- **EMNLP venue fit 위험**: NLP 리뷰어가 finance backtest 평가를 어색해 할 수 있음 → §1, §4 에서 LLM-centric framing 의 강도가 결정적.
- **저자 응답 일관성**: 일부 항목(예: K743-W6 의 "rule-based filter 비교는 future work")을 rebuttal 에선 미뤘으므로, revision 에선 이를 실제로 수행했음을 명확히 보여야 함 — 안 그러면 같은 reviewer 가 다시 reject 할 위험.

---

## 10. 다음 액션 (구체)

> 사용자가 진행 승인 시 다음 순서로 작업.

1. ✅ `revision/` 폴더 생성 — **완료**
2. ✅ `revision/revision_todo.md` 작성 — **완료**
3. ⬜ `revision/notes/novelty_repositioning.md` 작성 (C1, C2 contribution 정식 안)
4. ⬜ `revision/notes/reviewer_response_strategy.md` 작성 (각 weakness ↔ revision 항목 매핑 표)
5. ⬜ `FaVOR_paper_tex.zip` 압축 해제 → `revision/paper_emnlp/` 에 사본
6. ⬜ E1 실험 디자인 문서 (`revision/exp/E01_llm_vs_rule/DESIGN.md`) 작성
7. ⬜ E7 — AlphaForge 결과 복원 가능 여부 확인 (`../AlphaForge/exp_AFF_calc_result.ipynb` 점검)

---

## Appendix A. 점수가 어디까지 올라야 accept 인가 (추정)

EMNLP 평균적으로 borderline accept = 3.5 / 5 수준. 현재 평균이 (Novelty 2.25 + Tech 2.25 + Pres 2.75 + Repro 2.25) / 4 ≈ 2.4. 즉 **모든 축에서 +1 이상** 끌어올려야 borderline. 현실적으로:

- Novelty 2.25 → 3.5 (C1, C2 재정의 + E1, E5 결과)
- Tech 2.25 → 3.5 (E1, E3, E7 의 정량 결과 + W-03, W-04 글)
- Repro 2.25 → 4.0 (W-10 + 데이터셋 공개)
- Pres 2.75 → 3.5 (W-01~W-11 전반)

이 조합이 달성되면 EMNLP main 으로 borderline accept 권역 진입 가능.
