# FaVOR Paper — Author Responses (Rebuttal)

> **Paper**: FaVOR — An LLM-based Multi-Agent Framework for Financial Factor Mining
> **Number of Reviewers**: 4 (m3Gj, K743, QYCP, T5Q4)

---

## Reviewer m3Gj

We thank the reviewer for the insightful comment.

### [W1, Q1]
We agree that the quantitative validation criteria should be stated more explicitly. In the revision, we will report all thresholds used in the pipeline.

Specifically:
- **Stage 2:** Factors are partitioned into five quantile bins; those with fewer than three valid bins are discarded. Refinement is limited to three iterations.
- **Stage 3:** Strictness is evaluated at q50, q70, q90 with a monotonicity threshold of 0.7. A combination is **PASS** if cross-ticker pass rate ≥50%, and highly generalizable if ≥70%. It is forwarded if ticker-level pass ≥50% or cross-ticker monotonic improvement is observed.
- Combined-signal quantile is 0.9, trigger window $k=1,\dots,5$, stop-loss −10%.
- Parallel Stage 3 explores thresholds in [0.55, 0.95] with step 0.05 (50 trials).

We will also report the train/validation/test split and transaction costs.

### [W2, Q2]
We clarify that *"GPT-4o achieves the strongest overall performance"* refers to a **risk-return balanced evaluation**, not dominance in each metric. While Claude-4.5-sonnet and Gemini-2.5-pro show higher AR and IR, GPT-4o achieves the most stable MDD, indicating stronger downside control. We will revise the wording accordingly.

### [W3, Q3]
We agree that strategy-level statistics are necessary to confirm that performance is not driven by a few extreme trades.

The table summarizes trade-level distribution and confirms that performance is broadly distributed across many trades rather than driven by a small number of extreme outcomes.

**Strategy-Level Statistics**

| Metric | Value |
|---|---|
| Trades | 2,674 |
| Holding period | 5.60 days |
| Mean return | 0.61% |
| Profit factor | 1.35 |
| p10 | −5.92% |
| p90 | +7.18% |

### [W4, Q4]
We acknowledge that year-by-year or regime-based analysis could add insight. However, our fixed split (**2015–2019 / 2020 / 2021–2025**) already spans both stable and volatile market environments, indicating that the framework is not tied to a particular year or regime. We consider finer regime analysis future work.

### [W5, Q5]
We report variance across independent runs. The low variance across independent runs indicates that validation outcomes are consistent, supporting the stability of LLM-based decisions.

**Out-of-Sample Performance Summary**

| Metric | Var |
|---|---|
| IR | 0.004 |
| Return | 0.003 |
| MDD | 0.015 |
| Turnover | 0.005 |

---

## Reviewer K743

We thank the reviewer for the insightful comment.

### [W1]
Monotonic Strictness is not intended to assume a linear or universally monotonic relationship between factors and market outcomes. It serves as a **conservative selection mechanism** applied after decomposition and validation (Stages 1–2), testing whether stricter conditions remain directionally consistent with the intended economic rationale. Thus, it functions as an empirical consistency check rather than a global monotonicity assumption.

### [W2]
Restricting inputs to daily OHLCV data is a **deliberate design choice**. The framework targets **market-observable economic hypotheses** whose mechanisms are reflected in price and volume. Hypotheses are decomposed into **observable market conditions** measurable from OHLCV, enabling **reproducible, data-grounded validation** without external information.

### [W3]
The operator library is not merely restrictive but ensures **executability and semantic validity**. Without it, outputs may be non-executable. The DSL remains **expressive**, covering time-series, cross-sectional, smoothing, regression, and logical operations, providing a **rich yet controlled search space**.

### [W4]
The LLM operates in a **structured validation setting**, not open-ended reasoning. It evaluates quantile-based statistics under **predefined criteria**. Decisions are grounded in **numerical evidence and consistency checks**, with prompts enforcing deterministic structure.

### [W5]
Transaction cost assumptions are standardized across all methods, ensuring fair comparison. Market impact and slippage are **execution-dependent** and cannot be reliably modeled without a detailed simulator. Our focus is **reproducible evaluation**.

### [W6]
We agree that comparison with rule-based filters would be valuable but is beyond scope. Stage 2 is not reducible to fixed rules; it requires **context-dependent interpretation** (e.g., evidence selection, contradiction checks), evaluating **semantic-statistical alignment** beyond simple thresholds.

### [W7]
Prompts guide hypothesis generation but do not encode specific ideas. They define **roles and constraints** to ensure reproducibility and enforce separation between stages.

### [W8]
The framework first decomposes a hypothesis into distinct observable conditions, so the factor combinations capture different aspects of a market state rather than redundant signals. Thus, the integration stage does not create multicollinearity concerns.

---

## Reviewer QYCP

We thank the reviewer for the insightful comment.

### [W1]
Our contribution is not hypothesis generation itself, but enforcing **hypothesis–factor consistency** through validation stages. FaVOR decomposes hypotheses into observable conditions, evaluates factors using **distributional evidence** (e.g., monotonicity and cross-ticker consistency), and recombines only validated factors. This addresses a consistency failure mode not explicitly handled in prior LLM-based factor mining frameworks.

### [W2 & Q2]
The current ablation evaluates stage-level effects rather than isolating the contribution of the **Stage-2 LLM validator**. Stage-2 is not equivalent to executing a fixed checklist. While quantile-based OHLCV statistics are computed deterministically, the decision requires **context-dependent interpretation** of distributional evidence, including identifying supporting features and inconsistencies with the intended economic rationale.

Simpler rule-based filters (e.g., monotonicity or effect-size thresholds) were explored but found insufficiently expressive. Direct comparison with rule-based validation is left as future work.

### [W3]
Monotonic Strictness is not intended as a universal assumption that performance must strictly increase as threshold σ becomes more stringent. Rather, it serves as a **selection principle**: after decomposition and validation (Stages 1–2), combinations more consistent with the intended economic state are expected, on average, to yield **higher win rates and fewer false positives** under stricter filtering.

Its role is to evaluate **directional consistency** between economic rationale and empirical behavior, rather than to impose a global monotonicity assumption.

### [Q1]
Although Gemini-2.5-pro and Claude-4.5-sonnet show higher AR and IR, backbone selection considered overall **risk–return balance** including MDD. GPT-4o shows the most stable MDD in Table 4 and was therefore used for Table 1. We will clarify this criterion.

### [Q3]
Figure 5 is intended to illustrate the **iterative refinement process** rather than to demonstrate guaranteed monotonic performance improvement across outer-loop iterations. We agree that, based on five iterations, the current wording (e.g., "progressively more effective exploration") may sound stronger than intended. In the revision, we will soften this phrasing to avoid implying systematic monotonic gains.

---

## Reviewer T5Q4

We thank the reviewer for the insightful comment.

### [W1]
Our contribution is not hypothesis generation itself, but enforcing **hypothesis–factor consistency** through validation stages. FaVOR decomposes hypotheses into observable conditions, evaluates candidate factors using monotonicity and cross-ticker consistency criteria, and the Integration stage recombines only validated factors. This addresses the hypothesis–factor consistency gap not explicitly addressed in prior LLM-based factor mining work.

### [W2 & W4]
We agree that Sections 3.3–3.4 may make quantitative criteria and parameter roles appear insufficiently explicit. However, the framework already specifies fixed thresholds and reproducible evaluation settings. In Section 3.3, Stage 2 uses a 5-quantile partition, and Stage 3 applies monotonicity threshold 0.7 and cross-ticker pass-rate 0.5. In Section 3.4, the Integration stage applies combined-signal quantile 0.9 under predefined train/validation/test splits (2015–2019 / 2020 / 2021–2025) with consistent transaction cost assumptions.

The framework does **not rely on extensive hyperparameter tuning**, as most parameters are fixed design parameters:

| Stage | Parameter | Value |
|---|---|---|
| 2 | Quantile bins | 5 |
| 3 | Monotonicity τ | 0.7 |
| 3 | Pass-rate ρ | 0.5 |
| 4 | Signal quantile q | 0.9 |

Configuration snapshots and stage-wise artifacts ensure reproducibility. We will clarify parameter roles and explicitly report thresholds in Sections 3.3–3.4.

### [W3]
We agree that strong baselines are important. The paper already includes widely used benchmarks in quantitative finance, including quantitative signals (Alpha158, MACD), ML/DL models (Linear, MLP, LightGBM, XGBoost, Transformer), and recent LLM-based frameworks (R&D-Agent-Quant, AlphaAgent).

RL-based factor discovery methods were not included because they optimize return under a different search formulation, whereas our objective is validation-centered factor construction focusing on hypothesis–factor consistency.

### [W5]
The Cartesian product does **not imply unrestricted combinatorial expansion**. Hypotheses are decomposed into observable conditions, only validated factors are retained, and each observation generates at most 2–3 candidate formulas. Integration operates on reduced validated sets using a shared threshold σ, limiting search space growth while testing whether jointly satisfied conditions better capture the intended economic state.
