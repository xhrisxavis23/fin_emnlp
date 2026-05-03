# FaVOR Paper Review Summary

> **Paper**: FaVOR — An LLM-based Multi-Agent Framework for Financial Factor Mining
> **Number of Reviewers**: 4 (m3Gj, K743, QYCP, T5Q4)

---

## Score Summary

| Criterion | m3Gj | K743 | QYCP | T5Q4 |
|---|---|---|---|---|
| Relevance | 3 (Moderate) | 2 (Low) | 4 (High) | 4 (High) |
| Novelty | 3 (Moderate) | 2 (Low) | 2 (Low) | 2 (Low) |
| Technical Quality | 3 (Moderate) | 2 (Low) | 2 (Low) | 2 (Low) |
| Presentation | 3 (Moderate) | 3 (Moderate) | 3 (Moderate) | 2 (Low) |
| Reproducibility | 2 (Low) | 3 (Moderate) | 2 (Low) | 2 (Low) |
| Reviewer Confidence | 1 (Poor) | 3 (Moderate) | 3 (Moderate) | 4 (High) |

---

## Reviewer m3Gj

### Paper Summary
This paper proposes FaVOR, an LLM-based multi-agent framework for financial factor mining that prioritizes empirical validation of factor consistency—the alignment between a factor's formula and its economic rationale. The framework operates through three stages: First, decomposing hypotheses into observable market conditions and generating candidate factors. Second, validating each factor's distributional behavior against its intended economic meaning via an LLM agent. Third, integrating validated factors under a Monotonic Strictness principle. Experiments on CSI 500 and S&P 500 show strong performance over quantitative, ML/DL, and LLM-based baselines.

### Paper Strengths
1. The paper addresses the absence of empirical validation in current LLM-based factor mining approaches, which is a recognized gap between automated methods and traditional finance practice.
2. The decomposition–validation–integration pipeline follows a logical progression, and the separation of hypothesis atomization from factor-level verification provides a clear organizational principle for the system.
3. The paper provides concrete examples of how the validation agent distinguishes construct-aligned factors from misaligned ones using distributional evidence, which helps clarify the mechanism's operation.
4. The stage-wise ablation (Table 3) isolates the contribution of each component, showing progressive performance degradation as stages are removed.
5. The inclusion of complete prompt templates for all four agents in the appendix contributes to the transparency and potential reproducibility of the framework.

### Paper Weaknesses
1. **Validation criteria lack explicit quantitative thresholds.** Several validation criteria are described in qualitative terms without specifying concrete numerical values. Providing these thresholds would improve clarity and reproducibility.
2. **Text-table inconsistency in LLM backbone ablation.** In the Effect of LLM Backbone ablation study, the textual description does not accurately reflect the results shown in the corresponding table. This should be corrected to avoid confusion.
3. **Strategy-level statistics would strengthen the analysis.** The stepwise pattern in the cumulative return curves raises a natural question about whether performance might be concentrated in a small number of trades. Reporting trade count, per-trade return distribution, and holding period statistics would be helpful.
4. **Sub-period analysis could further enrich the evaluation.** The cross-market evaluation already provides useful evidence of generalizability. A supplementary year-by-year or regime-conditional breakdown would further strengthen the analysis.
5. **LLM judgment stability deserves further discussion.** It would be helpful to see some analysis of validation decision consistency across independent runs for the same candidate factors, as well as variance estimates for the main results.

### Questions and Suggestions for Rebuttal
1. If the authors could provide the explicit quantitative thresholds used in the validation criteria, as this would help readers better understand the mechanism.
2. It would be helpful if the authors could revisit this part for clarity.
3. We would like to learn more about the strategy-level statistics, such as total trade count, average holding period, and per-trade return distribution, if available.
4. If feasible, a year-by-year performance breakdown for both markets would be a valuable addition to the current evaluation.
5. Given the stochastic nature of LLM outputs, it would be reassuring if the authors could share some analysis on the consistency of validation decisions across independent runs.

---

## Reviewer K743

### Paper Summary
This method introduce an LLM-based multi-agent framework for quantitative factor mining. The system attempts to bridge the gap between qualitative economic hypotheses and quantitative signals by enforcing factor consistency through a three-stage pipeline: decomposition, validation, and integration. It aims to filter out spurious correlations before backtesting by using LLMs to evaluate distributional statistics of generated factors. The framework is evaluated on the CSI 500 and S&P 500 indices, demonstrating improvements over baseline methods.

### Paper Strengths
1. The framework addresses a relevant issue in automated quantitative finance, specifically the tendency of data-driven models to overfit to historical returns without underlying economic logic.
2. The separation of tasks into distinct agents provides a clear, structured pipeline that mirrors traditional quantitative research workflows.
3. The empirical results show consistent improvements across multiple evaluation metrics and distinct equity universes.

### Paper Weaknesses
1. The reliance on Monotonic Strictness assumes that valid economic factors must exhibit monotonic or linear relationships with market responses, which oversimplifies complex, non-linear market dynamics.
2. The framework restricts its inputs strictly to daily OHLCV data, which contradicts the broader claim of validating general economic hypotheses.
3. The predefined operator library heavily restricts the search space. The performance gains stem from constrained optimization rather than the LLMs reasoning capabilities.
4. Using an LLM to interpret hard statistical metrics introduces unnecessary risks of hallucination and inconsistency.
5. Transaction cost assumptions are overly simplistic and do not account for market impact, slippage, or turnover rates.
6. The ablation study removes entire pipeline stages but fails to isolate the specific contribution of the LLM in the validation phase against a simple, hard-coded statistical filter.
7. The hypothesis generation process appears heavily guided by highly specific, handcrafted prompts.
8. The integration stage multiplies combinations of factors, potentially leading to severe multicollinearity issues.

### Questions and Suggestions for Rebuttal
See above.

---

## Reviewer QYCP

### Paper Summary
This paper introduces FaVOR, a multi-agent framework that uses LLMs to automate financial factor mining. The framework aims to ensure that generated factors are not only profitable but also economically meaningful and robust. FaVOR employs a three-stage process: (1) Decomposing a high-level economic hypothesis into observable market conditions, (2) Validating candidate factors against these conditions using statistical and semantic checks, and (3) Integrating validated factors under a "Monotonic Strictness" principle before backtesting.

### Paper Strengths
- The paper addresses a timely and significant problem in quantitative finance: the tendency for LLM-based factor mining systems to produce high-performing but spurious factors that lack economic interpretability and robustness. The focus on the factor consistency is a valuable direction.
- The proposed FaVOR framework is well-structured and conceptually intuitive. The three-stage pipeline Decomposition, Validation, Integration provides a clear and systematic approach to instill more rigor into the automated factor generation process.
- The paper provides a very detailed description of the agentic framework, including the full prompts for each agent in the appendix. This level of transparency is commendable and significantly aids in understanding the methodology and potentially reproducing the work.

### Paper Weaknesses
1. The proposed framework, while well-engineered, is primarily a combination of existing concepts. The use of LLMs for hypothesis generation and code translation has been explored in prior work (e.g., AlphaAgent, R&D-Agent-Quant, which are cited). The idea of decomposing a complex problem into sub-tasks for different agents is a standard pattern in multi-agent systems. The statistical validation itself relies on basic descriptive statistics. The main contribution appears to be the specific assembly of these parts into a single pipeline, which feels more like an engineering contribution than a fundamental advance in data mining or AI.
2. The ablation in Table 3 simply shows that removing core stages of the proposed pipeline (Stage 2 and Stage 3) degrades performance. This is an expected outcome and provides little insight. A far more informative ablation would have been to replace the LLM-based validation agent A_V with a simple hard-coded script that implements the exact same heuristic checks from Section 3.3.2. This would test whether the LLM's "reasoning" provides any value beyond just executing a predefined checklist, which is a key unstated assumption of the paper.
3. The authors introduce Monotonic Strictness as a core principle for validating factor combinations, positing that performance must monotonically improve as the signal threshold σ becomes more stringent. This is an economically naive and overly strong assumption. Many robust financial strategies exhibit a sweet spot for their parameters, where performance peaks and then declines as thresholds become too strict, due to the trade-off between signal precision and the number of trading opportunities.

### Questions and Suggestions for Rebuttal
1. Why were the main results in Table 1 reported using GPT-4o when Table 4 shows that other LLMs like Gemini-2.5-pro achieve significantly better performance? Please provide the full results for Table 1 using your best-performing LLM backbone.
2. To better isolate the contribution of the LLM-based validation agent, could you run an ablation study where Stage 2 is performed by a simple, non-AI script that directly implements the statistical checks from Section 3.3.2? This would help clarify if the LLM's reasoning capability is essential or if it is merely executing a checklist.
3. The evidence for iterative improvement in Figure 5 appears weak due to high variance and a non-monotonic trend over just 5 rounds. Can you provide a more robust analysis over more rounds or runs to substantiate the claim of "progressively more effective exploration"?

---

## Reviewer T5Q4

### Paper Summary
This paper proposes FaVOR, a three-stage framework that uses LLMs to automate alpha factor mining for quantitative trading. The pipeline works as follows: (1) an LLM decomposes a high-level investment hypothesis into candidate factors with mathematical formulas; (2) each candidate factor is individually backtested for predictive power; (3) validated factors are combined into a composite signal via an integration module. The framework is evaluated on CSI 500 and S&P 500 markets using backtests over 2021 - 2025.

### Paper Strengths
- **S1.** The emphasis on empirical validation before factor integration is a sensible design choice that distinguishes this work from prior LLM-for-trading papers that tend to generate factors without checking whether they actually work individually.
- **S2.** Cross-market evaluation on both Chinese and US stock markets adds credibility.
- **S3.** The paper provides concrete factor examples (Table 2) with their mathematical formulations and IC values, which makes the approach more transparent than many black-box trading system papers.

### Paper Weaknesses
- **W1. Lack of Enough Novelty and insight to the community.** This work consists of a hypothesis generation module, factor evaluation module, factor integration module and an optimization module, which is somewhat similar to AlphaAgent and R&D-agent. Specifically, the hypothesis generation module is quite similar to AlphaAgent, which CANNOT be treated as a contribution.
- **W2. Somewhat poor writing.** Specifically in Section 3.3, criteria are overly abstract and loosely structured, lacking formal statistical tests, quantitative thresholds, and clear reproducibility, which weakens methodological rigor and empirical verifiability.
- **W3. Related Baseline comparisons are not sufficient.** Many strong, well known and open-sourced DL and RL based baselines are not discussed and compared [1], [2], [3].
- **W4. Lack of implementation details and hyperparameter sensitivity analysis.** There are many hyperparameters in Section 3.3 and Section 3.4, HOWEVER, I did not see detailed specifications of these hyperparameter values or any discussion of parameter sensitivity analysis, which weakens the paper's robustness and reproducibility.
- **W5. Lack of Cost Analysis.** In Section 3.4, the authors introduce a new factor integration module, which utilizes Cartesian products to integrate factors, which may raise serious cost concerns when the number of factors are scaled.

### References
- [1] Shuo Yu, Hongyan Xue, Xiang Ao, Feiyang Pan, Jia He, Dandan Tu, and Qing He. 2023. *Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning*. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23). Association for Computing Machinery, New York, NY, USA, 5476–5486.
- [2] Hao Shi, Weili Song, Xinting Zhang, Jiahe Shi, Cuicui Luo, Xiang Ao, Hamid Arian, and Luis Angel Seco. 2025. *AlphaForge: a framework to mine and dynamically combine formulaic alpha factors*. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence (AAAI'25/IAAI'25/EAAI'25), Vol. 39. AAAI Press, Article 1392, 12524–12532.
- [3] Zhoufan Zhu and Ke Zhu. *AlphaQCM: Alpha Discovery in Finance with Distributional Reinforcement Learning*, Forty-second International Conference on Machine Learning (ICML), 2025.

### Questions and Suggestions for Rebuttal
See in the weakness part.

---

## Overall Synthesis

### Common Strengths
- **Timely problem framing**: Addresses the real issue of missing economic meaning/consistency in LLM-based factor mining (m3Gj, K743, QYCP).
- **Clear pipeline structure**: The decomposition–validation–integration progression is intuitive and well-organized (m3Gj, K743, QYCP).
- **Reproducibility effort**: Full prompts disclosed in the appendix (m3Gj, QYCP).
- **Cross-market evaluation**: Both CSI 500 and S&P 500 (T5Q4).

### Common Weaknesses (Recurring Concerns)
1. **Lack of novelty** (K743, QYCP, T5Q4): Similar to prior work such as AlphaAgent and R&D-Agent; the contribution feels like an engineering assembly.
2. **Limitation of ablation** (m3Gj positive, K743 & QYCP negative): The ablation does not isolate whether the LLM-based validation agent provides value beyond a simple hard-coded statistical filter.
3. **Economically naive Monotonic Strictness assumption** (K743, QYCP): Ignores non-linear market dynamics and the sweet-spot phenomenon.
4. **Missing quantitative thresholds and hyperparameters** (m3Gj, T5Q4): Validation criteria are qualitative; no sensitivity analysis.
5. **Insufficient transaction cost and strategy-level statistics** (m3Gj, K743): Slippage, market impact, trade count, and holding period statistics are not reported.
6. **Insufficient baselines** (T5Q4): Strong open-source DL/RL baselines are not compared.
7. **Scalability, multicollinearity, and cost concerns** (K743, T5Q4): Cartesian-product integration may cause multicollinearity and cost explosion as factors scale.
8. **LLM judgment stability** (m3Gj, K743): No analysis of consistency or variance across independent runs.

### Suggested Rebuttal Priorities
1. **Add an ablation comparing LLM-based validation vs a hard-coded statistical filter** (key request from QYCP and K743).
2. **Specify quantitative thresholds and report hyperparameter sensitivity** (m3Gj, T5Q4).
3. **Compare against DL/RL baselines such as AlphaForge and AlphaQCM** (T5Q4).
4. **Provide strategy-level statistics + year-by-year / regime-conditional breakdowns** (m3Gj).
5. **Relax or discuss the Monotonic Strictness assumption with respect to the sweet-spot phenomenon** (K743, QYCP).
6. **Analyze run-to-run consistency of LLM outputs** (m3Gj).
7. **Refine the transaction cost model and analyze the cost of Cartesian-product integration** (K743, T5Q4).
8. **Fix text-table inconsistency and re-report Table 1 with the best-performing backbone (Gemini-2.5-pro)** (m3Gj, QYCP).
