"""
================================================================================
STAGE 1: Formula Generation
Hypothesis setting → observation decomposition → formula construction → formula value computation
================================================================================

[Purpose of Stage 1]
Transform a finance hypothesis into a form that can be validated on data.
The hypothesis is decomposed into observable conditions (observations), and each observation is
implemented via formulas that use OHLCV data only.

[Key principles]
- The hypothesis does not directly define a prediction rule or a trading strategy.
- The hypothesis defines an event as a conjunction (AND) of observable conditions.
- A formula is an implementation for detecting a condition, not a factor/alpha.
- Use OHLCV raw data only; do not introduce new indicators or exogenous signals.

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import polars as pl

from agent.formula_agent import FormulaAgent
from agent.hypothesis_agent import HypothesisAgent
from agent.observation_agent import ObservationAgent
from util.run_context import RunContext
from run.config import RDConfig
from run.util.config_utils import resolve_cfg, resolve_model, resolve_stage1_params
from run.util.pipeline_utils import (
    compute_formula_values,
    FormulaComputeResult,
)


@dataclass(frozen=True)
class Stage1Result:
    """
    Stage 1 output container.

    NOTE: `ohlcv_df` and `formula_df` are multi-ticker panels.
    Stage2/3 will split by ticker for validation and then aggregate.
    """
    hypothesis: dict[str, Any]
    observation_plan: dict[str, Any]
    formula_bundle: dict[str, Any]
    ohlcv_df: pl.DataFrame           # multi-ticker panel (timestamp, ticker, OHLCV)
    formula_df: pl.DataFrame         # multi-ticker panel (timestamp, ticker, formula values)
    price_with_formulas: pl.DataFrame
    failed_formulas: list[dict[str, Any]] = None  # formulas that failed to evaluate


# ════════════════════════════════════════════════════════════════════════════════
# Step 0: Hypothesis setting
# ════════════════════════════════════════════════════════════════════════════════
#
# Purpose
# Define the finance hypothesis to validate.
#
# Structure
# A finance hypothesis typically includes:
#   1) event definition: conditions that hold simultaneously
#   2) expected outcome: price behavior after the event
#   3) time horizon: the window over which the outcome is observed
#
# Principles
# - Do not encode a direct prediction rule/strategy.
# - Define the event via observable conditions.
# - Explicitly state the expectation/outcome.
#
# Input
# - A natural-language finance hypothesis
#
# Output
# - A structured hypothesis object (schemas/hypothesis.py), including:
#   - hypothesis_id
#   - hypothesis_name (summary)
#   - behavioral_description (including "why")
#   - horizon_days
#
# ════════════════════════════════════════════════════════════════════════════════

def step0_hypothesis_setting(
    *,
    concept: str,
    model: str,
    run_ctx: Optional[RunContext],
    allowed_columns: list[str],
    hypothesis_memory: list[Any] | None = None,
    knowledge: str = "",
    feedback: str = "",
) -> dict[str, Any]:
    agent = HypothesisAgent(model=model, run_ctx=run_ctx)
    return agent.purpose_hypothesis(
        concept=concept,
        metadata=allowed_columns,
        hypothesis_memory=hypothesis_memory,
        knowledge=knowledge,
        feedback=feedback,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Step 1: Hypothesis decomposition
# ════════════════════════════════════════════════════════════════════════════════
#
# Purpose
# Decompose the hypothesis into observable conditions (observations).
#
# Observation definition
# - An observation is a conceptual condition; the data implementation is not fixed yet.
# - Each observation is a condition that must hold for the hypothesis event.
# - The logical AND of observations defines the event.
#
# Decomposition principles
# 1) independence: each observation should be measurable independently
# 2) completeness: the combined observations fully define the event
# 3) separate conditions vs outcomes:
#    - condition observations: used to decide whether the event happened (validated in Stage 2)
#    - outcome observations: used as the outcome variable (used in Stage 3)
#
# Why this matters
# - Converts an abstract hypothesis into concrete, testable conditions
# - Enables independent validation of each condition's implementation
# - Helps diagnose which condition fails when the hypothesis fails
#
# Input
# - The structured hypothesis from Step 0
#
# Output
# - An observation plan object (schemas/observation.py), including:
#   - hypothesis_id
#   - observations[]
#     - observation_id
#     - description
#
# ════════════════════════════════════════════════════════════════════════════════

def step1_hypothesis_decomposition(
    *,
    hypothesis: dict[str, Any],
    model: str,
    run_ctx: Optional[RunContext],
) -> dict[str, Any]:
    agent = ObservationAgent(model=model, run_ctx=run_ctx)
    return agent.plan_observations(hypothesis=hypothesis)


# ════════════════════════════════════════════════════════════════════════════════
# Step 2: Observation formula construction
# ════════════════════════════════════════════════════════════════════════════════
#
# Purpose
# Design candidate formulas that implement each observation in data.
#
# Key constraints
#
# 1) Data constraint: OHLCV raw data only
#    - Use Open/High/Low/Close/Volume only
#    - Do not introduce derived indicators (RSI/MACD/...) or exogenous signals
#    - Rationale: extra indicators make "does the formula implement the observation?" ambiguous
#
# 2) Formula nature
#    - A formula is an implementation to detect a condition
#    - Not a factor/alpha
#    - Goal: implementation correctness, not return predictiveness
#
# 3) Multiple candidates
#    - Each observation may have multiple candidate formulas
#    - Stage 2 validates which formula best implements the observation
#    - This separates "observation definition" from "implementation"
#
# Design guidelines
# 1) simplicity: prefer interpretable formulas
# 2) direct mapping: reflect the observation definition directly
# 3) minimal parameters: keep lookback windows minimal
# 4) handle boundary conditions: early periods, missing values, etc.
#
# Input
# - Observations from Step 1
#
# Output
# - A formula bundle (schemas/behavioral_formula.py), including:
#   - hypothesis_id
#   - observation_descriptions
#   - formulas[] (evidence formulas)
#     - name (used as a column name)
#     - kind: "evidence"
#     - observation_id (optional; may be validated 1:1 by FormulaAgent)
#     - definition: DSL expression (no comparisons/logical ops)
#     - polarity: "higher_is_more_true" / "lower_is_more_true"
#     - description
#   - notes (optional)
#
# ════════════════════════════════════════════════════════════════════════════════

def step2_observation_formula_construction(
    *,
    hypothesis: dict[str, Any],
    observation_plan: dict[str, Any],
    model: str,
    run_ctx: Optional[RunContext],
    allowed_columns: list[str],
    knowledge: str = "",
    formula_memory: list[Any] | None = None,
    refine_rounds: int = 1,
) -> dict[str, Any]:
    agent = FormulaAgent(model=model, run_ctx=run_ctx)
    return agent.purpose_formula(
        hypothesis=hypothesis,
        metadata=allowed_columns,
        knowledge=knowledge,
        formula_memory=formula_memory,
        refine_rounds=refine_rounds,
        observation_plan=observation_plan,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Step 3: Formula value computation (expression evaluator)
# ════════════════════════════════════════════════════════════════════════════════
#
# Purpose
# Compute time-series values from the formula definitions produced in Step 2.
#
# Current implementation (important)
# - Instead of LLM-generated Python code, this repo uses the expression evaluator from `coder/factor_coder`
#   to compute formula values directly.
# - Formulas run through: DSL expression -> parsing/normalization -> evaluation.
# - The CoSTEER code-generation backend (Python function generation/validation) is not currently wired into
#   the Stage 1 runner. If needed, consider integrating an alternate backend (e.g.,
#   `agent/costeer_full_code_agent.py`) into Step 3.
#
# I/O requirements
#
# 1) Input data format
#    - Polars panel DataFrame required columns: ['timestamp','ticker','open','high','low','close','volume']
#    - timestamp must be sortable (string/date)
#
# 2) Output data format
#    - price_with_formulas: original panel with formula value columns appended (polars)
#    - each formula becomes a separate column
#    - column name: `formula['name']` (from FormulaAgent)
#
# 3) Computation quality requirements
#    - use vectorized ops (pandas/numpy-based evaluator)
#    - handle missing values explicitly
#    - handle boundary conditions (initial lookback periods)
#
# Validation / safety
# - raise an exception on parse/eval failures (include cause and expression)
# - if a formula column collides with an existing column name, overwrite on join
#
# Input
# - formulas from Step 2
# - OHLCV data
#
# Output
# - price_with_formulas and (per-ticker) pandas `ohlcv_df`, `formula_df`
# - This output is used as the input to Stage 2 (Observation Formula Validation)
#
# ════════════════════════════════════════════════════════════════════════════════

def step3_formula_value_computation(
    *,
    price_df: pl.DataFrame,
    formula_bundle: dict[str, Any],
) -> FormulaComputeResult:
    formulas = formula_bundle.get("formulas", []) if isinstance(formula_bundle, dict) else []
    return compute_formula_values(price_df, formulas=formulas)


# ───────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ───────────────────────────────────────────────────────────────────────────────

def run_stage1(
    *,
    concept: str,
    price_df: pl.DataFrame,
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    allowed_columns: list[str] | None = None,
    knowledge: str = "",
    feedback: str = "",
    hypothesis_memory: list[Any] | None = None,
    formula_memory: list[Any] | None = None,
    refine_rounds: int | None = None,
    cfg: RDConfig | None = None,
    max_eval_retries: int = 2,
) -> Stage1Result:
    """
    Run Stage 1: generate hypothesis -> decompose observations -> construct formulas -> compute formula values.

    NOTE: output is a multi-ticker panel.
    Stage2/3 will split by ticker for validation and then aggregate.

    Args:
        max_eval_retries: number of retries to regenerate formulas via FormulaAgent on evaluation failure (default: 2)
    """
    import logging

    if not isinstance(price_df, pl.DataFrame):
        raise TypeError("price_df must be a polars.DataFrame")

    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)

    allowed_columns, refine_rounds = resolve_stage1_params(
        cfg=cfg,
        allowed_columns=allowed_columns,
        refine_rounds=refine_rounds,
    )

    hypothesis = step0_hypothesis_setting(
        concept=concept,
        model=model,
        run_ctx=run_ctx,
        allowed_columns=allowed_columns,
        hypothesis_memory=hypothesis_memory,
        knowledge=knowledge,
        feedback=feedback,
    )

    # Log generated hypothesis.
    if run_ctx:
        import json
        run_ctx.log("\n")
        run_ctx.log("📋 GENERATED HYPOTHESIS")
        run_ctx.log(json.dumps(hypothesis, ensure_ascii=False, indent=2, default=str))
        run_ctx.log("\n")

    observation_plan = step1_hypothesis_decomposition(
        hypothesis=hypothesis,
        model=model,
        run_ctx=run_ctx,
    )

    # Log observation plan.
    if run_ctx:
        import json
        run_ctx.log("\n")
        run_ctx.log("🔍 OBSERVATION PLAN")
        run_ctx.log(json.dumps(observation_plan, ensure_ascii=False, indent=2, default=str))
        run_ctx.log("\n")

    formula_bundle = step2_observation_formula_construction(
        hypothesis=hypothesis,
        observation_plan=observation_plan,
        model=model,
        run_ctx=run_ctx,
        allowed_columns=allowed_columns,
        knowledge=knowledge,
        formula_memory=formula_memory,
        refine_rounds=refine_rounds,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: formula evaluation + retry loop on failures
    # ═══════════════════════════════════════════════════════════════════════════
    for eval_attempt in range(max_eval_retries + 1):
        formulas = formula_bundle.get("formulas", []) if isinstance(formula_bundle, dict) else []
        formula_names = [str(f.get("name")) for f in formulas if isinstance(f, dict) and f.get("name")]

        compute_result = step3_formula_value_computation(price_df=price_df, formula_bundle=formula_bundle)
        price_with_formulas = compute_result.df
        failed_formulas = compute_result.failed_formulas

        # Stop once all formulas evaluate successfully.
        if not failed_formulas:
            if eval_attempt > 0:
                logging.info(f"Stage1: All formulas now evaluate successfully after {eval_attempt} retry(s).")
            break

        # If this was the last attempt, keep failures and proceed.
        if eval_attempt >= max_eval_retries:
            logging.warning(
                f"Stage1: {len(failed_formulas)} formula(s) still failing after {max_eval_retries} retries. "
                f"Names: {[f['name'] for f in failed_formulas]}. "
                "These will be passed to Stage2 as FAIL."
            )
            break

        # If there are evaluation failures, request FormulaAgent to fix them.
        logging.warning(
            f"Stage1: {len(failed_formulas)} formula(s) failed to evaluate (attempt {eval_attempt + 1}). "
            f"Names: {[f['name'] for f in failed_formulas]}. "
            "Requesting FormulaAgent to fix..."
        )

        # Format evaluation errors as diagnostics for refinement.
        eval_error_diagnostics = {
            "evaluation_errors": [
                {
                    "formula_name": ff["name"],
                    "definition": ff["definition"],
                    "error": ff["error"],
                    "fix_suggestion": _suggest_fix_for_eval_error(ff["error"]),
                }
                for ff in failed_formulas
            ],
            "error_summary": (
                f"{len(failed_formulas)} formula(s) failed during evaluation. "
                "These formulas have syntax errors or use invalid function signatures. "
                "Please fix the formulas based on the error messages."
            ),
        }

        # Ask FormulaAgent to refine the bundle.
        formula_agent = FormulaAgent(model=model, run_ctx=run_ctx)
        formula_bundle = formula_agent.refine_behavioral_bundle(
            hypothesis=hypothesis,
            current_bundle=formula_bundle,
            diagnostics=eval_error_diagnostics,
            metadata=allowed_columns,
            knowledge=knowledge,
            focus="Fix evaluation errors. The formulas have syntax errors or invalid function signatures.",
            refine_rounds=1,
            observation_plan=observation_plan,
        )

    # Keep multi-ticker panel outputs (Stage2/3 handle per-ticker aggregation).
    ohlcv_cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    ohlcv_df = price_with_formulas.select([c for c in ohlcv_cols if c in price_with_formulas.columns])
    # Exclude failed formulas from formula_df
    valid_formula_names = [n for n in formula_names if n not in {f["name"] for f in failed_formulas}]
    formula_df = price_with_formulas.select(["timestamp", "ticker"] + valid_formula_names)

    return Stage1Result(
        hypothesis=hypothesis,
        observation_plan=observation_plan,
        formula_bundle=formula_bundle,
        ohlcv_df=ohlcv_df,
        formula_df=formula_df,
        price_with_formulas=price_with_formulas,
        failed_formulas=failed_formulas,
    )


def _suggest_fix_for_eval_error(error: str) -> str:
    """Generate a fix suggestion from an evaluator error message."""
    error_lower = error.lower()

    if "takes 1 positional argument but 2 were given" in error_lower:
        return "This function takes only 1 argument. Remove the extra argument (e.g., ZSCORE(x) instead of ZSCORE(x, 20))."
    if "takes 2 positional arguments but" in error_lower:
        return "Check the function signature - it expects exactly 2 arguments."
    if "unexpected keyword argument" in error_lower:
        return "Remove the invalid keyword argument from the function call."
    if "not defined" in error_lower or "name" in error_lower and "is not defined" in error_lower:
        return "Use only allowed functions from the function library. Check function names for typos."
    if "division by zero" in error_lower:
        return "Add a small epsilon (e.g., 1e-8) to the denominator to avoid division by zero."
    if "invalid syntax" in error_lower:
        return "Check the formula syntax - ensure parentheses are balanced and operators are valid."

    return "Review the function signature and arguments in the function library documentation."


# ════════════════════════════════════════════════════════════════════════════════
# Stage 1: methodological notes
# ════════════════════════════════════════════════════════════════════════════════
#
# Stage 1 is responsible for converting a hypothesis into a testable form.
#
# The key idea is separating three layers:
#
# 1) Hypothesis vs. implementation
#    - hypothesis: abstract finance idea
#    - implementation: formulas computable on data
#
# 2) Concept vs. measurement
#    - observation: conceptual condition (what to observe)
#    - formula: measurement method (how to observe)
#
# 3) Conditions vs. outcomes
#    - condition observations: define the event
#    - outcome observations: validate the hypothesis
#
# Benefits:
# - each component can be validated independently
# - failure diagnosis becomes easier
# - formula iteration can happen independently of the hypothesis itself
#
# ════════════════════════════════════════════════════════════════════════════════
