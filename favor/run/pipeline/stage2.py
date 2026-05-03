"""
================================================================================
STAGE 2: Observation Formula Validation
Observation-implementation validation (first core validation step; LLM-driven decision)
================================================================================

[Purpose of Stage 2]
Validate whether each Stage 1 formula actually implements the intended observation condition in data.

[Key question]
"Does this formula implement the observation condition defined by the hypothesis?"

[Key principle: return-agnostic validation]
- Do not use returns, predictiveness, or strategy performance.
- Validate only whether the formula consistently separates the raw OHLCV distribution.
- Compare against raw-data distributions (not additional indicators).

[Validation flow (LLM-driven)]
(i)   Compute distribution summaries
(ii)  Infer expected distribution changes from the observation description
(iii) Compare with observed changes -> PASS/FAIL
(iv)  If FAIL, propose improvements

[Distribution elements]
- MAG (range): H - L
- DIR (direction): C - O (positive=up, negative=down)
- VOL (volume): V
- POS (relative position): (C - L) / (H - L)

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import polars as pl

from agent.validation_agent import (
    DistributionStats,
    FormulaValidationResult,
    MonotonicityResult,
    ValidationAgent,
)
from util.run_context import RunContext
from run.config import RDConfig
from run.util.config_utils import resolve_cfg, resolve_model, resolve_stage2_params


# ════════════════════════════════════════════════════════════════════════════════
# Step 1: Conditional partition by formula value
# ════════════════════════════════════════════════════════════════════════════════
#
# Partition the series into bins using only the formula value f_t (e.g., quantiles).
# This enables conditional distribution analysis of raw OHLCV given the formula value.
#
# ════════════════════════════════════════════════════════════════════════════════

def step1_partition_by_formula_value(
    *,
    formula_values: pd.Series,
    n_quantiles: int | None = None,
    polarity: str = "higher_is_more_true",
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    cfg: RDConfig | None = None,
) -> tuple[pd.Series | None, dict[str, int] | None, int]:
    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)
    n_quantiles, _ = resolve_stage2_params(
        cfg=cfg,
        n_quantiles=n_quantiles,
        monotonicity_threshold=None,
    )

    agent = ValidationAgent(model=model, run_ctx=run_ctx, n_quantiles=n_quantiles)
    return agent._partition_by_formula_value(formula_values, n_quantiles, polarity=polarity)


# ════════════════════════════════════════════════════════════════════════════════
# Step 2: Raw OHLCV distribution observation
# ════════════════════════════════════════════════════════════════════════════════
#
# Observe how raw OHLCV-derived elements change across bins (e.g., mean/median/std/quantiles).
# Note: visualization is not implemented here.
#
# ════════════════════════════════════════════════════════════════════════════════

def step2_observe_raw_distribution(
    *,
    ohlcv_df: pd.DataFrame,
    quantile_labels: pd.Series,
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    cfg: RDConfig | None = None,
) -> dict[str, dict[str, DistributionStats]]:
    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)
    agent = ValidationAgent(model=model, run_ctx=run_ctx)
    return agent._observe_raw_distribution(ohlcv_df, quantile_labels)


# ════════════════════════════════════════════════════════════════════════════════
# Step 3: Monotonic shift verification (deprecated; replaced by LLM-based decision)
# ════════════════════════════════════════════════════════════════════════════════

# def step3_verify_monotonic_shift(
#     *,
#     distribution_by_element: dict[str, dict[str, DistributionStats]],
#     polarity: str,
#     monotonicity_threshold: float | None = None,
#     model: str | None = None,
#     run_ctx: Optional[RunContext] = None,
#     cfg: RDConfig | None = None,
# ) -> list[MonotonicityResult]:
#     cfg = resolve_cfg(cfg)
#     model = resolve_model(model, cfg)
#     _, monotonicity_threshold = resolve_stage2_params(
#         cfg=cfg,
#         n_quantiles=None,
#         monotonicity_threshold=monotonicity_threshold,
#     )
#     agent = ValidationAgent(
#         model=model,
#         run_ctx=run_ctx,
#         monotonicity_threshold=monotonicity_threshold,
#     )
#     return agent._verify_monotonic_shift(distribution_by_element, polarity)


# ════════════════════════════════════════════════════════════════════════════════
# Step 4: Decision (deprecated; replaced by LLM-based decision)
# ════════════════════════════════════════════════════════════════════════════════

# def step4_make_decision(
#     *,
#     monotonicity_results: list[MonotonicityResult],
#     quantile_counts: dict[str, int],
#     monotonicity_threshold: float | None = None,
#     model: str | None = None,
#     run_ctx: Optional[RunContext] = None,
#     cfg: RDConfig | None = None,
# ) -> tuple[str, float, list[str], list[str]]:
#     cfg = resolve_cfg(cfg)
#     model = resolve_model(model, cfg)
#     _, monotonicity_threshold = resolve_stage2_params(
#         cfg=cfg,
#         n_quantiles=None,
#         monotonicity_threshold=monotonicity_threshold,
#     )
#     agent = ValidationAgent(
#         model=model,
#         run_ctx=run_ctx,
#         monotonicity_threshold=monotonicity_threshold,
#     )
#     return agent._make_decision(monotonicity_results, quantile_counts)



@dataclass
class Stage2Result:
    """
    Stage 2 output container (panel pooling mode).

    pooling_info: panel pooling metadata (tickers, n_samples, etc.)
    """
    hypothesis_id: str
    summary: dict[str, Any]
    passed_formulas: list[dict[str, Any]]
    report_md: str
    # Panel pooling metadata.
    pooling_info: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregated_summary(self) -> dict[str, Any]:
        """Returns aggregated info (pooling info) for backward compatibility."""
        return self.pooling_info


# ════════════════════════════════════════════════════════════════════════════════
# Panel pooling helpers
# ════════════════════════════════════════════════════════════════════════════════

def _pool_panel_data(
    ohlcv_df: pl.DataFrame,
    formula_df: pl.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Pool multi-ticker panel data into a single dataset for validation.

    Args:
        ohlcv_df: OHLCV panel (ticker, timestamp, O, H, L, C, V)
        formula_df: formula-value panel (ticker, timestamp, formula columns)

    Returns:
        (pooled_ohlcv, pooled_formula, pooling_info)
    """
    # Extract ticker list.
    tickers = ohlcv_df.select(pl.col("ticker").unique()).to_series().to_list()
    tickers = [t for t in tickers if t is not None]

    if not tickers:
        raise ValueError("No tickers found in ohlcv_df")

    # Pool and convert to pandas for analysis; keep ticker and sort by (ticker, timestamp).
    pooled_ohlcv = (
        ohlcv_df
        .sort(["ticker", "timestamp"])
        .to_pandas()
    )

    pooled_formula = (
        formula_df
        .sort(["ticker", "timestamp"])
        .to_pandas()
    )

    # Pooling metadata.
    n_samples_per_ticker = ohlcv_df.group_by("ticker").len().to_pandas().set_index("ticker")["len"].to_dict()

    pooling_info = {
        "n_tickers": len(tickers),
        "tickers": tickers,
        "total_samples": len(pooled_ohlcv),
        "n_samples_per_ticker": n_samples_per_ticker,
    }

    return pooled_ohlcv, pooled_formula, pooling_info


def validate_single_formula(
    *,
    formula: dict[str, Any],
    ohlcv_df: pd.DataFrame,
    formula_values: pd.Series,
    n_quantiles: int | None = None,
    monotonicity_threshold: float | None = None,
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    cfg: RDConfig | None = None,
) -> FormulaValidationResult:
    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)
    n_quantiles, monotonicity_threshold = resolve_stage2_params(
        cfg=cfg,
        n_quantiles=n_quantiles,
        monotonicity_threshold=monotonicity_threshold,
    )
    agent = ValidationAgent(
        model=model,
        run_ctx=run_ctx,
        n_quantiles=n_quantiles,
        monotonicity_threshold=monotonicity_threshold,
    )
    return agent.validate_formula(formula=formula, ohlcv_df=ohlcv_df, formula_values=formula_values)


def run_stage2(
    *,
    formula_bundle: dict[str, Any],
    ohlcv_df: pl.DataFrame,
    formula_df: pl.DataFrame,
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    n_quantiles: int | None = None,
    monotonicity_threshold: float | None = None,
    cfg: RDConfig | None = None,
    eval_failed_formulas: list[dict[str, Any]] | None = None,
) -> Stage2Result:
    """
    Run Stage 2: validate formulas (observation implementations) and build the PASS pool for Stage 3.

    NOTE: panel pooling mode.
    - `ohlcv_df` and `formula_df` are multi-ticker panels (polars DataFrames).
    - All tickers are pooled into a single dataset and validated together.
    - Pooling increases sample size and stabilizes distribution estimates.

    Args:
        formula_bundle: bundle produced by FormulaAgent
        ohlcv_df: OHLCV panel (polars; includes ticker column)
        formula_df: formula-value panel (polars; includes ticker column)
        model: LLM model name
        run_ctx: RunContext
        n_quantiles: number of quantile bins
        monotonicity_threshold: monotonicity threshold (agent-specific)
        cfg: RDConfig
        eval_failed_formulas: formulas that failed to evaluate in Stage 1 (auto-mark as FAIL)
    """
    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)
    n_quantiles, monotonicity_threshold = resolve_stage2_params(
        cfg=cfg,
        n_quantiles=n_quantiles,
        monotonicity_threshold=monotonicity_threshold,
    )

    hypothesis_id = formula_bundle.get("hypothesis_id", "unknown")
    all_formulas = formula_bundle.get("formulas", []) if isinstance(formula_bundle, dict) else []

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel pooling: pool all tickers into a single dataset.
    # ═══════════════════════════════════════════════════════════════════════════
    pooled_ohlcv, pooled_formula, pooling_info = _pool_panel_data(
        ohlcv_df=ohlcv_df,
        formula_df=formula_df,
    )

    # Drop ticker/timestamp and convert to pandas for ValidationAgent.
    # Timestamp is not used as an index in this pooling-based validation path.
    ohlcv_cols = [c for c in pooled_ohlcv.columns if c not in ("ticker", "timestamp")]
    pooled_ohlcv_pd = pooled_ohlcv[ohlcv_cols].reset_index(drop=True)

    formula_cols = [c for c in pooled_formula.columns if c not in ("ticker", "timestamp")]
    pooled_formula_pd = pooled_formula[formula_cols].reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Validate on pooled data.
    # ═══════════════════════════════════════════════════════════════════════════
    agent = ValidationAgent(
        model=model,
        run_ctx=run_ctx,
        n_quantiles=n_quantiles,
        monotonicity_threshold=monotonicity_threshold,
    )

    validation_summary = agent.validate_formula_bundle(
        formula_bundle=formula_bundle,
        ohlcv_df=pooled_ohlcv_pd,
        formula_df=pooled_formula_pd,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Add evaluation-failed formulas as automatic FAIL.
    # ═══════════════════════════════════════════════════════════════════════════
    if eval_failed_formulas:
        # Append evaluation failures to existing results.
        existing_results = validation_summary.get("results", [])
        existing_failed = validation_summary.get("failed_formulas", [])

        for ff in eval_failed_formulas:
            fname = ff.get("name", "")
            # Add FAIL record.
            existing_results.append({
                "formula_id": fname,
                "formula_name": fname,
                "obs_id": "",
                "verdict": "FAIL",
                "reasoning": f"Formula evaluation failed: {ff.get('error', 'unknown error')}",
                "quantile_counts": {},
                "evidence_packet": {},
                "primary_evidence": [],
                "distribution_by_element": {},
                "distribution_summary": "",
            })
            if fname not in existing_failed:
                existing_failed.append(fname)

        # Update summary fields.
        n_eval_failed = len(eval_failed_formulas)
        validation_summary["results"] = existing_results
        validation_summary["failed_formulas"] = existing_failed
        validation_summary["total_formulas"] = validation_summary.get("total_formulas", 0) + n_eval_failed
        validation_summary["failed"] = validation_summary.get("failed", 0) + n_eval_failed

        # Recompute pass_rate.
        total = validation_summary["total_formulas"]
        passed = validation_summary.get("passed", 0)
        validation_summary["pass_rate"] = passed / total if total > 0 else 0.0

        if run_ctx:
            run_ctx.log(f"Stage2: Added {n_eval_failed} evaluation-failed formula(s) as automatic FAIL")

    # ═══════════════════════════════════════════════════════════════════════════
    # Consolidate results.
    # ═══════════════════════════════════════════════════════════════════════════
    passed_names = set(validation_summary.get("passed_formulas", []) or [])

    # De-duplicate: if multiple formulas share the same name, keep the last one
    # (refinement may produce repeated names).
    name_to_formula = {}
    for f in all_formulas:
        if isinstance(f, dict):
            fname = str(f.get("name") or "")
            if fname in passed_names:
                name_to_formula[fname] = f  # Overwrite with the latest version.

    passed_formulas = list(name_to_formula.values())

    # Generate report.
    report_md = _generate_pooled_report(hypothesis_id, validation_summary, pooling_info)

    # Build summary.
    summary = {
        "hypothesis_id": hypothesis_id,
        "total_formulas": validation_summary.get("total_formulas", 0),
        "passed": validation_summary.get("passed", 0),
        "failed": validation_summary.get("failed", 0),
        "pass_rate": validation_summary.get("pass_rate", 0.0),
        "passed_formulas": list(passed_names),
        "failed_formulas": validation_summary.get("failed_formulas", []),
        "overall_verdict": validation_summary.get("overall_verdict", "UNKNOWN"),
        "results": validation_summary.get("results", []),
        # Pooling info
        "pooling_info": pooling_info,
    }

    return Stage2Result(
        hypothesis_id=str(hypothesis_id),
        summary=summary,
        passed_formulas=passed_formulas,
        report_md=report_md,
        pooling_info=pooling_info,
    )


def _generate_pooled_report(
    hypothesis_id: str,
    validation_summary: dict[str, Any],
    pooling_info: dict[str, Any],
) -> str:
    """Generate a Markdown report for panel pooling validation output."""
    lines = []
    lines.append("# Stage 2: Observation Formula Validation Report (Panel Pooling)")
    lines.append("")
    lines.append(f"**Hypothesis ID**: {hypothesis_id}")
    lines.append(f"**Overall Verdict**: {validation_summary.get('overall_verdict', 'N/A')}")
    lines.append("")

    # Pooling info.
    lines.append("## Panel Pooling Info")
    lines.append(f"- **Number of Tickers**: {pooling_info.get('n_tickers', 0)}")
    lines.append(f"- **Tickers**: {pooling_info.get('tickers', [])}")
    lines.append(f"- **Total Samples**: {pooling_info.get('total_samples', 0):,}")
    lines.append("")

    # Samples per ticker.
    n_samples_per_ticker = pooling_info.get("n_samples_per_ticker", {})
    if n_samples_per_ticker:
        lines.append("### Samples per Ticker")
        for ticker, n in sorted(n_samples_per_ticker.items()):
            lines.append(f"- {ticker}: {n:,}")
        lines.append("")

    # Validation summary.
    lines.append("## Validation Summary")
    lines.append(f"- **Total Formulas**: {validation_summary.get('total_formulas', 0)}")
    lines.append(f"- **Passed**: {validation_summary.get('passed', 0)}")
    lines.append(f"- **Failed**: {validation_summary.get('failed', 0)}")
    lines.append(f"- **Pass Rate**: {validation_summary.get('pass_rate', 0):.1%}")
    lines.append("")

    # Formula-level results.
    lines.append("## Formula-level Results")
    for result in validation_summary.get("results", []):
        lines.append(f"\n### {result.get('formula_name', 'Unknown')}")
        lines.append(f"- **Observation ID**: {result.get('obs_id', 'N/A')}")
        lines.append(f"- **Verdict**: {result.get('verdict', 'N/A')}")

        reasoning = result.get("reasoning", "")
        if reasoning:
            lines.append(f"- **Reasoning**: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")

    # Overall summary.
    lines.append("\n## Summary")
    lines.append(f"- **Passed Formulas**: {validation_summary.get('passed_formulas', [])}")
    lines.append(f"- **Failed Formulas**: {validation_summary.get('failed_formulas', [])}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# Stage 2: methodological notes
# ════════════════════════════════════════════════════════════════════════════════
#
# Stage 2 implements a core differentiator of the framework.
#
# Compared to common factor research:
#
# Traditional factor research:
#   formula -> return predictiveness (IC) -> factor evaluation
#   Issue: unclear what a formula is measuring
#
# This framework:
#   formula -> raw distribution separation -> validate observation implementation -> (Stage 3) validate outcomes/returns
#   Benefit: validate semantic correctness before performance
#
# Validation logic:
#
# If a formula implements an observation well:
# - raw distributions should differ between low vs high formula values
# - the difference should be directionally consistent (often monotone with formula value)
# - regardless of returns, this is evidence the formula measures something consistently
#
# Link to Stage 3:
#
# Formulas that pass Stage 2:
# - are empirically validated implementations of observations
# - can be combined in Stage 3 to validate hypothesis structure
# - make failure diagnosis easier (which obs/formula is responsible)
#
# ════════════════════════════════════════════════════════════════════════════════
