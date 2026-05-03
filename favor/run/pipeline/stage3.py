"""
================================================================================
STAGE 3: Hypothesis Instance Validation
================================================================================

This stage validates whether a hypothesis is *structurally consistent* in the data.

High-level flow:
- Delegate per-ticker strictness evaluation to `HypothesisValidationAgent`.
- Aggregate ticker-level verdicts for generalizability.
- Select qualified combinations using a simple 2-tier filter:
  - A) ticker-level pass-rate threshold
  - B) cross-ticker monotone improvement (direction-only; no magnitude threshold)

================================================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd
import polars as pl

from agent.hypothesis_validation_agent import HypothesisValidationAgent
from schemas.hypothesis_validation_dataclasses import HypothesisValidationResult
from run.config import RDConfig
from run.util.config_utils import resolve_cfg, resolve_model, resolve_stage3_params
from util.run_context import RunContext


# ════════════════════════════════════════════════════════════════════════════════
# Step 1: Hypothesis instance construction
# ════════════════════════════════════════════════════════════════════════════════
#
# Concept:
# - A hypothesis instance is the AND of one formula per observation: H_t = obs1_t ∧ ... ∧ obsN_t.
# - Strictness is controlled by quantile thresholds (looser → more signals; stricter → fewer signals).
#
# Implementation:
# - This module delegates instance construction and evaluation to `HypothesisValidationAgent`.

def _extract_hypothesis_id(hypothesis: dict[str, Any] | None) -> str | None:
    if not isinstance(hypothesis, dict):
        return None
    hyp_list = hypothesis.get("hypotheses", [])
    hyp_obj = hyp_list[0] if isinstance(hyp_list, list) and hyp_list and isinstance(hyp_list[0], dict) else hypothesis
    hid = (hyp_obj or {}).get("hypothesis_id") or (hyp_obj or {}).get("id")
    return str(hid) if isinstance(hid, str) and hid.strip() else None


def _extract_horizon_days(hypothesis: dict[str, Any] | None) -> int | None:
    if not isinstance(hypothesis, dict):
        return None
    hyp_list = hypothesis.get("hypotheses", [])
    hyp_obj = hyp_list[0] if isinstance(hyp_list, list) and hyp_list and isinstance(hyp_list[0], dict) else hypothesis
    h = (hyp_obj or {}).get("horizon_days")
    if isinstance(h, int) and h > 0:
        return h
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Step 2: Outcome variable definition
# ════════════════════════════════════════════════════════════════════════════════
#
# The default outcome variable is a forward return over `horizon_days`.
# This uses future prices and is intended for retrospective validation only.

def step2_define_outcome_variable(
    *,
    ohlcv_df: pd.DataFrame,
    horizon_days: int,
    model: str,
    run_ctx: Optional[RunContext] = None,
    strictness_grid: dict[str, float] | None = None,
    monotonicity_threshold: float = 0.7,
) -> pd.Series:
    """
    Wrapper that reuses the agent's internal outcome-variable implementation.
    """
    agent = HypothesisValidationAgent(
        model=model,
        run_ctx=run_ctx,
        strictness_grid=strictness_grid,
        horizon_days=horizon_days,
        monotonicity_threshold=monotonicity_threshold,
    )
    return agent._define_outcome_variable(ohlcv_df)


# ════════════════════════════════════════════════════════════════════════════════
# Step 3: Strictness grid evaluation
# ════════════════════════════════════════════════════════════════════════════════
#
# Evaluate how signal/outcome characteristics change across strictness levels.
# Detailed metric definitions and decision logic live in `HypothesisValidationAgent`.

def step3_evaluate_strictness_grid(
    *,
    hypothesis_id: str,
    passed_formulas: list[dict[str, Any]],
    ohlcv_df: pd.DataFrame,
    formula_df: pd.DataFrame,
    model: str,
    run_ctx: Optional[RunContext] = None,
    strictness_grid: dict[str, float] | None = None,
    horizon_days: int = 5,
    monotonicity_threshold: float = 0.7,
    use_random_grid: bool = True,
    random_grid_steps: int = 10,
    oos_ohlcv_df: pd.DataFrame | None = None,
    oos_formula_df: pd.DataFrame | None = None,
) -> HypothesisValidationResult:
    """
    Run the full Stage3 evaluation (grid → monotonicity → decision).

    Args:
        oos_ohlcv_df: out-of-sample OHLCV (record-keeping only; not used for pass/fail)
        oos_formula_df: out-of-sample formula values (record-keeping only)
    """
    agent = HypothesisValidationAgent(
        model=model,
        run_ctx=run_ctx,
        strictness_grid=strictness_grid,
        horizon_days=horizon_days,
        monotonicity_threshold=monotonicity_threshold,
        use_random_grid=use_random_grid,
        random_grid_steps=random_grid_steps,
    )
    return agent.validate_hypothesis(
        hypothesis_id=hypothesis_id,
        passed_formulas=passed_formulas,
        ohlcv_df=ohlcv_df,
        formula_df=formula_df,
        oos_ohlcv_df=oos_ohlcv_df,
        oos_formula_df=oos_formula_df,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Step 4: Monotonicity verification
# ════════════════════════════════════════════════════════════════════════════════
#
# Per-ticker monotonicity checks and PASS/FAIL decisions are implemented inside
# `HypothesisValidationAgent`.
#
# Cross-ticker combination selection is implemented in `run_stage3()` below:
# - A) ticker-level pass-rate threshold (cfg.stage3.combination_pass_rate_threshold)
# - B) cross-ticker monotone improvement on aggregated statistics:
#      - s2_ratio = S2/(S1+S2) should be non-increasing with at least one strict decrease, OR
#      - mean_return (signal-count weighted) should be non-decreasing with at least one strict increase
#
# ════════════════════════════════════════════════════════════════════════════════

def _format_stage3_report(result: HypothesisValidationResult) -> str:
    lines: list[str] = []
    lines.append("# Stage 3: Hypothesis Instance Validation Report")
    lines.append("")
    lines.append(f"**Hypothesis ID**: {result.hypothesis_id}")
    lines.append(f"**Overall Verdict**: {result.overall_verdict}")
    lines.append(f"**Confidence**: {result.confidence:.3f}")
    lines.append("")
    if result.key_findings:
        lines.append("## Key Findings")
        for k in result.key_findings[:20]:
            lines.append(f"- {k}")
        lines.append("")
    lines.append("## Conclusion")
    lines.append(result.conclusion)
    lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# Step 5: Strategy execution (as a measurement tool)
# ════════════════════════════════════════════════════════════════════════════════
#
# Strategy simulation lives inside `HypothesisValidationAgent`.
# Stage3 uses backtest-like metrics as a measurement device, not as an objective to maximize.
#
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Stage3Result:
    """
    Stage 3 output container (supports multi-ticker aggregation).

    - ticker_results: per-ticker validation payloads (dict-serialized HypothesisValidationResult)
    - aggregated_result: cross-ticker aggregation summary
    - passed_combinations: combinations qualified by the 2-tier filter
    - combination_stats: per-combination diagnostics (e.g., pass_rate, s2_ratio_improvement)
    """
    hypothesis_id: str
    result: dict[str, Any]
    report_md: str
    # Multi-ticker aggregation results
    ticker_results: dict[str, dict[str, Any]] = field(default_factory=dict)  # {ticker: result}
    aggregated_result: dict[str, Any] = field(default_factory=dict)
    # Qualified combinations (each combination is a list of formula dicts).
    passed_combinations: list[list[dict[str, Any]]] = field(default_factory=list)
    # Per-combination diagnostics (e.g., s2_ratio_improvement).
    combination_stats: dict[tuple, dict[str, Any]] = field(default_factory=dict)


def _aggregate_stage3_ticker_results(
    ticker_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate per-ticker Stage3 results and evaluate generalizability.
    """
    if not ticker_results:
        return {}

    tickers = list(ticker_results.keys())
    n_tickers = len(tickers)

    # Aggregate ticker-level verdicts (only include evaluated tickers).
    ticker_verdicts = {}
    n_passed = 0
    n_failed = 0
    n_evaluated = 0  # Number of tickers that produced valid evaluation outputs.

    for t, r in ticker_results.items():
        verdict = r.get("overall_verdict", "UNKNOWN")
        # Consider a ticker evaluated if it has strictness results or an explicit PASS verdict.
        has_results = len(r.get("strictness_results", [])) > 0 or verdict == "PASS"

        if has_results or verdict in ("PASS", "FAIL"):
            ticker_verdicts[t] = verdict
            n_evaluated += 1
            if verdict == "PASS":
                n_passed += 1
            elif verdict == "FAIL":
                n_failed += 1

    pass_rate = n_passed / n_evaluated if n_evaluated > 0 else 0.0

    # Aggregate confidence values across tickers.
    confidences = [r.get("confidence", 0.0) for r in ticker_results.values()]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Direction-consistency check across tickers.
    monotonicity_directions = {}
    for ticker, result in ticker_results.items():
        mono_results = result.get("monotonicity_results", [])
        for mr in mono_results:
            metric = mr.get("metric_name", "")
            direction = mr.get("direction", "")
            if metric not in monotonicity_directions:
                monotonicity_directions[metric] = []
            monotonicity_directions[metric].append(direction)

    # Compute direction consistency.
    direction_consistency = {}
    for metric, directions in monotonicity_directions.items():
        if not directions:
            continue
        most_common = max(set(directions), key=directions.count)
        consistency = directions.count(most_common) / len(directions)
        direction_consistency[metric] = {
            "dominant_direction": most_common,
            "consistency": consistency,
        }

    # Aggregated decision.
    if pass_rate >= 0.7:
        aggregated_verdict = "PASS"
        generalizability = "HIGH"
    elif pass_rate >= 0.5:
        aggregated_verdict = "PASS"
        generalizability = "MEDIUM"
    else:
        aggregated_verdict = "FAIL"
        generalizability = "NONE"

    return {
        "n_tickers": n_tickers,
        "n_evaluated": n_evaluated,
        "tickers": tickers,
        "ticker_verdicts": ticker_verdicts,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "pass_rate": pass_rate,
        "avg_confidence": avg_confidence,
        "direction_consistency": direction_consistency,
        "aggregated_verdict": aggregated_verdict,
        "generalizability": generalizability,
    }


def run_stage3(
    *,
    passed_formulas: list[dict[str, Any]],
    ohlcv_df: pl.DataFrame,
    formula_df: pl.DataFrame,
    hypothesis: dict[str, Any] | None = None,
    hypothesis_id: str | None = None,
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    strictness_grid: dict[str, float] | None = None,
    horizon_days: int | None = None,
    monotonicity_threshold: float | None = None,
    use_random_grid: bool | None = None,
    random_grid_steps: int | None = None,
    cfg: RDConfig | None = None,
    oos_ohlcv_df: pl.DataFrame | None = None,
    oos_formula_df: pl.DataFrame | None = None,
) -> Stage3Result:
    """
    Run Stage 3: evaluate AND-instances of Stage2-passed formulas over a strictness grid.

    Notes (multi-ticker aggregation):
    - `ohlcv_df` and `formula_df` are multi-ticker panels (polars DataFrames).
    - Validation is performed per ticker and then aggregated.
    - The final verdict is based on `aggregated_verdict`.

    Args:
        oos_ohlcv_df: out-of-sample OHLCV (record-keeping only; not used for pass/fail)
        oos_formula_df: out-of-sample formula values (record-keeping only)
    """
    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)

    hypothesis_id = hypothesis_id or _extract_hypothesis_id(hypothesis) or "unknown"
    horizon_from_hyp = _extract_horizon_days(hypothesis)
    horizon_days, monotonicity_threshold = resolve_stage3_params(
        cfg=cfg,
        horizon_days=horizon_days if horizon_days is not None else horizon_from_hyp,
        monotonicity_threshold=monotonicity_threshold,
    )

    strictness_grid = strictness_grid or cfg.stage3.strictness_grid
    use_random_grid = use_random_grid if use_random_grid is not None else cfg.stage3.use_random_grid
    random_grid_steps = random_grid_steps if random_grid_steps is not None else cfg.stage3.random_grid_steps

    # Extract ticker universe.
    tickers = ohlcv_df.select(pl.col("ticker").unique()).to_series().to_list()
    tickers = [t for t in tickers if t is not None]

    if not tickers:
        raise ValueError("No tickers found in ohlcv_df")

    # Validate per ticker.
    ticker_results: dict[str, dict[str, Any]] = {}

    # Optional progress bar (falls back to plain iteration if tqdm isn't available).
    try:
        import sys
        from tqdm.auto import tqdm  # type: ignore

        ticker_iter = tqdm(tickers, desc="Stage3 tickers", disable=not sys.stderr.isatty())
    except Exception:
        ticker_iter = tickers

    for ticker in ticker_iter:
        # Filter per-ticker data and convert to pandas (agent expects pandas inputs).
        ticker_ohlcv = (
            ohlcv_df.filter(pl.col("ticker") == ticker)
            .sort("timestamp")
            .drop("ticker")
            .to_pandas()
            .set_index("timestamp")
        )
        ticker_formula = (
            formula_df.filter(pl.col("ticker") == ticker)
            .sort("timestamp")
            .drop("ticker")
            .to_pandas()
            .set_index("timestamp")
        )

        if ticker_ohlcv.empty or ticker_formula.empty:
            continue

        try:
            if hasattr(ticker_iter, "set_postfix_str"):
                ticker_iter.set_postfix_str(f"ticker={ticker}")
        except Exception:
            pass

        # Filter out-of-sample data (record-keeping / IC logging only).
        ticker_oos_ohlcv = None
        ticker_oos_formula = None
        if oos_ohlcv_df is not None and oos_formula_df is not None:
            oos_ticker_ohlcv_pl = oos_ohlcv_df.filter(pl.col("ticker") == ticker)
            oos_ticker_formula_pl = oos_formula_df.filter(pl.col("ticker") == ticker)
            if not oos_ticker_ohlcv_pl.is_empty() and not oos_ticker_formula_pl.is_empty():
                ticker_oos_ohlcv = (
                    oos_ticker_ohlcv_pl
                    .sort("timestamp")
                    .drop("ticker")
                    .to_pandas()
                    .set_index("timestamp")
                )
                ticker_oos_formula = (
                    oos_ticker_formula_pl
                    .sort("timestamp")
                    .drop("ticker")
                    .to_pandas()
                    .set_index("timestamp")
                )

        # Reuse the existing `step3_evaluate_strictness_grid` logic.
        result_obj = step3_evaluate_strictness_grid(
            hypothesis_id=hypothesis_id,
            passed_formulas=passed_formulas,
            ohlcv_df=ticker_ohlcv,
            formula_df=ticker_formula,
            model=model,
            run_ctx=run_ctx,
            strictness_grid=strictness_grid,
            horizon_days=horizon_days,
            monotonicity_threshold=monotonicity_threshold,
            use_random_grid=use_random_grid,
            random_grid_steps=random_grid_steps,
            oos_ohlcv_df=ticker_oos_ohlcv,
            oos_formula_df=ticker_oos_formula,
        )
        ticker_results[ticker] = asdict(result_obj)

    # Aggregate ticker-level results.
    aggregated_result = _aggregate_stage3_ticker_results(ticker_results)

    # ════════════════════════════════════════════════════════════════════════════════
    # Combination-level aggregation and filtering
    # ════════════════════════════════════════════════════════════════════════════════

    # 1) Track which tickers evaluated each combination and their PASS/FAIL verdicts.
    combination_pass_count: dict[tuple, list[tuple[str, str]]] = {}  # combo_key -> [(ticker, verdict), ...]
    combination_quadrant_stats: dict[tuple, dict] = {}  # combo_key -> {level: {S1: [], S2: [], ..., strictness_value, mean_return}}

    for ticker, ticker_result in ticker_results.items():
        # All evaluated combination results for this ticker.
        all_combo_results = ticker_result.get("all_combination_results", [])

        for combo_result in all_combo_results:
            combination = combo_result.get("combination", [])
            verdict = combo_result.get("verdict", "FAIL")

            # Combination key: sorted tuple of formula names.
            combo_key = tuple(sorted(f["name"] for f in combination))

            # Track ticker verdicts.
            if combo_key not in combination_pass_count:
                combination_pass_count[combo_key] = []
            combination_pass_count[combo_key].append((ticker, verdict))

            # Collect quadrant stats (for cross-ticker aggregation).
            if combo_key not in combination_quadrant_stats:
                combination_quadrant_stats[combo_key] = {}

            strictness_results = combo_result.get("strictness_results", [])
            for s in strictness_results:
                level = s.get("strictness_level")
                quad = s.get("quadrant_stats")
                strictness_value = s.get("strictness_value")
                mean_return = s.get("mean_return")

                if level and quad:
                    if level not in combination_quadrant_stats[combo_key]:
                        combination_quadrant_stats[combo_key][level] = {
                            "S1": [], "S2": [], "S3": [], "S4": [],
                            "strictness_value": None,
                            "mean_return": [],
                        }
                    if combination_quadrant_stats[combo_key][level].get("strictness_value") is None and strictness_value is not None:
                        combination_quadrant_stats[combo_key][level]["strictness_value"] = float(strictness_value)

                    combination_quadrant_stats[combo_key][level]["S1"].append(quad.get("s1_true_positive", 0))
                    combination_quadrant_stats[combo_key][level]["S2"].append(quad.get("s2_false_positive", 0))
                    combination_quadrant_stats[combo_key][level]["S3"].append(quad.get("s3_true_negative", 0))
                    combination_quadrant_stats[combo_key][level]["S4"].append(quad.get("s4_false_negative", 0))
                    combination_quadrant_stats[combo_key][level]["mean_return"].append(float(mean_return or 0.0))

    # 2) Compute pass rate and cross-ticker monotone improvement signals per combination.
    qualified_combinations = []
    combination_stats = {}  # Diagnostics for logging/debugging.

    for combo_key, ticker_verdicts in combination_pass_count.items():
        n_pass = sum(1 for _, v in ticker_verdicts if v == "PASS")
        n_eval = len(ticker_verdicts)
        pass_rate = n_pass / n_eval if n_eval > 0 else 0.0

        # Filter B: cross-ticker (micro-averaged) monotone improvement across strictness.
        # - Aggregate counts across tickers at each strictness level, then compute:
        #   - s2_ratio = S2/(S1+S2)  (lower is better)
        #   - mean_return: signal-count weighted average mean return (higher is better)
        # - B is satisfied if, as strictness increases:
        #   - s2_ratio is non-increasing with at least one strict decrease, OR
        #   - mean_return is non-decreasing with at least one strict increase.
        quad_stats = combination_quadrant_stats.get(combo_key, {})
        s2_ratio_improvement = 0.0
        has_s2_improvement = False
        s2_ratios_by_level: list[float] = []
        mean_returns_by_level: list[float] = []
        mean_return_improvement = 0.0
        has_mean_return_improvement = False

        # Sort strictness levels by strictness_value when available; otherwise fall back to level name.
        level_items = []
        for level_name, level_data in quad_stats.items():
            if not isinstance(level_data, dict):
                continue
            sv = level_data.get("strictness_value")
            level_items.append((level_name, sv))
        if level_items:
            if all(isinstance(sv, (int, float)) for _, sv in level_items):
                level_items.sort(key=lambda x: float(x[1]))
            else:
                level_items.sort(key=lambda x: str(x[0]))

        if len(level_items) >= 2:
            for level_name, _ in level_items:
                s1 = sum(quad_stats[level_name]["S1"])
                s2 = sum(quad_stats[level_name]["S2"])
                denom = (s1 + s2)
                s2_ratio = (s2 / denom) if denom > 0 else 0.0
                s2_ratios_by_level.append(float(s2_ratio))

                # mean_return: ticker-level mean_return weighted by signal_count (S1+S2).
                means = quad_stats[level_name].get("mean_return", []) or []
                weights = [
                    float(a + b)
                    for a, b in zip(quad_stats[level_name]["S1"], quad_stats[level_name]["S2"])
                ]
                w_sum = float(sum(weights))
                if w_sum > 0 and means:
                    mean_return = float(sum(m * w for m, w in zip(means, weights)) / w_sum)
                else:
                    mean_return = 0.0
                mean_returns_by_level.append(mean_return)

            # ΔS2 ratio is for logging/diagnostics only (not used as a threshold).
            s2_ratio_improvement = s2_ratios_by_level[0] - s2_ratios_by_level[-1]

            # Monotone decrease condition: non-increasing + at least one strict decrease.
            eps = 1e-12
            non_increasing = all(
                s2_ratios_by_level[i + 1] <= s2_ratios_by_level[i] + eps
                for i in range(len(s2_ratios_by_level) - 1)
            )
            has_decrease = any(
                s2_ratios_by_level[i + 1] < s2_ratios_by_level[i] - eps
                for i in range(len(s2_ratios_by_level) - 1)
            )
            has_s2_improvement = non_increasing and has_decrease

            # Monotone increase condition: non-decreasing + at least one strict increase.
            mean_return_improvement = mean_returns_by_level[-1] - mean_returns_by_level[0]
            non_decreasing = all(
                mean_returns_by_level[i + 1] >= mean_returns_by_level[i] - eps
                for i in range(len(mean_returns_by_level) - 1)
            )
            has_increase = any(
                mean_returns_by_level[i + 1] > mean_returns_by_level[i] + eps
                for i in range(len(mean_returns_by_level) - 1)
            )
            has_mean_return_improvement = non_decreasing and has_increase

        combination_stats[combo_key] = {
            "n_pass": n_pass,
            "n_eval": n_eval,
            "pass_rate": pass_rate,
            "s2_ratio_improvement": s2_ratio_improvement,
            "has_s2_improvement": has_s2_improvement,
            "s2_ratios_by_level": s2_ratios_by_level,
            "mean_return_improvement": mean_return_improvement,
            "has_mean_return_improvement": has_mean_return_improvement,
            "mean_returns_by_level": mean_returns_by_level,
        }

        # Filter A: ticker-level pass_rate threshold.
        pass_rate_threshold = cfg.stage3.combination_pass_rate_threshold
        qualified_by_pass_rate = pass_rate >= pass_rate_threshold

        # Filter B: cross-ticker monotone improvement (direction-only; no magnitude threshold).
        qualified_by_s2_improvement = has_s2_improvement or has_mean_return_improvement

        # Qualified if (A OR B).
        if qualified_by_pass_rate or qualified_by_s2_improvement:
            # Recover the original combination object (from the first ticker that contains it).
            for ticker, ticker_result in ticker_results.items():
                all_combo_results = ticker_result.get("all_combination_results", [])
                for combo_result in all_combo_results:
                    combination = combo_result.get("combination", [])
                    if tuple(sorted(f["name"] for f in combination)) == combo_key:
                        qualified_combinations.append(combination)
                        break
                else:
                    continue
                break

    all_passed_combinations = qualified_combinations

    print(f"\n[Stage3] Combination filtering (2-tier system):")
    print(f"  - Total unique combinations evaluated: {len(combination_pass_count)}")
    print(f"  - Filter A (pass_rate >= {cfg.stage3.combination_pass_rate_threshold:.0%}): "
          f"{sum(1 for s in combination_stats.values() if s['pass_rate'] >= cfg.stage3.combination_pass_rate_threshold)} combos")
    print(f"  - Filter B (S2_ratio↓ OR mean_return↑ monotonic): "
          f"{sum(1 for s in combination_stats.values() if (s.get('has_s2_improvement') or s.get('has_mean_return_improvement')))} combos")
    print(f"  - Qualified combinations (A OR B): {len(qualified_combinations)}")
    print(f"\n  Top combinations by criteria:")

    # Print qualified vs. non-qualified combinations (for quick inspection).
    qualified_keys = {tuple(sorted(f["name"] for f in combo)) for combo in qualified_combinations}

    for i, (combo_key, stats) in enumerate(sorted(combination_stats.items(),
                                                    key=lambda x: (x[1]["pass_rate"], x[1]["s2_ratio_improvement"]),
                                                    reverse=True)[:10], 1):
        is_qualified = combo_key in qualified_keys
        status = "✓" if is_qualified else "✗"
        reason = []
        if stats["pass_rate"] >= cfg.stage3.combination_pass_rate_threshold:
            reason.append(f"A: pass_rate={stats['pass_rate']:.1%}")
        if stats.get("has_s2_improvement"):
            reason.append(f"B(s2): monotone ΔS2={stats['s2_ratio_improvement']:+.3f}")
        if stats.get("has_mean_return_improvement"):
            reason.append(f"B(r): monotone Δμ={stats.get('mean_return_improvement', 0.0):+.4f}")
        reason_str = " | ".join(reason) if reason else "no criteria met"

        print(f"    {status} [{i}] {combo_key[:3]}...")
        print(f"        pass={stats['n_pass']}/{stats['n_eval']} ({stats['pass_rate']:.1%}), "
              f"ΔS2={stats.get('s2_ratio_improvement', 0.0):+.3f}, "
              f"Δμ={stats.get('mean_return_improvement', 0.0):+.4f}")
        print(f"        → {reason_str}")

    # Auto-generate hypothesis instance IDs.
    combination_ids = [
        f"{hypothesis_id}_instance_{i+1:03d}"
        for i in range(len(all_passed_combinations))
    ]

    # Generate report.
    report_md = _generate_aggregated_stage3_report(
        hypothesis_id, ticker_results, aggregated_result, all_passed_combinations, combination_ids
    )

    # Build the lightweight output summary based on aggregated_result.
    result = {
        "hypothesis_id": hypothesis_id,
        "n_tickers": aggregated_result.get("n_tickers", 0),
        "overall_verdict": aggregated_result.get("aggregated_verdict", "UNKNOWN"),
        "generalizability": aggregated_result.get("generalizability", "NONE"),
        "pass_rate": aggregated_result.get("pass_rate", 0.0),
        "avg_confidence": aggregated_result.get("avg_confidence", 0.0),
        "ticker_verdicts": aggregated_result.get("ticker_verdicts", {}),
        "direction_consistency": aggregated_result.get("direction_consistency", {}),
        "n_passed_combinations": len(all_passed_combinations),
        "passed_combination_ids": combination_ids,
        "passed_combination_names": [
            [f.get("name", "unknown") for f in combo]
            for combo in all_passed_combinations
        ],
    }

    return Stage3Result(
        hypothesis_id=hypothesis_id,
        result=result,
        report_md=report_md,
        ticker_results=ticker_results,
        aggregated_result=aggregated_result,
        passed_combinations=all_passed_combinations,
        combination_stats=combination_stats,
    )


def _generate_aggregated_stage3_report(
    hypothesis_id: str,
    ticker_results: dict[str, dict[str, Any]],
    aggregated_result: dict[str, Any],
    passed_combinations: list[list[dict[str, Any]]] | None = None,
    combination_ids: list[str] | None = None,
) -> str:
    """Generate a Markdown report for multi-ticker Stage3 aggregation."""
    lines = []
    lines.append("# Stage 3: Hypothesis Instance Validation Report (Multi-ticker)")
    lines.append("")
    lines.append(f"**Hypothesis ID**: {hypothesis_id}")
    lines.append(f"**Overall Verdict**: {aggregated_result.get('aggregated_verdict', 'N/A')}")
    lines.append(f"**Generalizability**: {aggregated_result.get('generalizability', 'N/A')}")
    lines.append(f"**Number of Tickers**: {aggregated_result.get('n_tickers', 0)}")
    lines.append("")

    # Aggregation summary.
    n_eval = aggregated_result.get("n_evaluated", aggregated_result.get("n_tickers", 0))
    lines.append("## Aggregation Summary")
    lines.append(f"- **Pass Rate**: {aggregated_result.get('pass_rate', 0):.1%} ({aggregated_result.get('n_passed', 0)}/{n_eval} evaluated tickers)")
    lines.append(f"- **Avg Confidence**: {aggregated_result.get('avg_confidence', 0):.3f}")
    lines.append(f"- **Failed**: {aggregated_result.get('n_failed', 0)} tickers")
    lines.append("")

    # Passed combinations.
    if passed_combinations:
        lines.append("## Passed Combinations")
        lines.append(f"Total qualified combinations: {len(passed_combinations)}")
        lines.append("")
        for i, combo in enumerate(passed_combinations):
            combo_names = [f.get("name", "unknown") for f in combo]
            combo_id = combination_ids[i] if combination_ids and i < len(combination_ids) else f"instance_{i+1:03d}"
            lines.append(f"{i+1}. **{combo_id}**: `{' + '.join(combo_names)}`")
        lines.append("")

    # Direction consistency.
    direction_consistency = aggregated_result.get("direction_consistency", {})
    if direction_consistency:
        lines.append("## Direction Consistency (across tickers)")
        for metric, info in direction_consistency.items():
            lines.append(f"- **{metric}**: {info.get('dominant_direction', 'N/A')} (consistency: {info.get('consistency', 0):.1%})")
        lines.append("")

    # Per-ticker verdicts.
    lines.append("## Ticker-level Verdicts")
    ticker_verdicts = aggregated_result.get("ticker_verdicts", {})
    for ticker, verdict in ticker_verdicts.items():
        lines.append(f"- {ticker}: {verdict}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# Stage 3: methodological notes
# ════════════════════════════════════════════════════════════════════════════════
#
# Stage 3 provides a final check of whether the hypothesis is structurally consistent in the data.
#
# Rationale for monotonicity-based validation
#
# If a hypothesis describes a real and stable event in the data, then making its definition stricter should
# typically sharpen the outcome. This is expressed as a monotone relationship between strictness and
# performance-like measurements.
#
# If the monotone relationship does not hold, common explanations include:
# - the observation decomposition is not appropriate,
# - the formulas do not capture the intended observations, or
# - the hypothesis itself is not supported by the data.
#
# Position in the overall framework
#
# - Stage 1: transform a hypothesis into a testable form
# - Stage 2: verify formulas implement observations (return-agnostic)
# - Stage 3: verify the hypothesis is structurally consistent in the data
#
# Practical takeaways
#
# After Stage 3, you can often infer:
#
# 1) hypothesis structural validity (consistency of the observation structure)
#
# 2) whether the decomposed observations capture the event well
#
# 3) operational implications:
#    - which strictness level is reasonable
#    - the trade-off between signal frequency and quality
#
# ════════════════════════════════════════════════════════════════════════════════
