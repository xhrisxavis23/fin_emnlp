"""
================================================================================
Stage 3: Hypothesis Instance Validation Agent
Hypothesis Structure Validation Agent (second core validation stage)
================================================================================

[Purpose]
Build hypothesis instances by combining formulas validated in Stage 2, and verify whether
strengthening the hypothesis definition (higher strictness) makes the hypothesis-implied
outcome characteristics clearer.

[Core Question]
"As we make the hypothesis definition stricter, do the outcome characteristics predicted by
the hypothesis become clearer?"

[Principles]
- The strategy is an experimental instrument, not the research objective.
- Strategy performance is a summary signal used to assess structural consistency.
- This research does not propose a "profit-maximizing strategy".
- Instead, it identifies which financial hypotheses exhibit a consistent observable structure in data.

[Validation Steps]
1. Build hypothesis instances (H_t = obs1_t ∧ obs2_t ∧ ... ∧ obsN_t)
2. Define outcome variables (Forward Returns)
3. Evaluate a strictness grid
4. Verify monotonicity (Strictness ↑ → Performance ↑)
5. Run a strategy (measurement tool)
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from agent.base_agent import BaseAgent
from schemas.hypothesis_validation_dataclasses import (
    HypothesisValidationResult,
    ICMetrics,
    MonotonicityVerification,
    QuadrantStats,
    StrictnessLevelResult,
)
from util.run_context import RunContext

logger = logging.getLogger(__name__)


# ============================================================================
# Hypothesis Validation Agent (Stage 3)
# ============================================================================

class HypothesisValidationAgent(BaseAgent):
    """
    Stage 3: Hypothesis Instance Validation Agent

    Purpose:
    - Build hypothesis instances and evaluate a strictness grid
    - Verify monotonicity: Strictness ↑ → Performance ↑
    - Use strategy execution only as a measurement tool

    Core validation logic:
    1. Build hypothesis instances via logical AND combinations of Stage 2 PASS formulas
    2. Create hypothesis instances per threshold (strictness) level
    3. Measure performance metrics at each level
    4. Confirm monotonic improvement as strictness increases
    """

    # Default strictness grid
    DEFAULT_STRICTNESS_GRID = {
        # strictness_value in (0, 1): larger => stricter (fewer events).
        # Interpretation:
        # - higher_is_more_true: values >= quantile(strictness_value)  -> keeps top (1 - strictness_value)
        # - lower_is_more_true : values <= quantile(1 - strictness_value) -> keeps bottom (1 - strictness_value)
        "very_loose": 0.2,
        "loose": 0.3,
        "medium": 0.5,
        "strict": 0.7,
        "very_strict": 0.9,
    }

    def __init__(
        self,
        model: str,
        run_ctx: Optional[RunContext] = None,
        strictness_grid: Optional[Dict[str, float]] = None,
        horizon_days: int = 5,
        monotonicity_threshold: float = 0.7,
        use_random_grid: bool = True,
        random_grid_steps: int = 10,
    ):
        """
        Initialize HypothesisValidationAgent.

        Args:
            model: LLM model name (used for interpretation/logging)
            run_ctx: RunContext for logging
            strictness_grid: Per-level thresholds (quantiles). If None, use default or a random grid.
            horizon_days: Time horizon (days) for outcome calculation
            monotonicity_threshold: Threshold for monotonicity verdict
            use_random_grid: If True, use a progressive random grid instead of a fixed grid
            random_grid_steps: Number of steps to generate when using a random grid
        """
        super().__init__(model=model, run_ctx=run_ctx)
        self.strictness_grid = strictness_grid or self.DEFAULT_STRICTNESS_GRID
        self.horizon_days = horizon_days
        self.monotonicity_threshold = monotonicity_threshold
        self.use_random_grid = use_random_grid
        self.random_grid_steps = random_grid_steps

    def _group_formulas_by_observation(
        self,
        passed_formulas: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group passed formulas by `observation_id`.

        Args:
            passed_formulas: List of formulas that passed Stage 2

        Returns:
            Dict[obs_id, List[formula]]: formulas grouped per `observation_id`
        """
        obs_groups: Dict[str, List[Dict[str, Any]]] = {}

        for formula in passed_formulas:
            if not isinstance(formula, dict):
                continue

            # Extract observation_id (support multiple field names)
            obs_id = (
                formula.get("observation_id") or
                formula.get("obs_id") or
                formula.get("name", "unknown")
            )
            obs_id = str(obs_id).strip()

            if obs_id not in obs_groups:
                obs_groups[obs_id] = []
            obs_groups[obs_id].append(formula)

        return obs_groups

    def _generate_formula_combinations(
        self,
        obs_groups: Dict[str, List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate all combinations of passed formulas across observations.

        Example:
            obs1 = {factor1, factor2}
            obs2 = {factor3, factor4}
            obs3 = {factor5}
            => combinations:
                [{factor1, factor3, factor5},
                 {factor1, factor4, factor5},
                 {factor2, factor3, factor5},
                 {factor2, factor4, factor5}]

        Args:
            obs_groups: Formula list per `observation_id`

        Returns:
            List[List[formula]]: Each combination contains one formula per observation
        """
        if not obs_groups:
            return []

        # Get sorted obs_ids for consistent ordering
        obs_ids = sorted(obs_groups.keys())

        # Get formula lists for each observation
        formula_lists = [obs_groups[obs_id] for obs_id in obs_ids]

        # Generate all combinations using itertools.product
        combinations = list(itertools.product(*formula_lists))

        # Convert tuples to lists
        return [list(combo) for combo in combinations]

    def _generate_progressive_grid(
        self,
        formula_names: List[str],
        n_steps: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate a uniform percentile grid (top 10%, 20%, 30%, ...).

        Algorithm:
        1. Split the 0.10–0.90 range into `n_steps` evenly spaced percentiles
        2. For each step:
           - Create `level_name`: "top{percentile}pct"
           - Assign the same threshold to every formula (the percentile-derived threshold)
        3. Reduce combinations drastically (N^M → N)

        Purpose:
        - Use uniform percentile thresholds for strictness monotonicity verification
        - Reduce combination count by applying the same threshold to all formulas
        - Keep flexibility: Stage 4 (Optuna) can later tune per-formula thresholds

        Args:
            formula_names: List of formula names
            n_steps: Number of strictness levels to generate

        Returns:
            Dict[level_name, Dict[formula_name, threshold]]
            Example: {"top10pct": {"factor1": 0.90, "factor2": 0.90},
                 "top20pct": {"factor1": 0.80, "factor2": 0.80}, ...}
        """
        grid: Dict[str, Dict[str, float]] = {}

        # NOTE:
        # - default: evenly split 0.10–0.90 into `n_steps` (top10%–top90%)
        # - special-case (n_steps=3): use 0.50/0.70/0.90 quantiles for cleaner mean-reversion strictness ladder
        if n_steps == 3:
            thresholds = [0.5, 0.7, 0.9]
            for thr in thresholds:
                level_name = f"q{int(thr * 100)}"
                grid[level_name] = {fname: thr for fname in formula_names}
            return grid

        # Evenly split 0.10–0.90 (top 10%–top 90%)
        percentiles = np.linspace(0.10, 0.90, n_steps)

        for pct in percentiles:
            # Top X% → threshold = 1 - X/100
            # Example: top 10% → 0.90, top 20% → 0.80
            threshold = 1.0 - pct
            level_name = f"top{int(pct*100)}pct"

            # Apply the same threshold to all formulas
            grid[level_name] = {fname: threshold for fname in formula_names}

        return grid

    def validate_hypothesis(
        self,
        hypothesis_id: str,
        passed_formulas: List[Dict[str, Any]],
        ohlcv_df: pd.DataFrame,
        formula_df: pd.DataFrame,
        oos_ohlcv_df: Optional[pd.DataFrame] = None,
        oos_formula_df: Optional[pd.DataFrame] = None,
    ) -> HypothesisValidationResult:
        """
        Build hypothesis instances and evaluate the strictness grid.

        Args:
            hypothesis_id: Hypothesis ID
            passed_formulas: Formulas that passed Stage 2 (multiple per observation possible)
            ohlcv_df: Raw OHLCV data (in-sample)
            formula_df: DataFrame containing computed formula values (in-sample)
            oos_ohlcv_df: Out-of-sample OHLCV data (for IC logging only; not used for the verdict)
            oos_formula_df: Out-of-sample formula value DataFrame (for IC logging only)

        Returns:
            HypothesisValidationResult: Validation result

        Note:
            - Each observation can have multiple passed formulas
            - obs1 = {factor1, factor2}, obs2 = {factor3, factor4, factor5}
            - combinations: {factor1, factor3}, {factor1, factor4}, ..., {factor2, factor5}
            - OOS data is used only for IC logging; it does not affect the validation verdict
        """
        logger.info(f"Starting Stage 3 validation for hypothesis: {hypothesis_id}")
        logger.info(f"Using {len(passed_formulas)} passed formulas from Stage 2")

        # Group formulas by observation_id
        obs_groups = self._group_formulas_by_observation(passed_formulas)
        logger.info(f"Formulas grouped into {len(obs_groups)} observations")
        for obs_id, formulas in obs_groups.items():
            logger.info(f"  - {obs_id}: {len(formulas)} formulas")

        # Generate all combinations
        combinations = self._generate_formula_combinations(obs_groups)
        n_combinations = len(combinations)
        logger.info(f"Generated {n_combinations} formula combinations")

        if n_combinations == 0:
            logger.warning("No formula combinations generated!")
            return HypothesisValidationResult(
                hypothesis_id=hypothesis_id,
                overall_verdict="FAIL",
                confidence=0.0,
                strictness_results=[],
                monotonicity_results=[],
                key_findings=["No valid formula combinations"],
                conclusion="No formula combinations could be generated from passed formulas",
            )

        # Step 1: Define outcome variable (forward returns)
        outcome_series = self._define_outcome_variable(ohlcv_df)

        # Step 2: Evaluate all combinations and collect PASS results
        passed_combinations = []  # combinations that PASS
        all_combination_results = []  # evaluation results for all combinations

        for combo_idx, combo in enumerate(combinations):
            combo_names = [f.get("name") for f in combo]
            logger.info(f"Evaluating combination {combo_idx + 1}/{n_combinations}: {combo_names}")

            # Choose which grid to use
            if self.use_random_grid:
                formula_names = [f.get("name") for f in combo if f.get("name")]
                target_grid = self._generate_progressive_grid(formula_names, self.random_grid_steps)
            else:
                target_grid = self.strictness_grid

            # Evaluate strictness grid
            strictness_results = []

            # Iterate over grid
            def get_sort_key(item):
                k, v = item
                if isinstance(v, dict):
                    return sum(v.values()) / len(v) if v else 0
                return v

            sorted_grid = sorted(target_grid.items(), key=get_sort_key)

            for level_name, threshold_or_dict in sorted_grid:
                # Compute a representative value (float) for logging/reporting
                if isinstance(threshold_or_dict, dict):
                    representative_value = sum(threshold_or_dict.values()) / len(threshold_or_dict) if threshold_or_dict else 0.0
                else:
                    representative_value = threshold_or_dict

                result = self._evaluate_strictness_level(
                    level_name=level_name,
                    threshold_input=threshold_or_dict,
                    representative_value=representative_value,
                    passed_formulas=combo,
                    formula_df=formula_df,
                    outcome_series=outcome_series,
                )
                strictness_results.append(result)

            # Monotonicity verification
            monotonicity_results = self._verify_monotonicity(strictness_results)

            # Final verdict
            verdict, confidence, findings, conclusion = self._make_final_decision(
                strictness_results, monotonicity_results
            )

            # Store combination result
            combination_result = {
                "combination": combo,
                "combination_names": combo_names,
                "verdict": verdict,
                "confidence": confidence,
                "strictness_results": strictness_results,
                "monotonicity_results": monotonicity_results,
                "findings": findings,
                "conclusion": conclusion,
            }
            all_combination_results.append(combination_result)

            # Collect only PASS combinations
            if verdict == "PASS":
                passed_combinations.append(combo)
                logger.info(f"✓ Combination {combo_idx + 1}: PASS (confidence={confidence:.2f})")
            else:
                logger.info(f"✗ Combination {combo_idx + 1}: FAIL (confidence={confidence:.2f})")

        # Build final result
        logger.info(f"Stage 3 validation complete: {len(passed_combinations)} PASS / {n_combinations} total combinations")

        if not passed_combinations:
            # All combinations FAIL
            logger.warning("No combinations passed validation!")
            return HypothesisValidationResult(
                hypothesis_id=hypothesis_id,
                overall_verdict="FAIL",
                confidence=0.0,
                passed_combinations=[],
                all_combination_results=all_combination_results,  # ✅ return results for all combinations, too
                key_findings=["No combinations passed validation"],
                conclusion="All combinations failed monotonicity verification. Review the hypothesis observation conditions.",
            )

        # Pick the PASS combination with the highest confidence (for reporting)
        best_passed = max(all_combination_results, key=lambda x: x["confidence"] if x["verdict"] == "PASS" else -1)

        # IC/ICIR computation (representative combo, logging only)
        # NOTE: IC is not used for the verdict; disabled for performance.
        # representative_combo = best_passed["combination"]
        # combo_names = best_passed["combination_names"]

        # # IS IC computation
        # is_signal = self._compute_and_signal_value(representative_combo, formula_df)
        # is_ic_mean, is_ic_std, is_icir, is_ic_pos_ratio, is_n_days = self._compute_ic_metrics(
        #     is_signal, outcome_series
        # )

        # # OOS IC computation (only when data is provided)
        # oos_ic_mean, oos_ic_std, oos_icir, oos_ic_pos_ratio, oos_n_days = 0.0, 0.0, 0.0, 0.0, 0
        # if oos_ohlcv_df is not None and oos_formula_df is not None and not oos_ohlcv_df.empty:
        #     oos_outcome = self._define_outcome_variable(oos_ohlcv_df)
        #     oos_signal = self._compute_and_signal_value(representative_combo, oos_formula_df)
        #     oos_ic_mean, oos_ic_std, oos_icir, oos_ic_pos_ratio, oos_n_days = self._compute_ic_metrics(
        #         oos_signal, oos_outcome
        #     )

        # # Build ICMetrics
        # ic_metrics = ICMetrics(
        #     is_ic_mean=is_ic_mean,
        #     is_ic_std=is_ic_std,
        #     is_icir=is_icir,
        #     is_ic_positive_ratio=is_ic_pos_ratio,
        #     is_n_days=is_n_days,
        #     oos_ic_mean=oos_ic_mean,
        #     oos_ic_std=oos_ic_std,
        #     oos_icir=oos_icir,
        #     oos_ic_positive_ratio=oos_ic_pos_ratio,
        #     oos_n_days=oos_n_days,
        # )

        # logger.info(f"IC Metrics (representative) - IS: IC={is_ic_mean:.4f}, ICIR={is_icir:.4f}, OOS: IC={oos_ic_mean:.4f}, ICIR={oos_icir:.4f}")

        combo_names = best_passed["combination_names"]

        # Return final result (PASS combinations + performance info for all combinations)
        return HypothesisValidationResult(
            hypothesis_id=hypothesis_id,
            overall_verdict="PASS",
            confidence=best_passed["confidence"],
            strictness_results=best_passed["strictness_results"],
            monotonicity_results=best_passed["monotonicity_results"],
            # ic_metrics=ic_metrics,  # IC computation disabled
            passed_combinations=passed_combinations,  # keep the raw list
            all_combination_results=all_combination_results,  # used by Stage 4 to pick the best
            key_findings=best_passed["findings"] + [
                f"Passed combinations: {len(passed_combinations)}/{n_combinations}",
                f"Representative combination: {combo_names}",
            ],
            conclusion=best_passed["conclusion"] + f"\n\n{len(passed_combinations)} combinations passed validation.",
        )

    def _define_outcome_variable(
        self,
        ohlcv_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Define the outcome variable.

        Default: forward return over `horizon_days`.
        """
        # Find Close column
        close_col = None
        for col in ohlcv_df.columns:
            if col.lower() in ['close', 'c']:
                close_col = col
                break

        if close_col is None:
            raise ValueError("Close price column not found in ohlcv_df")

        close_prices = ohlcv_df[close_col]

        # Forward return: (Close[t+n] - Close[t]) / Close[t]
        forward_return = close_prices.shift(-self.horizon_days) / close_prices - 1

        return forward_return

    def _construct_hypothesis_instance(
        self,
        passed_formulas: List[Dict[str, Any]],
        formula_df: pd.DataFrame,
        threshold_input: float | Dict[str, float],
    ) -> pd.Series:
        """
        Construct a hypothesis instance: H_t = obs1_t ∧ obs2_t ∧ ... ∧ obsN_t

        Each observation becomes True when its formula value meets the threshold.
        H_t is True only when all observations are True.

        Note:
            - `passed_formulas` already contains one formula per observation
            - (selected during the combination generation stage)

        Args:
            passed_formulas: Selected formulas (one per observation)
            formula_df: Formula value DataFrame
            threshold_input: Threshold input (quantile). float (global) or dict (per-formula)

        Returns:
            Hypothesis instance time series (boolean)
        """
        if not passed_formulas:
            return pd.Series(False, index=formula_df.index)

        obs_signals = []

        for formula in passed_formulas:
            formula_name = formula.get("name", "")
            polarity = formula.get("polarity", "higher_is_more_true")

            if formula_name not in formula_df.columns:
                logger.warning(f"Formula {formula_name} not in formula_df, skipping")
                continue

            values = formula_df[formula_name]

            # Determine threshold
            if isinstance(threshold_input, dict):
                threshold = threshold_input.get(formula_name, 0.5)  # fallback to median
            else:
                threshold = threshold_input

            # Quantile-based thresholding
            if polarity == "higher_is_more_true":
                # strictness_value ↑ => keep only more extreme upper tail (stricter)
                quantile_threshold = values.quantile(threshold)
                obs_signal = values >= quantile_threshold
            else:
                # strictness_value ↑ => keep only more extreme lower tail (stricter)
                quantile_threshold = values.quantile(1 - threshold)
                obs_signal = values <= quantile_threshold

            obs_signals.append(obs_signal)

        if not obs_signals:
            return pd.Series(False, index=formula_df.index)

        # Logical AND: H_t is True only when all obs are True
        hypothesis_instance = obs_signals[0]
        for signal in obs_signals[1:]:
            hypothesis_instance = hypothesis_instance & signal

        return hypothesis_instance

    def _evaluate_strictness_level(
        self,
        level_name: str,
        threshold_input: float | Dict[str, float],
        representative_value: float,
        passed_formulas: List[Dict[str, Any]],
        formula_df: pd.DataFrame,
        outcome_series: pd.Series,
    ) -> StrictnessLevelResult:
        """
        Evaluate a single strictness level.

        Args:
            level_name: Strictness level name (e.g., "loose", "strict")
            threshold_input: Threshold input (float or dict)
            representative_value: Representative threshold value for logging/reporting (float)
            passed_formulas: Selected formulas
            formula_df: Formula value DataFrame
            outcome_series: Outcome variable series

        Returns:
            StrictnessLevelResult
        """
        # Build hypothesis instance
        h_t = self._construct_hypothesis_instance(passed_formulas, formula_df, threshold_input)

        # Use only valid data (exclude missing values)
        valid_mask = ~outcome_series.isna() & ~h_t.isna()
        h_t_valid = h_t[valid_mask]
        outcome_valid = outcome_series[valid_mask]

        total_samples = len(h_t_valid)
        signal_count = h_t_valid.sum()

        if signal_count == 0:
            # logger.debug(f"No signals at strictness level {level_name}")
            return StrictnessLevelResult(
                strictness_level=level_name,
                strictness_value=representative_value,
                signal_count=0,
                signal_frequency=0.0,
            )

        # Signal characteristics
        signal_frequency = signal_count / total_samples

        # Outcomes when the signal is active/inactive
        signal_outcomes = outcome_valid[h_t_valid]
        no_signal_outcomes = outcome_valid[~h_t_valid]

        # Return statistics
        mean_return = float(signal_outcomes.mean())
        median_return = float(signal_outcomes.median())
        std_return = float(signal_outcomes.std())
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

        # Win/loss distribution
        wins = signal_outcomes[signal_outcomes > 0]
        losses = signal_outcomes[signal_outcomes <= 0]
        win_rate = len(wins) / len(signal_outcomes) if len(signal_outcomes) > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')

        # Quadrant analysis
        quadrant = QuadrantStats(
            s1_true_positive=int(((h_t_valid == True) & (outcome_valid > 0)).sum()),
            s2_false_positive=int(((h_t_valid == True) & (outcome_valid <= 0)).sum()),
            s3_true_negative=int(((h_t_valid == False) & (outcome_valid <= 0)).sum()),
            s4_false_negative=int(((h_t_valid == False) & (outcome_valid > 0)).sum()),
        )

        return StrictnessLevelResult(
            strictness_level=level_name,
            strictness_value=representative_value,
            signal_count=int(signal_count),
            signal_frequency=signal_frequency,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.0,
            quadrant_stats=quadrant,
        )

    def _verify_monotonicity(
        self,
        strictness_results: List[StrictnessLevelResult],
    ) -> List[MonotonicityVerification]:
        """
        Verify monotonic performance improvement as strictness increases.

        Targets:
        1. Decrease in S2 ratio (lower false-positive rate)
        2. Increase in precision (S1/(S1+S2))
        3. Increase in mean return
        4. Increase in Sharpe ratio
        """
        results = []

        # Sort by strictness
        sorted_results = sorted(strictness_results, key=lambda x: x.strictness_value)

        # IMPORTANT:
        # - Levels with signal_count == 0 have "unobservable" performance metrics, so exclude them.
        # - Otherwise, the region where signals disappear as strictness increases can skew comparisons
        #   and create fake improving/degrading signals.
        sorted_results = [r for r in sorted_results if getattr(r, "signal_count", 0) > 0]
        strictness_values = [r.strictness_value for r in sorted_results]

        # Metrics to verify
        metrics = {
            "precision": [r.quadrant_stats.precision if r.quadrant_stats else np.nan for r in sorted_results],
            "mean_return": [r.mean_return for r in sorted_results],
            "sharpe_ratio": [r.sharpe_ratio for r in sorted_results],
            "s2_ratio": [
                r.quadrant_stats.s2_false_positive / (r.quadrant_stats.s1_true_positive + r.quadrant_stats.s2_false_positive)
                if r.quadrant_stats and (r.quadrant_stats.s1_true_positive + r.quadrant_stats.s2_false_positive) > 0
                else np.nan
                for r in sorted_results
            ],
        }

        # S2 ratio should decrease, so invert sign
        improving_direction = {
            "precision": 1,
            "mean_return": 1,
            "sharpe_ratio": 1,
            "s2_ratio": -1,  # decrease is improvement
        }

        for metric_name, values in metrics.items():
            # Use only valid values
            valid_pairs = [(s, v) for s, v in zip(strictness_values, values) if not np.isnan(v)]
            if len(valid_pairs) < 3:
                continue

            _, m_vals = zip(*valid_pairs)

            # Monotonicity check based on adjacent-level comparisons
            expected_direction = improving_direction[metric_name]
            n_pairs = len(m_vals) - 1
            improving_count = sum(
                1 for i in range(n_pairs)
                if (m_vals[i+1] - m_vals[i]) * expected_direction > 0
            )
            degrading_count = sum(
                1 for i in range(n_pairs)
                if (m_vals[i+1] - m_vals[i]) * expected_direction < 0
            )
            monotonicity_score = improving_count / n_pairs if n_pairs > 0 else 0.0

            # Monotonicity verdict: monotonic if score meets threshold
            is_monotonic = monotonicity_score >= self.monotonicity_threshold

            # Direction label: improving vs degrading vs non_monotonic
            if is_monotonic:
                direction = "improving"
            elif degrading_count / n_pairs >= self.monotonicity_threshold:
                direction = "degrading"
            else:
                direction = "non_monotonic"

            results.append(MonotonicityVerification(
                metric_name=metric_name,
                values=list(m_vals),
                is_monotonic=is_monotonic,
                direction=direction,
                spearman_corr=monotonicity_score,  # used as monotonicity score
                spearman_pvalue=0.0,  # unused (always 0)
            ))

        return results

    def _make_final_decision(
        self,
        strictness_results: List[StrictnessLevelResult],
        monotonicity_results: List[MonotonicityVerification],
    ) -> Tuple[str, float, List[str], str]:
        """
        Make the final verdict.

        PASS:
        - Monotonic improvement in at least one of {precision, mean_return}

        FAIL:
        - Neither precision nor mean_return improves monotonically
        """
        findings = []

        # Check signal count
        total_signals = sum(r.signal_count for r in strictness_results)
        if total_signals == 0:
            return "FAIL", 0.0, ["No signals triggered"], "The hypothesis instance is not observed in the data."

        # Target metrics (precision, mean_return)
        # NOTE: In the current implementation, win_rate == precision, so we avoid redundant metrics.
        target_metric_names = {"precision", "mean_return"}
        target_metrics = [r for r in monotonicity_results if r.metric_name in target_metric_names]
        improving_metrics = [r for r in target_metrics if r.direction == "improving"]

        # Summary per strictness level
        for r in strictness_results:
            if r.quadrant_stats:
                findings.append(
                    f"{r.strictness_level}: signals={r.signal_count}, "
                    f"precision={r.quadrant_stats.precision:.2%}, "
                    f"mean_return={r.mean_return:.4f}"
                )

        # Monotonicity summary
        for r in monotonicity_results:
            status = "✓" if r.direction == "improving" else "✗"
            findings.append(
                f"{status} {r.metric_name}: {r.direction} "
                f"(score={r.spearman_corr:.3f})"
            )

        # Verdict: PASS if at least one of {precision, mean_return} is improving
        if len(improving_metrics) >= 1:
            verdict = "PASS"
            confidence = np.mean([r.spearman_corr for r in improving_metrics if r.spearman_corr > 0])
            improving_names = ", ".join([r.metric_name for r in improving_metrics])
            conclusion = (
                "The hypothesis is structurally valid in the data. "
                f"Monotonic improvement is confirmed for {len(improving_metrics)} metric(s) ({improving_names})."
            )
        else:
            verdict = "FAIL"
            confidence = 0.0
            degrading = [r for r in target_metrics if r.direction == "degrading"]
            if degrading:
                degrading_names = ", ".join([r.metric_name for r in degrading])
                conclusion = (
                    "The hypothesis structure is not valid in the data. "
                    f"As strictness increased, performance worsened for metric(s): {degrading_names}."
                )
            else:
                conclusion = (
                    "No consistent relationship is found between hypothesis structure and performance. "
                    "Neither precision nor mean_return improves monotonically."
                )

        return verdict, float(confidence), findings, conclusion

    # ========================================================================
    # IC computation helpers (not used for verdict; disabled)
    # ========================================================================

    # def _compute_ic_metrics(
    #     self,
    #     signal_series: pd.Series,
    #     outcome_series: pd.Series,
    # ) -> Tuple[float, float, float, float, int]:
    #     """
    #     Compute IC (Information Coefficient) metrics.

    #     IC = mean of daily correlation between (signal, forward_return)
    #     ICIR = IC_mean / IC_std

    #     Args:
    #         signal_series: Signal values for an AND combination (continuous or 0/1)
    #         outcome_series: Forward return

    #     Returns:
    #         (ic_mean, ic_std, icir, ic_positive_ratio, n_days)
    #     """
    #     # Use only valid data
    #     valid_mask = ~signal_series.isna() & ~outcome_series.isna()
    #     signal_valid = signal_series[valid_mask]
    #     outcome_valid = outcome_series[valid_mask]

    #     if len(signal_valid) < 10:
    #         return 0.0, 0.0, 0.0, 0.0, 0

    #     # Daily IC (date-based cross-sectional correlation)
    #     # For a single ticker, use rolling correlation across the full period.
    #     # Here we compute overall IC for a single-ticker case.
    #     try:
    #         # Spearman rank correlation (IC typically uses rank correlation)
    #         ic, _ = stats.spearmanr(signal_valid, outcome_valid)
    #         if np.isnan(ic):
    #             ic = 0.0
    #     except Exception:
    #         ic = 0.0

    #     # Rolling IC (window=20)
    #     window = min(20, len(signal_valid) // 5)
    #     if window < 5:
    #         window = 5

    #     rolling_ics = []
    #     for i in range(window, len(signal_valid)):
    #         window_signal = signal_valid.iloc[i - window:i]
    #         window_outcome = outcome_valid.iloc[i - window:i]
    #         try:
    #             corr, _ = stats.spearmanr(window_signal, window_outcome)
    #             if not np.isnan(corr):
    #                 rolling_ics.append(corr)
    #         except Exception:
    #             continue

    #     if len(rolling_ics) < 2:
    #         # Fallback: return overall IC only
    #         return ic, 0.0, 0.0, 1.0 if ic > 0 else 0.0, len(signal_valid)

    #     ic_mean = float(np.mean(rolling_ics))
    #     ic_std = float(np.std(rolling_ics))
    #     icir = ic_mean / ic_std if ic_std > 0 else 0.0
    #     ic_positive_ratio = sum(1 for x in rolling_ics if x > 0) / len(rolling_ics)

    #     return ic_mean, ic_std, icir, ic_positive_ratio, len(rolling_ics)

    # def _compute_and_signal_value(
    #     self,
    #     passed_formulas: List[Dict[str, Any]],
    #     formula_df: pd.DataFrame,
    # ) -> pd.Series:
    #     """
    #     Compute a continuous signal value for an AND combination.

    #     Normalize each formula value to [0, 1] and multiply to represent AND strength.
    #     Used for IC computation.
    #     """
    #     signal_values = []

    #     for formula in passed_formulas:
    #         name = formula.get("name")
    #         direction = formula.get("direction", "higher_is_more_true")

    #         if name not in formula_df.columns:
    #             continue

    #         values = formula_df[name]

    #         # Min-max normalization to [0, 1]
    #         v_min, v_max = values.min(), values.max()
    #         if v_max - v_min > 1e-10:
    #             normalized = (values - v_min) / (v_max - v_min)
    #         else:
    #             normalized = pd.Series(0.5, index=values.index)

    #         # Invert depending on direction
    #         if direction == "lower_is_more_true":
    #             normalized = 1 - normalized

    #         signal_values.append(normalized)

    #     if not signal_values:
    #         return pd.Series(dtype=float)

    #     # AND combination: multiplication (higher when all conditions are strong)
    #     combined = signal_values[0]
    #     for sv in signal_values[1:]:
    #         combined = combined * sv

    #     return combined

    def generate_validation_report(
        self,
        result: HypothesisValidationResult,
    ) -> str:
        """
        Generate a human-readable report for the validation result.
        """
        lines = []
        lines.append("# Stage 3: Hypothesis Instance Validation Report")
        lines.append("")
        lines.append(f"**Hypothesis ID**: {result.hypothesis_id}")
        lines.append(f"**Overall Verdict**: {result.overall_verdict}")
        lines.append(f"**Confidence**: {result.confidence:.2f}")
        lines.append("")

        # Conclusion
        lines.append("## Conclusion")
        lines.append(result.conclusion)
        lines.append("")

        # Strictness grid
        lines.append("## Strictness Grid Evaluation")
        lines.append("")
        lines.append("| Level | Threshold | Signals | Precision | Mean Return | Sharpe |")
        lines.append("|-------|-----------|---------|-----------|-------------|--------|")
        for r in result.strictness_results:
            precision = r.quadrant_stats.precision if r.quadrant_stats else 0.0
            lines.append(
                f"| {r.strictness_level} | {r.strictness_value:.0%} | {r.signal_count} | "
                f"{precision:.1%} | {r.mean_return:.4f} | {r.sharpe_ratio:.2f} |"
            )
        lines.append("")

        # Quadrant analysis
        lines.append("## Quadrant Analysis (strictest level)")
        if result.strictness_results:
            strictest = max(result.strictness_results, key=lambda x: x.strictness_value)
            if strictest.quadrant_stats:
                q = strictest.quadrant_stats
                lines.append("```")
                lines.append("                    │ Return > 0    │ Return <= 0  │")
                lines.append("────────────────────┼───────────────┼──────────────│")
                lines.append(f" Signal = 1 (Entry) │ S1: {q.s1_true_positive:>7}   │ S2: {q.s2_false_positive:>7}  │")
                lines.append(f" Signal = 0 (No)    │ S4: {q.s4_false_negative:>7}   │ S3: {q.s3_true_negative:>7}  │")
                lines.append("```")
                lines.append("")
                lines.append(f"- **Precision** (S1/(S1+S2)): {q.precision:.1%}")
                lines.append(f"- **Recall** (S1/(S1+S4)): {q.recall:.1%}")
                lines.append(f"- **F1 Score**: {q.f1_score:.3f}")
        lines.append("")

        # Monotonicity verification
        lines.append("## Monotonicity Verification")
        lines.append("")
        lines.append("Strictness ↑ → Performance change (adjacent-level comparison):")
        for r in result.monotonicity_results:
            status = "✓" if r.direction == "improving" else ("✗" if r.direction == "degrading" else "−")
            lines.append(f"- {status} **{r.metric_name}**: {r.direction} (score={r.spearman_corr:.3f})")
        lines.append("")

        # Key findings
        if result.key_findings:
            lines.append("## Key Findings")
            for f in result.key_findings:
                lines.append(f"- {f}")

        return "\n".join(lines)


# ============================================================================
# Backward Compatibility Alias
# ============================================================================
# Compatibility alias for legacy imports that expect `DiagnosticsAgent`
DiagnosticsAgent = HypothesisValidationAgent
