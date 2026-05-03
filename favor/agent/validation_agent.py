"""
================================================================================
Stage 2: Observation Formula Validation Agent
Observation-implementation validation agent (LLM-driven decision)
================================================================================

[Purpose]
Validate whether a formula actually implements the intended observation condition (obs) in data.

[Key change: LLM-driven judgment]
Previous: code computed monotonicity scores and made a threshold-based decision.
Now: code computes distribution statistics, and the LLM judges based on the observation description.

[Validation flow]
(i)   Compute distribution summaries (analysis tool role)
(ii)  The LLM infers how distributions should change if the observation is true
(iii) Compare with observed distribution shifts -> PASS/FAIL
(iv)  If FAIL, propose formula improvements
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from agent.base_agent import BaseAgent
from prompts.validation_agent_prompts import (
    DISTRIBUTION_JUDGMENT_SYSTEM_PROMPT,
    DISTRIBUTION_JUDGMENT_USER_TEMPLATE,
)
from schemas.validation import DISTRIBUTION_JUDGMENT_TOOL
from schemas.validation_dataclasses import (
    DistributionStats,
    FormulaValidationResult,
    LLMJudgment,
)
from util.llm_client import call_llm
from util.run_context import RunContext

logger = logging.getLogger(__name__)


# ============================================================================
# Validation Agent
# ============================================================================

class ValidationAgent(BaseAgent):
    """
    Stage 2: LLM-driven observation formula validation agent.

    Key change: the LLM reads the observation description and infers expected directionality.

    Flow:
    1) Quantile partition by formula value (Q1..Qk)
    2) Compute per-quantile raw OHLCV distribution statistics
    3) Provide structured evidence to the LLM
    4) LLM decides PASS/FAIL based on the observation description
    5) Optionally suggest improvements on FAIL
    """

    # Basic elements derived directly from raw OHLCV.
    RAW_ELEMENTS = {
        "MAG": "H - L (range)",
        "DIR": "C - O (direction: positive=up, negative=down)",
        "VOL": "V (volume)",
        "POS": "(C - L) / (H - L) (relative position: 0=close at low, 1=close at high)",
    }

    def __init__(
        self,
        model: str,
        run_ctx: Optional[RunContext] = None,
        n_quantiles: int = 5,
        monotonicity_threshold: float = 0.8,  # Legacy compatibility (not used for decision).
        use_llm_analysis: bool = True,  # Legacy compatibility (always True).
    ):
        """
        Initialize ValidationAgent.

        Args:
            model: LLM model name
            run_ctx: RunContext for logging
            n_quantiles: number of quantile bins (default: 5)
        """
        super().__init__(model=model, run_ctx=run_ctx)
        self.n_quantiles = n_quantiles
        self.monotonicity_threshold = monotonicity_threshold  # Legacy compatibility.

    def validate_formula(
        self,
        formula: Dict[str, Any],
        ohlcv_df: pd.DataFrame,
        formula_values: pd.Series,
    ) -> FormulaValidationResult:
        """
        Validate whether a single formula implements its observation in data.

        Args:
            formula: formula metadata (name, obs_id, definition, polarity, obs_description, etc.)
            ohlcv_df: raw OHLCV data (columns: Open, High, Low, Close, Volume)
            formula_values: formula time series aligned to ohlcv_df index

        Returns:
            FormulaValidationResult: validation result
        """
        formula_id = formula.get("formula_id", formula.get("name", "unknown"))
        formula_name = formula.get("name", "unknown")
        obs_id = formula.get("obs_id", formula.get("observation_id", "unknown"))
        obs_description = formula.get("obs_description", "")
        definition = formula.get("definition", "")
        polarity = formula.get("polarity", "higher_is_more_true")

        logger.info(f"Validating formula: {formula_name} (obs: {obs_id})")

        # (i) Compute distribution summaries (analysis tool role).
        quantile_labels, quantile_counts, actual_n_bins = self._partition_by_formula_value(
            formula_values,
            self.n_quantiles,
            polarity=polarity,
        )

        # On partition failure: return FAIL (standardized; no SKIP).
        if quantile_labels is None:
            logger.warning(
                f"Formula {formula_name}: partition failed "
                f"(actual_bins={actual_n_bins}, min_required={self.MIN_BINS_FOR_VALIDATION})"
            )
            return FormulaValidationResult(
                formula_id=formula_id,
                formula_name=formula_name,
                obs_id=obs_id,
                verdict="FAIL",
                reasoning=(
                    f"Quantile partition failed: actual bins ({actual_n_bins}) < minimum required "
                    f"({self.MIN_BINS_FOR_VALIDATION}). The data may have too many duplicate values or be "
                    "highly skewed."
                ),
                quantile_counts={},
                evidence_packet={},
                primary_evidence=[],
                distribution_by_element={},
            )

        distribution_by_element = self._observe_raw_distribution(
            ohlcv_df, quantile_labels
        )
        # Build a JSON-only evidence packet (text summary is for logging/debugging only).
        evidence_packet = self._build_evidence_packet(
            distribution_by_element,
            quantile_counts,
            polarity=polarity,
            actual_n_bins=actual_n_bins,
        )

        # (ii) & (iii) LLM judgment based on obs description + evidence packet.
        judgment = self._llm_judge_obs_support(
            formula_name=formula_name,
            definition=definition,
            polarity=polarity,
            obs_id=obs_id,
            obs_description=obs_description,
            evidence_packet=evidence_packet,
        )

        # Generate a text summary for logging/debugging.
        distribution_summary = self._generate_text_summary(
            distribution_by_element,
            quantile_counts,
            polarity=polarity,
            actual_n_bins=actual_n_bins,
        )

        # (iv) Optional improvement hints on FAIL (not formula rewriting).
        # improvement_hints = None
        # if judgment.verdict == "FAIL":
        #     improvement_hints = self._llm_suggest_improvement(
        #         formula=formula,
        #         obs_description=obs_description,
        #         evidence_packet=evidence_packet,
        #         judgment=judgment,
        #     )

        result = FormulaValidationResult(
            formula_id=formula_id,
            formula_name=formula_name,
            obs_id=obs_id,
            verdict=judgment.verdict,
            reasoning=judgment.reasoning,
            quantile_counts=quantile_counts,
            evidence_packet=evidence_packet,
            primary_evidence=judgment.primary_evidence,
            distribution_by_element=distribution_by_element,
            # improvement_hints=improvement_hints,
            distribution_summary=distribution_summary,
        )

        self._save_stage2_validation_record(
            formula=formula,
            obs_description=obs_description,
            distribution_summary=distribution_summary,
            evidence_packet=evidence_packet,
            judgment=judgment,
            result=result,
        )

        logger.info(f"Validation result for {formula_name}: {judgment.verdict}")
        return result

    def _save_stage2_validation_record(
        self,
        *,
        formula: Dict[str, Any],
        obs_description: str,
        distribution_summary: str,
        evidence_packet: Dict[str, Any],
        judgment: LLMJudgment,
        result: FormulaValidationResult,
    ) -> None:
        """
        Save agent-level I/O for a single Stage2 validation into a JSON file.

        - Input: formula meta + obs_description + distribution_summary + evidence_packet
        - Output: raw LLM response + parsed judgment + final FormulaValidationResult
        """
        if not self.run_ctx:
            return

        try:
            ts = datetime.now()
            ts_str = ts.strftime("%H%M%S_%f")
            safe_formula_name = (result.formula_name or "unknown").replace("/", "_")
            out_path = self.run_ctx.root_dir / f"logs/agents/react_validation_agent_{ts_str}_{safe_formula_name}.json"

            record = {
                "timestamp": ts.isoformat(),
                "stage": "stage2",
                "kind": "formula_validation",
                "formula": {
                    "formula_id": result.formula_id,
                    "formula_name": result.formula_name,
                    "definition": (formula or {}).get("definition", ""),
                    "polarity": (formula or {}).get("polarity", ""),
                    "obs_id": result.obs_id,
                    "obs_description": obs_description,
                },
                "input": {
                    "distribution_summary": distribution_summary,
                    "evidence_packet": evidence_packet,
                },
                "output": {
                    "llm_raw_response": getattr(judgment, "raw_response", None),
                    "llm_judgment": asdict(judgment),
                    "result": asdict(result),
                },
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save Stage2 validation record: {e}")

    def validate_formula_bundle(
        self,
        formula_bundle: Dict[str, Any],
        ohlcv_df: pd.DataFrame,
        formula_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Validate an entire formula bundle.

        Args:
            formula_bundle: bundle produced by FormulaAgent (contains formulas list)
            ohlcv_df: raw OHLCV data
            formula_df: DataFrame with all formula columns

        Returns:
            A summary dict of validation results
        """
        hypothesis_id = formula_bundle.get("hypothesis_id", "unknown")
        formulas = formula_bundle.get("formulas", [])

        # Build observation description map.
        obs_desc_map: Dict[str, str] = {}
        for item in formula_bundle.get("observation_descriptions", []) if isinstance(formula_bundle, dict) else []:
            if not isinstance(item, dict):
                continue
            oid = str(item.get("observation_id") or "").strip()
            desc = str(item.get("description") or "").strip()
            if oid and desc:
                obs_desc_map[oid] = desc

        results: List[FormulaValidationResult] = []
        passed_formulas: List[str] = []
        failed_formulas: List[str] = []

        # Optional progress bar (falls back to plain iteration if tqdm isn't available).
        def _iter_formulas(items: list[Any]):
            try:
                import sys
                from tqdm.auto import tqdm  # type: ignore

                return tqdm(
                    items,
                    desc=f"Stage2 formulas ({hypothesis_id})",
                    total=len(items),
                    disable=not sys.stderr.isatty(),
                )
            except Exception:
                return items

        formula_iter = _iter_formulas(formulas if isinstance(formulas, list) else [])

        for formula in formula_iter:
            if not isinstance(formula, dict):
                continue

            formula_name = formula.get("name", "")
            if not formula_name or formula_name not in formula_df.columns:
                logger.warning(f"Formula {formula_name} not found in formula_df, skipping")
                continue

            try:
                if hasattr(formula_iter, "set_postfix_str"):
                    formula_iter.set_postfix_str(f"formula={formula_name}")
            except Exception:
                pass

            formula_values = formula_df[formula_name]

            # Inject obs_description when available.
            obs_id = str(formula.get("obs_id", formula.get("observation_id", "")) or "").strip()
            if obs_id and obs_id in obs_desc_map and "obs_description" not in formula:
                formula = dict(formula)
                formula["obs_description"] = obs_desc_map.get(obs_id, "")

            result = self.validate_formula(
                formula=formula,
                ohlcv_df=ohlcv_df,
                formula_values=formula_values,
            )
            results.append(result)

            try:
                if hasattr(formula_iter, "set_postfix_str"):
                    formula_iter.set_postfix_str(f"formula={formula_name} verdict={result.verdict}")
            except Exception:
                pass

            if result.verdict == "PASS":
                passed_formulas.append(formula_name)
            elif result.verdict == "FAIL":
                failed_formulas.append(formula_name)

        # Overall summary.
        total = len(results)
        pass_rate = len(passed_formulas) / total if total > 0 else 0.0

        summary = {
            "hypothesis_id": hypothesis_id,
            "total_formulas": total,
            "passed": len(passed_formulas),
            "failed": len(failed_formulas),
            "pass_rate": pass_rate,
            "passed_formulas": passed_formulas,
            "failed_formulas": failed_formulas,
            "results": [asdict(r) for r in results],
            "overall_verdict": "PASS" if pass_rate >= 0.5 and len(failed_formulas) == 0 else "FAIL",
        }

        return summary

    # ========================================================================
    # Distribution computation (analysis tool role)
    # ========================================================================

    # Minimum number of bins required for validation.
    MIN_BINS_FOR_VALIDATION = 3

    def _partition_by_formula_value(
        self,
        formula_values: pd.Series,
        n_quantiles: int,
        polarity: str = "higher_is_more_true",
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, int]], int]:
        """
        Partition a series into quantile bins using only the formula values.

        Args:
            formula_values: formula value time series
            n_quantiles: target number of quantile bins

        Returns:
            Tuple of:
                - quantile_labels: per-sample bin label series (None on failure)
                - quantile_counts: per-bin sample counts (None on failure)
                - actual_n_bins: actual number of bins created (0 on failure)
        """
        valid_values = formula_values.dropna()

        if len(valid_values) < n_quantiles * 10:
            logger.warning(f"Insufficient samples for {n_quantiles}-quantile partition: {len(valid_values)}")
            n_quantiles = max(self.MIN_BINS_FOR_VALIDATION, len(valid_values) // 10)

        # Use labels=False to get integer codes; retbins=True to inspect actual cut points.
        try:
            quantile_codes, bins = pd.qcut(
                valid_values,
                q=n_quantiles,
                labels=False,
                duplicates='drop',
                retbins=True
            )
        except ValueError as e:
            logger.warning(f"qcut failed: {e}")
            return None, None, 0

        # Actual number of bins created.
        actual_n_bins = len(bins) - 1

        # Guard: require at least MIN_BINS_FOR_VALIDATION bins.
        if actual_n_bins < self.MIN_BINS_FOR_VALIDATION:
            logger.warning(
                f"Insufficient bins for validation: {actual_n_bins} < {self.MIN_BINS_FOR_VALIDATION}. "
                f"Data may have too many duplicate values."
            )
            return None, None, actual_n_bins

        pol = str(polarity).strip().lower()
        if pol in ("lower_is_more_true", "lower", "inverse"):
            quantile_codes = (actual_n_bins - 1) - quantile_codes

        # Build ordered labels (Categorical preserves ordering).
        label_list = [f"Q{i+1}" for i in range(actual_n_bins)]
        label_map = {i: label_list[i] for i in range(actual_n_bins)}
        quantile_labels = quantile_codes.map(label_map).astype(
            pd.CategoricalDtype(categories=label_list, ordered=True)
        )

        full_labels = pd.Series(index=formula_values.index, dtype=object)
        full_labels[valid_values.index] = quantile_labels

        # Create counts sorted by categorical order.
        counts = quantile_labels.value_counts().sort_index().to_dict()
        return full_labels, {str(k): int(v) for k, v in counts.items()}, actual_n_bins

    def _observe_raw_distribution(
        self,
        ohlcv_df: pd.DataFrame,
        quantile_labels: pd.Series,
    ) -> Dict[str, Dict[str, DistributionStats]]:
        """
        Observe raw OHLCV-derived element distributions within each quantile bin.
        """
        # Normalize OHLCV column names.
        col_map = {}
        for col in ohlcv_df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'o']:
                col_map['Open'] = col
            elif col_lower in ['high', 'h']:
                col_map['High'] = col
            elif col_lower in ['low', 'l']:
                col_map['Low'] = col
            elif col_lower in ['close', 'c']:
                col_map['Close'] = col
            elif col_lower in ['volume', 'v', 'vol']:
                col_map['Volume'] = col

        # Compute raw elements (MAG, DIR, VOL, POS).
        raw_elements = {}

        if 'High' in col_map and 'Low' in col_map:
            raw_elements['MAG'] = ohlcv_df[col_map['High']] - ohlcv_df[col_map['Low']]

        if 'Close' in col_map and 'Open' in col_map:
            raw_elements['DIR'] = ohlcv_df[col_map['Close']] - ohlcv_df[col_map['Open']]

        if 'High' in col_map and 'Low' in col_map and 'Close' in col_map:
            hl_range = ohlcv_df[col_map['High']] - ohlcv_df[col_map['Low']]
            raw_elements['POS'] = (
                (ohlcv_df[col_map['Close']] - ohlcv_df[col_map['Low']]) / hl_range.replace(0, np.nan)
            )

        if 'Volume' in col_map:
            raw_elements['VOL'] = ohlcv_df[col_map['Volume']]

        # Compute per-element, per-quantile distribution stats.
        distribution_by_element = {}
        unique_quantiles = sorted([q for q in quantile_labels.unique() if pd.notna(q)])

        for element_name, element_series in raw_elements.items():
            distribution_by_element[element_name] = {}

            for q_label in unique_quantiles:
                mask = quantile_labels == q_label
                values = element_series[mask].dropna()

                if len(values) < 5:
                    continue

                distribution_by_element[element_name][str(q_label)] = DistributionStats(
                    mean=float(values.mean()),
                    median=float(values.median()),
                    std=float(values.std()),
                    iqr=float(values.quantile(0.75) - values.quantile(0.25)),
                    skewness=float(stats.skew(values)),
                    kurtosis=float(stats.kurtosis(values)),
                    q10=float(values.quantile(0.10)),
                    q25=float(values.quantile(0.25)),
                    q75=float(values.quantile(0.75)),
                    q90=float(values.quantile(0.90)),
                    count=len(values),
                )

        return distribution_by_element

    def _build_evidence_packet(
        self,
        distribution_by_element: Dict[str, Dict[str, DistributionStats]],
        quantile_counts: Dict[str, int],
        polarity: str = "higher_is_more_true",
        actual_n_bins: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Convert distribution stats into a JSON evidence packet.

        Notes:
        - The LLM should primarily rely on the structured JSON evidence.
        - The text summary is kept only for logging/debugging.

        Returns:
            JSON-serializable evidence packet
        """
        sorted_bins = sorted(quantile_counts.keys())

        # NOTE:
        # `_partition_by_formula_value()` already re-orients bins according to `polarity`
        # (i.e., Q1 is always "observation weaker" and Qk is always "observation stronger").
        # Keep this invariant explicit for the LLM.
        bin_order = "Q1=obs_weak → Qk=obs_strong"

        # Build JSON evidence packet.
        evidence_packet = {
            "meta": {
                "polarity": polarity,
                "actual_n_bins": actual_n_bins or len(sorted_bins),
                "bin_order": bin_order,
                "bins_oriented_for_obs_strength": True,
            },
            "bins": sorted_bins,
            "counts": quantile_counts,
            "features": {},
        }

        for element_name in ["VOL", "MAG", "DIR", "POS"]:
            if element_name not in distribution_by_element:
                continue

            q_stats = distribution_by_element[element_name]
            sorted_qs = sorted(q_stats.keys())

            if len(sorted_qs) < 2:
                continue

            # Collect each metric as a per-bin array.
            means = [round(q_stats[q].mean, 6) for q in sorted_qs]
            stds = [round(q_stats[q].std, 6) for q in sorted_qs]
            medians = [round(q_stats[q].median, 6) for q in sorted_qs]
            q10s = [round(q_stats[q].q10, 6) for q in sorted_qs]
            q90s = [round(q_stats[q].q90, 6) for q in sorted_qs]
            skewnesses = [round(q_stats[q].skewness, 6) for q in sorted_qs]
            kurtoses = [round(q_stats[q].kurtosis, 6) for q in sorted_qs]

            # Simple monotonicity scoring helper.
            def calc_monotonicity(values):
                n_pairs = len(values) - 1
                if n_pairs > 0:
                    increasing = sum(1 for i in range(n_pairs) if values[i+1] > values[i])
                    decreasing = sum(1 for i in range(n_pairs) if values[i+1] < values[i])
                    score = max(increasing, decreasing) / n_pairs
                    direction = "increasing" if increasing >= decreasing else "decreasing"
                    return score, direction
                return 0.0, "unknown"

            # Compute monotonicity for each metric.
            mean_mono_score, mean_mono_dir = calc_monotonicity(means)
            median_mono_score, median_mono_dir = calc_monotonicity(medians)
            std_mono_score, std_mono_dir = calc_monotonicity(stds)
            q10_mono_score, q10_mono_dir = calc_monotonicity(q10s)
            q90_mono_score, q90_mono_dir = calc_monotonicity(q90s)
            skew_mono_score, skew_mono_dir = calc_monotonicity(skewnesses)
            kurt_mono_score, kurt_mono_dir = calc_monotonicity(kurtoses)

            # Q1 -> Qk percent change.
            pct_change = ((means[-1] - means[0]) / abs(means[0]) * 100) if means[0] != 0 else 0.0
            skew_pct_change = ((skewnesses[-1] - skewnesses[0]) / abs(skewnesses[0]) * 100) if skewnesses[0] != 0 else 0.0
            kurt_pct_change = ((kurtoses[-1] - kurtoses[0]) / abs(kurtoses[0]) * 100) if kurtoses[0] != 0 else 0.0

            evidence_packet["features"][element_name] = {
                "mean": means,
                "median": medians,
                "std": stds,
                "q10": q10s,
                "q90": q90s,
                "skewness": skewnesses,
                "kurtosis": kurtoses,
                "count": [q_stats[q].count for q in sorted_qs],
                "q1_to_qk_change_pct": round(pct_change, 2),
                "skewness_change_pct": round(skew_pct_change, 2),
                "kurtosis_change_pct": round(kurt_pct_change, 2),
                "monotonicity": {
                    "mean": {"score": round(mean_mono_score, 3), "direction": mean_mono_dir},
                    "median": {"score": round(median_mono_score, 3), "direction": median_mono_dir},
                    "std": {"score": round(std_mono_score, 3), "direction": std_mono_dir},
                    "q10": {"score": round(q10_mono_score, 3), "direction": q10_mono_dir},
                    "q90": {"score": round(q90_mono_score, 3), "direction": q90_mono_dir},
                    "skewness": {"score": round(skew_mono_score, 3), "direction": skew_mono_dir},
                    "kurtosis": {"score": round(kurt_mono_score, 3), "direction": kurt_mono_dir},
                },
            }

        return evidence_packet

    def _generate_text_summary(
        self,
        distribution_by_element: Dict[str, Dict[str, DistributionStats]],
        quantile_counts: Dict[str, int],
        polarity: str = "higher_is_more_true",
        actual_n_bins: Optional[int] = None,
    ) -> str:
        """
        Generate a text summary of distribution stats (logging/debugging only).

        Returns:
            Text summary
        """
        sorted_bins = sorted(quantile_counts.keys())

        # `_partition_by_formula_value()` already re-orients bins according to polarity.
        pol = str(polarity).strip().lower()
        bin_order = "Q1=obs_weak → Qk=obs_strong"

        lines = []
        lines.append(f"[bin order] Q1→Qk = obs strength (weak→strong). polarity={polarity}")
        if actual_n_bins is not None:
            lines.append(f"[n_bins] {actual_n_bins}")

        count_str = ", ".join([f"{k}:{v}" for k, v in sorted(quantile_counts.items())])
        lines.append(f"[samples] {count_str}")
        lines.append("")

        for element_name, element_desc in self.RAW_ELEMENTS.items():
            if element_name not in distribution_by_element:
                continue

            q_stats = distribution_by_element[element_name]
            sorted_qs = sorted(q_stats.keys())

            if len(sorted_qs) < 2:
                continue

            lines.append(f"### [{element_name}] {element_desc}")

            first_q, last_q = sorted_qs[0], sorted_qs[-1]
            first_stats, last_stats = q_stats[first_q], q_stats[last_q]

            # mean/std changes
            if first_stats.mean != 0:
                mean_pct = ((last_stats.mean - first_stats.mean) / abs(first_stats.mean)) * 100
                lines.append(f"mean: {first_stats.mean:.4f} → {last_stats.mean:.4f} ({mean_pct:+.1f}%)")
            else:
                lines.append(f"mean: {first_stats.mean:.4f} → {last_stats.mean:.4f}")

            if first_stats.std != 0:
                std_pct = ((last_stats.std - first_stats.std) / abs(first_stats.std)) * 100
                lines.append(f"std: {first_stats.std:.4f} → {last_stats.std:.4f} ({std_pct:+.1f}%)")
            else:
                lines.append(f"std: {first_stats.std:.4f} → {last_stats.std:.4f}")

            # skewness/kurtosis changes
            if first_stats.skewness != 0:
                skew_pct = ((last_stats.skewness - first_stats.skewness) / abs(first_stats.skewness)) * 100
                lines.append(f"skewness: {first_stats.skewness:.4f} → {last_stats.skewness:.4f} ({skew_pct:+.1f}%)")
            else:
                lines.append(f"skewness: {first_stats.skewness:.4f} → {last_stats.skewness:.4f}")

            if first_stats.kurtosis != 0:
                kurt_pct = ((last_stats.kurtosis - first_stats.kurtosis) / abs(first_stats.kurtosis)) * 100
                lines.append(f"kurtosis: {first_stats.kurtosis:.4f} → {last_stats.kurtosis:.4f} ({kurt_pct:+.1f}%)")
            else:
                lines.append(f"kurtosis: {first_stats.kurtosis:.4f} → {last_stats.kurtosis:.4f}")

            lines.append(f"q90: {first_stats.q90:.4f} → {last_stats.q90:.4f}")

            # Per-quantile paths.
            means_str = [f"{q}={q_stats[q].mean:.4f}" for q in sorted_qs]
            lines.append(f"mean path: {' → '.join(means_str)}")

            skew_str = [f"{q}={q_stats[q].skewness:.4f}" for q in sorted_qs]
            lines.append(f"skewness path: {' → '.join(skew_str)}")

            kurt_str = [f"{q}={q_stats[q].kurtosis:.4f}" for q in sorted_qs]
            lines.append(f"kurtosis path: {' → '.join(kurt_str)}")
            lines.append("")

        return "\n".join(lines)

    # ========================================================================
    # LLM judgment
    # ========================================================================

    def _llm_judge_obs_support(
        self,
        formula_name: str,
        definition: str,
        polarity: str,
        obs_id: str,
        obs_description: str,
        evidence_packet: Dict[str, Any],
    ) -> LLMJudgment:
        """
        Ask the LLM to judge PASS/FAIL from the observation description and structured evidence.

        Notes:
        - Uses a tool to force structured output.
        - Uses JSON evidence as the primary input (text summary is optional).
        - Keeps the prompt compact to save tokens.
        """
        user_prompt = DISTRIBUTION_JUDGMENT_USER_TEMPLATE.format(
            formula_name=formula_name,
            definition=definition,
            polarity=polarity,
            obs_id=obs_id,
            obs_description=obs_description,
            evidence_json=json.dumps(evidence_packet, ensure_ascii=False, indent=2),
        )

        try:
            response = call_llm(
                model=self.model,
                system_prompt=DISTRIBUTION_JUDGMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=[DISTRIBUTION_JUDGMENT_TOOL],
                target_tool_name="distribution_judgment_tool",
                temperature=0.1,
                react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
                react_agent_name="validation_agent",
                context="Stage2: Distribution Validation",
            )

            # Parse tool response directly.
            if isinstance(response, dict):
                verdict = response.get("verdict", "FAIL").upper()
                primary_evidence = response.get("primary_evidence", [])
                reasoning = response.get("reasoning", "")

                judgment = LLMJudgment(
                    verdict=verdict,
                    reasoning=reasoning,
                    primary_evidence=primary_evidence,
                )
                judgment.raw_response = json.dumps(response, ensure_ascii=False)
                return judgment
            else:
                # Fallback: parse text response.
                parsed = self._parse_judgment_response(response)
                parsed.raw_response = response
                return parsed

        except Exception as e:
            logger.error(f"LLM judgment failed: {e}")
            return LLMJudgment(
                verdict="FAIL",
                reasoning=f"LLM call failed: {str(e)}",
                primary_evidence=[],
                raw_response=None,
            )

    def _parse_judgment_response(self, response: str) -> LLMJudgment:
        """
        Parse JSON from an LLM response.

        Expected schema:
        - verdict: PASS | FAIL
        - primary_evidence: list of numeric evidence citations
        - reasoning: short explanation (required for both PASS/FAIL)
        """
        # Extract JSON block.
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        try:
            data = json.loads(json_str)

            verdict = data.get("verdict", "FAIL").upper()

            # Extract primary_evidence.
            primary_evidence = data.get("primary_evidence", [])
            if not isinstance(primary_evidence, list):
                primary_evidence = []

            return LLMJudgment(
                verdict=verdict,
                reasoning=data.get("reasoning", ""),
                primary_evidence=primary_evidence,
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Try to infer verdict from plain text.
            verdict = "FAIL"
            if "PASS" in response.upper() and "FAIL" not in response.upper():
                verdict = "PASS"

            return LLMJudgment(
                verdict=verdict,
                reasoning=response[:500],
                primary_evidence=[],
            )

    # def _llm_suggest_improvement(
    #     self,
    #     formula: Dict[str, Any],
    #     obs_description: str,
    #     evidence_packet: Dict[str, Any],
    #     judgment: LLMJudgment,
    # ) -> Optional[Dict[str, Any]]:
    #     """
    #     Request improvement hints on FAIL.

    #     Notes:
    #     - Do not rewrite formulas completely.
    #     - Suggest only threshold/window adjustments.
    #     - Point out ambiguous observation definitions.
    #     - Suggest routing to a different observation if needed.
    #     """
    #     user_prompt = FORMULA_IMPROVEMENT_USER_TEMPLATE.format(
    #         formula_name=formula.get("name", "unknown"),
    #         definition=formula.get("definition", ""),
    #         obs_id=formula.get("obs_id", ""),
    #         obs_description=obs_description,
    #         judgment_reasoning=judgment.reasoning,
    #         evidence_json=json.dumps(evidence_packet, ensure_ascii=False, indent=2),
    #     )

    #     try:
    #         response = call_llm(
    #             model=self.model,
    #             system_prompt=FORMULA_IMPROVEMENT_SYSTEM_PROMPT,
    #             user_prompt=user_prompt,
    #             tools=None,
    #             temperature=0.3,
    #         )

    #         # Parse JSON
    #         json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    #         if json_match:
    #             return json.loads(json_match.group(1))
    #         else:
    #             return {"raw_response": response}

    #     except Exception as e:
    #         logger.warning(f"LLM improvement hint failed: {e}")
    #         return None

    # ========================================================================
    # Report generation
    # ========================================================================

    def generate_validation_report(
        self,
        validation_result: Dict[str, Any],
    ) -> str:
        """
        Generate a human-readable validation report.
        """
        lines = []
        lines.append("# Stage 2: Observation Formula Validation Report")
        lines.append("")
        lines.append(f"**Hypothesis ID**: {validation_result.get('hypothesis_id', 'N/A')}")
        lines.append(f"**Overall Verdict**: {validation_result.get('overall_verdict', 'N/A')}")
        lines.append("")

        # Summary.
        lines.append("## Summary")
        lines.append(f"- Total Formulas: {validation_result.get('total_formulas', 0)}")
        lines.append(f"- Passed: {validation_result.get('passed', 0)}")
        lines.append(f"- Failed: {validation_result.get('failed', 0)}")
        lines.append(f"- Pass Rate: {validation_result.get('pass_rate', 0):.1%}")
        lines.append("")

        # Per-formula results.
        lines.append("## Formula-level Results")
        for result in validation_result.get('results', []):
            lines.append(f"\n### {result.get('formula_name', 'Unknown')}")
            lines.append(f"- **Observation ID**: {result.get('obs_id', 'N/A')}")
            lines.append(f"- **Verdict**: {result.get('verdict', 'N/A')}")
            lines.append("")

            # Reasoning.
            reasoning = result.get('reasoning', '')
            if reasoning:
                lines.append("**Reasoning:**")
                lines.append(f"  {reasoning}")

            # Expected vs observed.
            expected = result.get('expected_direction', {})
            observed = result.get('observed_direction', {})
            if expected or observed:
                lines.append("\n**Expected vs Observed:**")
                for key in set(list(expected.keys()) + list(observed.keys())):
                    exp = expected.get(key, '-')
                    obs = observed.get(key, '-')
                    match = result.get('match_analysis', {}).get(key)
                    match_str = "✓" if match is True else ("✗" if match is False else "?")
                    lines.append(f"  - {key}: expected={exp}, observed={obs} [{match_str}]")

            # Improvement suggestions.
            improvement = result.get('improvement_suggestions')
            if improvement:
                lines.append("\n**Improvement Suggestions:**")
                if isinstance(improvement, dict):
                    if 'failure_analysis' in improvement:
                        lines.append(f"  - Analysis: {improvement['failure_analysis']}")
                    if 'improvement_direction' in improvement:
                        lines.append(f"  - Direction: {improvement['improvement_direction']}")
                    if 'suggested_formulas' in improvement:
                        for sf in improvement['suggested_formulas']:
                            lines.append(f"  - Suggested: {sf.get('name', 'N/A')}")
                            lines.append(f"    Definition: {sf.get('definition', 'N/A')}")

        # Link to Stage 3.
        lines.append("\n## Next Steps")
        passed = validation_result.get('passed_formulas', [])
        if passed:
            lines.append("The following formulas will be forwarded to Stage 3 (Hypothesis Instance Validation):")
            for f in passed:
                lines.append(f"  - {f}")
        else:
            lines.append("No formulas passed. Improve formulas before running Stage 3.")

        return "\n".join(lines)

    # ========================================================================
    # Legacy compatibility methods (kept for older callers).
    # Deprecated: replaced by LLM-based judgment.
    # ========================================================================

    # def _verify_monotonic_shift(
    #     self,
    #     distribution_by_element: Dict[str, Dict[str, DistributionStats]],
    #     polarity: str,
    # ) -> List[Dict[str, Any]]:
    #     """Legacy: monotonicity verification (not used for decision)."""
    #     results = []
    #     metrics_to_check = ['mean', 'median', 'std', 'iqr']

    #     for element_name, quantile_stats in distribution_by_element.items():
    #         if len(quantile_stats) < 3:
    #             continue

    #         sorted_quantiles = sorted(quantile_stats.keys())

    #         for metric in metrics_to_check:
    #             values = [getattr(quantile_stats[q], metric) for q in sorted_quantiles]
    #             n_pairs = len(values) - 1
    #             if n_pairs < 1:
    #                 continue

    #             increasing_count = sum(1 for i in range(n_pairs) if values[i+1] > values[i])
    #             decreasing_count = sum(1 for i in range(n_pairs) if values[i+1] < values[i])

    #             if increasing_count >= decreasing_count:
    #                 direction = "increasing"
    #                 mono_score = increasing_count / n_pairs
    #             else:
    #                 direction = "decreasing"
    #                 mono_score = decreasing_count / n_pairs

    #             results.append({
    #                 "metric_name": f"{element_name}_{metric}",
    #                 "direction": direction if mono_score >= 0.5 else "non_monotonic",
    #                 "monotonicity_score": mono_score,
    #                 "values_by_quantile": values,
    #             })

    #     return results

    # def _make_decision(
    #     self,
    #     monotonicity_results: List[Dict[str, Any]],
    #     quantile_counts: Dict[str, int],
    # ) -> Tuple[str, float, List[str], List[str]]:
    #     """Legacy: threshold-based decision logic (replaced by LLM judgment)."""
    #     # Kept for backward compatibility only.
    #     if not monotonicity_results:
    #         return "FAIL", 0.0, ["Monotonicity check unavailable"], []

    #     significant = [r for r in monotonicity_results if r.get("monotonicity_score", 0) >= self.monotonicity_threshold]

    #     if len(significant) >= 1:
    #         return "PASS", max(r["monotonicity_score"] for r in significant), [], []
    #     else:
    #         return "FAIL", 0.0, ["Weak monotonicity"], ["Consider improving the formula"]


# ============================================================================
# Legacy compatibility: keep the original dataclass name
# ============================================================================

MonotonicityResult = Dict[str, Any]  # Type-hint compatibility.
