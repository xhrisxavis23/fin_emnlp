"""
================================================================================
Refinement Pipeline (Stage 4 → Stage 1)
Hypothesis Refinement Pipeline
================================================================================

[Purpose]
- Improve the hypothesis itself based on Stage4 backtest results.
- If Stage4 performance is poor, provide feedback to the LLM to regenerate the hypothesis.

[Core Principles]
- Summarize Stage4 results (information_ratio, mean_return, etc.) and pass them as feedback.
- Encourage improvements to the hypothesis' underlying observation structure.
- Generate feedback using performance metrics only (avoid complex analyses).

[Refinement Strategy]
- If both IS and OOS are weak → the hypothesis is likely weak.
- If IS is good but OOS is bad → overfitting or a regime change.
- If only certain combinations work → revisit observation conditions for the others.

================================================================================
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def _extract_horizon_days(hypothesis: Dict[str, Any] | None) -> int | None:
    if not isinstance(hypothesis, dict):
        return None
    hyps = hypothesis.get("hypotheses", [])
    if isinstance(hyps, list) and hyps and isinstance(hyps[0], dict):
        h = hyps[0].get("horizon_days")
        return int(h) if isinstance(h, int) and h > 0 else None
    h = hypothesis.get("horizon_days")
    return int(h) if isinstance(h, int) and h > 0 else None


def _extract_hypotheses_list(hypothesis: Dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(hypothesis, dict):
        return []
    hyps = hypothesis.get("hypotheses", [])
    if isinstance(hyps, list) and hyps and all(isinstance(x, dict) for x in hyps):
        return [dict(x) for x in hyps]
    # Fallback: if a single hypothesis object was passed.
    if "hypothesis_id" in hypothesis or "behavioral_description" in hypothesis:
        return [dict(hypothesis)]
    return []


def build_hypothesis_feedback_from_stage4(
    stage4_summary: Dict[str, Any],
    hypothesis: Dict[str, Any] | None = None,
    observation_plan: Dict[str, Any] | None = None,
    formula_bundle: Dict[str, Any] | None = None,
) -> str:
    """
    Build a hypothesis-refinement feedback message from Stage4 backtest results.

    Args:
        stage4_summary: Summary of a Stage4 run.
        hypothesis: Stage1 hypothesis (optional, for context).
        observation_plan: Stage1 observation plan (optional, for context).
        formula_bundle: Stage1 formula bundle (optional, for context).

    Returns:
        Feedback string for refining the next hypothesis.
    """
    feedback_parts = []
    feedback_parts.append("=== Stage4 Backtest Feedback ===\n")

    # Validate overall summary structure
    if not stage4_summary:
        return "No Stage4 results available for feedback."

    # Add hypothesis behavioral_description (if provided)
    if hypothesis:
        behavioral_desc = hypothesis.get("behavioral_description", "")
        if behavioral_desc:
            feedback_parts.append(f"Current Hypothesis: {behavioral_desc}\n")

    # Extract per-combination results
    combination_results = stage4_summary.get("all_combinations", [])
    if not combination_results:
        feedback_parts.append("No combination results found.")
        feedback_parts.append("\nSuggestion: The hypothesis may need stronger observation conditions.")
        return "\n".join(feedback_parts)

    # Helper: map formula_name -> observation metadata
    formula_to_obs = {}
    if observation_plan and formula_bundle:
        obs_list = observation_plan.get("observations", [])
        formulas_list = formula_bundle.get("formulas", [])

        for obs in obs_list:
            obs_id = obs.get("obs_id", "")
            obs_desc = obs.get("description", "")
            for formula in formulas_list:
                if formula.get("obs_id") == obs_id:
                    formula_name = formula.get("formula_name", "")
                    formula_to_obs[formula_name] = {
                        "obs_id": obs_id,
                        "obs_description": obs_desc,
                    }

    # Helper: extract Information Ratio (qlib-standard structure)
    def _as_finite_float(v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return f if math.isfinite(f) else 0.0

    def _get_oos_ir(comb):
        v = comb.get("outsample", {}).get("return", {}).get("information_ratio", 0.0)
        return _as_finite_float(v)

    def _get_is_ir(comb):
        v = comb.get("insample", {}).get("return", {}).get("information_ratio", 0.0)
        return _as_finite_float(v)

    # Overall performance summary
    n_combinations = len(combination_results)
    feedback_parts.append(f"Evaluated {n_combinations} formula combination(s)\n")

    # Overall average performance
    avg_is_ir = sum(_get_is_ir(c) for c in combination_results) / len(combination_results)
    avg_oos_ir = sum(_get_oos_ir(c) for c in combination_results) / len(combination_results)

    feedback_parts.append("=== Overall Performance ===")
    feedback_parts.append(f"Average IS IR:  {avg_is_ir:.3f}")
    feedback_parts.append(f"Average OOS IR: {avg_oos_ir:.3f}")

    # Best/worst combinations by OOS IR
    sorted_by_oos = sorted(combination_results, key=_get_oos_ir, reverse=True)
    best_comb = sorted_by_oos[0] if sorted_by_oos else None
    worst_comb = sorted_by_oos[-1] if len(sorted_by_oos) > 1 else None

    # Best combination details
    if best_comb:
        feedback_parts.append("\n=== Best Performing Combination (Ranked by OOS IR) ===")
        best_formulas = best_comb.get("formula_names", [])
        best_is = _get_is_ir(best_comb)
        best_oos = _get_oos_ir(best_comb)

        feedback_parts.append(f"Combo {best_comb.get('combo_idx', 'N/A')} - HIGHEST OOS IR among all {n_combinations} combinations")
        feedback_parts.append(f"Performance: IS IR = {best_is:.3f}, OOS IR = {best_oos:.3f}")

        # Add observation-condition descriptions (if available)
        if formula_to_obs:
            feedback_parts.append("\nObservation Conditions (these worked well):")
            for fname in best_formulas:
                obs_info = formula_to_obs.get(fname, {})
                obs_desc = obs_info.get("obs_description", "")
                if obs_desc:
                    feedback_parts.append(f"  ✓ {obs_desc}")
                    feedback_parts.append(f"    (implemented by: {fname})")
                else:
                    feedback_parts.append(f"  ✓ {fname}")
        else:
            feedback_parts.append(f"\nFormulas: {', '.join(best_formulas)}")

        feedback_parts.append("\n→ Insight: These observation conditions generalized well to out-of-sample data.")
        feedback_parts.append("  Consider strengthening similar patterns in the next hypothesis.")

    # Worst combination details
    if worst_comb and worst_comb != best_comb:
        feedback_parts.append("\n=== Worst Performing Combination (Ranked by OOS IR) ===")
        worst_formulas = worst_comb.get("formula_names", [])
        worst_is = _get_is_ir(worst_comb)
        worst_oos = _get_oos_ir(worst_comb)

        feedback_parts.append(f"Combo {worst_comb.get('combo_idx', 'N/A')} - LOWEST OOS IR among all {n_combinations} combinations")
        feedback_parts.append(f"Performance: IS IR = {worst_is:.3f}, OOS IR = {worst_oos:.3f}")

        # Add observation-condition descriptions (if available)
        if formula_to_obs:
            feedback_parts.append("\nObservation Conditions (these failed in OOS):")
            for fname in worst_formulas:
                obs_info = formula_to_obs.get(fname, {})
                obs_desc = obs_info.get("obs_description", "")
                if obs_desc:
                    feedback_parts.append(f"  ✗ {obs_desc}")
                    feedback_parts.append(f"    (implemented by: {fname})")
                else:
                    feedback_parts.append(f"  ✗ {fname}")
        else:
            feedback_parts.append(f"\nFormulas: {', '.join(worst_formulas)}")

        feedback_parts.append("\n→ Insight: These observation conditions did not generalize to out-of-sample data.")

        # IS/OOS gap analysis
        is_oos_gap = worst_is - worst_oos
        if worst_is > 0.3 and is_oos_gap > 0.5:
            feedback_parts.append("  High IS performance but poor OOS suggests overfitting.")
            feedback_parts.append("  These conditions may be too specific to the in-sample period.")
        elif worst_is < 0.1:
            feedback_parts.append("  Both IS and OOS performance are weak.")
            feedback_parts.append("  These observation conditions may not capture the intended behavior.")

    feedback_parts.append("\n=== Suggestions for Next Iteration ===")
    feedback_parts.append("Consider the following improvements:")
    if best_comb and worst_comb:
        feedback_parts.append("- Strengthen observation patterns similar to the best combination")
        feedback_parts.append("- Reconsider or refine observation conditions from worst combination")
    feedback_parts.append("- Adjust hypothesis structure if IS/OOS gap is large")
    feedback_parts.append("- Ensure observation conditions are stable across different market regimes")

    return "\n".join(feedback_parts)


def generate_next_hypothesis_suggestions(
    successful_patterns: list[dict],
    all_combinations: list[dict],
    hypothesis: dict | None = None,
) -> str:
    """
    Analyze success/failure patterns and propose concrete directions for the next hypothesis.

    Args:
        successful_patterns: Details of successful combinations (OOS excess IR > 0).
        all_combinations: All combination results (for analyzing failures).
        hypothesis: Current hypothesis (optional).

    Returns:
        A concrete suggestion string for the next hypothesis direction.
    """
    suggestions = []
    suggestions.append("\n" + "="*80)
    suggestions.append("🎯 SPECIFIC SUGGESTIONS FOR NEXT HYPOTHESIS")
    suggestions.append("="*80 + "\n")

    # Helper: extract OOS excess IR
    def _get_oos_excess_ir(comb: dict) -> float:
        v = comb.get("outsample", {}).get("excess_return_with_cost", {}).get("information_ratio")
        if v is None:
            v = comb.get("data_split", {}).get("outsample", {}).get("excess_return_with_cost", {}).get("information_ratio")
        return float(v or 0.0)

    # Filter failed combinations
    failed_combinations = [
        comb for comb in all_combinations
        if _get_oos_excess_ir(comb) <= 0
    ]

    n_success = len(successful_patterns)
    n_fail = len(failed_combinations)
    n_total = len(all_combinations)

    suggestions.append(f"📊 Performance Summary:")
    suggestions.append(f"   - Successful combinations: {n_success}/{n_total} ({100*n_success/n_total if n_total > 0 else 0:.1f}%)")
    suggestions.append(f"   - Failed combinations: {n_fail}/{n_total}\n")

    # If success rate is too low, suggest a fundamental redesign
    if n_success == 0:
        suggestions.append("❌ NO SUCCESSFUL COMBINATIONS")
        suggestions.append("\n💡 Next Hypothesis Direction:")
        suggestions.append("   → FUNDAMENTALLY REDESIGN the hypothesis")
        if hypothesis:
            current_desc = hypothesis.get("behavioral_description", "")
            suggestions.append(f"   → Current: '{current_desc}'")
        suggestions.append("   → Try a COMPLETELY DIFFERENT behavioral pattern")
        suggestions.append("   → Consider opposite market conditions (e.g., if current was mean-reversion, try momentum)")
        suggestions.append("   → Simplify observation conditions (current may be too complex/specific)")
        return "\n".join(suggestions)

    # === 1. Analyze common traits of successful patterns ===
    suggestions.append("✅ SUCCESSFUL PATTERN ANALYSIS\n")

    # 1-1. Find common formula patterns
    formula_freq_success = {}
    obs_freq_success = {}
    for pattern in successful_patterns:
        for formula_name in pattern.get("formula_names", []):
            formula_freq_success[formula_name] = formula_freq_success.get(formula_name, 0) + 1

        # Extract observation conditions
        obs_plan = pattern.get("observation_plan", {})
        for obs in obs_plan.get("observations", []):
            obs_desc = obs.get("description", "")
            if obs_desc:
                obs_freq_success[obs_desc] = obs_freq_success.get(obs_desc, 0) + 1

    # Frequently appearing formulas
    if formula_freq_success:
        common_formulas = sorted(formula_freq_success.items(), key=lambda x: x[1], reverse=True)[:3]
        suggestions.append("   🔑 Most Effective Formulas:")
        for fname, count in common_formulas:
            suggestions.append(f"      • {fname} (appeared in {count}/{n_success} successful combinations)")

    # Frequently appearing observation conditions
    if obs_freq_success:
        common_obs = sorted(obs_freq_success.items(), key=lambda x: x[1], reverse=True)[:3]
        suggestions.append("\n   🎯 Most Effective Observation Conditions:")
        for obs_desc, count in common_obs:
            suggestions.append(f"      • '{obs_desc}' ({count}/{n_success} times)")

    # 1-2. Performance characteristics of successful patterns
    if successful_patterns:
        avg_success_oos_ir = sum(
            _get_oos_excess_ir(p.get("backtest_results", {}) or p)
            for p in successful_patterns
        ) / len(successful_patterns)

        suggestions.append(f"\n   📈 Average OOS Excess IR of successful patterns: {avg_success_oos_ir:.3f}")

    # === 2. Analyze failed patterns ===
    if failed_combinations:
        suggestions.append("\n\n❌ FAILED PATTERN ANALYSIS\n")

        # Formula frequency among failed combinations
        formula_freq_fail = {}
        for comb in failed_combinations:
            for fname in comb.get("formula_names", []):
                formula_freq_fail[fname] = formula_freq_fail.get(fname, 0) + 1

        # Formulas that frequently appear only in failures (rare in successes)
        problematic_formulas = []
        for fname, fail_count in formula_freq_fail.items():
            success_count = formula_freq_success.get(fname, 0)
            if fail_count > success_count and fail_count >= n_fail * 0.5:  # Appears in >=50% of failures
                problematic_formulas.append((fname, fail_count, success_count))

        if problematic_formulas:
            suggestions.append("   ⚠️  Formulas that frequently appeared in FAILURES:")
            for fname, fail_count, success_count in problematic_formulas[:3]:
                suggestions.append(f"      • {fname}: {fail_count} failures vs {success_count} successes")
                suggestions.append(f"        → Consider AVOIDING or REDESIGNING this observation")

    # === 3. Concrete directions for the next hypothesis ===
    suggestions.append("\n\n💡 CONCRETE DIRECTIONS FOR NEXT HYPOTHESIS\n")

    success_rate = n_success / n_total if n_total > 0 else 0

    if success_rate >= 0.5:
        # Success rate >= 50%: strengthen successful patterns
        suggestions.append("   ✨ SUCCESS RATE IS HIGH (≥50%) - BUILD ON WHAT WORKS\n")
        suggestions.append("   Strategy: REFINE and STRENGTHEN successful patterns")

        if common_obs:
            suggestions.append(f"\n   1️⃣  KEEP and ENHANCE these observation conditions:")
            for obs_desc, count in common_obs[:2]:
                suggestions.append(f"      → '{obs_desc}'")

        if common_formulas:
            suggestions.append(f"\n   2️⃣  REUSE these proven formulas in new combinations:")
            for fname, count in common_formulas[:2]:
                suggestions.append(f"      → {fname}")

        suggestions.append("\n   3️⃣  ADD complementary observations that could work well with existing patterns")
        suggestions.append("      → Look for conditions that could FILTER or STRENGTHEN current signals")

    elif success_rate >= 0.2:
        # Success rate 20-50%: mixed approach
        suggestions.append("   ⚙️  SUCCESS RATE IS MODERATE (20-50%) - MIXED APPROACH\n")
        suggestions.append("   Strategy: Keep what works, redesign what doesn't")

        if common_formulas:
            suggestions.append(f"\n   1️⃣  PRESERVE successful formulas but in NEW CONTEXTS:")
            for fname, count in common_formulas[:2]:
                suggestions.append(f"      → {fname}")

        if problematic_formulas:
            suggestions.append(f"\n   2️⃣  REPLACE or REDESIGN problematic observations:")
            for fname, _, _ in problematic_formulas[:2]:
                suggestions.append(f"      → Avoid: {fname}")

        suggestions.append("\n   3️⃣  TRY ALTERNATIVE observation angles:")
        suggestions.append("      → If current used price patterns, try volume/volatility")
        suggestions.append("      → If current used momentum, try mean reversion aspects")

    else:
        # Success rate < 20%: partial redesign
        suggestions.append("   🔄 SUCCESS RATE IS LOW (<20%) - PARTIAL REDESIGN NEEDED\n")
        suggestions.append("   Strategy: Major changes while preserving limited successful elements")

        if common_formulas and n_success > 0:
            suggestions.append(f"\n   1️⃣  ONLY keep the few successful elements:")
            for fname, count in common_formulas[:1]:
                suggestions.append(f"      → {fname} (was successful)")

        suggestions.append("\n   2️⃣  REDESIGN the hypothesis structure:")
        suggestions.append("      → Simplify observation conditions (current may be too complex)")
        suggestions.append("      → Change the core behavioral pattern")
        if hypothesis:
            current_desc = hypothesis.get("behavioral_description", "")
            suggestions.append(f"      → Current: '{current_desc}' → Try different angle")

        suggestions.append("\n   3️⃣  EXPLORE different market phenomena:")
        suggestions.append("      → If used technical patterns, try fundamental/sentiment signals")
        suggestions.append("      → If used short-term, try longer time horizons")

    # === 4. Combination size analysis ===
    if successful_patterns:
        combo_sizes_success = [len(p.get("formula_names", [])) for p in successful_patterns]
        avg_size_success = sum(combo_sizes_success) / len(combo_sizes_success) if combo_sizes_success else 0

        if failed_combinations:
            combo_sizes_fail = [len(c.get("formula_names", [])) for c in failed_combinations]
            avg_size_fail = sum(combo_sizes_fail) / len(combo_sizes_fail) if combo_sizes_fail else 0

            suggestions.append(f"\n\n📊 Combination Size Analysis:")
            suggestions.append(f"   - Successful avg: {avg_size_success:.1f} formulas")
            suggestions.append(f"   - Failed avg: {avg_size_fail:.1f} formulas")

            if avg_size_success > avg_size_fail + 0.5:
                suggestions.append(f"\n   💡 Insight: LARGER combinations worked better")
                suggestions.append(f"      → Next hypothesis should have MORE diverse observation conditions")
            elif avg_size_success < avg_size_fail - 0.5:
                suggestions.append(f"\n   💡 Insight: SMALLER combinations worked better")
                suggestions.append(f"      → Next hypothesis should be SIMPLER with fewer, stronger conditions")

    suggestions.append("\n" + "="*80)
    return "\n".join(suggestions)


def create_hypothesis_memory_entry(
    stage4_summary: Dict[str, Any],
    hypothesis: Dict[str, Any] | None = None,
    observation_plan: Dict[str, Any] | None = None,
    formula_bundle: Dict[str, Any] | None = None,
    stage2_summary: Dict[str, Any] | None = None,
    stage3_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Create a feedback entry to append to hypothesis_memory from Stage4 results.
    Includes details for successful combinations (OOS excess_return_with_cost IR > 0).

    Args:
        stage4_summary: Summary of a Stage4 run.
        hypothesis: Stage1 hypothesis (optional, for context).
        observation_plan: Stage1 observation plan (optional, for context).
        formula_bundle: Stage1 formula bundle (optional, for context).
        stage2_summary: Stage2 formula validation results (optional, for per-formula details).
        stage3_result: Stage3 combination validation results (optional, for per-combination details).

    Returns:
        A dict entry to append to hypothesis_memory.
    """
    feedback = build_hypothesis_feedback_from_stage4(
        stage4_summary,
        hypothesis=hypothesis,
        observation_plan=observation_plan,
        formula_bundle=formula_bundle,
    )

    # Filter successful combinations (OOS excess_return_with_cost IR > 0)
    combination_results = stage4_summary.get("all_combinations", [])

    def _get_oos_excess_ir(comb: Dict[str, Any]) -> float:
        """Extract information_ratio from OOS excess_return_with_cost."""
        v = comb.get("outsample", {}).get("excess_return_with_cost", {}).get("information_ratio")
        if v is None:
            v = comb.get("data_split", {}).get("outsample", {}).get("excess_return_with_cost", {}).get("information_ratio")
        return float(v or 0.0)

    # Successful combinations only
    successful_combinations = [
        comb for comb in combination_results
        if _get_oos_excess_ir(comb) > 0
    ]

    # Average performance across all combinations
    if combination_results:
        def _get_is_ir(comb: Dict[str, Any]) -> float:
            v = comb.get("insample", {}).get("return", {}).get("information_ratio")
            if v is None:
                v = comb.get("data_split", {}).get("insample", {}).get("strategy", {}).get("information_ratio")
            return float(v or 0.0)

        def _get_oos_ir(comb: Dict[str, Any]) -> float:
            v = comb.get("outsample", {}).get("return", {}).get("information_ratio")
            if v is None:
                v = comb.get("data_split", {}).get("outsample", {}).get("strategy", {}).get("information_ratio")
            return float(v or 0.0)

        avg_is_ir = sum(_get_is_ir(c) for c in combination_results) / len(combination_results)
        avg_oos_ir = sum(_get_oos_ir(c) for c in combination_results) / len(combination_results)
    else:
        avg_is_ir = 0.0
        avg_oos_ir = 0.0

    # Collect details for successful combinations
    successful_patterns = []
    for comb in successful_combinations:
        pattern = {
            "combo_idx": comb.get("combo_idx"),
            "formula_names": comb.get("formula_names", []),
            "optimal_thresholds": comb.get("optimal_thresholds", {}),
            "backtest_results": {
                "insample": comb.get("insample", {}),
                "outsample": comb.get("outsample", {}),
                "fixed_modes": comb.get("fixed_modes", {}),
            },
            "hypothesis": hypothesis,
            "observation_plan": observation_plan,
            "formulas": {},  # Per-formula details
            "stage2_validation": {},  # Stage2 validation results
            "stage3_validation": {},  # Stage3 validation results
        }

        # Map Stage2 formula validation results
        if stage2_summary and formula_bundle:
            formula_results = stage2_summary.get("results", [])
            formulas_list = formula_bundle.get("formulas", [])

            for formula_name in comb.get("formula_names", []):
                # Find formula definition
                formula_def = next((f for f in formulas_list if f.get("name") == formula_name), None)
                # Find Stage2 validation result
                validation_result = next((r for r in formula_results if r.get("formula_name") == formula_name), None)

                if formula_def or validation_result:
                    pattern["formulas"][formula_name] = {
                        "definition": formula_def,
                        "stage2_validation": validation_result,
                    }

        # Attach Stage3 combination validation details (if available)
        if stage3_result:
            combination_stats = stage3_result.get("combination_stats", [])
            combo_stat = next(
                (cs for cs in combination_stats if cs.get("combination") == comb.get("formula_names")),
                None
            )
            if combo_stat:
                pattern["stage3_validation"] = combo_stat

        successful_patterns.append(pattern)

    # Generate concrete next-hypothesis suggestions
    next_hypothesis_suggestions = generate_next_hypothesis_suggestions(
        successful_patterns=successful_patterns,
        all_combinations=combination_results,
        hypothesis=hypothesis,
    )

    return {
        # Provide the last hypothesis object too (lets Stage1 prompt see prior ids/descriptions/horizon).
        "hypotheses": _extract_hypotheses_list(hypothesis),
        "horizon_days": _extract_horizon_days(hypothesis),
        "feedback": feedback,
        "next_hypothesis_suggestions": next_hypothesis_suggestions,  # Concrete next-hypothesis directions
        "avg_is_ir": avg_is_ir,
        "avg_oos_ir": avg_oos_ir,
        # Backward-compat keys (main.py logs expect these names)
        "avg_is_sharpe": avg_is_ir,
        "avg_oos_sharpe": avg_oos_ir,
        "n_combinations": len(combination_results),
        "n_successful": len(successful_combinations),
        "successful_patterns": successful_patterns,  # Details for successful combinations
        "iteration_type": "stage4_to_stage1",
    }


def create_stage3_fail_memory_entry(
    stage3_result: Dict[str, Any],
    stage2_summary: Dict[str, Any] | None = None,
    hypothesis: Dict[str, Any] | None = None,
    observation_plan: Dict[str, Any] | None = None,
    formula_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Create a hypothesis_memory feedback entry when Stage3 produces zero passing combinations.

    Args:
        stage3_result: Stage3 run result.
        stage2_summary: Stage2 summary.
        hypothesis: Stage1 hypothesis.
        observation_plan: Stage1 observation plan.
        formula_bundle: Stage1 formula bundle.

    Returns:
        A dict entry to append to hypothesis_memory.
    """
    feedback_parts = []
    feedback_parts.append("=== Stage3 FAIL Feedback ===\n")
    feedback_parts.append("CRITICAL: No formula combinations passed Stage3 validation.\n")

    # Add hypothesis info
    if hypothesis:
        behavioral_desc = hypothesis.get("behavioral_description", "")
        if behavioral_desc:
            feedback_parts.append(f"Failed Hypothesis: {behavioral_desc}\n")

    # Stage2 summary
    if stage2_summary:
        n_total = stage2_summary.get("total_formulas", 0)
        n_passed = stage2_summary.get("passed", 0)
        n_failed = stage2_summary.get("failed", 0)
        feedback_parts.append(f"Stage2 Results: {n_passed}/{n_total} formulas passed validation")
        if n_failed > 0:
            failed_formulas = stage2_summary.get("failed_formulas", [])
            feedback_parts.append(f"  Failed formulas: {failed_formulas}")

    # Analyze Stage3 failure reasons
    feedback_parts.append("\n=== Failure Analysis ===")

    overall_verdict = stage3_result.get("overall_verdict", "UNKNOWN")
    n_combinations = stage3_result.get("n_combinations", 0)
    n_passed_combinations = stage3_result.get("n_passed_combinations", 0)

    feedback_parts.append(f"Overall Verdict: {overall_verdict}")
    feedback_parts.append(f"Total Combinations Tested: {n_combinations}")
    feedback_parts.append(f"Passed Combinations: {n_passed_combinations}")

    # Analyze monotonicity validation failures
    feedback_parts.append("\n=== Why Combinations Failed ===")
    feedback_parts.append("Stage3 validates that stricter thresholds should improve performance monotonically.")
    feedback_parts.append("Possible reasons for failure:")
    feedback_parts.append("  1. The hypothesis does not capture a real market pattern")
    feedback_parts.append("  2. The observation conditions are not well-defined")
    feedback_parts.append("  3. The formulas do not correctly implement the observation conditions")
    feedback_parts.append("  4. The hypothesis is too weak or too noisy to pass strictness tests")

    # Add observation plan details
    if observation_plan:
        observations = observation_plan.get("observations", [])
        if observations:
            feedback_parts.append("\n=== Observation Conditions That Failed ===")
            for obs in observations:
                obs_id = obs.get("observation_id", obs.get("obs_id", ""))
                obs_desc = obs.get("description", "")
                feedback_parts.append(f"  - {obs_id}: {obs_desc}")

    # Add formula bundle details
    if formula_bundle:
        formulas = formula_bundle.get("formulas", [])
        if formulas:
            feedback_parts.append("\n=== Formulas Used ===")
            for f in formulas:
                fname = f.get("name", "")
                fdef = f.get("definition", "")
                feedback_parts.append(f"  - {fname}: {fdef}")

    # Improvement suggestions
    feedback_parts.append("\n=== Suggestions for Next Iteration ===")
    feedback_parts.append("IMPORTANT: The current hypothesis structure is fundamentally weak.")
    feedback_parts.append("Consider the following major changes:")
    feedback_parts.append("  1. Rethink the core hypothesis - the current one may not describe a real pattern")
    feedback_parts.append("  2. Define clearer, more distinct observation conditions")
    feedback_parts.append("  3. Ensure observation conditions are truly independent and measurable")
    feedback_parts.append("  4. Consider a completely different approach to the market phenomenon")
    feedback_parts.append("\nDo NOT just tweak the formulas - the hypothesis itself needs rethinking.")

    feedback = "\n".join(feedback_parts)

    return {
        "hypotheses": _extract_hypotheses_list(hypothesis),
        "horizon_days": _extract_horizon_days(hypothesis),
        "feedback": feedback,
        "avg_is_ir": 0.0,
        "avg_oos_ir": 0.0,
        "avg_is_sharpe": 0.0,
        "avg_oos_sharpe": 0.0,
        "n_combinations": n_combinations,
        "n_passed_combinations": n_passed_combinations,
        "iteration_type": "stage3_fail_to_stage1",
        "failure_reason": "No combinations passed Stage3 monotonicity validation",
    }


def create_stage3_fail_memory_entry(
    stage3_result: Dict[str, Any],
    stage2_summary: Dict[str, Any] | None = None,
    hypothesis: Dict[str, Any] | None = None,
    observation_plan: Dict[str, Any] | None = None,
    formula_bundle: Dict[str, Any] | None = None,
    model: str = "gpt-4o-mini",
    run_ctx: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    When Stage3 produces zero passing combinations, ask an LLM to analyze the failure and generate feedback.

    Args:
        stage3_result: Stage3 run result.
        stage2_summary: Stage2 summary.
        hypothesis: Stage1 hypothesis.
        observation_plan: Stage1 observation plan.
        formula_bundle: Stage1 formula bundle.
        model: LLM model name.
        run_ctx: RunContext

    Returns:
        A dict entry to append to hypothesis_memory.
    """
    from util.llm_client import call_llm
    import json

    # Stage3 result info
    overall_verdict = stage3_result.get("overall_verdict", "UNKNOWN")
    n_combinations = stage3_result.get("n_combinations", 0)
    n_passed_combinations = stage3_result.get("n_passed_combinations", 0)

    # Build context
    context_parts = []
    context_parts.append("=== STAGE3 FAILURE ANALYSIS REQUEST ===\n")
    context_parts.append("All formula combinations failed Stage3 validation.")
    context_parts.append("Stage3 validates that stricter thresholds should improve performance monotonically.\n")

    # Hypothesis info
    if hypothesis:
        context_parts.append("=== HYPOTHESIS ===")
        context_parts.append(json.dumps(hypothesis, ensure_ascii=False, indent=2, default=str))
        context_parts.append("")

    # Observation Plan
    if observation_plan:
        context_parts.append("=== OBSERVATION PLAN ===")
        context_parts.append(json.dumps(observation_plan, ensure_ascii=False, indent=2, default=str))
        context_parts.append("")

    # Formula Bundle
    if formula_bundle:
        context_parts.append("=== FORMULA BUNDLE ===")
        # Extract formulas only
        formulas = formula_bundle.get("formulas", [])
        for f in formulas:
            context_parts.append(f"- {f.get('name', 'unknown')}: {f.get('definition', 'N/A')}")
            context_parts.append(f"  observation_id: {f.get('observation_id', 'N/A')}")
            context_parts.append(f"  polarity: {f.get('polarity', 'N/A')}")
        context_parts.append("")

    # Stage2 results
    if stage2_summary:
        context_parts.append("=== STAGE2 RESULTS ===")
        context_parts.append(f"Total formulas: {stage2_summary.get('total_formulas', 0)}")
        context_parts.append(f"Passed: {stage2_summary.get('passed', 0)}")
        context_parts.append(f"Failed: {stage2_summary.get('failed', 0)}")
        failed_formulas = stage2_summary.get("failed_formulas", [])
        if failed_formulas:
            context_parts.append(f"Failed formula names: {failed_formulas}")
        context_parts.append("")

    # Stage3 results
    context_parts.append("=== STAGE3 RESULTS ===")
    context_parts.append(f"Overall Verdict: {overall_verdict}")
    context_parts.append(f"Total Combinations Tested: {n_combinations}")
    context_parts.append(f"Passed Combinations: {n_passed_combinations}")

    # Add detailed results (if available)
    combination_details = stage3_result.get("combination_results", [])
    if combination_details:
        context_parts.append("\nCombination Details:")
        for detail in combination_details[:5]:  # Up to 5 items
            context_parts.append(f"  - Combo: {detail.get('formula_names', [])}")
            context_parts.append(f"    Verdict: {detail.get('verdict', 'N/A')}")
            context_parts.append(f"    Reason: {detail.get('reason', 'N/A')}")

    context = "\n".join(context_parts)

    # Ask the LLM for analysis
    system_prompt = """You are a quantitative research analyst specializing in hypothesis validation.

Your task is to analyze why a financial hypothesis failed Stage3 validation and provide actionable feedback for the next iteration.

Stage3 validates that:
1. Stricter observation thresholds should improve signal quality
2. The relationship between strictness and performance should be monotonic
3. The hypothesis structure should capture a real market pattern

When a hypothesis fails Stage3, it means the observation conditions do not form a coherent, monotonically improving signal.

Provide your analysis in English. Be specific about:
1. What went wrong with the hypothesis structure
2. Which observation conditions are problematic and why
3. Concrete suggestions for a completely new approach

Do NOT suggest minor formula tweaks. The hypothesis itself needs fundamental rethinking."""

    user_prompt = f"""Below is the hypothesis that failed Stage3 validation and the related information.

{context}

Analyze the information above and answer the following:

1. **Root-cause analysis**: Why did this hypothesis fail Stage3 validation? Be specific.

2. **Observation-condition issues**: Which observation conditions are problematic, and why did monotonic improvement not appear?

3. **Hypothesis redesign proposal**: Propose a different way to approach the same financial concept. Do not suggest minor formula tweaks; explain concretely how to change the hypothesis structure itself.

Your response will be provided as feedback to the LLM when generating the next hypothesis. Write clearly and concretely."""

    try:
        # LLM call (text-only; no tools)
        response = call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=None,
            target_tool_name=None,
            temperature=0.7,
            react_log_path=run_ctx.root_dir / "logs/agents" if run_ctx else None,
            react_agent_name="stage3_failure_analyzer",
            context="Stage3 Failure Analysis",
        )

        if isinstance(response, str):
            feedback = response
        elif isinstance(response, dict):
            feedback = response.get("content", str(response))
        else:
            feedback = str(response)

        logger.info(f"Stage3 failure analysis completed. Feedback length: {len(feedback)}")

    except Exception as e:
        logger.error(f"Failed to generate Stage3 failure analysis: {e}")
        # Fallback to template-based feedback
        feedback = _generate_template_stage3_feedback(
            stage3_result, stage2_summary, hypothesis, observation_plan, formula_bundle
        )

    return {
        "hypotheses": _extract_hypotheses_list(hypothesis),
        "horizon_days": _extract_horizon_days(hypothesis),
        "feedback": feedback,
        "avg_is_ir": 0.0,
        "avg_oos_ir": 0.0,
        "avg_is_sharpe": 0.0,
        "avg_oos_sharpe": 0.0,
        "n_combinations": n_combinations,
        "n_passed_combinations": n_passed_combinations,
        "iteration_type": "stage3_fail_to_stage1",
        "failure_reason": "No combinations passed Stage3 monotonicity validation",
    }


def _generate_template_stage3_feedback(
    stage3_result: Dict[str, Any],
    stage2_summary: Dict[str, Any] | None,
    hypothesis: Dict[str, Any] | None,
    observation_plan: Dict[str, Any] | None,
    formula_bundle: Dict[str, Any] | None,
) -> str:
    """Template-based feedback used when the LLM call fails."""
    feedback_parts = []
    feedback_parts.append("=== Stage3 FAIL Feedback ===\n")
    feedback_parts.append("CRITICAL: No formula combinations passed Stage3 validation.\n")

    if hypothesis:
        behavioral_desc = hypothesis.get("behavioral_description", "")
        if behavioral_desc:
            feedback_parts.append(f"Failed Hypothesis: {behavioral_desc}\n")

    feedback_parts.append("The hypothesis structure is fundamentally weak.")
    feedback_parts.append("Consider rethinking the core hypothesis - it may not describe a real pattern.")

    return "\n".join(feedback_parts)


# ============================================================================
# End of Refinement Pipeline (Stage 4 → Stage 1)
# ============================================================================
