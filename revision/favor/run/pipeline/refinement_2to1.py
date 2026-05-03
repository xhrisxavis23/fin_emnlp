"""
================================================================================
Refinement Pipeline (Stage 2 → Stage 1)
Refinement pipeline for failed observation-implementation formulas
================================================================================

[Purpose]
- Consume Stage 2 results that judge whether a formula plausibly implements the target observation (PASS/FAIL).
  If a formula FAILs, refine it using the LLM's rationale (why it failed) and structured numeric evidence.
- Keep PASS formulas unchanged; refine only FAIL formulas.
- The refined bundle is sent back to Stage 1 for re-validation.

[Principles]
- This refinement targets observation-implementation quality, not strategy performance/returns.
- Drive changes using the LLM rationale + numeric citations (`primary_evidence` / `evidence_packet`).
- Keep `observation_id` fixed; refine only the formula definitions.
- Preserve PASS formulas to keep the total bundle size stable.

[Refinement Strategy]
- Use rationale (natural language) + evidence to make safe edits (continuity, simplification, axis alignment, noise reduction, etc.).

================================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.formula_agent import FormulaAgent
from schemas.validation_dataclasses import FormulaValidationResult
from util.run_context import RunContext
from run.config import RDConfig
from run.util.config_utils import resolve_cfg, resolve_model

logger = logging.getLogger(__name__)


# ============================================================================
# Refinement Functions
# ============================================================================

def _build_diagnostics_from_validation_result(
    validation_result: FormulaValidationResult,
) -> Dict[str, Any]:
    """
    Convert a `FormulaValidationResult` into the diagnostics format expected by `FormulaAgent`.

    `FormulaAgent.refine_behavioral_bundle` consumes `diagnostics`, so Stage 2 results must be adapted.

    Args:
        validation_result: Stage 2 validation result

    Returns:
        diagnostics: Diagnostics payload for `FormulaAgent`
    """
    rationale = getattr(validation_result, "reasoning", "") or ""
    diagnostics = {
        "formula_id": validation_result.formula_id,
        "formula_name": validation_result.formula_name,
        "obs_id": validation_result.obs_id,
        "verdict": validation_result.verdict,
        "reasoning": validation_result.reasoning,
        "rationale": rationale,
        "quantile_counts": validation_result.quantile_counts,
        "evidence_packet": validation_result.evidence_packet,
        "primary_evidence": validation_result.primary_evidence,
        "distribution_summary": validation_result.distribution_summary,
    }

    # Legacy note: improvement_hints is no longer used; refinement relies on `reasoning` only.
    # if validation_result.improvement_hints:
    #     diagnostics["improvement_hints"] = validation_result.improvement_hints
    

    return diagnostics


def _extract_focus_from_diagnostics(diagnostics: Dict[str, Any]) -> str:
    """
    Extract a refinement focus string from diagnostics.

    Args:
        diagnostics: Diagnostics payload

    Returns:
        focus: Refinement focus string
    """
    base_focus = "Improve hypothesis-observation alignment using rationale and cited numeric evidence"
    rationale = str(diagnostics.get("rationale") or "").strip()

    # Extract key mismatches from primary_evidence
    primary_evidence = diagnostics.get("primary_evidence", [])
    key_metrics = []
    for ev in primary_evidence:
        if isinstance(ev, dict):
            # Support both "metric" and "feature" keys
            metric = ev.get("metric") or ev.get("feature")
            if metric:
                key_metrics.append(metric)

    # Build focus string
    focus_parts = [base_focus]

    if key_metrics:
        focus_parts.append(f"Focus on metrics: {', '.join(key_metrics)}")

    if rationale:
        focus_parts.append(f"Rationale: {rationale[:240]}")

    return "; ".join(focus_parts)


def _build_combined_diagnostics(
    validation_results: List[FormulaValidationResult],
) -> Dict[str, Any]:
    """
    Combine multiple FAIL validation results into a single diagnostics payload.

    Args:
        validation_results: Validation results for failed formulas

    Returns:
        combined_diagnostics: Combined diagnostics payload
    """
    failed_formula_names = []
    per_formula_diagnostics = []

    for vr in validation_results:
        failed_formula_names.append(vr.formula_name)
        per_formula_diagnostics.append({
            "formula_name": vr.formula_name,
            "obs_id": vr.obs_id,
            "verdict": vr.verdict,
            "reasoning": vr.reasoning,
            "primary_evidence": vr.primary_evidence,
            "distribution_summary": vr.distribution_summary,
        })

    return {
        "failed_formula_names": failed_formula_names,
        "per_formula_diagnostics": per_formula_diagnostics,
    }


def _extract_focus_from_combined_diagnostics(combined_diagnostics: Dict[str, Any]) -> str:
    """
    Extract a refinement focus string from combined diagnostics.
    """
    failed_names = combined_diagnostics.get("failed_formula_names", [])
    per_formula = combined_diagnostics.get("per_formula_diagnostics", [])

    focus_parts = [
        "Improve hypothesis-observation alignment",
        f"FAIL formulas to refine: {', '.join(failed_names)}",
    ]

    # Summarize rationale per formula (keep only a few for brevity)
    for diag in per_formula[:3]:  # include up to 3
        name = diag.get("formula_name", "")
        reasoning = str(diag.get("reasoning") or "")[:150]
        if reasoning:
            focus_parts.append(f"{name}: {reasoning}")

    return "; ".join(focus_parts)


def refine_failed_formula(
    *,
    hypothesis: Dict[str, Any],
    current_bundle: Dict[str, Any],
    validation_result: FormulaValidationResult,
    observation_plan: Dict[str, Any] = None,
    metadata: list = None,
    knowledge: str = "",
    refine_rounds: int = 1,
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    cfg: RDConfig | None = None,
    original_bundle: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Refine a single formula that failed Stage 2.

    This calls `FormulaAgent.refine_behavioral_bundle` to improve the formula using
    rationale (why FAIL) and structured evidence.

    Args:
        hypothesis: Hypothesis info
        current_bundle: Current formula bundle (full)
        validation_result: Stage 2 validation result (failed formula)
        observation_plan: Observation plan
        metadata: Allowed column list
        knowledge: Extra knowledge context
        refine_rounds: Self-correction rounds
        model: LLM model name
        run_ctx: RunContext
        cfg: RDConfig

    Returns:
        refined_bundle: Refined formula bundle
    """
    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)

    # Build diagnostics
    diagnostics = _build_diagnostics_from_validation_result(validation_result)

    # Extract focus string
    focus = _extract_focus_from_diagnostics(diagnostics)

    logger.info(
        f"Refining formula '{validation_result.formula_name}' "
        f"(verdict={validation_result.verdict})"
    )
    logger.info(f"Refinement focus: {focus}")

    # Refine bundle via FormulaAgent
    agent = FormulaAgent(model=model, run_ctx=run_ctx)

    refined_bundle = agent.refine_behavioral_bundle(
        hypothesis=hypothesis,
        current_bundle=current_bundle,
        diagnostics=diagnostics,
        metadata=metadata,
        knowledge=knowledge,
        focus=focus,
        refine_rounds=refine_rounds,
        observation_plan=observation_plan,
        original_bundle=original_bundle,  # Pass original bundle for name tracking
    )

    return refined_bundle


def refine_failed_formulas_batch(
    *,
    hypothesis: Dict[str, Any],
    current_bundle: Dict[str, Any],
    validation_results: List[FormulaValidationResult],
    observation_plan: Dict[str, Any] = None,
    metadata: list = None,
    knowledge: str = "",
    refine_rounds: int = 1,
    max_refinement_iterations: int = 3,
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    cfg: RDConfig | None = None,
) -> Dict[str, Any]:
    """
    Batch-refine multiple formulas that failed Stage 2.

    Provide the full bundle to the LLM, instructing it to modify only FAIL formulas.
    The LLM returns the full bundle, preserving PASS formulas unchanged.

    Args:
        hypothesis: Hypothesis info
        current_bundle: Current formula bundle
        validation_results: Stage 2 validation results (PASS + FAIL)
        observation_plan: Observation plan
        metadata: Allowed column list
        knowledge: Extra knowledge context
        refine_rounds: Self-correction rounds per refinement
        max_refinement_iterations: Maximum refinement iterations
        model: LLM model name
        run_ctx: RunContext
        cfg: RDConfig

    Returns:
        refined_bundle: Full formula bundle (only FAIL formulas modified)
    """
    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)

    # Split PASS/FAIL
    failed_results = [
        r for r in validation_results
        if r.verdict == "FAIL"
    ]
    passed_results = [
        r for r in validation_results
        if r.verdict == "PASS"
    ]

    if not failed_results:
        logger.info("No failed formulas to refine")
        return current_bundle

    failed_formula_names = [r.formula_name for r in failed_results]
    logger.info(
        f"Refining {len(failed_results)} failed formulas: {failed_formula_names}, "
        f"preserving {len(passed_results)} passed formulas"
    )

    # Combine diagnostics for all FAIL formulas
    combined_diagnostics = _build_combined_diagnostics(failed_results)
    focus = _extract_focus_from_combined_diagnostics(combined_diagnostics)

    logger.info(f"Refinement focus: {focus[:200]}...")

    # Refine the entire bundle in one shot via FormulaAgent.
    # (Instruct to modify FAIL formulas only; keep PASS formulas unchanged.)
    agent = FormulaAgent(model=model, run_ctx=run_ctx)

    try:
        refined_bundle = agent.refine_behavioral_bundle(
            hypothesis=hypothesis,
            current_bundle=current_bundle,  # pass the full bundle
            diagnostics=combined_diagnostics,
            metadata=metadata,
            knowledge=knowledge,
            focus=focus,
            refine_rounds=refine_rounds,
            observation_plan=observation_plan,
            original_bundle=current_bundle,
        )
        logger.info(f"Successfully refined {len(failed_formula_names)} formulas")
        return refined_bundle

    except Exception as e:
        logger.error(f"Failed to refine formulas: {e}", exc_info=True)
        return current_bundle


def build_refinement_feedback(
    validation_results: List[FormulaValidationResult],
) -> str:
    """
    Build a refinement feedback string from validation results.

    This feedback is intended to help `FormulaAgent` avoid repeating the same mistakes
    in subsequent generations.

    Args:
        validation_results: Stage 2 validation results

    Returns:
        feedback: Feedback string
    """
    failed_results = [r for r in validation_results if r.verdict == "FAIL"]

    if not failed_results:
        return "All formulas passed validation. No refinement needed."

    feedback_parts = []
    feedback_parts.append(f"=== Validation Feedback ({len(failed_results)} failures) ===\n")

    feedback_parts.append("Failure breakdown (by obs_id):")
    obs_counts: dict[str, int] = {}
    for result in failed_results:
        oid = str(getattr(result, "obs_id", "") or "unknown").strip() or "unknown"
        obs_counts[oid] = obs_counts.get(oid, 0) + 1

    for oid, count in sorted(obs_counts.items(), key=lambda x: -x[1]):
        feedback_parts.append(f"  - {oid}: {count} formulas")

    feedback_parts.append("\nDetailed failures:\n")

    # Detailed feedback for a subset of failed formulas
    for i, result in enumerate(failed_results[:5], 1):  # show up to 5
        feedback_parts.append(f"\n{i}. Formula: {result.formula_name} (obs: {result.obs_id})")
        rationale = getattr(result, "reasoning", "") or ""
        feedback_parts.append(f"   Rationale: {rationale[:200]}{'...' if len(rationale) > 200 else ''}")

        # Extract key info from primary_evidence
        if result.primary_evidence:
            feedback_parts.append("   Key evidence:")
            for ev in result.primary_evidence[:2]:  # show up to 2
                if isinstance(ev, dict):
                    # Support both "metric" and "feature" keys
                    metric = ev.get("metric") or ev.get("feature", "N/A")

                    # Extract patterns (support multiple formats)
                    if "expected" in ev and "observed" in ev:
                        expected = ev.get("expected", "N/A")
                        observed = ev.get("observed", "N/A")
                        feedback_parts.append(f"     - {metric}: expected={expected}, observed={observed}")
                    elif "pattern" in ev:
                        pattern = ev.get("pattern", "N/A")
                        stat = ev.get("stat", "")
                        bins = ev.get("bins", "")
                        feedback_parts.append(f"     - {metric} ({stat}): {pattern} trend across {bins}")
                    else:
                        feedback_parts.append(f"     - {metric}: {ev}")

    if len(failed_results) > 5:
        feedback_parts.append(f"\n... and {len(failed_results) - 5} more failures")

    return "\n".join(feedback_parts)


# ============================================================================
# Refinement Pipeline Runner
# ============================================================================

@dataclass
class RefinementResult:
    """Refinement result."""
    hypothesis_id: str
    original_bundle: Dict[str, Any]
    refined_bundle: Dict[str, Any]
    validation_results: List[FormulaValidationResult]
    n_failed: int
    n_refined: int
    refinement_feedback: str
    success: bool


def run_refinement_2to1(
    *,
    hypothesis: Dict[str, Any],
    current_bundle: Dict[str, Any],
    validation_results: List[FormulaValidationResult],
    observation_plan: Dict[str, Any] = None,
    metadata: list = None,
    knowledge: str = "",
    refine_rounds: int = 1,
    max_refinement_iterations: int = 3,
    model: str | None = None,
    run_ctx: Optional[RunContext] = None,
    cfg: RDConfig | None = None,
) -> RefinementResult:
    """
    Run the refinement pipeline: refine Stage 2 failed formulas and send them back to Stage 1.

    Args:
        hypothesis: Hypothesis info
        current_bundle: Current formula bundle
        validation_results: Stage 2 validation results
        observation_plan: Observation plan
        metadata: Allowed column list
        knowledge: Extra knowledge context
        refine_rounds: Self-correction rounds per refinement
        max_refinement_iterations: Maximum refinement iterations
        model: LLM model name
        run_ctx: RunContext
        cfg: RDConfig

    Returns:
        RefinementResult: Refinement result
    """
    cfg = resolve_cfg(cfg)
    model = resolve_model(model, cfg)

    hypothesis_id = hypothesis.get("hypothesis_id", "unknown")

    # Filter failed formulas
    failed_results = [r for r in validation_results if r.verdict == "FAIL"]
    n_failed = len(failed_results)

    logger.info(
        f"Starting refinement pipeline for hypothesis {hypothesis_id}: "
        f"{n_failed} failed formulas"
    )

    if n_failed == 0:
        logger.info("No failed formulas to refine")
        return RefinementResult(
            hypothesis_id=hypothesis_id,
            original_bundle=current_bundle,
            refined_bundle=current_bundle,
            validation_results=validation_results,
            n_failed=0,
            n_refined=0,
            refinement_feedback="No failures detected",
            success=True,
        )

    # Run batch refinement (provide PASS + FAIL together)
    refined_bundle = refine_failed_formulas_batch(
        hypothesis=hypothesis,
        current_bundle=current_bundle,
        validation_results=validation_results,  # provide PASS + FAIL together
        observation_plan=observation_plan,
        metadata=metadata,
        knowledge=knowledge,
        refine_rounds=refine_rounds,
        max_refinement_iterations=max_refinement_iterations,
        model=model,
        run_ctx=run_ctx,
        cfg=cfg,
    )

    # Build feedback
    refinement_feedback = build_refinement_feedback(failed_results)

    # Determine whether refinement succeeded (bundle changed)
    bundle_changed = (
        json.dumps(refined_bundle, sort_keys=True) !=
        json.dumps(current_bundle, sort_keys=True)
    )

    n_refined = n_failed if bundle_changed else 0

    logger.info(
        f"Refinement pipeline completed: "
        f"{n_refined}/{n_failed} formulas refined"
    )

    return RefinementResult(
        hypothesis_id=hypothesis_id,
        original_bundle=current_bundle,
        refined_bundle=refined_bundle,
        validation_results=validation_results,
        n_failed=n_failed,
        n_refined=n_refined,
        refinement_feedback=refinement_feedback,
        success=bundle_changed,
    )


# ============================================================================
# Utility Functions
# ============================================================================

def extract_failed_validation_results(
    stage2_summary: Dict[str, Any],
) -> List[FormulaValidationResult]:
    """
    Extract failed validation results from a Stage 2 summary.

    Args:
        stage2_summary: Stage 2 run summary

    Returns:
        failed_results: List of failed FormulaValidationResult
    """
    results = stage2_summary.get("results", [])
    failed_results = []

    for result_dict in results:
        if not isinstance(result_dict, dict):
            continue

        # Convert dict to FormulaValidationResult
        # NOTE: assume result_dict is already in `asdict()`-compatible shape
        if result_dict.get("verdict") == "FAIL":
            try:
                # Extract required fields only
                validation_result = FormulaValidationResult(
                    formula_id=result_dict.get("formula_id", ""),
                    formula_name=result_dict.get("formula_name", ""),
                    obs_id=result_dict.get("obs_id", ""),
                    verdict=result_dict.get("verdict", "FAIL"),
                    reasoning=result_dict.get("reasoning", ""),
                    quantile_counts=result_dict.get("quantile_counts", {}),
                    evidence_packet=result_dict.get("evidence_packet", {}),
                    primary_evidence=result_dict.get("primary_evidence", []),
                    distribution_by_element=result_dict.get("distribution_by_element", {}),
                    # improvement_hints=result_dict.get("improvement_hints"),
                    distribution_summary=result_dict.get("distribution_summary", ""),
                )
                failed_results.append(validation_result)
            except Exception as e:
                logger.error(f"Failed to parse validation result: {e}")
                continue

    return failed_results


def create_refinement_memory_entry(
    validation_results: List[FormulaValidationResult],
) -> Dict[str, Any]:
    """
    Create a feedback entry to append to `formula_memory` based on validation results.

    `FormulaAgent.purpose_formula` accepts a `formula_memory` parameter that may include
    prior failure feedback; append this entry to it.

    Args:
        validation_results: Stage 2 validation results

    Returns:
        memory_entry: Entry to add to formula_memory (dict with a 'feedback' field)
    """
    feedback = build_refinement_feedback(validation_results)

    return {
        "feedback": feedback,
        "n_failures": len([r for r in validation_results if r.verdict == "FAIL"]),
        "failure_rationales": [
            str(getattr(r, "reasoning", "") or "").strip()[:160]
            for r in validation_results
            if r.verdict == "FAIL" and str(getattr(r, "reasoning", "") or "").strip()
        ],
    }


# ============================================================================
# End of Refinement Pipeline
# ============================================================================
