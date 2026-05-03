"""
================================================================================
Hypothesis-Observation-Validation Framework
================================================================================

This module is the entry point for running the research pipeline:

- `run_pipeline()`: Stage1 → Stage2 (optional formula refinement loop) → Stage3 → Stage4.
- `run_outer_loop()`: Optional outer loop that repeats the pipeline and accumulates feedback
  that guides hypothesis regeneration.

Data split and leakage control:
- In-sample/out-of-sample periods are configured via `run/config.py` (`DataSplitConfig`).
- Stage2/Stage3 validate on in-sample only.
- Stage4 optimizes thresholds on in-sample and reports final metrics on out-of-sample.

Note:
- The pipeline assumes daily OHLCV-only observability.
- Comments and docstrings are kept in English for maintainability.
"""

from __future__ import annotations

import itertools
import math
import warnings

warnings.filterwarnings("ignore")

from typing import Any, Optional
from dataclasses import asdict

import polars as pl

from datetime import datetime
from run.config import RDConfig, load_price_data, load_rd_config
from run.pipeline.stage1 import Stage1Result, run_stage1
from run.pipeline.stage2 import run_stage2, Stage2Result
from run.pipeline.stage3 import run_stage3, Stage3Result
from run.pipeline.stage4 import run_stage4
from run.pipeline.refinement_2to1 import run_refinement_2to1
from run.pipeline.refinement_4to1 import (
    create_hypothesis_memory_entry,
    create_stage3_fail_memory_entry,
)
from run.util.pipeline_utils import compute_formula_values, FormulaComputeResult
from schemas.validation_dataclasses import FormulaValidationResult
from run.util.data_utils import standardize_price_columns
from util.run_context import RunContext


def _split_data_by_period(
    df: pl.DataFrame,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """
    Filter a Polars DataFrame by an inclusive timestamp range.

    Args:
        df: Polars DataFrame (must include a `timestamp` column)
        start_date: Start date (YYYY-MM-DD or YYYYMMDD)
        end_date: End date (YYYY-MM-DD or YYYYMMDD)

    Returns:
        Filtered DataFrame
    """
    # Normalize date strings (YYYY-MM-DD -> YYYYMMDD) for string comparisons.
    start_normalized = start_date.replace("-", "")
    end_normalized = end_date.replace("-", "")

    # Filter by string comparison (assumes timestamps are in YYYYMMDD-compatible format).
    filtered = df.filter(
        (pl.col("timestamp") >= start_normalized) &
        (pl.col("timestamp") <= end_normalized)
    )

    return filtered


def _parse_stage2_validation_results(stage2_summary: dict[str, Any]) -> list[FormulaValidationResult]:
    """
    Convert `stage2_summary["results"]` (dict-like / asdict payload) to FormulaValidationResult objects.

    NOTE:
    - refinement_2to1 needs the full PASS/FAIL set (PASS are preserved; FAIL are refined).
    """
    results = stage2_summary.get("results", [])
    parsed: list[FormulaValidationResult] = []

    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            parsed.append(
                FormulaValidationResult(
                    formula_id=str(r.get("formula_id", "") or ""),
                    formula_name=str(r.get("formula_name", "") or ""),
                    obs_id=str(r.get("obs_id", "") or ""),
                    verdict=str(r.get("verdict", "") or ""),
                    reasoning=str(r.get("reasoning", "") or ""),
                    quantile_counts=r.get("quantile_counts", {}) or {},
                    evidence_packet=r.get("evidence_packet", {}) or {},
                    primary_evidence=r.get("primary_evidence", []) or [],
                    distribution_by_element=r.get("distribution_by_element", {}) or {},
                    distribution_summary=str(r.get("distribution_summary", "") or ""),
                )
            )
        except Exception:
            # Skip partial/invalid entries to tolerate Stage2 summary schema changes.
            continue

    return parsed


def _get_missing_observations(
    *,
    observation_plan: dict[str, Any],
    passed_formulas: list[dict[str, Any]],
) -> list[str]:
    """
    Return observation_ids that appear in observation_plan but have zero passed formulas.

    NOTE:
    - Stage3 builds combinations from Stage2 `passed_formulas` grouped by observation_id.
    - If an observation has no surviving formula, the hypothesis instance cannot be constructed
      as intended (one formula per observation).
    """
    obs_items = observation_plan.get("observations", []) if isinstance(observation_plan, dict) else []
    expected_obs_ids: list[str] = []
    for item in obs_items:
        if not isinstance(item, dict):
            continue
        oid = str(item.get("observation_id") or item.get("obs_id") or "").strip()
        if oid:
            expected_obs_ids.append(oid)

    if not expected_obs_ids:
        return []

    obs_with_passed: set[str] = set()
    for f in passed_formulas or []:
        if not isinstance(f, dict):
            continue
        oid = str(f.get("observation_id") or f.get("obs_id") or "").strip()
        if oid:
            obs_with_passed.add(oid)

    missing = [oid for oid in expected_obs_ids if oid not in obs_with_passed]
    # stable ordering + de-dupe while preserving plan order
    seen: set[str] = set()
    out: list[str] = []
    for oid in missing:
        if oid in seen:
            continue
        seen.add(oid)
        out.append(oid)
    return out


def _build_stage3_combinations_without_validation(
    *,
    observation_plan: dict[str, Any] | None,
    passed_formulas: list[dict[str, Any]],
    max_combinations: int | None = None,
) -> list[list[dict[str, Any]]]:
    """
    Build hypothesis instances (combinations) without running Stage3's return/monotonicity validation.

    A combination is the AND of one formula per observation (obs1 ∧ obs2 ∧ ...).

    This is used when `pipeline_control.enable_stage3=False`:
    - We still construct combinations in the intended structure (one per observation).
    - We do NOT evaluate profitability/monotonicity in Stage3.
    - We intentionally avoid the old behavior of AND-ing *all* formulas into one combo (too strict).
    """
    if not passed_formulas:
        return []

    # Prefer observation plan order (stable + aligned with hypothesis decomposition).
    expected_obs_ids: list[str] = []
    if isinstance(observation_plan, dict):
        obs_items = observation_plan.get("observations", []) or []
        if isinstance(obs_items, list):
            for item in obs_items:
                if not isinstance(item, dict):
                    continue
                oid = str(item.get("observation_id") or item.get("obs_id") or "").strip()
                if oid:
                    expected_obs_ids.append(oid)

    # Group formulas by observation id (only keep obs_ids from the plan when available).
    obs_groups: dict[str, list[dict[str, Any]]] = {oid: [] for oid in expected_obs_ids} if expected_obs_ids else {}
    for f in passed_formulas:
        if not isinstance(f, dict):
            continue
        oid = str(f.get("observation_id") or f.get("obs_id") or "").strip()
        if expected_obs_ids:
            if oid in obs_groups:
                obs_groups[oid].append(f)
        else:
            if not oid:
                continue
            obs_groups.setdefault(oid, []).append(f)

    if not obs_groups:
        # If there is no observation metadata, fall back to singleton combos
        # (still better than returning an empty set and producing no signals).
        singletons = [[f] for f in passed_formulas if isinstance(f, dict)]
        if isinstance(max_combinations, int) and max_combinations > 0:
            singletons = singletons[:max_combinations]
        else:
            singletons = singletons[:10]
        return singletons

    # Deterministic ordering within each observation group.
    for oid, lst in obs_groups.items():
        lst.sort(key=lambda x: str(x.get("name") or x.get("id") or ""))

    # If any expected observation has no formulas, we cannot construct instances.
    if expected_obs_ids and any(len(obs_groups.get(oid, [])) == 0 for oid in expected_obs_ids):
        return []

    # Build full cartesian-product combinations: one formula per observation.
    # NOTE: This can be large when each observation has many PASS formulas.
    obs_order = expected_obs_ids if expected_obs_ids else sorted(obs_groups.keys())
    groups = [obs_groups[oid] for oid in obs_order if obs_groups.get(oid)]
    if not groups:
        return []

    combos: list[list[dict[str, Any]]] = []
    for tup in itertools.product(*groups):
        combos.append(list(tup))
        if isinstance(max_combinations, int) and max_combinations > 0 and len(combos) >= max_combinations:
            break

    # Optional cap (defensive).
    if isinstance(max_combinations, int) and max_combinations > 0:
        combos = combos[:max_combinations]

    # De-duplicate while preserving order.
    seen: set[tuple[str, ...]] = set()
    uniq: list[list[dict[str, Any]]] = []
    for c in combos:
        key = tuple(str(f.get("name") or f.get("id") or "") for f in c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    return uniq


def run_pipeline(
    *,
    concept: str,
    cfg: RDConfig | None = None,
    run_ctx: Optional[RunContext] = None,
    price_df: pl.DataFrame | None = None,
    max_refinement_iterations: int | None = None,
    enable_refinement_loop: bool | None = None,
    hypothesis_memory: list[Any] | None = None,
    outer_iter: int | None = None,
) -> dict[str, Any]:
    """
    End-to-end runner: Stage1 → Stage2 (with refinement loop) → Stage3 → Stage4.

    - If `price_df` is provided, it will be used (after column standardization via DataConfig).
    - Otherwise, data is loaded using the settings in `run/config.py` (DataConfig/QlibConfig).

    NOTE: Multi-ticker aggregation is supported.
    - Stage1: generate hypothesis/observations/formulas and compute formula values (multi-ticker panel)
    - Stage2: Observation Formula Validation (with refinement loop)
    - Stage3: per-ticker validation and aggregation
    - Stage4: threshold optimization on in-sample, then evaluation on out-of-sample

    Refinement Loop:
    - If Stage2 produces FAIL formulas, run refinement_2to1.
    - Preserve PASS formulas; refine FAIL formulas only (do not regenerate Stage1 wholesale).
    - Recompute only refined formula values and re-run Stage2 for those formulas.
    - Repeat until all formulas PASS or max_refinement_iterations is reached.

    Args:
        concept: Finance hypothesis / concept to validate
        cfg: RDConfig
        run_ctx: RunContext
        price_df: Price data (if None, loaded from config)
        max_refinement_iterations: Max refinement iterations (defaults to cfg.refinement.max_inner_iterations)
        enable_refinement_loop: Enable refinement loop (defaults to cfg.refinement.enable_inner_loop)
    """
    cfg = cfg or load_rd_config()
    run_ctx = run_ctx or RunContext.create()

    # Resolve refinement settings.
    if max_refinement_iterations is None:
        max_refinement_iterations = cfg.refinement.max_inner_iterations
    if enable_refinement_loop is None:
        enable_refinement_loop = cfg.refinement.enable_inner_loop

    if price_df is None:
        price_df = load_price_data(cfg)

    price_df = standardize_price_columns(price_df, cfg.data)

    # Print dataset scope early (helps interpret pooling/Stage2+3 counts)
    asset_col = getattr(cfg.data, "asset_col", "ticker")
    n_unique_tickers = int(price_df.select(pl.col(asset_col).n_unique()).item())
    run_ctx.log(f"[Data] Unique tickers: {n_unique_tickers:,}")

    # ========================================================================
    # Train/Val/Test 기간 설정 (3-way split)
    # ========================================================================
    # Stage 2/3: Train 기간 사용
    train_start = cfg.data_split.train_start
    train_end = cfg.data_split.train_end
    # Stage 4 Optuna: Validation 기간 사용
    val_start = cfg.data_split.val_start
    val_end = cfg.data_split.val_end
    # Stage 4 최종 평가: Test 기간 사용
    test_start = cfg.data_split.test_start
    test_end = cfg.data_split.test_end

    # 하위 호환성을 위한 변수 (기존 코드에서 is_*, oos_* 사용)
    is_start = train_start
    is_end = train_end
    oos_start = test_start
    oos_end = test_end

    run_ctx.log(f"\n{'='*80}")
    run_ctx.log("Data Split (3-way)")
    run_ctx.log(f"{'='*80}")
    run_ctx.log(f"  Train (Stage2/3):      {train_start} ~ {train_end}")
    run_ctx.log(f"  Validation (Optuna):   {val_start} ~ {val_end}")
    run_ctx.log(f"  Test (Final Eval):     {test_start} ~ {test_end}")
    run_ctx.log(f"{'='*80}\n")

    # ========================================================================
    # Refinement loop: Stage1 ⇄ Stage2
    # ========================================================================
    refinement_history = []
    stage1: Stage1Result | None = None

    for iteration in range(max_refinement_iterations):
        run_ctx.log(f"\n{'='*80}")
        run_ctx.log(f"Refinement Iteration {iteration + 1}/{max_refinement_iterations}")
        run_ctx.log(f"{'='*80}\n")

        # Run Stage1 once (hypothesis/observation plan are fixed for the inner loop).
        # For subsequent iterations, refine only formulas that failed Stage2 and recompute formula values.
        if stage1 is None:
            stage1 = run_stage1(
                concept=concept,
                price_df=price_df,
                cfg=cfg,
                run_ctx=run_ctx,
                hypothesis_memory=hypothesis_memory if hypothesis_memory else None,
            )

            # Log failed formulas from Stage1 (if any - after Stage1's internal retry loop)
            if stage1.failed_formulas:
                run_ctx.log(f"\n⚠️  Stage1: {len(stage1.failed_formulas)} formula(s) still failing after internal retries:")
                for ff in stage1.failed_formulas:
                    run_ctx.log(f"   - {ff['name']}: {ff['error']}")
                run_ctx.log("These formulas will be marked as FAIL in Stage2.\n")

        # ═══════════════════════════════════════════════════════════════════════
        # Stage 2: validate formulas on in-sample only to avoid OOS leakage.
        # Formula values are computed on the full dataset, but validation is restricted to the IS window.
        # ═══════════════════════════════════════════════════════════════════════
        is_ohlcv_df = _split_data_by_period(stage1.ohlcv_df, is_start, is_end)
        is_formula_df = _split_data_by_period(stage1.formula_df, is_start, is_end)

        run_ctx.log(f"Stage2: Using IS data only ({is_start} ~ {is_end})")
        run_ctx.log(f"  IS samples: {len(is_ohlcv_df):,} rows")

        # Stage2 skip path.
        if not cfg.pipeline_control.enable_stage2:
            run_ctx.log("\n⚡ Stage2 SKIPPED (pipeline_control.enable_stage2=False)")
            run_ctx.log("   All formulas are assumed to PASS.\n")

            # Create a dummy result: treat all formulas as PASS.
            formulas = stage1.formula_bundle.get("formulas", [])
            all_formula_names = [f.get("name") for f in formulas if isinstance(f, dict) and f.get("name")]

            stage2 = Stage2Result(
                hypothesis_id=stage1.formula_bundle.get("hypothesis_id", ""),
                summary={
                    "total_formulas": len(formulas),
                    "passed": len(formulas),
                    "failed": 0,
                    "conditional": 0,
                    "pass_rate": 1.0,
                    "passed_formulas": all_formula_names,
                    "failed_formulas": [],
                    "conditional_formulas": [],
                    "results": [
                        {
                            "formula_id": f.get("id", f.get("name", "")),
                            "formula_name": f.get("name", ""),
                            "obs_id": f.get("obs_id", ""),
                            "verdict": "PASS",
                            "reasoning": "Stage2 skipped - formula assumed to pass",
                        }
                        for f in formulas if isinstance(f, dict)
                    ],
                    "skipped": True,
                },
                passed_formulas=formulas,
                report_md="# Stage2 Report\n\n**SKIPPED**: Stage2 validation was disabled via `pipeline_control.enable_stage2=False`.\n\nAll formulas are assumed to PASS.\n",
                pooling_info={"skipped": True},
            )
        else:
            stage2 = run_stage2(
                formula_bundle=stage1.formula_bundle,
                ohlcv_df=is_ohlcv_df,
                formula_df=is_formula_df,
                cfg=cfg,
                run_ctx=run_ctx,
                eval_failed_formulas=stage1.failed_formulas,
            )

        # Inspect validation results.
        n_failed = stage2.summary.get("failed", 0)
        n_total = stage2.summary.get("total_formulas", 0)
        pass_rate = stage2.summary.get("pass_rate", 0.0)

        run_ctx.log(f"\nStage2 Result (Iteration {iteration + 1}):")
        run_ctx.log(f"  Total formulas: {n_total}")
        run_ctx.log(f"  Passed: {stage2.summary.get('passed', 0)}")
        run_ctx.log(f"  Failed: {n_failed}")
        run_ctx.log(f"  Pass rate: {pass_rate:.1%}")

        # Persist this iteration's summary.
        refinement_history.append({
            "iteration": iteration + 1,
            "n_total": n_total,
            "n_passed": stage2.summary.get("passed", 0),
            "n_failed": n_failed,
            "pass_rate": pass_rate,
            "failed_formulas": stage2.summary.get("failed_formulas", []),
        })

        # Stop early if all formulas PASS.
        if n_failed == 0:
            run_ctx.log(f"\n✅ All formulas PASSED! (Iteration {iteration + 1})")
            run_ctx.log("Proceeding to Stage3...\n")
            break

        # Stop if refinement loop is disabled.
        if not enable_refinement_loop:
            run_ctx.log(f"\n⚠️  Refinement loop disabled. Stopping with {n_failed} failures.")
            break

        # Stop if this is the last iteration.
        if iteration == max_refinement_iterations - 1:
            run_ctx.log(f"\n⚠️  Max refinement iterations ({max_refinement_iterations}) reached.")
            run_ctx.log(f"Proceeding to Stage3 with {n_failed} failures...\n")
            break

        # Prepare refinement input from failed formulas.
        run_ctx.log(f"\n🔄 Refinement needed: {n_failed} formulas failed")
        run_ctx.log("Refining FAIL formulas only (PASS formulas preserved)...")

        validation_results = _parse_stage2_validation_results(stage2.summary)
        if not validation_results:
            run_ctx.log("Warning: No validation results parsed from Stage2 summary; stopping refinement loop.")
            break

        refinement = run_refinement_2to1(
            hypothesis=stage1.hypothesis,
            current_bundle=stage1.formula_bundle,
            validation_results=validation_results,  # PASS + FAIL
            observation_plan=stage1.observation_plan,
            metadata=cfg.stage1.allowed_ohlcv_columns,
            refine_rounds=cfg.stage1.refine_rounds,
            run_ctx=run_ctx,
            cfg=cfg,
        )

        if not refinement.success:
            run_ctx.log("Warning: refinement_2to1 produced no bundle changes; stopping refinement loop.")
            break

        # Recompute formula values only (keep hypothesis/observation plan fixed).
        refined_bundle = refinement.refined_bundle
        formulas = refined_bundle.get("formulas", []) if isinstance(refined_bundle, dict) else []
        # Extract formula names and remove duplicates (safety check)
        formula_names = list(dict.fromkeys(str(f.get("name")) for f in formulas if isinstance(f, dict) and f.get("name")))

        compute_result = compute_formula_values(price_df, formulas=formulas)
        price_with_formulas = compute_result.df
        failed_formulas = compute_result.failed_formulas

        # Log any failed formulas during refinement
        if failed_formulas:
            run_ctx.log(f"\n⚠️  Refinement: {len(failed_formulas)} formula(s) still failing:")
            for ff in failed_formulas:
                run_ctx.log(f"   - {ff['name']}: {ff['error']}")

        ohlcv_cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
        ohlcv_df = price_with_formulas.select([c for c in ohlcv_cols if c in price_with_formulas.columns])
        # Exclude failed formulas from formula_df
        valid_formula_names = [n for n in formula_names if n not in {f["name"] for f in failed_formulas}]
        formula_df = price_with_formulas.select(["timestamp", "ticker"] + valid_formula_names)

        # ═══════════════════════════════════════════════════════════════════════
        # Optimization: skip PASS formulas; re-validate only formulas that previously failed.
        # ═══════════════════════════════════════════════════════════════════════
        # Names of formulas that failed in the previous Stage2 run.
        failed_names_from_prev_stage2 = set(stage2.summary.get("failed_formulas", []))

        # Select refined formulas that previously failed (re-validation targets).
        refined_formulas_to_revalidate = [
            f for f in formulas
            if isinstance(f, dict) and f.get("name") in failed_names_from_prev_stage2
        ]

        # Preserve previous PASS validation results.
        passed_validation_results = [
            r for r in validation_results
            if r.verdict == "PASS"
        ]

        run_ctx.log(f"\n🔍 Re-validating only {len(refined_formulas_to_revalidate)} refined formulas (skipping {len(passed_validation_results)} PASS formulas)...")

        if refined_formulas_to_revalidate:
            # Partial bundle containing only formulas to re-validate.
            partial_bundle = {
                "hypothesis_id": refined_bundle.get("hypothesis_id"),
                "formulas": refined_formulas_to_revalidate,
            }

            # Extract refined formula names.
            refined_names_to_revalidate = [f.get("name") for f in refined_formulas_to_revalidate if f.get("name")]

            # Re-validate using in-sample data only (refined formulas only).
            is_ohlcv_df_partial = _split_data_by_period(ohlcv_df, is_start, is_end)
            is_formula_df_partial = _split_data_by_period(
                formula_df.select(["timestamp", "ticker"] + [n for n in refined_names_to_revalidate if n in formula_df.columns]),
                is_start,
                is_end
            )

            run_ctx.log(f"  Partial Stage2: Using IS data only ({is_start} ~ {is_end})")
            run_ctx.log(f"  IS samples: {len(is_ohlcv_df_partial):,} rows")

            # Run partial re-validation.
            stage2_partial = run_stage2(
                formula_bundle=partial_bundle,
                ohlcv_df=is_ohlcv_df_partial,
                formula_df=is_formula_df_partial,
                cfg=cfg,
                run_ctx=run_ctx,
                eval_failed_formulas=[ff for ff in failed_formulas if ff.get("name") in refined_names_to_revalidate],
            )

            # ═══════════════════════════════════════════════════════════════════════
            # Merge: reuse PASS results + partial re-validation results.
            # ═══════════════════════════════════════════════════════════════════════
            # Partial re-validation results.
            new_validation_results = stage2_partial.summary.get("results", [])

            # Convert PASS results to dicts for merging.
            passed_results_dict = [asdict(r) for r in passed_validation_results]

            # Combine results.
            combined_results = passed_results_dict + new_validation_results

            # Recompute summary fields.
            combined_passed = [r for r in combined_results if r.get("verdict") == "PASS"]
            combined_failed = [r for r in combined_results if r.get("verdict") == "FAIL"]
            combined_conditional = [r for r in combined_results if r.get("verdict") == "CONDITIONAL"]

            n_combined_total = len(combined_results)
            n_combined_passed = len(combined_passed)
            n_combined_failed = len(combined_failed)
            combined_pass_rate = n_combined_passed / n_combined_total if n_combined_total > 0 else 0.0

            # Update stage2.summary (used by the next iteration).
            stage2.summary["results"] = combined_results
            stage2.summary["total_formulas"] = n_combined_total
            stage2.summary["passed"] = n_combined_passed
            stage2.summary["failed"] = n_combined_failed
            stage2.summary["conditional"] = len(combined_conditional)
            stage2.summary["pass_rate"] = combined_pass_rate
            stage2.summary["passed_formulas"] = [r.get("formula_name") for r in combined_passed]
            stage2.summary["failed_formulas"] = [r.get("formula_name") for r in combined_failed]
            stage2.summary["conditional_formulas"] = [r.get("formula_name") for r in combined_conditional]

            run_ctx.log(f"\n✅ Partial re-validation completed:")
            run_ctx.log(f"  Re-validated: {len(new_validation_results)} formulas")
            run_ctx.log(f"  Reused PASS: {len(passed_validation_results)} formulas")
            run_ctx.log(f"  Combined total: {n_combined_total} formulas")
            run_ctx.log(f"  Combined PASS: {n_combined_passed} ({combined_pass_rate:.1%})")
            run_ctx.log(f"  Combined FAIL: {n_combined_failed}")

        # Update stage1 for the next iteration.
        stage1 = Stage1Result(
            hypothesis=stage1.hypothesis,
            observation_plan=stage1.observation_plan,
            formula_bundle=refined_bundle,
            ohlcv_df=ohlcv_df,
            formula_df=formula_df,
            price_with_formulas=price_with_formulas,
            failed_formulas=failed_formulas,
        )

        run_ctx.log(
            f"\nRefinement applied: refined={refinement.n_refined}/{refinement.n_failed}; "
            "proceeding to next iteration...\n"
        )

    # If inner-loop refinement completes but some observation has zero surviving formulas,
    # do not proceed to Stage3/4. In outer-loop mode, this will trigger hypothesis regeneration.
    missing_obs = _get_missing_observations(
        observation_plan=stage1.observation_plan,
        passed_formulas=stage2.passed_formulas,
    )
    stage2.summary["obs_coverage"] = {
        "n_observations_in_plan": len((stage1.observation_plan or {}).get("observations", []) or []),
        "missing_observation_ids": missing_obs,
        "coverage_ok": len(missing_obs) == 0,
    }

    if missing_obs:
        run_ctx.log("\n❌ Stage2 COVERAGE FAIL: some observations have zero PASS formulas after refinement.")
        run_ctx.log(f"   Missing obs: {missing_obs}")
        run_ctx.log("   Skipping Stage3/Stage4 to force hypothesis regeneration.\n")

        stage3 = Stage3Result(
            hypothesis_id=stage1.hypothesis.get("id", ""),
            result={
                "overall_verdict": "FAIL",
                "pass_rate": 0.0,
                "n_passed_combinations": 0,
                "n_total_combinations": 0,
                "reason": "Missing observation coverage (no PASS formula for at least one observation)",
                "missing_observation_ids": missing_obs,
                "skipped": True,
            },
            report_md=(
                "# Stage3 Report\n\n"
                "**SKIPPED**: Stage2 did not produce at least one PASS formula for every observation.\n\n"
                f"- Missing observation_ids: {missing_obs}\n"
            ),
            ticker_results={},
            aggregated_result={"n_tickers": 0, "skipped": True},
            passed_combinations=[],
            combination_stats={},
        )

        # Save artifacts (without Stage4). Use iteration-scoped saves when outer loop is enabled.
        if outer_iter is not None:
            run_ctx.save_json_with_iter("specs/hypothesis.json", outer_iter, stage1.hypothesis)
            run_ctx.save_json_with_iter("specs/observation_plan.json", outer_iter, stage1.observation_plan)
            run_ctx.save_json_with_iter("specs/formula_bundle.json", outer_iter, stage1.formula_bundle)
            run_ctx.save_parquet_with_iter("data/price_with_formulas.parquet", outer_iter, stage1.price_with_formulas)
            run_ctx.save_json_with_iter("specs/stage2_summary.json", outer_iter, stage2.summary)
            run_ctx.save_json_with_iter("specs/stage3_result.json", outer_iter, stage3.result)
            run_ctx.save_json_with_iter("specs/stage3_ticker_details.json", outer_iter, stage3.ticker_results)
            run_ctx.save_text_with_iter("reports/stage2.md", outer_iter, stage2.report_md)
            run_ctx.save_text_with_iter("reports/stage3.md", outer_iter, stage3.report_md)
            if refinement_history:
                run_ctx.save_json_with_iter(
                    "specs/refinement_history.json",
                    outer_iter,
                    {
                        "iterations": refinement_history,
                        "final_iteration": len(refinement_history),
                        "converged": stage2.summary.get("failed", 0) == 0,
                    },
                )
        else:
            run_ctx.save_json("specs/hypothesis.json", stage1.hypothesis)
            run_ctx.save_json("specs/observation_plan.json", stage1.observation_plan)
            run_ctx.save_json("specs/formula_bundle.json", stage1.formula_bundle)
            run_ctx.save_parquet("data/price_with_formulas.parquet", stage1.price_with_formulas)
            run_ctx.save_json("specs/stage2_summary.json", stage2.summary)
            run_ctx.save_json("specs/stage3_result.json", stage3.result)
            run_ctx.save_json("specs/stage3_ticker_details.json", stage3.ticker_results)
            run_ctx.save_text("reports/stage2.md", stage2.report_md)
            run_ctx.save_text("reports/stage3.md", stage3.report_md)
            if refinement_history:
                run_ctx.save_json(
                    "specs/refinement_history.json",
                    {
                        "iterations": refinement_history,
                        "final_iteration": len(refinement_history),
                        "converged": stage2.summary.get("failed", 0) == 0,
                    },
                )

        run_ctx.save_json(
            "run_config.json",
            {
                "run_id": run_ctx.run_id,
                "concept": concept,
                "timestamp": datetime.now().isoformat(),
                "config": cfg.dict(),
                "pipeline_control": {
                    "enable_stage2": cfg.pipeline_control.enable_stage2,
                    "enable_stage3": cfg.pipeline_control.enable_stage3,
                },
                "data_split": {
                    "train_start": train_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "test_start": test_start,
                    "test_end": test_end,
                },
                "stage4_skipped": True,
                "stage3_verdict": "FAIL",
                "outer_loop_used": outer_iter is not None,
            },
        )

        return {
            "run_id": run_ctx.run_id,
            "data_split": {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            },
            "refinement": {
                "iterations": refinement_history,
                "final_iteration": len(refinement_history),
                "converged": stage2.summary.get("failed", 0) == 0,
            },
            "stage1": {
                "hypothesis": stage1.hypothesis,
                "observation_plan": stage1.observation_plan,
                "formula_bundle": stage1.formula_bundle,
            },
            "stage2": {
                "summary": stage2.summary,
                "aggregated_summary": stage2.aggregated_summary,
                "n_tickers": stage2.aggregated_summary.get("n_tickers", 0),
                "data_period": f"{is_start} ~ {is_end} (IS only)",
            },
            "stage3": {
                "result": stage3.result,
                "aggregated_result": stage3.aggregated_result,
                "n_tickers": stage3.aggregated_result.get("n_tickers", 0),
                "data_period": f"{is_start} ~ {is_end} (IS only)",
                "verdict": "FAIL",
                "reason": "Missing observation coverage - Stage3/4 skipped",
            },
            "stage4": None,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 3: combination validation (validate on IS; OOS is used for IC recording only)
    # ═══════════════════════════════════════════════════════════════════════════
    # Recompute IS slice based on the final stage1 outputs after the inner refinement loop.
    is_ohlcv_df = _split_data_by_period(stage1.ohlcv_df, is_start, is_end)
    is_formula_df = _split_data_by_period(stage1.formula_df, is_start, is_end)

    # Build OOS slices for IC recording (not used for pass/fail decisions).
    oos_ohlcv_df = _split_data_by_period(stage1.ohlcv_df, oos_start, oos_end)
    oos_formula_df = _split_data_by_period(stage1.formula_df, oos_start, oos_end)

    run_ctx.log(f"\nStage3: Validation using IS data ({is_start} ~ {is_end})")
    run_ctx.log(f"  IS samples: {len(is_ohlcv_df):,} rows")
    run_ctx.log(f"  OOS samples (for IC recording only): {len(oos_ohlcv_df):,} rows")

    # Stage3 skip path.
    if not cfg.pipeline_control.enable_stage3:
        run_ctx.log("\n⚡ Stage3 SKIPPED (pipeline_control.enable_stage3=False)")

        # If both Stage2 and Stage3 are disabled, pick the first formula from each observation
        # and combine them into a single strategy.
        if not cfg.pipeline_control.enable_stage2:
            run_ctx.log(
                "   Both Stage2 and Stage3 are disabled; using FIRST formula per observation as a single strategy.\n"
            )
            # Group formulas by observation_id and pick the first from each
            obs_first_formulas: dict[str, dict] = {}
            for f in stage2.passed_formulas:
                if not isinstance(f, dict):
                    continue
                obs_id = str(f.get("observation_id") or f.get("obs_id") or "").strip()
                if obs_id and obs_id not in obs_first_formulas:
                    obs_first_formulas[obs_id] = f

            if obs_first_formulas:
                # Combine first formula from each obs into one combination
                combinations = [list(obs_first_formulas.values())]
                run_ctx.log(f"   Combined first formula from {len(obs_first_formulas)} observation(s).")
            else:
                # Fallback: if no obs_id found, use the first formula overall
                first_formula = stage2.passed_formulas[0] if stage2.passed_formulas else None
                combinations = [[first_formula]] if first_formula else []
        else:
            # Stage2 enabled, Stage3 disabled: pick one random combination for quick performance check.
            import random
            run_ctx.log(
                "   Stage3 validation is disabled; picking ONE random combination for performance evaluation.\n"
            )
            all_combinations = _build_stage3_combinations_without_validation(
                observation_plan=stage1.observation_plan,
                passed_formulas=stage2.passed_formulas,
                max_combinations=None,
            )
            if all_combinations:
                combinations = [random.choice(all_combinations)]
                run_ctx.log(f"   Selected 1 random combination out of {len(all_combinations)} possible.")
            else:
                combinations = []

        stage3 = Stage3Result(
            hypothesis_id=stage1.hypothesis.get("id", ""),
            result={
                "overall_verdict": "PASS",
                "pass_rate": 1.0,
                "n_passed_combinations": len(combinations),
                "n_total_combinations": len(combinations),
                "skipped": True,
            },
            report_md=(
                "# Stage3 Report\n\n"
                "**SKIPPED**: Stage3 validation was disabled via `pipeline_control.enable_stage3=False`.\n\n"
                + (
                    "Both Stage2 and Stage3 are disabled; using FIRST formula per observation as a single strategy.\n\n"
                    if not cfg.pipeline_control.enable_stage2
                    else "Stage3 disabled; picked ONE random combination for quick performance evaluation.\n\n"
                )
                + f"- Evaluated combinations: {len(combinations)}\n"
            ),
            ticker_results={},
            aggregated_result={"skipped": True, "n_tickers": 0},
            passed_combinations=combinations,
            combination_stats={},
        )
    else:
        stage3 = run_stage3(
            passed_formulas=stage2.passed_formulas,
            ohlcv_df=is_ohlcv_df,
            formula_df=is_formula_df,
            hypothesis=stage1.hypothesis,
            cfg=cfg,
            run_ctx=run_ctx,
            oos_ohlcv_df=oos_ohlcv_df,
            oos_formula_df=oos_formula_df,
        )

    # Inspect Stage3 results: proceed to Stage4 only if there is at least one passed combination.
    stage3_verdict = stage3.result.get("overall_verdict", "UNKNOWN")
    stage3_pass_rate = stage3.result.get("pass_rate", 0.0)
    n_passed_combinations = stage3.result.get("n_passed_combinations", 0)

    run_ctx.log(f"\nStage3 Result:")
    run_ctx.log(f"  Overall Verdict: {stage3_verdict}")
    run_ctx.log(f"  Pass Rate: {stage3_pass_rate:.1%}")
    run_ctx.log(f"  Qualified Combinations: {n_passed_combinations}")

    # Stage3 decision: even if the ticker-level verdict is FAIL, proceed if there exists at least one passed combination
    # (e.g., passed the secondary filter in the 2-tier selection logic).
    should_skip_stage4 = (stage3_verdict == "FAIL" and n_passed_combinations == 0)

    if should_skip_stage4:
        run_ctx.log("\n❌ Stage3 FAIL: The hypothesis structure is not valid.")
        run_ctx.log("   Monotonic improvement as strictness increases was not confirmed; skipping Stage4.")
        run_ctx.log("   Revisit observation decomposition and/or formula implementation.\n")

        # Save artifacts (without Stage4). Use iteration-scoped saves when outer loop is enabled.
        if outer_iter is not None:
            run_ctx.save_json_with_iter("specs/hypothesis.json", outer_iter, stage1.hypothesis)
            run_ctx.save_json_with_iter("specs/observation_plan.json", outer_iter, stage1.observation_plan)
            run_ctx.save_json_with_iter("specs/formula_bundle.json", outer_iter, stage1.formula_bundle)
            run_ctx.save_parquet_with_iter("data/price_with_formulas.parquet", outer_iter, stage1.price_with_formulas)
            run_ctx.save_json_with_iter("specs/stage2_summary.json", outer_iter, stage2.summary)
            run_ctx.save_json_with_iter("specs/stage3_result.json", outer_iter, stage3.result)
            # Save Stage3 ticker-level details (e.g., mean_return / sharpe by strictness level).
            run_ctx.save_json_with_iter("specs/stage3_ticker_details.json", outer_iter, stage3.ticker_results)
            run_ctx.save_text_with_iter("reports/stage2.md", outer_iter, stage2.report_md)
            run_ctx.save_text_with_iter("reports/stage3.md", outer_iter, stage3.report_md)
        else:
            run_ctx.save_json("specs/hypothesis.json", stage1.hypothesis)
            run_ctx.save_json("specs/observation_plan.json", stage1.observation_plan)
            run_ctx.save_json("specs/formula_bundle.json", stage1.formula_bundle)
            run_ctx.save_parquet("data/price_with_formulas.parquet", stage1.price_with_formulas)
            run_ctx.save_json("specs/stage2_summary.json", stage2.summary)
            run_ctx.save_json("specs/stage3_result.json", stage3.result)
            # Save Stage3 ticker-level details (e.g., mean_return / sharpe by strictness level).
            run_ctx.save_json("specs/stage3_ticker_details.json", stage3.ticker_results)
            run_ctx.save_text("reports/stage2.md", stage2.report_md)
            run_ctx.save_text("reports/stage3.md", stage3.report_md)

        # Save refinement history.
        if refinement_history:
            refinement_data = {
                "iterations": refinement_history,
                "final_iteration": len(refinement_history),
                "converged": stage2.summary.get("failed", 0) == 0,
            }
            if outer_iter is not None:
                run_ctx.save_json_with_iter("specs/refinement_history.json", outer_iter, refinement_data)
            else:
                run_ctx.save_json("specs/refinement_history.json", refinement_data)

        # Save run config snapshot.
        run_ctx.save_json("run_config.json", {
            "run_id": run_ctx.run_id,
            "concept": concept,
            "timestamp": datetime.now().isoformat(),
            "config": cfg.dict(),
            "pipeline_control": {
                "enable_stage2": cfg.pipeline_control.enable_stage2,
                "enable_stage3": cfg.pipeline_control.enable_stage3,
            },
            "data_split": {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            },
            "stage4_skipped": True,
            "stage3_verdict": "FAIL",
            "outer_loop_used": outer_iter is not None,
        })

        return {
            "run_id": run_ctx.run_id,
            "data_split": {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            },
            "refinement": {
                "iterations": refinement_history,
                "final_iteration": len(refinement_history),
                "converged": stage2.summary.get("failed", 0) == 0,
            },
            "stage1": {
                "hypothesis": stage1.hypothesis,
                "observation_plan": stage1.observation_plan,
                "formula_bundle": stage1.formula_bundle,
            },
            "stage2": {
                "summary": stage2.summary,
                "aggregated_summary": stage2.aggregated_summary,
                "n_tickers": stage2.aggregated_summary.get("n_tickers", 0),
                "data_period": f"{is_start} ~ {is_end} (IS only)",
            },
            "stage3": {
                "result": stage3.result,
                "aggregated_result": stage3.aggregated_result,
                "n_tickers": stage3.aggregated_result.get("n_tickers", 0),
                "data_period": f"{is_start} ~ {is_end} (IS only)",
                "verdict": "FAIL",
                "reason": "Monotonicity verification failed - Stage4 skipped",
            },
            "stage4": None,  # Stage4 skipped
        }

    run_ctx.log("\n✅ Stage3 PASS: The hypothesis structure is valid. Proceeding to Stage4.")
    run_ctx.log(f"  Stage3 passed combinations: {len(stage3.passed_combinations)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 4: Optuna-based backtest (optimize on IS → final evaluation on OOS)
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage4 receives the full panel and applies the configured IS/OOS split internally.
    # Only Stage3-passed combinations are evaluated.

    run_ctx.log(f"\nStage4: Train ({train_start} ~ {train_end}) → Val/Optuna ({val_start} ~ {val_end}) → Test ({test_start} ~ {test_end})")

    stage4 = run_stage4(
        hypothesis_id=stage3.hypothesis_id,
        passed_combinations=stage3.passed_combinations,  # Only Stage3-passed combinations
        passed_formulas=stage2.passed_formulas,  # Stage2 PASS pool (may be used depending on Stage4 implementation)
        ohlcv_df=stage1.ohlcv_df,
        formula_df=stage1.formula_df,
        hypothesis=stage1.hypothesis,
        cfg=cfg,
        run_ctx=run_ctx,
        verbose=True,
        outer_iter=outer_iter,
        combination_stats=stage3.combination_stats,  # Provide Stage3 stats (e.g., S2 improvement) for ranking/selection
    )

    # Save artifacts.
    # If outer_iter is provided, save per-iteration; otherwise, save as single-run outputs.
    run_ctx.log(f"\n[DEBUG] Starting artifact saves (outer_iter={outer_iter})...")
    if outer_iter is not None:
        run_ctx.log(f"[DEBUG] Saving with _with_iter methods (outer_iter={outer_iter})")
        run_ctx.save_json_with_iter("specs/hypothesis.json", outer_iter, stage1.hypothesis)
        run_ctx.log("[DEBUG] Saved hypothesis.json")
        run_ctx.save_json_with_iter("specs/observation_plan.json", outer_iter, stage1.observation_plan)
        run_ctx.log("[DEBUG] Saved observation_plan.json")
        run_ctx.save_json_with_iter("specs/formula_bundle.json", outer_iter, stage1.formula_bundle)
        run_ctx.log("[DEBUG] Saved formula_bundle.json")
        run_ctx.save_parquet_with_iter("data/price_with_formulas.parquet", outer_iter, stage1.price_with_formulas)
        run_ctx.log("[DEBUG] Saved price_with_formulas.parquet")
        run_ctx.save_json_with_iter("specs/stage2_summary.json", outer_iter, stage2.summary)
        run_ctx.log("[DEBUG] Saved stage2_summary.json")
        run_ctx.save_json_with_iter("specs/stage3_result.json", outer_iter, stage3.result)
        run_ctx.log("[DEBUG] Saved stage3_result.json")
        # Save Stage3 ticker-level details (e.g., mean_return / sharpe by strictness level).
        run_ctx.save_json_with_iter("specs/stage3_ticker_details.json", outer_iter, stage3.ticker_results)
        run_ctx.log("[DEBUG] Saved stage3_ticker_details.json")
        run_ctx.save_text_with_iter("reports/stage2.md", outer_iter, stage2.report_md)
        run_ctx.log("[DEBUG] Saved stage2.md")
        run_ctx.save_text_with_iter("reports/stage3.md", outer_iter, stage3.report_md)
        run_ctx.log("[DEBUG] Saved stage3.md")
    else:
        run_ctx.save_json("specs/hypothesis.json", stage1.hypothesis)
        run_ctx.save_json("specs/observation_plan.json", stage1.observation_plan)
        run_ctx.save_json("specs/formula_bundle.json", stage1.formula_bundle)
        run_ctx.save_parquet("data/price_with_formulas.parquet", stage1.price_with_formulas)
        run_ctx.save_json("specs/stage2_summary.json", stage2.summary)
        run_ctx.save_json("specs/stage3_result.json", stage3.result)
        # Save Stage3 ticker-level details (e.g., mean_return / sharpe by strictness level).
        run_ctx.save_json("specs/stage3_ticker_details.json", stage3.ticker_results)
        run_ctx.save_text("reports/stage2.md", stage2.report_md)
        run_ctx.save_text("reports/stage3.md", stage3.report_md)
    # Stage4 saves its own artifacts (e.g., stage4_summary.json, stage4_is_daily.parquet, stage4_oos_daily.parquet, stage4.md).

    # Save inner-loop refinement history.
    if refinement_history:
        refinement_data = {
            "iterations": refinement_history,
            "final_iteration": len(refinement_history),
            "converged": stage2.summary.get("failed", 0) == 0,
        }
        if outer_iter is not None:
            run_ctx.save_json_with_iter("specs/refinement_history.json", outer_iter, refinement_data)
        else:
            run_ctx.save_json("specs/refinement_history.json", refinement_data)

    # Save run config snapshot.
    run_ctx.save_json("run_config.json", {
        "run_id": run_ctx.run_id,
        "concept": concept,
        "timestamp": datetime.now().isoformat(),
        "config": cfg.dict(),
        "pipeline_control": {
            "enable_stage2": cfg.pipeline_control.enable_stage2,
            "enable_stage3": cfg.pipeline_control.enable_stage3,
        },
        "data_split": {
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        },
        "stage4_skipped": False,
        "stage3_verdict": "PASS",
        "outer_loop_used": False,
    })

    # Save LLM usage summary.
    run_ctx.save_llm_usage()

    return {
        "run_id": run_ctx.run_id,
        "data_split": {
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        },
        "refinement": {
            "inner_loop": {
                "iterations": refinement_history,
                "final_iteration": len(refinement_history),
                "converged": stage2.summary.get("failed", 0) == 0,
            } if refinement_history else None,
        },
        "stage1": {
            "hypothesis": stage1.hypothesis,
            "observation_plan": stage1.observation_plan,
            "formula_bundle": stage1.formula_bundle,
        },
        "stage2": {
            "summary": stage2.summary,
            "aggregated_summary": stage2.aggregated_summary,
            "n_tickers": stage2.aggregated_summary.get("n_tickers", 0),
            "data_period": f"{is_start} ~ {is_end} (IS only)",
        },
        "stage3": {
            "result": stage3.result,
            "aggregated_result": stage3.aggregated_result,
            "n_tickers": stage3.aggregated_result.get("n_tickers", 0),
            "data_period": f"{is_start} ~ {is_end} (IS only)",
        },
        "stage4": {
            "summary": stage4.summary,
            "result": stage4.result,
            "train_period": f"{train_start} ~ {train_end}",
            "val_period": f"{val_start} ~ {val_end}",
            "test_period": f"{test_start} ~ {test_end}",
        },
    }


def run_outer_loop(
    *,
    concept: str,
    cfg: RDConfig | None = None,
    run_ctx: Optional[RunContext] = None,
    price_df: pl.DataFrame | None = None,
    max_refinement_iterations: int | None = None,
    enable_refinement_loop: bool | None = None,
    max_outer_iterations: int | None = None,
    enable_outer_loop: bool | None = None,
) -> dict[str, Any]:
    """
    Outer loop: hypothesis-level refinement driven by Stage4 outcomes.

    Each iteration:
    1) Run `run_pipeline()` (Stage1 → Stage2 → Stage3 → Stage4)
    2) Analyze results and append feedback to `hypothesis_memory`
    3) Repeat up to `max_outer_iterations`

    Args:
        concept: Finance hypothesis / concept to validate
        cfg: RDConfig
        run_ctx: RunContext
        price_df: Price data
        max_refinement_iterations: Max inner-loop iterations
        enable_refinement_loop: Enable inner-loop refinement
        max_outer_iterations: Max outer-loop iterations
        enable_outer_loop: Enable outer-loop hypothesis refinement

    Returns:
        Final pipeline result (from the last iteration)
    """
    cfg = cfg or load_rd_config()
    run_ctx = run_ctx or RunContext.create()

    # Resolve outer-loop settings.
    if max_outer_iterations is None:
        max_outer_iterations = cfg.refinement.max_outer_iterations
    if enable_outer_loop is None:
        enable_outer_loop = cfg.refinement.enable_outer_loop

    # If the outer loop is disabled, run the pipeline once.
    if not enable_outer_loop:
        return run_pipeline(
            concept=concept,
            cfg=cfg,
            run_ctx=run_ctx,
            price_df=price_df,
            max_refinement_iterations=max_refinement_iterations,
            enable_refinement_loop=enable_refinement_loop,
            hypothesis_memory=None,
        )

    run_ctx.log(f"\n{'='*80}")
    run_ctx.log("OUTER LOOP: Hypothesis-level Refinement")
    run_ctx.log(f"{'='*80}")
    run_ctx.log(f"Outer iterations: {max_outer_iterations}")
    run_ctx.log(f"{'='*80}\n")

    # Hypothesis memory: accumulate Stage4/Stage3 feedback across outer iterations.
    hypothesis_memory = []
    outer_history = []
    final_result = None

    for outer_iter in range(max_outer_iterations):
        run_ctx.log(f"\n{'='*80}")
        run_ctx.log(f"OUTER ITERATION {outer_iter + 1}/{max_outer_iterations}")
        run_ctx.log(f"{'='*80}\n")

        # Run the inner pipeline (Stage1 → Stage2 → Stage3 → Stage4).
        result = run_pipeline(
            concept=concept,
            cfg=cfg,
            run_ctx=run_ctx,
            price_df=price_df,
            max_refinement_iterations=max_refinement_iterations,
            enable_refinement_loop=enable_refinement_loop,
            hypothesis_memory=hypothesis_memory if hypothesis_memory else None,
            outer_iter=outer_iter + 1,  # 1-indexed
        )

        final_result = result

        # Extract Stage1 outputs (used by multiple feedback generators).
        stage1_result = result.get("stage1")
        hypothesis = stage1_result.get("hypothesis") if stage1_result else None
        observation_plan = stage1_result.get("observation_plan") if stage1_result else None
        formula_bundle = stage1_result.get("formula_bundle") if stage1_result else None

        # Extract Stage2/Stage3 outputs.
        stage2_result = result.get("stage2", {})
        stage2_summary = stage2_result.get("summary", {}) if stage2_result else {}
        stage3_result = result.get("stage3", {})

        # Extract Stage4 outputs (may be None if Stage4 was skipped).
        stage4_result = result.get("stage4")

        # Handle Stage3 FAIL (Stage4 skipped).
        if stage4_result is None:
            run_ctx.log("\n⚠️  Stage4 was skipped (Stage3 FAIL).")

            # Stop if this is the last outer iteration.
            if outer_iter == max_outer_iterations - 1:
                run_ctx.log(f"\n❌ Max iterations reached with Stage3 FAIL. Stopping.")
                break

            # Generate feedback for hypothesis regeneration based on the Stage3 failure.
            run_ctx.log("🔄 Analyzing Stage3 failure and generating feedback for hypothesis regeneration...\n")

            stage3_data = stage3_result.get("result", {}) if stage3_result else {}
            memory_entry = create_stage3_fail_memory_entry(
                stage3_result=stage3_data,
                stage2_summary=stage2_summary,
                hypothesis=hypothesis,
                observation_plan=observation_plan,
                formula_bundle=formula_bundle,
                model=cfg.llm.model_name,
                run_ctx=run_ctx,
            )
            if isinstance(memory_entry, dict):
                memory_entry["outer_iter"] = outer_iter + 1
                memory_entry["run_id"] = result.get("run_id")
            hypothesis_memory.append(memory_entry)

            # Record iteration summary.
            outer_history.append({
                "iteration": outer_iter + 1,
                "avg_is_sharpe": 0.0,
                "avg_oos_sharpe": 0.0,
                "n_combinations": memory_entry.get("n_combinations", 0),
                "stage3_fail": True,
                "failure_reason": memory_entry.get("failure_reason", "Stage3 FAIL"),
            })

            run_ctx.log("Added Stage3 FAIL feedback to hypothesis_memory:")
            run_ctx.log(f"  - Iteration: {outer_iter + 1}")
            run_ctx.log(f"  - Failure: {memory_entry.get('failure_reason', 'Unknown')}")
            run_ctx.log("\n🔄 Restarting pipeline with NEW HYPOTHESIS...\n")
            continue  # Continue to the next outer iteration.

        stage4_summary = stage4_result.get("summary", {})

        # Analyze performance.
        combination_results = stage4_summary.get("all_combinations", [])
        if not combination_results:
            run_ctx.log("\n⚠️  No combination results in Stage4.")

            # Stop if this is the last outer iteration.
            if outer_iter == max_outer_iterations - 1:
                run_ctx.log(f"\n❌ Max iterations reached with no results. Stopping.")
                break

            # Treat as a Stage3-failure-like case for regeneration feedback.
            stage3_data = stage3_result.get("result", {}) if stage3_result else {}
            memory_entry = create_stage3_fail_memory_entry(
                stage3_result=stage3_data,
                stage2_summary=stage2_summary,
                hypothesis=hypothesis,
                observation_plan=observation_plan,
                formula_bundle=formula_bundle,
                model=cfg.llm.model_name,
                run_ctx=run_ctx,
            )
            if isinstance(memory_entry, dict):
                memory_entry["outer_iter"] = outer_iter + 1
                memory_entry["run_id"] = result.get("run_id")
            hypothesis_memory.append(memory_entry)

            outer_history.append({
                "iteration": outer_iter + 1,
                "avg_is_sharpe": 0.0,
                "avg_oos_sharpe": 0.0,
                "n_combinations": 0,
                "stage3_fail": True,
            })

            run_ctx.log("\n🔄 Restarting pipeline with NEW HYPOTHESIS...\n")
            continue

        # Compute average Sharpe (=information_ratio). Prefer the qlib-style structure, with backward compatibility.
        def _get_is_ir(comb: dict[str, Any]) -> tuple[float, bool]:
            v = comb.get("insample", {}).get("return", {}).get("information_ratio")
            if v is None:
                v = comb.get("data_split", {}).get("insample", {}).get("strategy", {}).get("information_ratio")
            try:
                f = float(v or 0.0)
            except (TypeError, ValueError):
                return 0.0, False
            if not math.isfinite(f):
                return 0.0, True
            return f, False

        def _get_oos_ir(comb: dict[str, Any]) -> tuple[float, bool]:
            v = comb.get("outsample", {}).get("return", {}).get("information_ratio")
            if v is None:
                v = comb.get("data_split", {}).get("outsample", {}).get("strategy", {}).get("information_ratio")
            try:
                f = float(v or 0.0)
            except (TypeError, ValueError):
                return 0.0, False
            if not math.isfinite(f):
                return 0.0, True
            return f, False

        is_irs: list[float] = []
        oos_irs: list[float] = []
        n_is_nonfinite = 0
        n_oos_nonfinite = 0
        for c in combination_results:
            v, bad = _get_is_ir(c)
            is_irs.append(v)
            n_is_nonfinite += int(bad)
            v, bad = _get_oos_ir(c)
            oos_irs.append(v)
            n_oos_nonfinite += int(bad)

        avg_is_sharpe = sum(is_irs) / len(is_irs)
        avg_oos_sharpe = sum(oos_irs) / len(oos_irs)

        run_ctx.log(f"\nOuter Iteration {outer_iter + 1} Result:")
        run_ctx.log(f"  Average IS IR:  {avg_is_sharpe:.3f}")
        run_ctx.log(f"  Average OOS IR: {avg_oos_sharpe:.3f}")
        if n_is_nonfinite or n_oos_nonfinite:
            run_ctx.log(
                f"  Note: replaced non-finite IR values with 0.0 "
                f"(IS {n_is_nonfinite}/{len(is_irs)}, OOS {n_oos_nonfinite}/{len(oos_irs)}; often happens when return std=0 / no trades)."
            )

        # Record iteration summary (including refinement info when available).
        refinement_info = result.get("refinement", {})
        inner_loop_info = refinement_info.get("inner_loop") if refinement_info else None

        outer_history.append({
            "iteration": outer_iter + 1,
            "avg_is_sharpe": avg_is_sharpe,
            "avg_oos_sharpe": avg_oos_sharpe,
            "n_combinations": len(combination_results),
            "refinement": {
                "inner_loop": inner_loop_info,
            } if inner_loop_info else None,
        })

        # If this is the last iteration, stop without generating next-iteration feedback.
        if outer_iter == max_outer_iterations - 1:
            run_ctx.log(f"\n✅ All {max_outer_iterations} outer iterations completed.")
            break

        # Generate feedback and append to hypothesis_memory.
        run_ctx.log(f"\n🔄 Generating hypothesis-level feedback for next iteration...\n")

        memory_entry = create_hypothesis_memory_entry(
            stage4_summary,
            hypothesis=hypothesis,
            observation_plan=observation_plan,
            formula_bundle=formula_bundle,
            stage2_summary=stage2_summary,
            stage3_result=stage3_result.get("result", {}) if stage3_result else {},
        )
        if isinstance(memory_entry, dict):
            memory_entry["outer_iter"] = outer_iter + 1
            memory_entry["run_id"] = result.get("run_id")
        hypothesis_memory.append(memory_entry)

        run_ctx.log("Added feedback to hypothesis_memory:")
        run_ctx.log(f"  - Iteration: {outer_iter + 1}")
        avg_is_for_log = float(memory_entry.get("avg_is_sharpe", memory_entry.get("avg_is_ir", 0.0)) or 0.0)
        avg_oos_for_log = float(memory_entry.get("avg_oos_sharpe", memory_entry.get("avg_oos_ir", 0.0)) or 0.0)
        run_ctx.log(f"  - Avg IS IR: {avg_is_for_log:.3f}")
        run_ctx.log(f"  - Avg OOS IR: {avg_oos_for_log:.3f}")
        n_successful = memory_entry.get("n_successful", 0)
        run_ctx.log(f"  - Successful combinations (OOS excess IR > 0): {n_successful}/{memory_entry.get('n_combinations', 0)}")
        if n_successful > 0:
            run_ctx.log(f"  - Stored {n_successful} successful pattern(s) with full details (hypothesis/obs/formulas/validation)")

        # Print concrete suggestions for the next hypothesis (if provided).
        next_suggestions = memory_entry.get("next_hypothesis_suggestions", "")
        if next_suggestions:
            run_ctx.log(next_suggestions)

        run_ctx.log("\n🔄 Restarting pipeline with updated hypothesis_memory...\n")

    # Persist outer-loop history.
    if outer_history:
        run_ctx.save_json("specs/outer_loop_history.json", {
            "iterations": outer_history,
            "final_iteration": len(outer_history),
        })

        # Best-effort: update run_config.json with outer_loop_used=true.
        try:
            import json
            config_path = run_ctx.root_dir / "run_config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    run_config = json.load(f)

                # Add the outer_loop_used flag.
                run_config["outer_loop_used"] = True

                # Persist.
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(run_config, f, indent=2, ensure_ascii=False)

                run_ctx.log("Updated run_config.json with outer_loop_used flag.")
        except Exception as e:
            run_ctx.log(f"Warning: Failed to update run_config.json: {e}")

    # Attach outer-loop info to the final result.
    if final_result:
        final_result["outer_loop"] = {
            "enabled": True,
            "iterations": outer_history,
            "final_iteration": len(outer_history),
        }

    # Save LLM usage summary.
    run_ctx.save_llm_usage()

    return final_result


# Usage helper when executed directly.
if __name__ == "__main__":
    print("Usage: python run_pipeline.py [concept]")
    print("       python -m run.main (from project root)")
    print()
    print("Example:")
    print('  cd /home/user/fin/FinAgent_4090/workspace/kms/01_15_new')
    print('  python run_pipeline.py "Mean Reversion after Panic Selling"')
