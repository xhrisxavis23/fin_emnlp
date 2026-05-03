"""
================================================================================
STAGE 4 PARALLEL (Per-Combo Backtest + Combo Parallelism)
================================================================================

Intent:
- Phase 1: optimize each combination (Optuna or fixed thresholds)
- Phase 2: run IS/OOS backtest PER COMBINATION (not vectorized over all combos)
- Both phases can run in parallel across combinations (combo-level parallelism).

This is useful when the vectorized multi-combo simulation becomes a bottleneck and
you want to distribute combos across CPU cores.

Env vars:
  - STAGE4_COMBO_WORKERS: number of parallel workers across combinations (default: 1)
  - STAGE4_OPTUNA_N_JOBS: Optuna n_jobs per combination (default: 8; if combo_workers>1 and not set, defaults to 1)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl

from util.run_context import RunContext
from run.config import RDConfig, load_rd_config

from run.pipeline.stage4 import (
    Stage4Result,
    _prepare_panel,
    _split_in_out_sample,
    _compute_thresholds,
    _apply_signal,
    _compute_trade_metrics,
    _calc_excess_metrics,
    _calculate_benchmark_returns,
    _compute_metrics_qlib,
    _build_port_analysis_df,
    _create_objective,
)

# Reuse the proven per-signal backtest helpers from stage4_parallel (but run them per combo).
from run.pipeline.stage4_parallel import (  # noqa: E402
    _apply_stage4_env_overrides,
    _simulate_positions_vectorized,
    _calculate_returns_vectorized,
    _compute_metrics_from_returns_dict,
    _build_report_df_from_returns,
    _combinations_to_multiindex_df,
)

logger = logging.getLogger(__name__)

_G_IS_PANEL: pd.DataFrame | None = None
_G_OOS_PANEL: pd.DataFrame | None = None
_G_CFG: RDConfig | None = None
_G_USE_OPTUNA: bool = False
_G_OPTUNA_N_JOBS: int = 1
_G_IS_BENCH: pd.Series | None = None
_G_OOS_BENCH: pd.Series | None = None


def _set_globals(
    *,
    is_panel: pd.DataFrame,
    oos_panel: pd.DataFrame,
    cfg: RDConfig,
    use_optuna: bool,
    optuna_n_jobs: int,
    is_bench_series: pd.Series,
    oos_bench_series: pd.Series,
) -> None:
    global _G_IS_PANEL, _G_OOS_PANEL, _G_CFG, _G_USE_OPTUNA, _G_OPTUNA_N_JOBS, _G_IS_BENCH, _G_OOS_BENCH
    _G_IS_PANEL = is_panel
    _G_OOS_PANEL = oos_panel
    _G_CFG = cfg
    _G_USE_OPTUNA = use_optuna
    _G_OPTUNA_N_JOBS = optuna_n_jobs
    _G_IS_BENCH = is_bench_series
    _G_OOS_BENCH = oos_bench_series


def _run_one_combo(task: tuple[int, list[dict[str, Any]], list[str]]) -> dict[str, Any]:
    """
    Worker function (must be top-level for multiprocessing).
    Runs:
      1) per-combo threshold optimization (Optuna or fixed)
      2) per-combo IS/OOS backtest
    """
    combo_idx, combination, formula_names = task

    if _G_IS_PANEL is None or _G_OOS_PANEL is None or _G_CFG is None or _G_IS_BENCH is None or _G_OOS_BENCH is None:
        raise RuntimeError("Stage4 per-combo globals are not initialized.")

    cfg = _G_CFG
    is_panel = _G_IS_PANEL
    oos_panel = _G_OOS_PANEL

    if _G_USE_OPTUNA:
        import optuna

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42 + combo_idx),
        )
        objective = _create_objective(
            is_panel=is_panel,
            passed_formulas=combination,
            cfg=cfg,
            verbose=False,
        )
        study.optimize(
            objective,
            n_trials=cfg.stage4.n_trials,
            n_jobs=_G_OPTUNA_N_JOBS,
            show_progress_bar=False,
        )

        best_params = study.best_params
        optimal_thresholds: Dict[str, float] = {}
        for fname in formula_names:
            key = f"threshold_{fname}"
            if key in best_params:
                optimal_thresholds[fname] = best_params[key]

        is_information_ratio_optuna = float(study.best_value)
    else:
        fixed_quantiles = list(getattr(cfg.stage4, "fixed_quantiles", [0.8]) or [0.8])
        default_threshold = fixed_quantiles[0] if fixed_quantiles else 0.8
        optimal_thresholds = {fname: float(default_threshold) for fname in formula_names}
        is_information_ratio_optuna = 0.0

    final_thresholds = _compute_thresholds(
        train_panel=is_panel,
        passed_formulas=combination,
        threshold_dict=optimal_thresholds,
    )

    is_signal = _apply_signal(panel=is_panel, thresholds=final_thresholds, passed_formulas=combination)
    oos_signal = _apply_signal(panel=oos_panel, thresholds=final_thresholds, passed_formulas=combination)

    base_cols = ["timestamp", "ticker", "close", "high", "low"]
    is_bt_panel = is_panel.loc[:, [c for c in base_cols if c in is_panel.columns]].copy()
    oos_bt_panel = oos_panel.loc[:, [c for c in base_cols if c in oos_panel.columns]].copy()

    col_name = f"signal_c{combo_idx}"
    is_bt_panel[col_name] = is_signal.values
    oos_bt_panel[col_name] = oos_signal.values

    # IS backtest (single signal column)
    is_positions, is_trades = _simulate_positions_vectorized(
        panel=is_bt_panel,
        signal_columns=[col_name],
        cfg=cfg,
    )
    is_returns = _calculate_returns_vectorized(
        panel=is_bt_panel,
        positions=is_positions,
        cfg=cfg,
    )
    is_metrics = _compute_metrics_from_returns_dict(is_returns[col_name])
    is_report_df = _build_report_df_from_returns(is_returns[col_name], _G_IS_BENCH)
    is_trade_metrics = _compute_trade_metrics(is_trades[col_name])
    is_excess_without_cost = _calc_excess_metrics(is_report_df["return"] - is_report_df["bench"])
    is_excess_with_cost = _calc_excess_metrics(is_report_df["return"] - is_report_df["bench"] - is_report_df["cost"])

    # OOS backtest (single signal column)
    oos_positions, oos_trades = _simulate_positions_vectorized(
        panel=oos_bt_panel,
        signal_columns=[col_name],
        cfg=cfg,
    )
    oos_returns = _calculate_returns_vectorized(
        panel=oos_bt_panel,
        positions=oos_positions,
        cfg=cfg,
    )
    oos_metrics = _compute_metrics_from_returns_dict(oos_returns[col_name])
    oos_report_df = _build_report_df_from_returns(oos_returns[col_name], _G_OOS_BENCH)
    oos_trade_metrics = _compute_trade_metrics(oos_trades[col_name])
    oos_excess_without_cost = _calc_excess_metrics(oos_report_df["return"] - oos_report_df["bench"])
    oos_excess_with_cost = _calc_excess_metrics(oos_report_df["return"] - oos_report_df["bench"] - oos_report_df["cost"])

    return {
        "combo_idx": combo_idx,
        "combination": combination,
        "formula_names": formula_names,
        "optimal_thresholds": optimal_thresholds,
        "final_thresholds": final_thresholds,
        "backtest_mode": "per_combo_parallel",
        "is_information_ratio_optuna": is_information_ratio_optuna,
        "is_metrics": is_metrics,
        "is_excess_without_cost": is_excess_without_cost,
        "is_excess_with_cost": is_excess_with_cost,
        "is_trade_metrics": is_trade_metrics,
        "is_report_df": is_report_df,
        "is_trades": [t.__dict__ for t in is_trades[col_name]],
        "oos_metrics": oos_metrics,
        "oos_excess_without_cost": oos_excess_without_cost,
        "oos_excess_with_cost": oos_excess_with_cost,
        "oos_trade_metrics": oos_trade_metrics,
        "oos_report_df": oos_report_df,
        "oos_trades": [t.__dict__ for t in oos_trades[col_name]],
    }


def run_stage4_parallel_per_combo(
    *,
    hypothesis_id: str,
    passed_formulas: List[Dict[str, Any]],
    passed_combinations: List[List[Dict[str, Any]]],
    ohlcv_df: pl.DataFrame,
    formula_df: pl.DataFrame,
    hypothesis: Optional[Dict[str, Any]] = None,
    cfg: Optional[RDConfig] = None,
    run_ctx: Optional[RunContext] = None,
    verbose: bool = True,
    outer_iter: int | None = None,
    **_kwargs,
) -> Stage4Result:
    cfg = cfg or load_rd_config()
    _apply_stage4_env_overrides(cfg)

    use_optuna = bool(getattr(cfg.stage4, "enable_optuna", False))

    if hypothesis:
        hyp_list = hypothesis.get("hypotheses", [])
        hyp_obj = hyp_list[0] if isinstance(hyp_list, list) and hyp_list else hypothesis
        h = (hyp_obj or {}).get("horizon_days")
        if isinstance(h, int) and h > 0:
            cfg.stage4.horizon_days = h

    if cfg.stage4.trigger_kmax is None:
        cfg.stage4.trigger_kmax = cfg.stage4.horizon_days

    if verbose:
        print(f"[Stage4-PerCombo] Starting for hypothesis: {hypothesis_id}")
        print(f"[Stage4-PerCombo] In-Sample:     {cfg.data_split.in_sample_start} ~ {cfg.data_split.in_sample_end}")
        print(f"[Stage4-PerCombo] Out-of-Sample: {cfg.data_split.out_sample_start} ~ {cfg.data_split.out_sample_end}")
        print(f"[Stage4-PerCombo] Combinations: {len(passed_combinations)}, Horizon: {cfg.stage4.horizon_days}d")
        if use_optuna:
            print(f"[Stage4-PerCombo] Optuna trials: {cfg.stage4.n_trials}")

    if not passed_combinations:
        empty = pl.DataFrame({"timestamp": [], "gross_return": [], "net_return": []})
        return Stage4Result(
            hypothesis_id=hypothesis_id,
            config={},
            summary={"hypothesis_id": hypothesis_id, "error": "No passed_combinations; skip backtest."},
            report_md="# Stage 4: Backtest (Per-Combo)\n\nNo passed combinations. Skipped.",
            is_daily_panel=empty,
            oos_daily_panel=empty,
        )

    max_combinations_to_evaluate = cfg.stage4.max_combinations_to_evaluate
    combination_stats = _kwargs.get("combination_stats", {})

    if max_combinations_to_evaluate <= 0 or max_combinations_to_evaluate >= len(passed_combinations):
        combinations_to_evaluate = passed_combinations
    else:
        if combination_stats:
            combo_with_stats = []
            for combo in passed_combinations:
                combo_key = tuple(sorted(f["name"] for f in combo))
                stats = combination_stats.get(combo_key, {})
                s2_improvement = stats.get("s2_ratio_improvement", 0.0)
                combo_with_stats.append((combo, s2_improvement))
            combo_with_stats.sort(key=lambda x: x[1], reverse=True)
            combinations_to_evaluate = [combo for combo, _ in combo_with_stats[:max_combinations_to_evaluate]]
        else:
            combinations_to_evaluate = passed_combinations[:max_combinations_to_evaluate]

    if verbose:
        print(f"[Stage4-PerCombo] Evaluating {len(combinations_to_evaluate)} / {len(passed_combinations)} combinations")

    # Build unified panel with all needed formulas.
    all_formula_names = sorted({str(f["name"]) for combo in combinations_to_evaluate for f in combo if isinstance(f, dict) and f.get("name")})
    panel = _prepare_panel(
        ohlcv_df=ohlcv_df,
        formula_df=formula_df,
        formula_names=all_formula_names,
    )

    is_panel, oos_panel = _split_in_out_sample(
        panel,
        cfg.data_split.in_sample_start, cfg.data_split.in_sample_end,
        cfg.data_split.out_sample_start, cfg.data_split.out_sample_end,
    )

    # Fix universe
    is_tickers = set(is_panel["ticker"].unique().tolist())
    oos_tickers = set(oos_panel["ticker"].unique().tolist())
    common_tickers = sorted(is_tickers & oos_tickers)
    if not common_tickers:
        raise ValueError("No common tickers between IS and OOS")
    is_panel = is_panel[is_panel["ticker"].isin(common_tickers)].reset_index(drop=True)
    oos_panel = oos_panel[oos_panel["ticker"].isin(common_tickers)].reset_index(drop=True)

    # Benchmark
    is_benchmark_returns = _calculate_benchmark_returns(
        start_date=cfg.data_split.in_sample_start,
        end_date=cfg.data_split.in_sample_end,
        benchmark_code=cfg.qlib.benchmark,
    )
    oos_benchmark_returns = _calculate_benchmark_returns(
        start_date=cfg.data_split.out_sample_start,
        end_date=cfg.data_split.out_sample_end,
        benchmark_code=cfg.qlib.benchmark,
    )
    is_benchmark_metrics = _compute_metrics_qlib(is_benchmark_returns)
    oos_benchmark_metrics = _compute_metrics_qlib(oos_benchmark_returns)
    is_bench_series = is_benchmark_returns["gross"]
    oos_bench_series = oos_benchmark_returns["gross"]

    combo_workers = int(str(os.getenv("STAGE4_COMBO_WORKERS", "1") or "1").strip() or "1")
    combo_workers = max(1, combo_workers)
    optuna_n_jobs = int(str(os.getenv("STAGE4_OPTUNA_N_JOBS", "8") or "8").strip() or "8")
    optuna_n_jobs = max(1, optuna_n_jobs)
    if combo_workers > 1 and os.getenv("STAGE4_OPTUNA_N_JOBS") is None:
        optuna_n_jobs = 1

    # Pre-filter tasks
    combo_tasks: list[tuple[int, list[dict[str, Any]], list[str]]] = []
    for combo_idx, combination in enumerate(combinations_to_evaluate, 1):
        formula_names = [str(f.get("name") or "") for f in combination if isinstance(f, dict) and f.get("name")]
        formula_names = [n for n in formula_names if n and n in is_panel.columns]
        if not formula_names:
            continue
        combo_tasks.append((combo_idx, combination, formula_names))

    if not combo_tasks:
        raise ValueError("No valid combinations after preprocessing")

    if verbose:
        print(f"\n[Stage4-PerCombo] ═══ Phase 1+2: Per-Combo Optimize + Backtest ═══")
        print(f"[Stage4-PerCombo] workers={combo_workers}, optuna_n_jobs={optuna_n_jobs}")

    # Prefer fork-based multiprocessing to avoid pickling huge panels.
    executor_kind = "sequential"
    results: list[dict[str, Any]] = []

    if combo_workers > 1:
        try:
            ctx = mp.get_context("fork")
            executor_kind = "process(fork)"
        except Exception:
            ctx = None
            executor_kind = "thread"

        _set_globals(
            is_panel=is_panel,
            oos_panel=oos_panel,
            cfg=cfg,
            use_optuna=use_optuna,
            optuna_n_jobs=optuna_n_jobs,
            is_bench_series=is_bench_series,
            oos_bench_series=oos_bench_series,
        )

        if ctx is not None:
            with ProcessPoolExecutor(max_workers=combo_workers, mp_context=ctx) as ex:
                futs = {ex.submit(_run_one_combo, t): t[0] for t in combo_tasks}
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    if verbose:
                        print(f"[Stage4-PerCombo]   (done) Combo {r['combo_idx']}: IS IR={r['is_metrics']['information_ratio']:.3f}, OOS IR={r['oos_metrics']['information_ratio']:.3f}")
        else:
            # Fallback: threads (may not speed up much due to GIL)
            with ThreadPoolExecutor(max_workers=combo_workers) as ex:
                futs = {ex.submit(_run_one_combo, t): t[0] for t in combo_tasks}
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    if verbose:
                        print(f"[Stage4-PerCombo]   (done) Combo {r['combo_idx']}: IS IR={r['is_metrics']['information_ratio']:.3f}, OOS IR={r['oos_metrics']['information_ratio']:.3f}")
    else:
        _set_globals(
            is_panel=is_panel,
            oos_panel=oos_panel,
            cfg=cfg,
            use_optuna=use_optuna,
            optuna_n_jobs=optuna_n_jobs,
            is_bench_series=is_bench_series,
            oos_bench_series=oos_bench_series,
        )
        for t in combo_tasks:
            r = _run_one_combo(t)
            results.append(r)
            if verbose:
                print(f"[Stage4-PerCombo]   (done) Combo {r['combo_idx']}: IS IR={r['is_metrics']['information_ratio']:.3f}, OOS IR={r['oos_metrics']['information_ratio']:.3f}")

    results.sort(key=lambda x: x["combo_idx"])

    if verbose:
        print(f"[Stage4-PerCombo] Executor: {executor_kind}")
        print(f"[Stage4-PerCombo] Evaluated {len(results)} combinations")

    all_combinations_summary: list[dict[str, Any]] = []
    for r in results:
        combo_summary: dict[str, Any] = {
            "combo_idx": r["combo_idx"],
            "formula_names": r["formula_names"],
            "optimal_thresholds": r["optimal_thresholds"],
            "insample": {
                "return": {
                    "mean": r["is_metrics"]["mean_return"],
                    "std": r["is_metrics"]["ann_vol"] / np.sqrt(252) if r["is_metrics"]["ann_vol"] else 0,
                    "annualized_return": r["is_metrics"]["ann_return"],
                    "information_ratio": r["is_metrics"]["information_ratio"],
                    "max_drawdown": r["is_metrics"]["max_drawdown"],
                    "net_return": r["is_metrics"]["net_return"],
                    "avg_holdings": r["is_metrics"]["avg_holdings"],
                    "avg_turnover": r["is_metrics"]["avg_turnover"],
                },
                "excess_return_without_cost": {
                    "mean": r["is_excess_without_cost"]["mean_return"],
                    "std": r["is_excess_without_cost"]["ann_vol"] / np.sqrt(252) if r["is_excess_without_cost"]["ann_vol"] else 0,
                    "annualized_return": r["is_excess_without_cost"]["ann_return"],
                    "information_ratio": r["is_excess_without_cost"]["information_ratio"],
                    "max_drawdown": r["is_excess_without_cost"]["max_drawdown"],
                    "net_return": r["is_excess_without_cost"]["net_return"],
                },
                "excess_return_with_cost": {
                    "mean": r["is_excess_with_cost"]["mean_return"],
                    "std": r["is_excess_with_cost"]["ann_vol"] / np.sqrt(252) if r["is_excess_with_cost"]["ann_vol"] else 0,
                    "annualized_return": r["is_excess_with_cost"]["ann_return"],
                    "information_ratio": r["is_excess_with_cost"]["information_ratio"],
                    "max_drawdown": r["is_excess_with_cost"]["max_drawdown"],
                    "net_return": r["is_excess_with_cost"]["net_return"],
                },
                "trade_metrics": r["is_trade_metrics"],
            },
            "outsample": {
                "return": {
                    "mean": r["oos_metrics"]["mean_return"],
                    "std": r["oos_metrics"]["ann_vol"] / np.sqrt(252) if r["oos_metrics"]["ann_vol"] else 0,
                    "annualized_return": r["oos_metrics"]["ann_return"],
                    "information_ratio": r["oos_metrics"]["information_ratio"],
                    "max_drawdown": r["oos_metrics"]["max_drawdown"],
                    "net_return": r["oos_metrics"]["net_return"],
                    "avg_holdings": r["oos_metrics"]["avg_holdings"],
                    "avg_turnover": r["oos_metrics"]["avg_turnover"],
                },
                "excess_return_without_cost": {
                    "mean": r["oos_excess_without_cost"]["mean_return"],
                    "std": r["oos_excess_without_cost"]["ann_vol"] / np.sqrt(252) if r["oos_excess_without_cost"]["ann_vol"] else 0,
                    "annualized_return": r["oos_excess_without_cost"]["ann_return"],
                    "information_ratio": r["oos_excess_without_cost"]["information_ratio"],
                    "max_drawdown": r["oos_excess_without_cost"]["max_drawdown"],
                    "net_return": r["oos_excess_without_cost"]["net_return"],
                },
                "excess_return_with_cost": {
                    "mean": r["oos_excess_with_cost"]["mean_return"],
                    "std": r["oos_excess_with_cost"]["ann_vol"] / np.sqrt(252) if r["oos_excess_with_cost"]["ann_vol"] else 0,
                    "annualized_return": r["oos_excess_with_cost"]["ann_return"],
                    "information_ratio": r["oos_excess_with_cost"]["information_ratio"],
                    "max_drawdown": r["oos_excess_with_cost"]["max_drawdown"],
                    "net_return": r["oos_excess_with_cost"]["net_return"],
                },
                "trade_metrics": r["oos_trade_metrics"],
            },
        }
        all_combinations_summary.append(combo_summary)

    summary: dict[str, Any] = {
        "hypothesis_id": hypothesis_id,
        "backtest_mode": "per_combo_parallel",
        "evaluation_modes": ["optuna"] if use_optuna else ["fixed"],
        "n_trials": cfg.stage4.n_trials,
        "horizon_days": cfg.stage4.horizon_days,
        "in_sample_period": f"{cfg.data_split.in_sample_start} ~ {cfg.data_split.in_sample_end}",
        "out_sample_period": f"{cfg.data_split.out_sample_start} ~ {cfg.data_split.out_sample_end}",
        "transaction_costs": {
            "open_cost": cfg.qlib.open_cost,
            "close_cost": cfg.qlib.close_cost,
            "min_cost": cfg.qlib.min_cost,
        },
        "benchmark": {
            "insample": {
                "information_ratio": is_benchmark_metrics["information_ratio"],
                "net_return": is_benchmark_metrics["net_return"],
                "mean": is_benchmark_metrics["mean_return"],
                "std": is_benchmark_metrics["ann_vol"] / np.sqrt(252) if is_benchmark_metrics["ann_vol"] else 0,
                "annualized_return": is_benchmark_metrics["ann_return"],
                "max_drawdown": is_benchmark_metrics["max_drawdown"],
            },
            "outsample": {
                "information_ratio": oos_benchmark_metrics["information_ratio"],
                "net_return": oos_benchmark_metrics["net_return"],
                "mean": oos_benchmark_metrics["mean_return"],
                "std": oos_benchmark_metrics["ann_vol"] / np.sqrt(252) if oos_benchmark_metrics["ann_vol"] else 0,
                "annualized_return": oos_benchmark_metrics["ann_return"],
                "max_drawdown": oos_benchmark_metrics["max_drawdown"],
            },
        },
        "n_combinations_evaluated": len(all_combinations_summary),
        "all_combinations": all_combinations_summary,
        "executor": executor_kind,
        "combo_workers": combo_workers,
        "optuna_n_jobs": optuna_n_jobs,
    }

    # Minimal report (avoid relying on legacy report generator shape).
    lines = [
        "# Stage 4: Backtest (Per-Combo Parallel)",
        "",
        f"- hypothesis_id: `{hypothesis_id}`",
        f"- combinations_evaluated: {len(all_combinations_summary)}",
        f"- executor: `{executor_kind}` (workers={combo_workers})",
        "",
        "## Benchmark",
        f"- IS IR: {summary['benchmark']['insample']['information_ratio']:.3f}, net_return: {summary['benchmark']['insample']['net_return']:.3f}",
        f"- OOS IR: {summary['benchmark']['outsample']['information_ratio']:.3f}, net_return: {summary['benchmark']['outsample']['net_return']:.3f}",
        "",
    ]
    for c in all_combinations_summary:
        is_ir = c["insample"]["return"]["information_ratio"]
        oos_ir = c["outsample"]["return"]["information_ratio"]
        lines.append(f"- combo {c['combo_idx']}: IS IR={is_ir:.3f}, OOS IR={oos_ir:.3f} ({', '.join(c['formula_names'])})")
    report_md = "\n".join(lines) + "\n"

    config_dict = {
        "in_sample_start": cfg.data_split.in_sample_start,
        "in_sample_end": cfg.data_split.in_sample_end,
        "out_sample_start": cfg.data_split.out_sample_start,
        "out_sample_end": cfg.data_split.out_sample_end,
        "n_trials": cfg.stage4.n_trials,
        "threshold_min": cfg.stage4.threshold_min,
        "threshold_max": cfg.stage4.threshold_max,
        "horizon_days": cfg.stage4.horizon_days,
        "lookback_window": cfg.stage4.lookback_window,
        "ref_price_fn": cfg.stage4.ref_price_fn,
        "trigger_price_field": cfg.stage4.trigger_price_field,
        "trigger_op": cfg.stage4.trigger_op,
        "trigger_kmin": cfg.stage4.trigger_kmin,
        "trigger_kmax": cfg.stage4.trigger_kmax,
        "open_cost": cfg.qlib.open_cost,
        "close_cost": cfg.qlib.close_cost,
        "min_cost": cfg.qlib.min_cost,
        "init_cash": cfg.qlib.init_cash,
        "backtest_mode": "per_combo_parallel",
        "combo_workers": combo_workers,
        "optuna_n_jobs": optuna_n_jobs,
    }

    if run_ctx is not None:
        iter_prefix = f"iter_{outer_iter}" if outer_iter is not None else "iter_1"
        if outer_iter is not None:
            run_ctx.save_json_with_iter("specs/stage4_summary.json", outer_iter, summary)
            run_ctx.save_text_with_iter("reports/stage4.md", outer_iter, report_md)
        else:
            run_ctx.save_json("specs/stage4_summary.json", summary)
            run_ctx.save_text("reports/stage4.md", report_md)

        for r in results:
            combo_idx = r["combo_idx"]
            base_dir = f"qlib_artifacts/{iter_prefix}/combo_{combo_idx}"

            oos_dir = f"{base_dir}/oos"
            if r.get("oos_report_df") is not None:
                run_ctx.save_pickle(f"{oos_dir}/report_normal_1day.pkl", r["oos_report_df"])
            oos_analysis = _build_port_analysis_df(r.get("oos_report_df"), freq="1day")
            run_ctx.save_pickle(f"{oos_dir}/port_analysis_1day.pkl", oos_analysis)
            oos_trades = r.get("oos_trades", [])
            if oos_trades:
                run_ctx.save_pickle(f"{oos_dir}/trades.pkl", oos_trades)

            is_dir = f"{base_dir}/is"
            if r.get("is_report_df") is not None:
                run_ctx.save_pickle(f"{is_dir}/report_normal_1day.pkl", r["is_report_df"])
            is_analysis = _build_port_analysis_df(r.get("is_report_df"), freq="1day")
            run_ctx.save_pickle(f"{is_dir}/port_analysis_1day.pkl", is_analysis)
            is_trades = r.get("is_trades", [])
            if is_trades:
                run_ctx.save_pickle(f"{is_dir}/trades.pkl", is_trades)

        # qlib_res.csv and multiindex csv
        summary_rows = []
        bench_mean = oos_benchmark_metrics.get("mean_return")
        bench_std = (oos_benchmark_metrics.get("ann_vol") or 0.0) / np.sqrt(252)
        bench_ann = oos_benchmark_metrics.get("ann_return")
        bench_ir = oos_benchmark_metrics.get("information_ratio", oos_benchmark_metrics.get("sharpe"))
        bench_mdd = oos_benchmark_metrics.get("max_drawdown")

        for c in all_combinations_summary:
            ex_wo = c.get("outsample", {}).get("excess_return_without_cost", {}) or {}
            ex_w = c.get("outsample", {}).get("excess_return_with_cost", {}) or {}
            summary_rows.append({
                "combo_idx": c.get("combo_idx"),
                "formula_names": "_".join(c.get("formula_names", [])),
                "benchmark_mean": bench_mean,
                "benchmark_std": bench_std,
                "benchmark_annualized_return": bench_ann,
                "benchmark_information_ratio": bench_ir,
                "benchmark_max_drawdown": bench_mdd,
                "excess_return_without_cost_mean": ex_wo.get("mean"),
                "excess_return_without_cost_std": ex_wo.get("std"),
                "excess_return_without_cost_annualized_return": ex_wo.get("annualized_return"),
                "excess_return_without_cost_information_ratio": ex_wo.get("information_ratio"),
                "excess_return_without_cost_max_drawdown": ex_wo.get("max_drawdown"),
                "excess_return_with_cost_mean": ex_w.get("mean"),
                "excess_return_with_cost_std": ex_w.get("std"),
                "excess_return_with_cost_annualized_return": ex_w.get("annualized_return"),
                "excess_return_with_cost_information_ratio": ex_w.get("information_ratio"),
                "excess_return_with_cost_max_drawdown": ex_w.get("max_drawdown"),
            })
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res.csv", pd.DataFrame(summary_rows))

        multiindex_df = _combinations_to_multiindex_df(all_combinations_summary)
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res_multiindex.csv", multiindex_df)

        if verbose:
            print(f"[Stage4-PerCombo] Artifacts saved to qlib_artifacts/{iter_prefix}/")

    empty_panel = pl.DataFrame({"timestamp": [], "gross_return": [], "net_return": []})
    return Stage4Result(
        hypothesis_id=hypothesis_id,
        config=config_dict,
        summary=summary,
        report_md=report_md,
        is_daily_panel=empty_panel,
        oos_daily_panel=empty_panel,
    )


# Alias for monkey-patching symmetry
run_stage4 = run_stage4_parallel_per_combo
