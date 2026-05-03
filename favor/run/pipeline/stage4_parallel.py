"""
================================================================================
STAGE 4 PARALLEL: Vectorized Backtest for Phase 2
================================================================================

[Optimized version]
- Phase 1: Find the best threshold per combination via Optuna (same as legacy approach)
- Phase 2: Generate signals for all combinations at once → run vectorized backtests once for IS and once for OOS

[Key changes]
1. Keep Optuna optimization as-is (threshold search)
2. Generate all combination signals simultaneously during final evaluation
3. Vectorized backtest: compute all combination results with one IS run + one OOS run

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

import numpy as np
import pandas as pd
import polars as pl

from qlib.contrib.evaluate import risk_analysis

from util.run_context import RunContext
from run.config import RDConfig, load_rd_config

if TYPE_CHECKING:
    import optuna

# Import from original stage4 for compatibility
from run.pipeline.stage4 import (
    Stage4Result,
    TradeRecord,
    _prepare_panel,
    _split_in_out_sample,
    _compute_thresholds,
    _apply_signal,
    _simulate_positions,
    _calculate_portfolio_returns,
    _compute_metrics_from_returns,
    _run_qlib_backtest,
    _compute_metrics_qlib,
    _compute_trade_metrics,
    _calc_excess_metrics,
    _calculate_benchmark_returns,
    _build_port_analysis_df,
    _generate_report_all_combinations,
    _create_objective,
)

logger = logging.getLogger(__name__)

def _apply_stage4_env_overrides(cfg: RDConfig) -> None:
    """
    Apply Stage4 overrides from environment variables.

    This is intentionally kept in the *parallel* implementations so that
    run/config.py remains a clean, mostly-static config definition.
    """
    import os

    def _parse_bool_env(v: str) -> bool:
        s = str(v).strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        raise ValueError("Use 'True' or 'False'.")

    env_enable_optuna = os.getenv("STAGE4_ENABLE_OPTUNA")
    if env_enable_optuna is not None:
        try:
            cfg.stage4.enable_optuna = _parse_bool_env(env_enable_optuna)
        except Exception:
            pass

    env_n_trials = os.getenv("STAGE4_N_TRIALS")
    if env_n_trials is not None:
        try:
            cfg.stage4.n_trials = int(str(env_n_trials).strip())
        except Exception:
            pass

    env_max_combos = os.getenv("STAGE4_MAX_COMBINATIONS_TO_EVALUATE")
    if env_max_combos is not None:
        try:
            cfg.stage4.max_combinations_to_evaluate = int(str(env_max_combos).strip())
        except Exception:
            pass

    env_fixed_quantiles = os.getenv("STAGE4_FIXED_QUANTILES")
    if env_fixed_quantiles is not None:
        raw = str(env_fixed_quantiles).strip()
        if raw == "" or raw.lower() in {"none", "null", "[]"}:
            cfg.stage4.fixed_quantiles = []
        else:
            parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
            parsed: list[float] = []
            for p in parts:
                try:
                    q = float(p)
                except Exception:
                    continue
                if 0.0 < q < 1.0:
                    parsed.append(q)
            cfg.stage4.fixed_quantiles = parsed


# ════════════════════════════════════════════════════════════════════════════
# Vectorized Position Simulation (Phase 2)
# ════════════════════════════════════════════════════════════════════════════

def _simulate_positions_vectorized(
    *,
    panel: pd.DataFrame,
    signal_columns: List[str],
    cfg: RDConfig,
) -> Tuple[Dict[str, pd.Series], Dict[str, List[TradeRecord]]]:
    """
    Vectorized position simulation: process all signals in a single loop.

    Returns:
        positions: Dict[signal_col -> position Series]
        trades: Dict[signal_col -> List[TradeRecord]]
    """
    horizon_days = cfg.stage4.horizon_days
    lookback_window = cfg.stage4.lookback_window
    ref_price_fn = str(cfg.stage4.ref_price_fn).strip().lower()
    trigger_price_field = str(cfg.stage4.trigger_price_field).strip().lower()
    trigger_op = str(cfg.stage4.trigger_op).strip().lower()
    trigger_kmin = cfg.stage4.trigger_kmin
    trigger_kmax = cfg.stage4.trigger_kmax if cfg.stage4.trigger_kmax is not None else horizon_days
    stop_loss_threshold = getattr(cfg.stage4, "stop_loss_threshold", -0.05)

    # Initialize
    positions = {col: pd.Series(False, index=panel.index) for col in signal_columns}
    all_trades: Dict[str, List[TradeRecord]] = {col: [] for col in signal_columns}

    timestamps = panel["timestamp"].astype(str).tolist()

    for ticker, idx in panel.groupby("ticker", sort=False).groups.items():
        loc = pd.Index(idx)
        closes = panel.loc[loc, "close"].astype(float).to_numpy()
        highs = panel.loc[loc, "high"].astype(float).to_numpy()
        lows = panel.loc[loc, "low"].astype(float).to_numpy()
        ts = [timestamps[i] for i in loc]

        # Get signals for this ticker
        ticker_signals = {col: panel.loc[loc, col].astype(bool).to_numpy() for col in signal_columns}

        # State per signal
        states = {col: {
            "in_pos": False,
            "entry_i": -1,
            "entry_price": np.nan,
            "pref": np.nan,
        } for col in signal_columns}

        for i in range(len(closes)):
            for col in signal_columns:
                state = states[col]
                sig = ticker_signals[col]
                exited_today = False

                if state["in_pos"]:
                    k = i - state["entry_i"]
                    entry_price = state["entry_price"]
                    pref = state["pref"]

                    # 0) Stop loss
                    if entry_price > 0:
                        cumulative_return = (closes[i] / entry_price) - 1.0
                        if stop_loss_threshold is not None and cumulative_return <= float(stop_loss_threshold):
                            exit_price = closes[i]
                            ret = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                            all_trades[col].append(TradeRecord(
                                ticker=str(ticker),
                                entry_date=ts[state["entry_i"]],
                                entry_price=entry_price,
                                exit_date=ts[i],
                                exit_price=exit_price,
                                exit_reason="stop_loss",
                                holding_days=k,
                                return_pct=ret,
                            ))
                            state["in_pos"] = False
                            state["entry_i"] = -1
                            state["entry_price"] = np.nan
                            state["pref"] = np.nan
                            exited_today = True

                    # 1) Trigger exit
                    if (not exited_today) and (not np.isnan(pref)) and (k >= trigger_kmin) and (k <= trigger_kmax):
                        if trigger_price_field == "high":
                            obs = highs[i]
                        elif trigger_price_field == "low":
                            obs = lows[i]
                        else:
                            obs = closes[i]

                        triggered = (obs >= pref) if trigger_op == "gte" else (obs <= pref)
                        if triggered:
                            exit_price = closes[i]
                            ret = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                            all_trades[col].append(TradeRecord(
                                ticker=str(ticker),
                                entry_date=ts[state["entry_i"]],
                                entry_price=entry_price,
                                exit_date=ts[i],
                                exit_price=exit_price,
                                exit_reason="trigger",
                                holding_days=k,
                                return_pct=ret,
                            ))
                            state["in_pos"] = False
                            state["entry_i"] = -1
                            state["entry_price"] = np.nan
                            state["pref"] = np.nan
                            exited_today = True

                    # 2) Time stop
                    if (not exited_today) and (k >= horizon_days):
                        exit_price = closes[i]
                        ret = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                        all_trades[col].append(TradeRecord(
                            ticker=str(ticker),
                            entry_date=ts[state["entry_i"]],
                            entry_price=entry_price,
                            exit_date=ts[i],
                            exit_price=exit_price,
                            exit_reason="time_stop",
                            holding_days=k,
                            return_pct=ret,
                        ))
                        state["in_pos"] = False
                        state["entry_i"] = -1
                        state["entry_price"] = np.nan
                        state["pref"] = np.nan
                        exited_today = True

                # Entry
                if (not state["in_pos"]) and (not exited_today) and sig[i]:
                    state["in_pos"] = True
                    state["entry_i"] = i
                    state["entry_price"] = closes[i]

                    # P_ref calculation
                    start = max(0, i - lookback_window)
                    end = i
                    if end <= start:
                        state["pref"] = np.nan
                    else:
                        if ref_price_fn == "max_high":
                            state["pref"] = float(np.nanmax(highs[start:end]))
                        elif ref_price_fn == "min_low":
                            state["pref"] = float(np.nanmin(lows[start:end]))
                        elif ref_price_fn == "q50_close":
                            state["pref"] = float(np.nanquantile(closes[start:end], 0.5))
                        else:
                            state["pref"] = np.nan

                positions[col].loc[loc[i]] = state["in_pos"]

    return positions, all_trades


def _calculate_returns_vectorized(
    *,
    panel: pd.DataFrame,
    positions: Dict[str, pd.Series],
    cfg: RDConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Vectorized portfolio return calculation.
    """
    panel = panel.copy()
    panel["ret1"] = panel.groupby("ticker")["close"].shift(-1) / panel["close"] - 1.0

    results = {}

    for col, position in positions.items():
        valid = position & panel["ret1"].notna()

        denom = valid.groupby(panel["timestamp"]).transform("sum")
        weight = (valid.astype(float) / denom.replace(0, np.nan)).fillna(0.0)

        gross = (weight * panel["ret1"].fillna(0.0)).groupby(panel["timestamp"]).sum()
        holdings = valid.groupby(panel["timestamp"]).sum().astype(int)

        # Turnover
        wdf = panel.loc[:, ["timestamp", "ticker"]].copy()
        wdf["weight"] = weight.values
        wdf = wdf.sort_values(["ticker", "timestamp"], kind="mergesort")
        wdf["prev_weight"] = wdf.groupby("ticker")["weight"].shift(1).fillna(0.0)
        wdf["weight_change"] = wdf["weight"] - wdf["prev_weight"]

        buy_turnover = wdf.groupby("timestamp")["weight_change"].apply(lambda x: x.clip(lower=0).sum())
        sell_turnover = wdf.groupby("timestamp")["weight_change"].apply(lambda x: (-x).clip(lower=0).sum())
        turnover = 0.5 * (buy_turnover + sell_turnover)

        if not turnover.empty:
            turnover.iloc[0] = 0.0

        cost = (buy_turnover * cfg.qlib.open_cost + sell_turnover * cfg.qlib.close_cost).reindex(gross.index).fillna(0.0)
        net = gross - cost

        eq_gross = (1.0 + gross).cumprod()
        eq_net = (1.0 + net).cumprod()

        results[col] = {
            "gross": gross,
            "net": net,
            "eq_gross": eq_gross,
            "eq_net": eq_net,
            "holdings": holdings,
            "turnover": turnover,
            "cost": cost,
        }

    return results


def _compute_metrics_from_returns_dict(returns: Dict[str, Any]) -> Dict[str, float]:
    """Calculate metrics from returns dict"""
    net = returns["net"]
    eq_net = returns["eq_net"]
    eq_gross = returns["eq_gross"]
    holdings = returns["holdings"]
    turnover = returns["turnover"]

    n_days = int(net.shape[0])

    if n_days > 1:
        try:
            risk_metrics = risk_analysis(net, freq="day")
            mean_ret = float(risk_metrics.loc["mean", "risk"])
            ann_ret = float(risk_metrics.loc["annualized_return", "risk"])
            ann_vol = float(risk_metrics.loc["std", "risk"] * np.sqrt(252))
            information_ratio = float(risk_metrics.loc["information_ratio", "risk"])
            mdd = float(risk_metrics.loc["max_drawdown", "risk"])
        except Exception:
            mean_ret = ann_ret = ann_vol = information_ratio = mdd = 0.0
    else:
        mean_ret = ann_ret = ann_vol = information_ratio = mdd = 0.0

    return {
        "n_days": n_days,
        "avg_holdings": float(holdings.mean()) if not holdings.empty else 0.0,
        "gross_return": float(eq_gross.iloc[-1] - 1.0) if not eq_gross.empty else 0.0,
        "net_return": float(eq_net.iloc[-1] - 1.0) if not eq_net.empty else 0.0,
        "mean_return": mean_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "information_ratio": information_ratio,
        "max_drawdown": mdd,
        "avg_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
    }


def _build_report_df_from_returns(returns: Dict[str, Any], bench: pd.Series = None) -> pd.DataFrame:
    """Build Qlib-style report_df from returns dict"""
    gross = returns["gross"]
    cost = returns["cost"]
    turnover = returns["turnover"]

    if bench is None:
        bench = pd.Series(0.0, index=gross.index)
    else:
        bench = bench.reindex(gross.index).fillna(0.0)

    return pd.DataFrame({
        "return": gross,
        "cost": cost,
        "bench": bench,
        "turnover": turnover,
    })


def _combinations_to_multiindex_df(combinations_summary: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert `all_combinations_summary` into a DataFrame with MultiIndex columns.

    Qlib-style structure:
    - metadata: combo_idx, formula_names
    - insample.return.{mean, std, annualized_return, information_ratio, max_drawdown, ...}
    - insample.excess_return_without_cost.{mean, std, ...}
    - insample.excess_return_with_cost.{mean, std, ...}
    - insample.trade_metrics.{n_trades, win_rate, ...}
    - outsample.* (same structure)
    """
    rows = []
    for combo in combinations_summary:
        row_dict = {}

        # Metadata
        row_dict[("meta", "combo_idx")] = combo.get("combo_idx")
        row_dict[("meta", "formula_names")] = "_".join(combo.get("formula_names", []))

        # Process insample and outsample
        for sample_type in ["insample", "outsample"]:
            sample_data = combo.get(sample_type, {})

            # Return metrics
            for metric_name, metric_value in sample_data.get("return", {}).items():
                row_dict[(f"{sample_type}.return", metric_name)] = metric_value

            # excess_return_without_cost metrics
            for metric_name, metric_value in sample_data.get("excess_return_without_cost", {}).items():
                row_dict[(f"{sample_type}.excess_return_without_cost", metric_name)] = metric_value

            # excess_return_with_cost metrics
            for metric_name, metric_value in sample_data.get("excess_return_with_cost", {}).items():
                row_dict[(f"{sample_type}.excess_return_with_cost", metric_name)] = metric_value

            # trade_metrics
            for metric_name, metric_value in sample_data.get("trade_metrics", {}).items():
                row_dict[(f"{sample_type}.trade_metrics", metric_name)] = metric_value

        rows.append(row_dict)

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Sort MultiIndex columns (put meta first)
    if not df.empty:
        meta_cols = [col for col in df.columns if col[0] == "meta"]
        other_cols = [col for col in df.columns if col[0] != "meta"]
        df = df[meta_cols + sorted(other_cols)]

    return df


# ════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_stage4_parallel(
    *,
    hypothesis_id: str,
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
    """
    Stage4 Parallel main entry point.

    Phase 1: Find the best threshold per combination via Optuna (legacy approach).
    Phase 2: Generate signals for all combinations at once → vectorized backtest (1 IS + 1 OOS).
    """
    cfg = cfg or load_rd_config()
    _apply_stage4_env_overrides(cfg)
    use_optuna = bool(getattr(cfg.stage4, "enable_optuna", False))

    # Extract horizon_days from hypothesis
    if hypothesis:
        hyp_list = hypothesis.get("hypotheses", [])
        hyp_obj = hyp_list[0] if isinstance(hyp_list, list) and hyp_list else hypothesis
        h = (hyp_obj or {}).get("horizon_days")
        if isinstance(h, int) and h > 0:
            cfg.stage4.horizon_days = h

    if cfg.stage4.trigger_kmax is None:
        cfg.stage4.trigger_kmax = cfg.stage4.horizon_days

    if verbose:
        print(f"[Stage4-Parallel] Starting for hypothesis: {hypothesis_id}")
        print(f"[Stage4-Parallel] In-Sample:     {cfg.data_split.in_sample_start} ~ {cfg.data_split.in_sample_end}")
        print(f"[Stage4-Parallel] Out-of-Sample: {cfg.data_split.out_sample_start} ~ {cfg.data_split.out_sample_end}")
        print(f"[Stage4-Parallel] Combinations: {len(passed_combinations)}, Horizon: {cfg.stage4.horizon_days}d")
        if use_optuna:
            print(f"[Stage4-Parallel] Optuna trials: {cfg.stage4.n_trials}")
        else:
            fixed_quantiles = list(getattr(cfg.stage4, "fixed_quantiles", [0.8]) or [0.8])
            default_threshold = fixed_quantiles[0] if fixed_quantiles else 0.8
            print(f"[Stage4-Parallel] Optuna: DISABLED (using fixed threshold: {default_threshold:.2f})")

    # Empty case
    if not passed_combinations:
        empty = pl.DataFrame({"timestamp": [], "gross_return": [], "net_return": []})
        return Stage4Result(
            hypothesis_id=hypothesis_id,
            config={},
            summary={"hypothesis_id": hypothesis_id, "error": "No passed_combinations; skip backtest."},
            report_md="# Stage 4: Backtest\n\nNo passed combinations. Skipped.",
            is_daily_panel=empty,
            oos_daily_panel=empty,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Combination selection
    # ═══════════════════════════════════════════════════════════════════════
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
        print(f"[Stage4-Parallel] Evaluating {len(combinations_to_evaluate)} / {len(passed_combinations)} combinations")

    # ═══════════════════════════════════════════════════════════════════════
    # Prepare a merged panel containing all formulas
    # ═══════════════════════════════════════════════════════════════════════
    all_formula_names = set()
    for combo in combinations_to_evaluate:
        for f in combo:
            if isinstance(f, dict) and f.get("name"):
                all_formula_names.add(str(f["name"]))
    all_formula_names = sorted(all_formula_names)

    if verbose:
        print(f"[Stage4-Parallel] Loading {len(all_formula_names)} unique formulas")

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

    # Fix universe (common IS/OOS tickers)
    is_tickers = set(is_panel["ticker"].unique().tolist())
    oos_tickers = set(oos_panel["ticker"].unique().tolist())
    common_tickers = sorted(is_tickers & oos_tickers)

    if not common_tickers:
        raise ValueError("No common tickers between IS and OOS")

    is_panel = is_panel[is_panel["ticker"].isin(common_tickers)].reset_index(drop=True)
    oos_panel = oos_panel[oos_panel["ticker"].isin(common_tickers)].reset_index(drop=True)

    if verbose:
        print(f"[Stage4-Parallel] Universe: {len(common_tickers)} tickers")
        print(f"[Stage4-Parallel] IS: {len(is_panel)} rows, OOS: {len(oos_panel)} rows")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Find best thresholds via Optuna (legacy approach)
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n[Stage4-Parallel] ═══ Phase 1: Optuna Optimization ═══")

    combo_optimized = []

    for combo_idx, combination in enumerate(combinations_to_evaluate, 1):
        formula_names = [
            str(f.get("name") or "") for f in combination
            if isinstance(f, dict) and f.get("name")
        ]
        formula_names = [n for n in formula_names if n and n in is_panel.columns]

        if not formula_names:
            continue

        if verbose:
            print(f"\n[Stage4-Parallel] Combo {combo_idx}/{len(combinations_to_evaluate)}: {formula_names}")

        if use_optuna:
            import optuna

            # Optuna optimization
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

            study.optimize(objective, n_trials=cfg.stage4.n_trials, n_jobs=8, show_progress_bar=False)

            # Extract optimal thresholds
            best_params = study.best_params
            optimal_thresholds: Dict[str, float] = {}
            for fname in formula_names:
                key = f"threshold_{fname}"
                if key in best_params:
                    optimal_thresholds[fname] = best_params[key]

            is_information_ratio_optuna = float(study.best_value)
            if verbose:
                print(f"[Stage4-Parallel]   IS IR: {study.best_value:.3f}, Thresholds: {optimal_thresholds}")
        else:
            fixed_quantiles = list(getattr(cfg.stage4, "fixed_quantiles", [0.8]) or [0.8])
            default_threshold = fixed_quantiles[0] if fixed_quantiles else 0.8
            optimal_thresholds = {fname: float(default_threshold) for fname in formula_names}
            is_information_ratio_optuna = 0.0
            if verbose:
                print(f"[Stage4-Parallel]   Fixed threshold: {default_threshold:.2f}, Thresholds: {optimal_thresholds}")

        combo_optimized.append({
            "combo_idx": combo_idx,
            "combination": combination,
            "formula_names": formula_names,
            "optimal_thresholds": optimal_thresholds,
            "is_information_ratio_optuna": is_information_ratio_optuna,
        })

    if not combo_optimized:
        raise ValueError("No valid combinations after Optuna optimization")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Generate all signals + vectorized backtest
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n[Stage4-Parallel] ═══ Phase 2: Vectorized Backtest ═══")
        print(f"[Stage4-Parallel] Generating signals for {len(combo_optimized)} combinations...")

    signal_meta = {}

    for combo_info in combo_optimized:
        combo_idx = combo_info["combo_idx"]
        combination = combo_info["combination"]
        formula_names = combo_info["formula_names"]
        optimal_thresholds = combo_info["optimal_thresholds"]

        col_name = f"signal_c{combo_idx}"

        final_thresholds = _compute_thresholds(
            train_panel=is_panel,
            passed_formulas=combination,
            threshold_dict=optimal_thresholds,
        )

        is_signal = _apply_signal(
            panel=is_panel,
            thresholds=final_thresholds,
            passed_formulas=combination,
        )

        is_panel[col_name] = is_signal.values

        oos_signal = _apply_signal(
            panel=oos_panel,
            thresholds=final_thresholds,
            passed_formulas=combination,
        )

        oos_panel[col_name] = oos_signal.values

        signal_meta[col_name] = {
            **combo_info,
            "final_thresholds": final_thresholds,
        }

    signal_columns = list(signal_meta.keys())

    if verbose:
        print(f"[Stage4-Parallel] Generated {len(signal_columns)} signal columns")
        print(f"[Stage4-Parallel] Running vectorized IS backtest (1 loop for all combos)...")

    # ═══════════════════════════════════════════════════════════════════════
    # IS vectorized backtest
    # ═══════════════════════════════════════════════════════════════════════
    is_positions, is_trades = _simulate_positions_vectorized(
        panel=is_panel,
        signal_columns=signal_columns,
        cfg=cfg,
    )

    is_returns = _calculate_returns_vectorized(
        panel=is_panel,
        positions=is_positions,
        cfg=cfg,
    )

    if verbose:
        print(f"[Stage4-Parallel] Running vectorized OOS backtest (1 loop for all combos)...")

    # ═══════════════════════════════════════════════════════════════════════
    # OOS vectorized backtest
    # ═══════════════════════════════════════════════════════════════════════
    oos_positions, oos_trades = _simulate_positions_vectorized(
        panel=oos_panel,
        signal_columns=signal_columns,
        cfg=cfg,
    )

    oos_returns = _calculate_returns_vectorized(
        panel=oos_panel,
        positions=oos_positions,
        cfg=cfg,
    )

    if verbose:
        print(f"[Stage4-Parallel] Backtest complete! Computing metrics...")

    # ═══════════════════════════════════════════════════════════════════════
    # Benchmark
    # ═══════════════════════════════════════════════════════════════════════
    is_benchmark_returns = _calculate_benchmark_returns(
        start_date=cfg.data_split.in_sample_start,
        end_date=cfg.data_split.in_sample_end,
        benchmark_code=cfg.qlib.benchmark,
    )
    is_benchmark_metrics = _compute_metrics_qlib(is_benchmark_returns)
    is_bench_series = is_benchmark_returns["gross"]

    oos_benchmark_returns = _calculate_benchmark_returns(
        start_date=cfg.data_split.out_sample_start,
        end_date=cfg.data_split.out_sample_end,
        benchmark_code=cfg.qlib.benchmark,
    )
    oos_benchmark_metrics = _compute_metrics_qlib(oos_benchmark_returns)
    oos_bench_series = oos_benchmark_returns["gross"]

    # ═══════════════════════════════════════════════════════════════════════
    # Collect per-combination results
    # ═══════════════════════════════════════════════════════════════════════
    all_combination_results = []

    for col_name in signal_columns:
        meta = signal_meta[col_name]
        combo_idx = meta["combo_idx"]
        combination = meta["combination"]
        formula_names = meta["formula_names"]
        optimal_thresholds = meta["optimal_thresholds"]
        final_thresholds = meta["final_thresholds"]

        is_metrics = _compute_metrics_from_returns_dict(is_returns[col_name])
        is_report_df = _build_report_df_from_returns(is_returns[col_name], is_bench_series)
        is_trade_list = is_trades[col_name]
        is_trade_metrics = _compute_trade_metrics(is_trade_list)

        is_excess_without_cost = _calc_excess_metrics(
            is_report_df["return"] - is_report_df["bench"]
        )
        is_excess_with_cost = _calc_excess_metrics(
            is_report_df["return"] - is_report_df["bench"] - is_report_df["cost"]
        )

        oos_metrics = _compute_metrics_from_returns_dict(oos_returns[col_name])
        oos_report_df = _build_report_df_from_returns(oos_returns[col_name], oos_bench_series)
        oos_trade_list = oos_trades[col_name]
        oos_trade_metrics = _compute_trade_metrics(oos_trade_list)

        oos_excess_without_cost = _calc_excess_metrics(
            oos_report_df["return"] - oos_report_df["bench"]
        )
        oos_excess_with_cost = _calc_excess_metrics(
            oos_report_df["return"] - oos_report_df["bench"] - oos_report_df["cost"]
        )

        if verbose:
            print(f"[Stage4-Parallel] Combo {combo_idx}: IS IR={is_metrics['information_ratio']:.3f}, OOS IR={oos_metrics['information_ratio']:.3f}")

        all_combination_results.append({
            "combo_idx": combo_idx,
            "combination": combination,
            "formula_names": formula_names,
            "optimal_thresholds": optimal_thresholds,
            "final_thresholds": final_thresholds,
            "backtest_mode": "vectorized",
            "is_information_ratio": is_metrics["information_ratio"],
            "is_metrics": is_metrics,
            "is_excess_without_cost": is_excess_without_cost,
            "is_excess_with_cost": is_excess_with_cost,
            "is_trade_metrics": is_trade_metrics,
            "is_report_df": is_report_df,
            "is_positions": [],
            "oos_information_ratio": oos_metrics["information_ratio"],
            "oos_metrics": oos_metrics,
            "oos_excess_without_cost": oos_excess_without_cost,
            "oos_excess_with_cost": oos_excess_with_cost,
            "oos_trade_metrics": oos_trade_metrics,
            "oos_report_df": oos_report_df,
            "oos_trades": [t.__dict__ if hasattr(t, '__dict__') else t for t in oos_trade_list],
            "oos_positions": [],
        })

    if verbose:
        print(f"\n[Stage4-Parallel] ═══════════════════════════════════════════════════════")
        print(f"[Stage4-Parallel] All {len(all_combination_results)} combinations evaluated")
        print(f"[Stage4-Parallel] ═══════════════════════════════════════════════════════")

    # ═══════════════════════════════════════════════════════════════════════
    # Build summary
    # ═══════════════════════════════════════════════════════════════════════
    all_combinations_summary = []
    for r in all_combination_results:
        # Qlib-style structure: insample/outsample as top-level, strategy → return
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
                "trade_metrics": {
                    "n_trades": r["is_trade_metrics"]["n_trades"],
                    "win_rate": r["is_trade_metrics"]["win_rate"],
                    "profit_factor": r["is_trade_metrics"]["profit_factor"],
                    "avg_trade_return": r["is_trade_metrics"]["avg_trade_return"],
                    "avg_holding_days": r["is_trade_metrics"].get("avg_holding_days", 0.0),
                },
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
                    "std": r["oos_excess_without_cost"]["ann_vol"] / np.sqrt(252) if r["oos_excess_with_cost"]["ann_vol"] else 0,
                    "annualized_return": r["oos_excess_with_cost"]["ann_return"],
                    "information_ratio": r["oos_excess_with_cost"]["information_ratio"],
                    "max_drawdown": r["oos_excess_with_cost"]["max_drawdown"],
                    "net_return": r["oos_excess_with_cost"]["net_return"],
                },
                "trade_metrics": {
                    "n_trades": r["oos_trade_metrics"]["n_trades"],
                    "win_rate": r["oos_trade_metrics"]["win_rate"],
                    "profit_factor": r["oos_trade_metrics"]["profit_factor"],
                    "avg_trade_return": r["oos_trade_metrics"]["avg_trade_return"],
                    "avg_holding_days": r["oos_trade_metrics"].get("avg_holding_days", 0.0),
                },
            },
        }
        all_combinations_summary.append(combo_summary)

    summary = {
        "hypothesis_id": hypothesis_id,
        "backtest_mode": "vectorized",
        "evaluation_modes": ["optuna"],
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
        "n_combinations_evaluated": len(all_combination_results),
        "all_combinations": all_combinations_summary,
    }

    report_md = _generate_report_all_combinations(
        hypothesis_id=hypothesis_id,
        cfg=cfg,
        all_combinations=all_combinations_summary,
        benchmark=summary["benchmark"],
    )

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
        "backtest_mode": "vectorized",
    }

    # Save results
    if run_ctx is not None:
        iter_prefix = f"iter_{outer_iter}" if outer_iter is not None else "iter_1"

        if outer_iter is not None:
            run_ctx.save_json_with_iter("specs/stage4_summary.json", outer_iter, summary)
            run_ctx.save_text_with_iter("reports/stage4.md", outer_iter, report_md)
        else:
            run_ctx.save_json("specs/stage4_summary.json", summary)
            run_ctx.save_text("reports/stage4.md", report_md)

        for r in all_combination_results:
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

        summary_rows = []
        # Required CSV columns (OOS-based)
        bench_mean = oos_benchmark_metrics.get("mean_return")
        bench_std = (oos_benchmark_metrics.get("ann_vol") or 0.0) / np.sqrt(252)
        bench_ann = oos_benchmark_metrics.get("ann_return")
        bench_ir = oos_benchmark_metrics.get("information_ratio", oos_benchmark_metrics.get("sharpe"))
        bench_mdd = oos_benchmark_metrics.get("max_drawdown")

        for r in all_combination_results:
            ex_wo = r.get("oos_excess_without_cost", {}) or {}
            ex_w = r.get("oos_excess_with_cost", {}) or {}
            summary_rows.append({
                "combo_idx": r["combo_idx"],
                "formula_names": "_".join(r.get("formula_names", [])),
                # Benchmark (OOS)
                "benchmark_mean": bench_mean,
                "benchmark_std": bench_std,
                "benchmark_annualized_return": bench_ann,
                "benchmark_information_ratio": bench_ir,
                "benchmark_max_drawdown": bench_mdd,
                # Excess return (OOS, without cost)
                "excess_return_without_cost_mean": ex_wo.get("mean_return"),
                "excess_return_without_cost_std": (ex_wo.get("ann_vol") or 0.0) / np.sqrt(252),
                "excess_return_without_cost_annualized_return": ex_wo.get("ann_return"),
                "excess_return_without_cost_information_ratio": ex_wo.get("information_ratio", ex_wo.get("sharpe")),
                "excess_return_without_cost_max_drawdown": ex_wo.get("max_drawdown"),
                # Excess return (OOS, with cost)
                "excess_return_with_cost_mean": ex_w.get("mean_return"),
                "excess_return_with_cost_std": (ex_w.get("ann_vol") or 0.0) / np.sqrt(252),
                "excess_return_with_cost_annualized_return": ex_w.get("ann_return"),
                "excess_return_with_cost_information_ratio": ex_w.get("information_ratio", ex_w.get("sharpe")),
                "excess_return_with_cost_max_drawdown": ex_w.get("max_drawdown"),
                # Backward-compatible columns
                "1day.excess_return_without_cost.information_ratio": ex_wo.get("information_ratio", ex_wo.get("sharpe")),
                "1day.excess_return_without_cost.annualized_return": ex_wo.get("ann_return"),
                "1day.excess_return_without_cost.max_drawdown": ex_wo.get("max_drawdown"),
                "1day.excess_return_with_cost.information_ratio": ex_w.get("information_ratio", ex_w.get("sharpe")),
                "1day.excess_return_with_cost.annualized_return": ex_w.get("ann_return"),
                "1day.excess_return_with_cost.max_drawdown": ex_w.get("max_drawdown"),
            })
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res.csv", pd.DataFrame(summary_rows))

        # Save MultiIndex CSV (qlib standard structure)
        multiindex_df = _combinations_to_multiindex_df(all_combinations_summary)
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res_multiindex.csv", multiindex_df)

        if verbose:
            print(f"[Stage4-Parallel] Artifacts saved to qlib_artifacts/{iter_prefix}/")

    if verbose:
        print(f"[Stage4-Parallel] Complete!")

    empty_panel = pl.DataFrame({"timestamp": [], "gross_return": [], "net_return": []})

    return Stage4Result(
        hypothesis_id=hypothesis_id,
        config=config_dict,
        summary=summary,
        report_md=report_md,
        is_daily_panel=empty_panel,
        oos_daily_panel=empty_panel,
    )


# Alias
run_stage4 = run_stage4_parallel
