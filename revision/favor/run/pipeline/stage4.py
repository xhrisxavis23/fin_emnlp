"""
================================================================================
STAGE 4: In-Sample / Out-of-Sample Split Backtest with Optuna Optimization
================================================================================

[Qlib backtest integration]
- Runs backtests via Qlib's `backtest_daily()` (with a legacy fallback).
- Preserves the existing strategy logic via a custom strategy wrapper (Trigger/Exit style) or
  Qlib's `TopkDropoutStrategy` depending on configuration.
- Uses `risk_analysis()` to compute performance metrics.

[Key behavior]
1. In-sample / out-of-sample 2-way split is configured via `cfg.data_split`.
2. Uses Optuna to optimize per-formula thresholds on the in-sample window.
3. Final evaluation applies the in-sample thresholds to the out-of-sample window.
4. Avoids look-ahead bias: out-of-sample data is not used during optimization.

================================================================================
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

warnings.filterwarnings("ignore")

# Suppress qlib logging spam (these are logging WARNING/INFO messages, not Python warnings).
for _name in [
    "qlib",
    "qlib.Initialization",
    "qlib.online operator",
    "qlib.BaseExecutor",
    "qlib.backtest",
    "qlib.backtest caller",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import polars as pl

from qlib.contrib.evaluate import risk_analysis

from util.run_context import RunContext
from run.config import RDConfig, load_rd_config

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@dataclass
class Stage4Result:
    """Stage 4 final output container (Qlib-style nested structure)."""
    hypothesis_id: str
    config: Dict[str, Any]
    summary: Dict[str, Any]  # Qlib-style nested structure for all results.
    report_md: str
    is_daily_panel: pl.DataFrame  # In-Sample daily panel
    oos_daily_panel: pl.DataFrame  # Out-of-Sample daily panel
    # Backward compatibility: some callers expect `.result`.
    # Default to the same dict as `summary` when not provided.
    result: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.result is None:
            self.result = self.summary


# ════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════════════

def _prepare_panel(
    *,
    ohlcv_df: pl.DataFrame,
    formula_df: pl.DataFrame,
    formula_names: List[str],
) -> pd.DataFrame:
    """Merge OHLCV and formula panels into a single pandas DataFrame."""
    ohlcv_cols = set(ohlcv_df.columns)
    required_ohlcv = {"timestamp", "ticker", "close", "high", "low"}
    if not required_ohlcv.issubset(ohlcv_cols):
        missing = sorted(required_ohlcv - ohlcv_cols)
        raise ValueError(f"ohlcv_df missing required columns: {missing}")

    # Include optional OHLCV fields when available.
    select_cols = ["timestamp", "ticker", "close", "high", "low"]
    if "open" in ohlcv_cols:
        select_cols.append("open")
    if "volume" in ohlcv_cols:
        select_cols.append("volume")

    base = ohlcv_df.select(select_cols).to_pandas()
    fcols = ["timestamp", "ticker", *[c for c in formula_names if c in formula_df.columns]]
    if fcols == ["timestamp", "ticker"]:
        raise ValueError("No formula columns found in formula_df")
    f = formula_df.select(fcols).to_pandas()

    merged = base.merge(f, on=["timestamp", "ticker"], how="inner")
    merged = merged.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(drop=True)
    return merged


def _split_in_out_sample(
    panel: pd.DataFrame,
    in_sample_start: str,
    in_sample_end: str,
    out_sample_start: str,
    out_sample_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """In-Sample / Out-of-Sample 2-way split"""
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])

    is_mask = (panel["timestamp"] >= in_sample_start) & (panel["timestamp"] <= in_sample_end)
    oos_mask = (panel["timestamp"] >= out_sample_start) & (panel["timestamp"] <= out_sample_end)

    is_panel = panel[is_mask].copy().reset_index(drop=True)
    oos_panel = panel[oos_mask].copy().reset_index(drop=True)

    return is_panel, oos_panel


# ════════════════════════════════════════════════════════════════════════════
# Signal Generation (Per-Formula Threshold Support)
# ════════════════════════════════════════════════════════════════════════════

def _compute_thresholds(
    *,
    train_panel: pd.DataFrame,
    passed_formulas: List[Dict[str, Any]],
    threshold_dict: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-formula per-ticker quantile thresholds on the training panel.
    """
    thresholds = {}

    for formula in passed_formulas:
        if not isinstance(formula, dict):
            continue
        name = str(formula.get("name") or "").strip()
        if not name or name not in train_panel.columns:
            continue
        if name not in threshold_dict:
            continue

        threshold = float(threshold_dict[name])
        if not (0 < threshold < 1):
            raise ValueError(f"threshold for {name} must be in (0, 1); got {threshold!r}")

        polarity = str(formula.get("polarity") or "higher_is_more_true").strip().lower()

        if polarity in ("lower_is_more_true", "lower", "inverse"):
            q = 1 - threshold
        else:
            q = threshold

        # Compute per-ticker quantiles on the training window.
        thr_by_ticker = train_panel.groupby("ticker")[name].quantile(q).to_dict()
        thresholds[name] = {
            "values": thr_by_ticker,
            "polarity": polarity,
            "q": q,
        }

    return thresholds

def _compute_thresholds_cached(
    *,
    train_panel: pd.DataFrame,
    passed_formulas: List[Dict[str, Any]],
    threshold_dict: Dict[str, float],
    cache: Dict[tuple, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    """
    Same as `_compute_thresholds`, but caches `(formula_name, q) -> thr_by_ticker`.

    cache key: (name, q)
    cache value: {ticker: threshold_value}
    """
    thresholds: Dict[str, Dict[str, Any]] = {}

    for formula in passed_formulas:
        if not isinstance(formula, dict):
            continue
        name = str(formula.get("name") or "").strip()
        if not name or name not in train_panel.columns:
            continue
        if name not in threshold_dict:
            continue

        threshold = float(threshold_dict[name])
        if not (0 < threshold < 1):
            raise ValueError(f"threshold for {name} must be in (0, 1); got {threshold!r}")

        polarity = str(formula.get("polarity") or "higher_is_more_true").strip().lower()
        q = 1 - threshold if polarity in ("lower_is_more_true", "lower", "inverse") else threshold

        # ⚠️ Stabilize float keys (avoid tiny floating-point differences).
        q_key = round(q, 6)

        key = (name, q_key)
        thr_by_ticker = cache.get(key)
        if thr_by_ticker is None:
            thr_by_ticker = train_panel.groupby("ticker")[name].quantile(q).to_dict()
            cache[key] = thr_by_ticker

        thresholds[name] = {
            "values": thr_by_ticker,
            "polarity": polarity,
            "q": q,
        }

    return thresholds


def _apply_signal(
    *,
    panel: pd.DataFrame,
    thresholds: Dict[str, Dict[str, Any]],
    passed_formulas: List[Dict[str, Any]],
) -> pd.Series:
    """
    Generate a boolean signal by applying precomputed thresholds (no look-ahead bias).
    """
    signal = pd.Series(True, index=panel.index)

    for formula in passed_formulas:
        if not isinstance(formula, dict):
            continue
        name = str(formula.get("name") or "").strip()
        if not name or name not in panel.columns:
            continue
        if name not in thresholds:
            continue

        thr_info = thresholds[name]
        thr_by_ticker = thr_info["values"]
        polarity = thr_info["polarity"]

        values = panel[name]
        thr = panel["ticker"].map(thr_by_ticker)

        # If a ticker has no threshold value, force the condition to be False.
        thr = thr.fillna(np.inf if polarity not in ("lower_is_more_true", "lower", "inverse") else -np.inf)

        if polarity in ("lower_is_more_true", "lower", "inverse"):
            cond = values.le(thr)
        else:
            cond = values.ge(thr)

        signal = signal & cond.fillna(False)

    return signal


# ════════════════════════════════════════════════════════════════════════════
# Qlib Backtest Integration
# ════════════════════════════════════════════════════════════════════════════

def _run_qlib_backtest(
    *,
    panel: pd.DataFrame,
    signal: pd.Series,
    cfg: RDConfig,
    use_native_qlib: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, float], List[Any], List[Dict[str, Any]], pd.DataFrame, str]:
    """
    Run a Qlib-style backtest (native Qlib path with a legacy fallback).

    Parameters
    ----------
    panel : pd.DataFrame
        OHLCV + formula data
    signal : pd.Series
        Boolean signal (index: panel index)
    cfg : RDConfig
        Backtest configuration
    use_native_qlib : bool
        If True, use Qlib `backtest_daily()`; otherwise, use the legacy implementation.

    Returns:
        (returns_dict, metrics_dict, trade_records, position_records, report_df, backtest_mode)
        - report_df: Qlib-style report DataFrame (return, cost, bench, turnover columns)
          - return = without_cost (gross)
          - return - cost = with_cost (net)
        - backtest_mode: "qlib_native" or "qlib_legacy"
    """
    if use_native_qlib:
        # `_run_native_qlib_backtest` returns `backtest_mode` as well.
        return _run_native_qlib_backtest(
            panel=panel,
            signal=signal,
            cfg=cfg,
        )
    else:
        result = _run_legacy_backtest(
            panel=panel,
            signal=signal,
            cfg=cfg,
        )
        return (*result, "qlib_legacy")


def _run_native_qlib_backtest(
    *,
    panel: pd.DataFrame,
    signal: pd.Series,
    cfg: RDConfig,
) -> Tuple[Dict[str, Any], Dict[str, float], List[Any], List[Dict[str, Any]], pd.DataFrame, str]:
    """
    Backtest implementation that calls Qlib `backtest_daily()` directly.

    Depending on `cfg.stage4.native_strategy`:
    - "trigger_exit": TriggerExitStrategy (horizon_days + stop_loss)
    - "topk_dropout": Qlib TopkDropoutStrategy (select top-k instruments)

    Returns:
        (returns_dict, metrics, trade_records, position_records, report_df, backtest_mode)
        - backtest_mode: "qlib_native" (success) or "qlib_legacy" (fallback)
    """
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from run.pipeline.strategy import TriggerExitStrategy, TriggerExitConfig

    # Convert panel timestamps to datetime.
    panel = panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])

    # Convert signal to the Qlib format (MultiIndex: datetime, instrument).
    signal_df = panel[["timestamp", "ticker"]].copy()
    # Qlib expects float signals.
    signal_df["signal"] = signal.values

    # Build a MultiIndex Series (datetime, instrument).
    signal_series = signal_df.set_index(["timestamp", "ticker"])["signal"]
    signal_series.index.names = ["datetime", "instrument"]

    # Ensure float64 for `isnan` compatibility.
    signal_series = signal_series.astype(np.float64)

    # ═══════════════════════════════════════════════════════════════════════════
    # Shift signal by 1 day: use (T-1) signals to trade on day T.
    # With `deal_price="close"` (below), this corresponds to a (T-1) signal executed at day T close.
    # ═══════════════════════════════════════════════════════════════════════════
    signal_series = signal_series.groupby(level="instrument").shift(1).fillna(0.0).astype(np.float64)

    # Extract date range.
    start_time = panel["timestamp"].min()
    end_time = panel["timestamp"].max()

    # Select strategy.
    native_strategy = getattr(cfg.stage4, "native_strategy", "trigger_exit")

    if native_strategy == "topk_dropout":
        # Use Qlib TopkDropoutStrategy.
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

        strategy = TopkDropoutStrategy(
            signal=signal_series,
            topk=getattr(cfg.stage4, "topk", 50),
            n_drop=getattr(cfg.stage4, "n_drop", 5),
            hold_thresh=getattr(cfg.stage4, "hold_thresh", 1),
            only_tradable=True,
            forbid_all_trade_at_limit=True,
            risk_degree=0.95,
        )
    else:
        # Use TriggerExitStrategy (default).
        trigger_config = TriggerExitConfig(
            horizon_days=cfg.stage4.horizon_days,
            stop_loss_threshold=getattr(cfg.stage4, "stop_loss_threshold", -0.05),
        )

        strategy = TriggerExitStrategy(
            signal=signal_series,
            config=trigger_config,
            risk_degree=0.95,
        )

    # Exchange config.
    # Qlib convention: (T-1) signal -> executed at day T deal_price (close here).
    exchange_kwargs = {
        "freq": "day",
        "limit_threshold": cfg.qlib.limit_threshold,  # CN: 0.1 (10%), US: None
        "deal_price": "close",  # Execute at day T close.
        "open_cost": cfg.qlib.open_cost,
        "close_cost": cfg.qlib.close_cost,
        "min_cost": cfg.qlib.min_cost,
    }

    try:
        # Suppress qlib's NaN warnings during backtest
        qlib_logger = logging.getLogger("qlib")
        original_level = qlib_logger.level
        qlib_logger.setLevel(logging.ERROR)

        try:
            # Run Qlib `backtest_daily()`.
            report_df, positions = backtest_daily(
                start_time=start_time,
                end_time=end_time,
                strategy=strategy,
                account=cfg.qlib.init_cash,
                benchmark=cfg.qlib.benchmark,
                exchange_kwargs=exchange_kwargs,
            )
        finally:
            # Restore original log level
            qlib_logger.setLevel(original_level)

        # Process results.
        if report_df is not None and len(report_df) > 0 and "return" in report_df.columns:
            # ═══════════════════════════════════════════════════════════════
            # Qlib report_df columns:
            # - return: daily portfolio return (without_cost / gross)
            # - cost: transaction costs
            # - return - cost: daily portfolio return (with_cost / net)
            # - bench: benchmark return
            # - turnover: turnover
            # ═══════════════════════════════════════════════════════════════
            daily_returns = report_df["return"]  # without_cost (gross)
            daily_cost = report_df.get("cost", pd.Series(0.0, index=daily_returns.index))
            daily_bench = report_df.get("bench", pd.Series(0.0, index=daily_returns.index))

            # Debug: Check if benchmark is calculated
            if "bench" not in report_df.columns:
                logger.warning("Benchmark column missing in report_df! Using zeros.")
            elif daily_bench.abs().sum() < 1e-6:
                logger.warning(f"Benchmark appears to be all zeros! Sum: {daily_bench.sum():.6f}")
            else:
                logger.debug(f"Benchmark calculated: mean={daily_bench.mean():.6f}, sum={daily_bench.sum():.6f}")

            # Qlib-style excess return:
            # excess_return_without_cost = return - bench
            # excess_return_with_cost = return - bench - cost
            #
            # Terminology:
            # - return = without_cost (gross)
            # - return - cost = with_cost (net)
            daily_returns_gross = daily_returns              # without_cost (gross)
            daily_returns_net = daily_returns - daily_cost   # with_cost (net)

            # Compute risk metrics on with_cost / net returns.
            risk_metrics = risk_analysis(daily_returns_net, freq="day")

            # Extract holdings and record per-instrument positions (if available).
            # positions is typically a dict[datetime, Position].
            holdings_series = pd.Series(0, index=daily_returns_net.index, dtype=int)
            position_records = []  # Per-day per-instrument position details.

            if positions:
                for date, pos in positions.items():
                    # Normalize date (drop time).
                    date_normalized = pd.Timestamp(date).normalize()
                    if date_normalized in holdings_series.index:
                        # Count holdings from Position.
                        try:
                            stock_list = pos.get_stock_list()
                            holdings_series.loc[date_normalized] = len(stock_list)

                            # Record per-instrument position details.
                            for stock_id in stock_list:
                                amount = pos.get_stock_amount(stock_id)
                                price = pos.get_stock_price(stock_id)
                                value = amount * price if amount and price else 0.0

                                position_records.append({
                                    'date': date_normalized.strftime('%Y-%m-%d'),
                                    'ticker': stock_id,
                                    'amount': float(amount) if amount else 0.0,
                                    'price': float(price) if price else 0.0,
                                    'value': float(value),
                                })
                        except Exception:
                            # If Position is missing or fails, keep 0.
                            pass

            # Build returns_dict (gross = without_cost, net = with_cost).
            eq_gross = (1 + daily_returns_gross).cumprod()
            eq_net = (1 + daily_returns_net).cumprod()
            returns_dict = {
                "gross": daily_returns_gross,  # return = without_cost (gross)
                "net": daily_returns_net,       # return - cost = with_cost (net)
                "eq_gross": eq_gross,
                "eq_net": eq_net,
                "holdings": holdings_series,
                "turnover": report_df.get("turnover", pd.Series(0.0, index=daily_returns_net.index)),
                "cost": daily_cost,
                "bench": daily_bench,  # Benchmark daily returns.
            }

            metrics = {
                "n_days": len(daily_returns_net),
                "avg_holdings": float(returns_dict["holdings"].mean()),
                "gross_return": float(eq_gross.iloc[-1] - 1.0),
                "net_return": float(eq_net.iloc[-1] - 1.0),
                "mean_return": float(risk_metrics.loc["mean", "risk"]),  # Mean daily return.
                "ann_return": float(risk_metrics.loc["annualized_return", "risk"]),
                "ann_vol": float(risk_metrics.loc["std", "risk"] * np.sqrt(252)),
                # Use Qlib `risk_analysis`'s "information_ratio" as the standard key.
                "information_ratio": float(risk_metrics.loc["information_ratio", "risk"]),
                # Backward compatibility (legacy code/comments may still call this "sharpe").
                "sharpe": float(risk_metrics.loc["information_ratio", "risk"]),
                "max_drawdown": float(risk_metrics.loc["max_drawdown", "risk"]),
                "avg_turnover": float(returns_dict["turnover"].mean()),
            }

            # Trade records (if the strategy provides them).
            trade_records = strategy.trade_records if hasattr(strategy, "trade_records") else []

            return returns_dict, metrics, trade_records, position_records, report_df, "qlib_native"

    except Exception as e:
        logger.warning(f"Native Qlib backtest failed: {e}, falling back to legacy")
        print(f"[Warning] Native Qlib backtest failed: {e}, falling back to legacy")

    # Fallback to legacy (mark backtest_mode as "qlib_legacy").
    result = _run_legacy_backtest(panel=panel, signal=signal, cfg=cfg)
    return (*result, "qlib_legacy")


def _get_benchmark_returns(
    *,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    benchmark: str,
) -> pd.Series:
    """
    Fetch benchmark daily returns (same source as the native Qlib path).
    """
    try:
        import qlib
        from qlib.data import D

        # Fetch benchmark close prices.
        bench_df = D.features(
            instruments=[benchmark],
            fields=["$close"],
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )

        if bench_df.empty:
            return pd.Series(dtype=float)

        # Extract close series from MultiIndex.
        bench_close = bench_df["$close"].droplevel("instrument")
        bench_returns = bench_close.pct_change().fillna(0.0)
        return bench_returns

    except Exception as e:
        logger.warning(f"Failed to get benchmark returns: {e}")
        return pd.Series(dtype=float)


def _run_legacy_backtest(
    *,
    panel: pd.DataFrame,
    signal: pd.Series,
    cfg: RDConfig,
) -> Tuple[Dict[str, Any], Dict[str, float], List["TradeRecord"], List[Dict[str, Any]], pd.DataFrame]:
    """
    Legacy backtest implementation (intended to match the native Qlib path).
    """
    # Position simulation (trigger-exit + horizon-days logic).
    position, trades = _simulate_positions(
        panel=panel,
        signal=signal,
        cfg=cfg,
    )

    # Portfolio returns with Qlib-Exchange-style fee model.
    returns_dict = _calculate_portfolio_returns_qlib(
        panel=panel,
        position=position,
        cfg=cfg,
    )

    # Performance metrics via Qlib `risk_analysis`.
    metrics = _compute_metrics_qlib(returns_dict)

    # Benchmark returns (same as native path).
    start_time = pd.to_datetime(panel["timestamp"].min())
    end_time = pd.to_datetime(panel["timestamp"].max())
    bench_returns = _get_benchmark_returns(
        start_time=start_time,
        end_time=end_time,
        benchmark=cfg.qlib.benchmark,
    )

    # Align benchmark index with gross returns.
    if not bench_returns.empty:
        bench_returns = bench_returns.reindex(returns_dict["gross"].index).fillna(0.0)
    else:
        bench_returns = pd.Series(0.0, index=returns_dict["gross"].index)

    # Convert to Qlib-style report_df.
    # - return = without_cost (gross)
    # - return - cost = with_cost (net)
    report_df = pd.DataFrame({
        "return": returns_dict["gross"],  # without_cost (gross)
        "cost": returns_dict["cost"],
        "bench": bench_returns,  # Use the actual benchmark, same as native.
        "turnover": returns_dict["turnover"],
    })

    return returns_dict, metrics, trades, [], report_df


def _calculate_portfolio_returns_qlib(
    *,
    panel: pd.DataFrame,
    position: pd.Series,
    cfg: RDConfig,
) -> Dict[str, Any]:
    """
    Qlib-style portfolio return calculation (consistent with the native path).

    Conventions:
    - `signal` is already shifted by 1 day in `_simulate_positions`.
    - `position[i] = True` means the portfolio holds the position on day i.
    - Uses close-to-close returns.
    - Applies Qlib Exchange-style fee model.
    """
    panel = panel.copy()

    # Close-to-close return: day T close -> day T+1 close.
    panel["ret1"] = panel.groupby("ticker")["close"].shift(-1) / panel["close"] - 1.0

    # No position shift needed: `position[i]` already represents holding on day i.
    valid = position & panel["ret1"].notna()

    denom = valid.groupby(panel["timestamp"]).transform("sum")
    weight = (valid.astype(float) / denom.replace(0, np.nan)).fillna(0.0)

    gross = (weight * panel["ret1"].fillna(0.0)).groupby(panel["timestamp"]).sum()
    holdings = valid.groupby(panel["timestamp"]).sum().astype(int)

    # Turnover calculation.
    wdf = panel.loc[:, ["timestamp", "ticker"]].copy()
    wdf["weight"] = weight.values
    wdf = wdf.sort_values(["ticker", "timestamp"], kind="mergesort")
    wdf["prev_weight"] = wdf.groupby("ticker")["weight"].shift(1).fillna(0.0)
    wdf["weight_change"] = wdf["weight"] - wdf["prev_weight"]

    # Split buy and sell turnover (Qlib style).
    wdf["buy_turnover"] = wdf["weight_change"].clip(lower=0)  # Positive change = buy.
    wdf["sell_turnover"] = (-wdf["weight_change"]).clip(lower=0)  # Negative change = sell.

    buy_turnover = wdf.groupby("timestamp")["buy_turnover"].sum()
    sell_turnover = wdf.groupby("timestamp")["sell_turnover"].sum()
    turnover = 0.5 * (buy_turnover + sell_turnover)

    if not turnover.empty:
        turnover.iloc[0] = 0.0

    # Qlib-style fees: open_cost for buys, close_cost for sells.
    cost = (buy_turnover * cfg.qlib.open_cost + sell_turnover * cfg.qlib.close_cost).reindex(gross.index).fillna(0.0)
    net = gross - cost

    eq_gross = (1.0 + gross).cumprod()
    eq_net = (1.0 + net).cumprod()

    return {
        "gross": gross,
        "net": net,
        "eq_gross": eq_gross,
        "eq_net": eq_net,
        "holdings": holdings,
        "turnover": turnover,
        "cost": cost,
    }


def _compute_metrics_qlib(returns: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute performance metrics via Qlib `risk_analysis`.
    """
    from qlib.contrib.evaluate import risk_analysis

    net = returns["net"]
    eq_net = returns["eq_net"]
    holdings = returns["holdings"]
    turnover = returns["turnover"]
    eq_gross = returns["eq_gross"]

    n_days = int(net.shape[0])

    if n_days > 1:
        # Qlib risk_analysis.
        risk_metrics = risk_analysis(net, freq="day")
        mean_ret = float(risk_metrics.loc["mean", "risk"])
        ann_ret = float(risk_metrics.loc["annualized_return", "risk"])
        ann_vol = float(risk_metrics.loc["std", "risk"] * np.sqrt(252))
        information_ratio = float(risk_metrics.loc["information_ratio", "risk"])
        mdd = float(risk_metrics.loc["max_drawdown", "risk"])
    else:
        mean_ret = 0.0
        ann_ret = 0.0
        ann_vol = 0.0
        information_ratio = 0.0
        mdd = 0.0

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


# ════════════════════════════════════════════════════════════════════════════
# Legacy Backtest (keep the previous implementation)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Single trade record."""
    ticker: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str  # "trigger" or "time_stop"
    holding_days: int
    return_pct: float


def _simulate_positions(
    *,
    panel: pd.DataFrame,
    signal: pd.Series,
    cfg: RDConfig,
) -> Tuple[pd.Series, List[TradeRecord]]:
    """
    Event-driven long-only position simulation per ticker.

    Conventions:
    - Shift signal by 1 day: (T-1) signal -> enter on day T.
    - Use the close price for entry/exit.
    """
    horizon_days = cfg.stage4.horizon_days
    lookback_window = cfg.stage4.lookback_window
    ref_price_fn = str(cfg.stage4.ref_price_fn).strip().lower()
    trigger_price_field = str(cfg.stage4.trigger_price_field).strip().lower()
    trigger_op = str(cfg.stage4.trigger_op).strip().lower()
    trigger_kmin = cfg.stage4.trigger_kmin
    trigger_kmax = cfg.stage4.trigger_kmax if cfg.stage4.trigger_kmax is not None else horizon_days
    stop_loss_threshold = getattr(cfg.stage4, "stop_loss_threshold", -0.05)

    # Shift signal by 1 day: (T-1) signal -> enter on day T.
    signal_shifted = signal.groupby(panel["ticker"]).shift(1).fillna(False).astype(bool)

    pos = pd.Series(False, index=panel.index)
    trades: List[TradeRecord] = []

    timestamps = panel["timestamp"].astype(str).tolist()

    for ticker, idx in panel.groupby("ticker", sort=False).groups.items():
        loc = pd.Index(idx)
        # Use shifted signal: `sig[i]` corresponds to the original signal at i-1.
        sig = signal_shifted.loc[loc].astype(bool).to_numpy()
        closes = panel.loc[loc, "close"].astype(float).to_numpy()  # Use close prices.
        highs = panel.loc[loc, "high"].astype(float).to_numpy()
        lows = panel.loc[loc, "low"].astype(float).to_numpy()
        ts = [timestamps[i] for i in loc]

        in_pos = False
        entry_i = -1
        entry_price = np.nan
        pref = np.nan

        for i in range(closes.shape[0]):
            exited_today = False

            if in_pos:
                k = i - entry_i

                # 0) Stop-loss (highest priority). Uses close prices.
                # k = i - entry_i is the holding day count (0 = entry day).
                # Start checking from the day after entry (k >= 1).
                # If close is NaN (suspended/untradable), skip stop-loss checks.
                if entry_price > 0 and k >= 1 and not np.isnan(closes[i]):
                    cumulative_return = (closes[i] / entry_price) - 1.0
                    if stop_loss_threshold is not None and cumulative_return <= float(stop_loss_threshold):
                        exit_price = closes[i]
                        ret = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                        trades.append(TradeRecord(
                            ticker=str(ticker),
                            entry_date=ts[entry_i],  # Actual entry date.
                            entry_price=entry_price,
                            exit_date=ts[i],
                            exit_price=exit_price,
                            exit_reason="stop_loss",
                            holding_days=k,  # Actual holding days.
                            return_pct=ret,
                        ))
                        in_pos = False
                        entry_i = -1
                        entry_price = np.nan
                        pref = np.nan
                        exited_today = True

                # # 1) Trigger exit
                # if (not exited_today) and (not np.isnan(pref)) and (k >= trigger_kmin) and (k <= trigger_kmax):
                #     if trigger_price_field == "high":
                #         obs = highs[i]
                #     elif trigger_price_field == "low":
                #         obs = lows[i]
                #     else:
                #         obs = closes[i]

                #     triggered = (obs >= pref) if trigger_op == "gte" else (obs <= pref)
                #     if triggered:
                #         exit_price = closes[i]
                #         ret = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                #         trades.append(TradeRecord(
                #             ticker=str(ticker),
                #             entry_date=ts[entry_i],
                #             entry_price=entry_price,
                #             exit_date=ts[i],
                #             exit_price=exit_price,
                #             exit_reason="trigger",
                #             holding_days=k,
                #             return_pct=ret,
                #         ))
                #         in_pos = False
                #         entry_i = -1
                #         entry_price = np.nan
                #         pref = np.nan
                #         exited_today = True

                # 2) Time stop. Uses close prices.
                # With shifted signals, `entry_i` is the actual entry day index in this panel.
                # k = i - entry_i is the holding day count (0 = entry day).
                # Exit when k >= horizon_days.
                # If close is NaN (suspended/untradable), keep holding.
                if (not exited_today) and (k >= horizon_days):
                    if not np.isnan(closes[i]):  # Exit only when tradable.
                        exit_price = closes[i]
                        ret = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                        trades.append(TradeRecord(
                            ticker=str(ticker),
                            entry_date=ts[entry_i],  # Actual entry date.
                            entry_price=entry_price,
                            exit_date=ts[i],
                            exit_price=exit_price,
                            exit_reason="time_stop",
                            holding_days=k,  # Actual holding days.
                            return_pct=ret,
                        ))
                        in_pos = False
                        entry_i = -1
                        entry_price = np.nan
                        pref = np.nan
                        exited_today = True

            # Entry: signal is already shifted by 1 day (sig[i] = original_signal[i-1]).
            # If sig[i] is True, enter on day i using the close price.
            if (not in_pos) and (not exited_today) and sig[i]:
                if not np.isnan(closes[i]):
                    in_pos = True
                    entry_i = i  # Actual entry day index.
                    entry_price = closes[i]  # Entry close price.

                # Compute P_ref.
                start = max(0, i - lookback_window)
                end = i
                if end <= start:
                    pref = np.nan
                else:
                    if ref_price_fn == "max_high":
                        pref = float(np.nanmax(highs[start:end]))
                    elif ref_price_fn == "min_low":
                        pref = float(np.nanmin(lows[start:end]))
                    elif ref_price_fn == "q50_close":
                        pref = float(np.nanquantile(closes[start:end], 0.5))
                    else:
                        raise ValueError(
                            f"Unsupported ref_price_fn={ref_price_fn!r}; "
                            "use one of: max_high, min_low, q50_close"
                        )

            pos.loc[loc[i]] = in_pos

    return pos, trades


def _calculate_portfolio_returns(
    *,
    panel: pd.DataFrame,
    position: pd.Series,
    cfg: RDConfig,
) -> Dict[str, Any]:
    """Portfolio return calculation (equal-weight) with a Qlib-style fee model."""
    panel = panel.copy()
    panel["ret1"] = panel.groupby("ticker")["close"].shift(-1) / panel["close"] - 1.0

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
    wdf["abs_diff"] = (wdf["weight"] - wdf["prev_weight"]).abs()
    turnover = 0.5 * wdf.groupby("timestamp")["abs_diff"].sum()
    if not turnover.empty:
        turnover.iloc[0] = 0.0

    # Qlib-style cost: open_cost for buys, close_cost for sells
    # Simplified: use average of open and close cost
    avg_cost = (cfg.qlib.open_cost + cfg.qlib.close_cost) / 2
    cost = turnover * avg_cost
    net = gross - cost.reindex(gross.index).fillna(0.0)

    eq_gross = (1.0 + gross).cumprod()
    eq_net = (1.0 + net).cumprod()

    return {
        "gross": gross,
        "net": net,
        "eq_gross": eq_gross,
        "eq_net": eq_net,
        "holdings": holdings,
        "turnover": turnover,
        "cost": cost,
    }


def _load_benchmark_index(
    *,
    start_date: str,
    end_date: str,
    benchmark_code: str = "SH000905",
) -> pd.Series:
    """
    Load benchmark index returns from Qlib

    Args:
        start_date: Start date
        end_date: End date
        benchmark_code: Benchmark index code
            - CN: "SH000905" (CSI500)
            - US: "GSPC" (S&P 500; "^GSPC" is also accepted)

    Returns:
        Daily returns series
    """
    from qlib.data import D

    # Qlib instrument keys are usually normalized (e.g. "GSPC" instead of "^GSPC").
    benchmark_code_norm = benchmark_code[1:] if benchmark_code.startswith("^") else benchmark_code

    result = D.features(
        [benchmark_code_norm],
        ["$close/Ref($close, 1)-1"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    if (result is None or len(result) == 0) and benchmark_code_norm != benchmark_code:
        result = D.features(
            [benchmark_code],
            ["$close/Ref($close, 1)-1"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )

    if result is None or len(result) == 0:
        raise ValueError(f"Failed to load benchmark index data ({benchmark_code}) from {start_date} to {end_date}")

    code_level = result.index.get_level_values(0)
    code_in_result = benchmark_code_norm if benchmark_code_norm in code_level else benchmark_code

    index_returns = result.loc[code_in_result].iloc[:, 0]
    index_returns = index_returns.fillna(0.0)

    return index_returns


def _calculate_benchmark_returns(
    *,
    start_date: str,
    end_date: str,
    benchmark_code: str = "SH000905",
) -> Dict[str, Any]:
    """
    Calculate benchmark returns

    Args:
        start_date: Start date
        end_date: End date
        benchmark_code: Benchmark index code (SH000905 for CSI500, GSPC/^GSPC for S&P500)

    Returns:
        Dictionary with benchmark returns
    """
    index_returns = _load_benchmark_index(start_date=start_date, end_date=end_date, benchmark_code=benchmark_code)

    gross_returns = index_returns
    net_returns = index_returns

    eq_gross = (1.0 + gross_returns).cumprod()
    eq_net = (1.0 + net_returns).cumprod()

    holdings = pd.Series(0, index=index_returns.index)
    turnover = pd.Series(0.0, index=index_returns.index)
    cost = pd.Series(0.0, index=index_returns.index)

    return {
        "gross": gross_returns,
        "net": net_returns,
        "eq_gross": eq_gross,
        "eq_net": eq_net,
        "holdings": holdings,
        "turnover": turnover,
        "cost": cost,
    }


def _build_port_analysis_df(report_df: pd.DataFrame | None, *, freq: str = "1day") -> pd.DataFrame:
    """
    Build a Qlib-style `port_analysis_1day.pkl` DataFrame.

    This matches the structure produced by Qlib's `PortAnaRecord`:
      - analysis["excess_return_without_cost"] = risk_analysis(report["return"] - report["bench"], freq=...)
      - analysis["excess_return_with_cost"]    = risk_analysis(report["return"] - report["bench"] - report["cost"], freq=...)
      - analysis_df = pd.concat(analysis)

    Format: pd.DataFrame
        index: MultiIndex (analysis_name, metric_name)
        column: "risk"

    Example:
        >>> df.loc[("excess_return_without_cost", "information_ratio"), "risk"]
        1.234
    """
    from qlib.contrib.evaluate import risk_analysis

    if report_df is None or len(report_df) == 0:
        return pd.DataFrame(columns=["risk"])

    if "return" not in report_df.columns:
        return pd.DataFrame(columns=["risk"])

    daily_returns = report_df["return"]
    daily_cost = report_df["cost"] if "cost" in report_df.columns else pd.Series(0.0, index=daily_returns.index)
    daily_bench = report_df["bench"] if "bench" in report_df.columns else pd.Series(0.0, index=daily_returns.index)

    analysis: dict[str, pd.DataFrame] = {
        "excess_return_without_cost": risk_analysis(daily_returns - daily_bench, freq=freq),
        "excess_return_with_cost": risk_analysis(daily_returns - daily_bench - daily_cost, freq=freq),
    }
    return pd.concat(analysis)


def _calc_excess_metrics(excess_return: pd.Series) -> Dict[str, float]:
    """
    Compute Qlib-style excess-return metrics.

    Qlib convention:
        # Here, return = without_cost (gross)
        excess_return_without_cost = return - bench
        excess_return_with_cost = return - bench - cost  # = (return - cost) - bench = with_cost (net) - bench

    Usage:
        _calc_excess_metrics(report_df["return"] - report_df["bench"])
        _calc_excess_metrics(report_df["return"] - report_df["bench"] - report_df["cost"])
    """
    from qlib.contrib.evaluate import risk_analysis

    if len(excess_return) < 2:
        return {
            "information_ratio": 0.0,
            "mean_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "max_drawdown": 0.0,
            "net_return": 0.0,
        }

    metrics = risk_analysis(excess_return, freq="day")
    cumulative_return = float((1 + excess_return).cumprod().iloc[-1] - 1.0)

    return {
        "information_ratio": float(metrics.loc["information_ratio", "risk"]),
        "mean_return": float(metrics.loc["mean", "risk"]),
        "ann_return": float(metrics.loc["annualized_return", "risk"]),
        "ann_vol": float(metrics.loc["std", "risk"] * np.sqrt(252)),
        "max_drawdown": float(metrics.loc["max_drawdown", "risk"]),
        "net_return": cumulative_return,
    }


def _max_drawdown_pct_from_equity_curve(eq_curve: pd.Series) -> float:
    """
    Standard max drawdown as a percentage (bounded in [-1, 0]) from an equity curve.

    NOTE: Qlib's `risk_analysis` default `mode="sum"` computes max_drawdown on `r.cumsum()`,
    which is not a percentage drawdown and can be less than -100% when formatted as percent.
    For reporting, we want the conventional definition based on compounded equity.
    """
    if eq_curve is None or len(eq_curve) == 0:
        return 0.0
    eq = pd.to_numeric(eq_curve, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    v = dd.min()
    try:
        return float(v)
    except Exception:
        return 0.0


def _compute_metrics_from_returns(returns: Dict[str, Any]) -> Dict[str, float]:
    """Compute performance metrics from a returns dict (via Qlib `risk_analysis`)."""
    from qlib.contrib.evaluate import risk_analysis

    net = returns["net"]
    eq_net = returns["eq_net"]
    holdings = returns["holdings"]
    turnover = returns["turnover"]
    eq_gross = returns["eq_gross"]

    n_days = int(net.shape[0])

    if n_days > 1:
        # Qlib risk_analysis.
        risk_metrics = risk_analysis(net, freq="day")
        mean_ret = float(risk_metrics.loc["mean", "risk"])
        ann_ret = float(risk_metrics.loc["annualized_return", "risk"])
        ann_vol = float(risk_metrics.loc["std", "risk"] * np.sqrt(252))
        information_ratio = float(risk_metrics.loc["information_ratio", "risk"])
        mdd = float(risk_metrics.loc["max_drawdown", "risk"])
    else:
        mean_ret = 0.0
        ann_ret = 0.0
        ann_vol = 0.0
        information_ratio = 0.0
        mdd = 0.0

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

        # Metadata.
        row_dict[("meta", "combo_idx")] = combo.get("combo_idx")
        row_dict[("meta", "formula_names")] = "_".join(combo.get("formula_names", []))

        # In-sample and out-of-sample payloads.
        for sample_type in ["insample", "outsample"]:
            sample_data = combo.get(sample_type, {})

            # return metrics
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

    # Build DataFrame.
    df = pd.DataFrame(rows)

    # Sort MultiIndex columns (meta first).
    if not df.empty:
        meta_cols = [col for col in df.columns if col[0] == "meta"]
        other_cols = [col for col in df.columns if col[0] != "meta"]
        df = df[meta_cols + sorted(other_cols)]

    return df


def _compute_trade_metrics(trades: List[Any]) -> Dict[str, float]:
    """Compute trade-based metrics (supports TradeRecord or dict)."""
    if not trades:
        return {
            "n_trades": 0,
            "avg_trade_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
        }

    # Support both TradeRecord objects and dict entries.
    returns = []
    holding_days = []
    for t in trades:
        if isinstance(t, dict):
            returns.append(t.get('return_pct', 0.0))
            holding_days.append(t.get('holding_days', 0))
        else:
            returns.append(t.return_pct)
            holding_days.append(t.holding_days)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    win_rate = len(wins) / len(returns) if returns else 0.0

    total_win = sum(wins) if wins else 0.0
    total_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = total_win / total_loss if total_loss > 0 else (float('inf') if total_win > 0 else 0.0)

    return {
        "n_trades": len(trades),
        "avg_trade_return": float(np.mean(returns)) if returns else 0.0,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if profit_factor != float('inf') else 999.99,
        "avg_holding_days": float(np.mean(holding_days)) if holding_days else 0.0,
    }


# ════════════════════════════════════════════════════════════════════════════
# Sequential Combination Evaluation
# ════════════════════════════════════════════════════════════════════════════

def _evaluate_single_combination_sequential(
    combo_idx: int,
    combination: List[Dict[str, Any]],
    is_panel: pd.DataFrame,
    oos_panel: pd.DataFrame,
    cfg: RDConfig,
    is_benchmark_returns_for_optuna: pd.Series,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single combination with Optuna optimization + IS/OOS backtest (sequential version).

    Returns:
        Dict containing all evaluation results for this combination
    """
    import optuna

    # Suppress optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Extract formula names
    formula_names = [
        str(f.get("name") or "") for f in combination
        if isinstance(f, dict) and f.get("name")
    ]
    formula_names = [n for n in formula_names if n]

    if is_panel.empty or oos_panel.empty:
        return {"combo_idx": combo_idx, "error": "Empty panel", "skipped": True}

    # Compute common tickers between IS and OOS panels
    is_tickers = set(is_panel["ticker"].unique().tolist())
    oos_tickers = set(oos_panel["ticker"].unique().tolist())
    common_tickers = sorted(is_tickers & oos_tickers)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Optuna Optimization (IS)
    # ═══════════════════════════════════════════════════════════════
    study = None
    trials_df = None

    if cfg.stage4.enable_optuna:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42 + combo_idx,
                constant_liar=True,
            ),
        )

        objective = _create_objective(
            is_panel=is_panel,
            passed_formulas=combination,
            cfg=cfg,
            benchmark_returns=is_benchmark_returns_for_optuna,
            verbose=False,
        )

        # Keep Optuna `n_jobs=1` (callers may parallelize at the combination level).
        study.optimize(
            objective,
            n_trials=cfg.stage4.n_trials,
            n_jobs=1,  # Combination-level parallelism, so inner loop stays sequential.
            show_progress_bar=False,
            catch=(Exception,),
        )

        combo_is_information_ratio = study.best_value
        best_params = study.best_params

        # Extract optimal thresholds
        optimal_thresholds: Dict[str, float] = {}
        for fname in formula_names:
            key = f"threshold_{fname}"
            if key in best_params:
                optimal_thresholds[fname] = best_params[key]

        # Keep a lightweight representation for aggregation/saving in the parent process.
        trials_df = study.trials_dataframe()
    else:
        # Fixed threshold
        fixed_quantiles = list(getattr(cfg.stage4, "fixed_quantiles", [0.8]) or [0.8])
        default_threshold = fixed_quantiles[0] if fixed_quantiles else 0.8
        optimal_thresholds = {fname: default_threshold for fname in formula_names}
        combo_is_information_ratio = 0.0

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: IS/OOS Evaluation
    # ═══════════════════════════════════════════════════════════════
    final_cache: Dict[tuple, Dict[str, float]] = {}
    final_thresholds = _compute_thresholds_cached(
        train_panel=is_panel,
        passed_formulas=combination,
        threshold_dict=optimal_thresholds,
        cache=final_cache,
    )

    # In-Sample evaluation (SKIPPED for speed)
    # is_signal = _apply_signal(
    #     panel=is_panel,
    #     thresholds=final_thresholds,
    #     passed_formulas=combination,
    # )
    # is_returns, is_metrics, is_trades, is_positions, is_report_df, backtest_mode = _run_qlib_backtest(
    #     panel=is_panel,
    #     signal=is_signal,
    #     cfg=cfg,
    # )
    # is_trade_metrics = _compute_trade_metrics(is_trades)
    # Dummy values for IS (skipped)
    is_signal = None
    is_returns = {}
    is_metrics = {
        "mean_return": 0.0,
        "ann_return": 0.0,
        "ann_vol": 0.0,
        "information_ratio": 0.0,
        "max_drawdown": 0.0,
        "net_return": 0.0,
        "avg_holdings": 0.0,
        "avg_turnover": 0.0,
    }
    is_trades = []
    is_positions = []
    is_report_df = pd.DataFrame({"return": [0.0], "bench": [0.0], "cost": [0.0]})
    backtest_mode = "skipped"
    is_trade_metrics = {
        "n_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_trade_return": 0.0,
        "avg_holding_days": 0.0,
    }

    # Excess return metrics (IS) - dummy
    # is_excess_without_cost = _calc_excess_metrics(
    #     is_report_df["return"] - is_report_df["bench"]
    # )
    # is_excess_with_cost = _calc_excess_metrics(
    #     is_report_df["return"] - is_report_df["bench"] - is_report_df["cost"]
    # )
    _dummy_excess = {
        "mean_return": 0.0,
        "ann_return": 0.0,
        "ann_vol": 0.0,
        "information_ratio": 0.0,
        "max_drawdown": 0.0,
        "net_return": 0.0,
    }
    is_excess_without_cost = _dummy_excess.copy()
    is_excess_with_cost = _dummy_excess.copy()

    # Out-of-Sample evaluation
    oos_signal = _apply_signal(
        panel=oos_panel,
        thresholds=final_thresholds,
        passed_formulas=combination,
    )
    oos_returns, oos_metrics, oos_trades, oos_positions, oos_report_df, _ = _run_qlib_backtest(
        panel=oos_panel,
        signal=oos_signal,
        cfg=cfg,
    )
    oos_trade_metrics = _compute_trade_metrics(oos_trades)

    # Excess return metrics (OOS)
    oos_excess_without_cost = _calc_excess_metrics(
        oos_report_df["return"] - oos_report_df["bench"]
    )
    oos_excess_with_cost = _calc_excess_metrics(
        oos_report_df["return"] - oos_report_df["bench"] - oos_report_df["cost"]
    )

    # ═══════════════════════════════════════════════════════════════
    # Fixed-quantile evaluation modes (optional)
    # ═══════════════════════════════════════════════════════════════
    fixed_quantiles = list(getattr(cfg.stage4, "fixed_quantiles", []) or [])
    fixed_modes: Dict[str, Dict[str, Any]] = {}
    for q in fixed_quantiles:
        try:
            q = float(q)
        except Exception:
            continue
        if not (0.0 < q < 1.0):
            continue

        mode_name = f"fixed_q{int(round(q * 100)):02d}"
        if mode_name in fixed_modes:
            continue

        threshold_dict = {fname: q for fname in formula_names}
        try:
            fixed_thresholds = _compute_thresholds(
                train_panel=is_panel,
                passed_formulas=combination,
                threshold_dict=threshold_dict,
            )
        except Exception:
            continue

        if not fixed_thresholds:
            continue

        # IS (SKIPPED for speed)
        # is_signal_fixed = _apply_signal(
        #     panel=is_panel,
        #     thresholds=fixed_thresholds,
        #     passed_formulas=combination,
        # )
        # is_returns_f, is_metrics_f, is_trades_f, is_positions_f, is_report_df_f, _ = _run_qlib_backtest(
        #     panel=is_panel,
        #     signal=is_signal_fixed,
        #     cfg=cfg,
        # )
        # is_trade_metrics_f = _compute_trade_metrics(is_trades_f)
        # is_excess_wo_f = _calc_excess_metrics(is_report_df_f["return"] - is_report_df_f["bench"])
        # is_excess_w_f = _calc_excess_metrics(is_report_df_f["return"] - is_report_df_f["bench"] - is_report_df_f["cost"])
        # Dummy values for IS (skipped)
        is_metrics_f = {
            "mean_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "information_ratio": 0.0,
            "max_drawdown": 0.0,
            "net_return": 0.0,
            "avg_holdings": 0.0,
            "avg_turnover": 0.0,
        }
        is_trade_metrics_f = {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_return": 0.0,
            "avg_holding_days": 0.0,
        }
        is_report_df_f = pd.DataFrame({"return": [0.0], "bench": [0.0], "cost": [0.0]})
        _dummy_excess_f = {
            "mean_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "information_ratio": 0.0,
            "max_drawdown": 0.0,
            "net_return": 0.0,
        }
        is_excess_wo_f = _dummy_excess_f.copy()
        is_excess_w_f = _dummy_excess_f.copy()

        # OOS
        oos_signal_fixed = _apply_signal(
            panel=oos_panel,
            thresholds=fixed_thresholds,
            passed_formulas=combination,
        )
        oos_returns_f, oos_metrics_f, oos_trades_f, oos_positions_f, oos_report_df_f, _ = _run_qlib_backtest(
            panel=oos_panel,
            signal=oos_signal_fixed,
            cfg=cfg,
        )
        oos_trade_metrics_f = _compute_trade_metrics(oos_trades_f)
        oos_excess_wo_f = _calc_excess_metrics(oos_report_df_f["return"] - oos_report_df_f["bench"])
        oos_excess_w_f = _calc_excess_metrics(oos_report_df_f["return"] - oos_report_df_f["bench"] - oos_report_df_f["cost"])

        fixed_modes[mode_name] = {
            "threshold_q": q,
            "threshold_dict": threshold_dict,
            "final_thresholds": fixed_thresholds,
            "is_metrics": is_metrics_f,
            "oos_metrics": oos_metrics_f,
            "is_excess_without_cost": is_excess_wo_f,
            "is_excess_with_cost": is_excess_w_f,
            "oos_excess_without_cost": oos_excess_wo_f,
            "oos_excess_with_cost": oos_excess_w_f,
            "is_trade_metrics": is_trade_metrics_f,
            "oos_trade_metrics": oos_trade_metrics_f,
            "is_report_df": is_report_df_f.to_dict() if is_report_df_f is not None else {},
            "oos_report_df": oos_report_df_f.to_dict() if oos_report_df_f is not None else {},
        }

    # Return all results
    return {
        "combo_idx": combo_idx,
        "combination": combination,
        "formula_names": formula_names,
        "common_tickers": common_tickers,
        "is_information_ratio": combo_is_information_ratio,
        "optimal_thresholds": optimal_thresholds,
        "final_thresholds": final_thresholds,
        "backtest_mode": backtest_mode,
        "is_metrics": is_metrics,
        "is_excess_without_cost": is_excess_without_cost,
        "is_excess_with_cost": is_excess_with_cost,
        "oos_information_ratio": oos_metrics.get("information_ratio", -999),
        "oos_metrics": oos_metrics,
        "oos_excess_without_cost": oos_excess_without_cost,
        "oos_excess_with_cost": oos_excess_with_cost,
        "is_trade_metrics": is_trade_metrics,
        "oos_trade_metrics": oos_trade_metrics,
        "oos_trades": oos_trades,
        "oos_positions": oos_positions,
        "is_positions": is_positions,
        "is_report_df": is_report_df,
        "oos_report_df": oos_report_df,
        "fixed_modes": fixed_modes if fixed_modes else None,
        # NOTE: Do not return the Study object itself (can be heavy / not reliably picklable under spawn).
        "study": None,
        "trials_df": trials_df,
        "skipped": False,
    }


# ════════════════════════════════════════════════════════════════════════════
# Optuna Optimization
# ════════════════════════════════════════════════════════════════════════════

def _create_objective(
    *,
    is_panel: pd.DataFrame,
    passed_formulas: List[Dict[str, Any]],
    cfg: RDConfig,
    benchmark_returns: pd.Series = None,
    verbose: bool = False,
):
    """Create an Optuna objective (optimize in-sample excess information ratio vs benchmark)."""
    formula_names = [f["name"] for f in passed_formulas if "name" in f]

    quantile_cache: Dict[tuple, Dict[str, float]] = {}  # ✅ Per-study (per-combination) cache.

    # Precompute structures for fast objective evaluation (avoid Python loops over all rows per trial).
    # Note: This fast path matches current _simulate_positions behavior where trigger-exit is disabled
    # (only stop-loss + time-stop). If trigger-exit is re-enabled later, this needs to be revisited.
    horizon_days = int(getattr(cfg.stage4, "horizon_days", 5))
    stop_loss_threshold = getattr(cfg.stage4, "stop_loss_threshold", -0.05)
    avg_cost = float((cfg.qlib.open_cost + cfg.qlib.close_cost) / 2)

    date_codes, unique_dates = pd.factorize(is_panel["timestamp"], sort=True)
    n_dates = int(len(unique_dates))

    closes_all = is_panel["close"].astype(float).to_numpy()
    tick_groups: list[np.ndarray] = []
    tick_closes: list[np.ndarray] = []

    groups = is_panel.groupby("ticker", sort=False).groups
    for idx in groups.values():
        loc = np.asarray(idx, dtype=np.int64)
        tick_groups.append(loc)
        tick_closes.append(closes_all[loc])

    ret1 = np.full(int(is_panel.shape[0]), np.nan, dtype=float)
    for loc, c in zip(tick_groups, tick_closes, strict=True):
        if c.shape[0] >= 2:
            ret1[loc[:-1]] = (c[1:] / c[:-1]) - 1.0

    def _simulate_positions_fast_from_signal(signal_arr: np.ndarray) -> np.ndarray:
        pos = np.zeros(signal_arr.shape[0], dtype=bool)
        for loc, c in zip(tick_groups, tick_closes, strict=True):
            sig_t = signal_arr[loc]
            if not sig_t.any():
                continue

            n = int(sig_t.shape[0])
            pos_t = np.zeros(n, dtype=bool)
            entries = np.flatnonzero(sig_t)
            if entries.size == 0:
                continue

            p = 0
            last_exit = -1
            while p < entries.size:
                e = int(entries[p])
                if e <= last_exit:
                    p += 1
                    continue

                exit_idx = min(e + horizon_days, n)

                if stop_loss_threshold is not None:
                    entry_price = float(c[e])
                    end = min(e + horizon_days, n - 1)
                    if entry_price > 0.0 and (e + 1) <= end:
                        future = (c[(e + 1):(end + 1)] / entry_price) - 1.0
                        bad = np.flatnonzero(future <= float(stop_loss_threshold))
                        if bad.size:
                            exit_idx = (e + 1) + int(bad[0])

                if exit_idx > e:
                    pos_t[e:exit_idx] = True
                last_exit = exit_idx

                p = int(np.searchsorted(entries, last_exit + 1, side="left"))

            pos[loc] = pos_t
        return pos

    def _calculate_portfolio_returns_fast(position: np.ndarray) -> Dict[str, Any]:
        valid = position & np.isfinite(ret1)
        if not valid.any():
            empty_idx = pd.Index(unique_dates)
            z = pd.Series(0.0, index=empty_idx)
            return {
                "gross": z,
                "net": z,
                "eq_gross": (1.0 + z).cumprod(),
                "eq_net": (1.0 + z).cumprod(),
                "holdings": pd.Series(0, index=empty_idx, dtype=int),
                "turnover": z,
                "cost": z,
            }

        holdings = np.bincount(date_codes, weights=valid.astype(np.int8), minlength=n_dates).astype(int)
        denom = holdings[date_codes]

        weight = np.zeros_like(ret1)
        mask = valid & (denom > 0)
        weight[mask] = 1.0 / denom[mask]

        ret1_safe = np.nan_to_num(ret1, nan=0.0)
        gross_by_date = np.bincount(date_codes, weights=weight * ret1_safe, minlength=n_dates).astype(float)

        absdiff = np.zeros_like(weight)
        for loc in tick_groups:
            w = weight[loc]
            if w.size:
                prev = np.empty_like(w)
                prev[0] = 0.0
                prev[1:] = w[:-1]
                absdiff[loc] = np.abs(w - prev)

        turnover_by_date = 0.5 * np.bincount(date_codes, weights=absdiff, minlength=n_dates).astype(float)
        if turnover_by_date.size:
            turnover_by_date[0] = 0.0

        cost_by_date = turnover_by_date * avg_cost

        idx = pd.Index(unique_dates)
        gross = pd.Series(gross_by_date, index=idx)
        cost = pd.Series(cost_by_date, index=idx)
        net = gross - cost

        return {
            "gross": gross,
            "net": net,
            "eq_gross": (1.0 + gross).cumprod(),
            "eq_net": (1.0 + net).cumprod(),
            "holdings": pd.Series(holdings, index=idx, dtype=int),
            "turnover": pd.Series(turnover_by_date, index=idx),
            "cost": cost,
        }

    def objective(trial: Any) -> float:
        # Sample per-formula thresholds.
        # Use suggest_categorical to avoid floating-point key issues in Optuna.
        step = getattr(cfg.stage4, "threshold_step", 0.05)
        threshold_choices = [
            round(cfg.stage4.threshold_min + i * step, 2)
            for i in range(int((cfg.stage4.threshold_max - cfg.stage4.threshold_min) / step) + 1)
        ]

        threshold_dict = {}
        for fname in formula_names:
            threshold_dict[fname] = trial.suggest_categorical(
                f"threshold_{fname}",
                threshold_choices
            )

        # 1) Compute thresholds on the in-sample window.
        thresholds = _compute_thresholds_cached(
            train_panel=is_panel,
            passed_formulas=passed_formulas,
            threshold_dict=threshold_dict,
            cache=quantile_cache,
        )

        # 2) Apply thresholds and evaluate on the in-sample window.
        is_signal = _apply_signal(
            panel=is_panel,
            thresholds=thresholds,
            passed_formulas=passed_formulas,
        )

        is_position = _simulate_positions_fast_from_signal(is_signal.to_numpy(dtype=bool))
        is_returns = _calculate_portfolio_returns_fast(is_position)

        # 3) Excess information ratio vs. benchmark.
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align portfolio gross returns and benchmark returns on the common index.
            portfolio_gross = is_returns["gross"]
            common_idx = portfolio_gross.index.intersection(benchmark_returns.index)
            if len(common_idx) > 1:
                excess_return = portfolio_gross.loc[common_idx] - benchmark_returns.loc[common_idx]
                excess_metrics = _calc_excess_metrics(excess_return)
                excess_ir = excess_metrics["information_ratio"]
            else:
                # If no common index, fall back to portfolio information_ratio.
                is_metrics = _compute_metrics_from_returns(is_returns)
                excess_ir = is_metrics["information_ratio"]
        else:
            # If no benchmark is provided, use portfolio information_ratio.
            is_metrics = _compute_metrics_from_returns(is_returns)
            excess_ir = is_metrics["information_ratio"]

        # Guard against nan/inf (prevents Optuna from repeatedly retrying identical params).
        if np.isnan(excess_ir) or np.isinf(excess_ir):
            excess_ir = -999.0  # Replace with a very low value.

        if verbose and trial.number % 10 == 0:
            print(f"  Trial {trial.number}: IS Excess IR={excess_ir:.3f}")

        return excess_ir

    return objective


# ════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_stage4(
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
    Run Stage 4 (Qlib-integrated backtest + optional Optuna optimization).

    Args:
        passed_combinations: combinations that passed Stage 3 (List[List[Dict]]).
            Each combination is a list of formula dicts, e.g. [[f1, f2], [f3, f4], ...].
        cfg: RDConfig (includes stage4, qlib, data_split settings)
    """
    # Optuna import
    import optuna

    cfg = cfg or load_rd_config()

    # Override horizon_days from hypothesis (if provided).
    if hypothesis:
        hyp_list = hypothesis.get("hypotheses", [])
        hyp_obj = hyp_list[0] if isinstance(hyp_list, list) and hyp_list else hypothesis
        h = (hyp_obj or {}).get("horizon_days")
        if isinstance(h, int) and h > 0:
            cfg.stage4.horizon_days = h

    if cfg.stage4.trigger_kmax is None:
        cfg.stage4.trigger_kmax = cfg.stage4.horizon_days

    if verbose:
        print(f"[Stage4] Starting Qlib-based backtest for hypothesis: {hypothesis_id}")
        print(f"[Stage4] Validation (Optuna): {cfg.data_split.val_start} ~ {cfg.data_split.val_end}")
        print(f"[Stage4] Test (Final Eval):   {cfg.data_split.test_start} ~ {cfg.data_split.test_end}")
        print(f"[Stage4] Combinations: {len(passed_combinations)}, Horizon: {cfg.stage4.horizon_days}d")
        if cfg.stage4.enable_optuna:
            print(f"[Stage4] Optuna trials: {cfg.stage4.n_trials}")
        else:
            fixed_quantiles = list(getattr(cfg.stage4, "fixed_quantiles", [0.8]) or [0.8])
            default_threshold = fixed_quantiles[0] if fixed_quantiles else 0.8
            print(f"[Stage4] Optuna: DISABLED (using fixed threshold: {default_threshold:.2f})")
        print(f"[Stage4] Transaction costs: open={cfg.qlib.open_cost:.4%}, close={cfg.qlib.close_cost:.4%}")

    # Empty case
    if not passed_combinations:
        empty = pl.DataFrame({
            "timestamp": [], "gross_return": [], "net_return": []
        })
        return Stage4Result(
            hypothesis_id=hypothesis_id,
            config={},  # Empty config dict for empty case
            summary={
                "hypothesis_id": hypothesis_id,
                "error": "No passed_combinations; skip backtest.",
            },
            report_md="# Stage 4: Backtest\n\nNo passed combinations. Skipped.",
            is_daily_panel=empty,
            oos_daily_panel=empty,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Combination selection: evaluate top-N or evaluate all
    # ═══════════════════════════════════════════════════════════════════════
    max_combinations_to_evaluate = cfg.stage4.max_combinations_to_evaluate  # default: 10
    selection_criterion = cfg.stage4.combination_selection_criterion  # default: "s2_improvement"

    # Use `combination_stats` forwarded from Stage 3 when available.
    combination_stats = _kwargs.get("combination_stats", {})

    # Select combinations to evaluate.
    if max_combinations_to_evaluate <= 0 or max_combinations_to_evaluate >= len(passed_combinations):
        # Evaluate all combinations.
        combinations_to_evaluate = passed_combinations
    else:
        if combination_stats:
            # Extract per-combination sort values.
            combo_with_stats = []
            for combo in passed_combinations:
                combo_key = tuple(sorted(f["name"] for f in combo))
                stats = combination_stats.get(combo_key, {})

                # Choose sort key based on criterion.
                if selection_criterion == "s2_improvement":
                    sort_value = stats.get("s2_ratio_improvement", 0.0)
                elif selection_criterion == "mean_return":
                    # Use the last value in mean_returns_by_level (strictest level).
                    mean_returns = stats.get("mean_returns_by_level", [])
                    sort_value = mean_returns[-1] if mean_returns else 0.0
                elif selection_criterion in ("sharpe", "information_ratio", "ir"):
                    # Stage 3 doesn't provide Sharpe/IR directly; use mean_return_improvement as a proxy.
                    sort_value = stats.get("mean_return_improvement", 0.0)
                elif selection_criterion == "pass_rate":
                    sort_value = stats.get("pass_rate", 0.0)
                else:
                    # Default: s2_improvement.
                    sort_value = stats.get("s2_ratio_improvement", 0.0)

                combo_with_stats.append((combo, sort_value, stats))

            # Sort descending and take top-N.
            combo_with_stats.sort(key=lambda x: x[1], reverse=True)
            combinations_to_evaluate = [combo for combo, _, _ in combo_with_stats[:max_combinations_to_evaluate]]

            if verbose:
                print(f"[Stage4] Top {max_combinations_to_evaluate} combinations by {selection_criterion}:")
                for i, (combo, val, _) in enumerate(combo_with_stats[:max_combinations_to_evaluate], 1):
                    combo_names = [f.get("name", "?") for f in combo]
                    print(f"  {i}. {combo_names} ({selection_criterion}: {val:+.4f})")
        else:
            # If no combination_stats, fall back to the first N combinations.
            combinations_to_evaluate = passed_combinations[:max_combinations_to_evaluate]

    if verbose:
        print(f"[Stage4] Evaluating {len(combinations_to_evaluate)} / {len(passed_combinations)} combinations (criterion: {selection_criterion})")

    # ═══════════════════════════════════════════════════════════════════════
    # Prepare base panel once (avoid per-combination merge/split)
    # ═══════════════════════════════════════════════════════════════════════
    all_formula_names: List[str] = []
    seen = set()
    for combo in combinations_to_evaluate:
        for f in combo:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                all_formula_names.append(name)

    panel = _prepare_panel(
        ohlcv_df=ohlcv_df,
        formula_df=formula_df,
        formula_names=all_formula_names,
    )

    # 2-way split: Validation (Optuna) / Test (Final Eval)
    val_panel, test_panel = _split_in_out_sample(
        panel,
        cfg.data_split.val_start, cfg.data_split.val_end,
        cfg.data_split.test_start, cfg.data_split.test_end,
    )
    is_panel, oos_panel = val_panel, test_panel

    # Fix the ticker universe to the IS/OOS intersection once.
    is_tickers = set(is_panel["ticker"].unique().tolist())
    oos_tickers = set(oos_panel["ticker"].unique().tolist())
    common_tickers = sorted(is_tickers & oos_tickers)

    if not common_tickers:
        raise ValueError("No common tickers between IS and OOS panels.")

    is_panel = is_panel[is_panel["ticker"].isin(common_tickers)].reset_index(drop=True)
    oos_panel = oos_panel[oos_panel["ticker"].isin(common_tickers)].reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Benchmark load (used for excess-IR optimization in Optuna)
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"[Stage4] Loading benchmark ({cfg.qlib.benchmark}) for Excess IR optimization...")

    # Validation 기간 벤치마크 (Optuna 최적화용)
    val_benchmark_returns_for_optuna = _load_benchmark_index(
        start_date=cfg.data_split.val_start,
        end_date=cfg.data_split.val_end,
        benchmark_code=cfg.qlib.benchmark,
    )
    is_benchmark_returns_for_optuna = val_benchmark_returns_for_optuna  # 하위 호환성

    # Test 기간 벤치마크 (최종 평가용)
    test_benchmark_returns = _load_benchmark_index(
        start_date=cfg.data_split.test_start,
        end_date=cfg.data_split.test_end,
        benchmark_code=cfg.qlib.benchmark,
    )
    oos_benchmark_returns = test_benchmark_returns  # 하위 호환성

    # 벤치마크 metrics 계산 (summary에서 사용)
    val_benchmark_dict = _calculate_benchmark_returns(
        start_date=cfg.data_split.val_start,
        end_date=cfg.data_split.val_end,
        benchmark_code=cfg.qlib.benchmark,
    )
    is_benchmark_metrics = _compute_metrics_from_returns(val_benchmark_dict)

    test_benchmark_dict = _calculate_benchmark_returns(
        start_date=cfg.data_split.test_start,
        end_date=cfg.data_split.test_end,
        benchmark_code=cfg.qlib.benchmark,
    )
    oos_benchmark_metrics = _compute_metrics_from_returns(test_benchmark_dict)

    if verbose:
        print(f"[Stage4] Benchmark loaded: Val={len(val_benchmark_returns_for_optuna)} days, Test={len(test_benchmark_returns)} days")

    # ═══════════════════════════════════════════════════════════════════════
    # Sequential Combination Evaluation (Phase 1 + Phase 2 combined)
    # ═══════════════════════════════════════════════════════════════════════
    n_combinations = len(combinations_to_evaluate)

    if verbose:
        print(f"\n[Stage4] ═══════════════════════════════════════════════════════")
        print(f"[Stage4] Starting SEQUENTIAL evaluation")
        print(f"[Stage4] Evaluating {n_combinations} combinations")
        print(f"[Stage4] ═══════════════════════════════════════════════════════")

    all_combination_results = []

    for combo_idx, combination in enumerate(combinations_to_evaluate, 1):
        try:
            # Skip combinations with missing formula columns in the pre-built panel.
            formula_names = [
                str(f.get("name") or "") for f in combination
                if isinstance(f, dict) and f.get("name")
            ]
            formula_names = [n for n in formula_names if n]
            missing_cols = [n for n in formula_names if n not in is_panel.columns]
            if missing_cols:
                if verbose:
                    print(f"[Stage4] ✗ Combination {combo_idx}: missing columns {missing_cols}")
                continue

            result = _evaluate_single_combination_sequential(
                combo_idx=combo_idx,
                combination=combination,
                is_panel=is_panel,
                oos_panel=oos_panel,
                cfg=cfg,
                is_benchmark_returns_for_optuna=is_benchmark_returns_for_optuna,
                verbose=verbose,
            )
            if result.get("skipped", False):
                if verbose:
                    print(f"[Stage4] ✗ Combination {combo_idx}: {result.get('error', 'skipped')}")
            else:
                all_combination_results.append(result)
                if verbose:
                    is_ir = result.get("is_information_ratio", 0)
                    oos_ir = result.get("oos_information_ratio", 0)
                    print(f"[Stage4] ✓ Combination {combo_idx}/{n_combinations}: IS IR={is_ir:.3f}, OOS IR={oos_ir:.3f}")
        except Exception as e:
            if verbose:
                print(f"[Stage4] ✗ Combination {combo_idx} failed: {e}")

    # Sort results by combo_idx for consistent ordering
    all_combination_results.sort(key=lambda x: x.get("combo_idx", 0))

    if verbose:
        print(f"\n[Stage4] ═══════════════════════════════════════════════════════")
        print(f"[Stage4] Evaluation complete: {len(all_combination_results)}/{n_combinations} succeeded")
        print(f"[Stage4] ═══════════════════════════════════════════════════════")

    # Validation: at least one combination must be evaluated successfully.
    if not all_combination_results:
        raise ValueError("No valid combinations found after evaluation")

    if verbose:
        print(f"\n[Stage4] ═══════════════════════════════════════════════════════")
        print(f"[Stage4] All {len(all_combination_results)} combinations evaluated")
        print(f"[Stage4] ═══════════════════════════════════════════════════════")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary/result construction (includes detailed results for all combinations).
    # ═══════════════════════════════════════════════════════════════════════

    # Put detailed results into all_combinations_summary (Qlib-style naming).
    all_combinations_summary = []
    for r in all_combination_results:
        # Qlib-style structure: put insample/outsample at top-level and map strategy -> return.
        combo_summary: dict[str, Any] = {
            "combo_idx": r["combo_idx"],
            "formula_names": r["formula_names"],
            "optimal_thresholds": r["optimal_thresholds"],
            "insample": {
                "return": {
                    "mean": r["is_metrics"]["mean_return"],
                    "std": r["is_metrics"]["ann_vol"] / np.sqrt(252),
                    "annualized_return": r["is_metrics"]["ann_return"],
                    "information_ratio": r["is_metrics"]["information_ratio"],
                    "max_drawdown": r["is_metrics"]["max_drawdown"],
                    "net_return": r["is_metrics"]["net_return"],
                    "avg_holdings": r["is_metrics"]["avg_holdings"],
                    "avg_turnover": r["is_metrics"]["avg_turnover"],
                },
                "excess_return_without_cost": {
                    "mean": r["is_excess_without_cost"]["mean_return"],
                    "std": r["is_excess_without_cost"]["ann_vol"] / np.sqrt(252),
                    "annualized_return": r["is_excess_without_cost"]["ann_return"],
                    "information_ratio": r["is_excess_without_cost"]["information_ratio"],
                    "max_drawdown": r["is_excess_without_cost"]["max_drawdown"],
                    "net_return": r["is_excess_without_cost"]["net_return"],
                },
                "excess_return_with_cost": {
                    "mean": r["is_excess_with_cost"]["mean_return"],
                    "std": r["is_excess_with_cost"]["ann_vol"] / np.sqrt(252),
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
                    "std": r["oos_metrics"]["ann_vol"] / np.sqrt(252),
                    "annualized_return": r["oos_metrics"]["ann_return"],
                    "information_ratio": r["oos_metrics"]["information_ratio"],
                    "max_drawdown": r["oos_metrics"]["max_drawdown"],
                    "net_return": r["oos_metrics"]["net_return"],
                    "avg_holdings": r["oos_metrics"]["avg_holdings"],
                    "avg_turnover": r["oos_metrics"]["avg_turnover"],
                },
                "excess_return_without_cost": {
                    "mean": r["oos_excess_without_cost"]["mean_return"],
                    "std": r["oos_excess_without_cost"]["ann_vol"] / np.sqrt(252),
                    "annualized_return": r["oos_excess_without_cost"]["ann_return"],
                    "information_ratio": r["oos_excess_without_cost"]["information_ratio"],
                    "max_drawdown": r["oos_excess_without_cost"]["max_drawdown"],
                    "net_return": r["oos_excess_without_cost"]["net_return"],
                },
                "excess_return_with_cost": {
                    "mean": r["oos_excess_with_cost"]["mean_return"],
                    "std": r["oos_excess_with_cost"]["ann_vol"] / np.sqrt(252),
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

        # Optional fixed modes (fixed_q80/fixed_q90/...)
        fixed_modes = r.get("fixed_modes") or {}
        if isinstance(fixed_modes, dict) and fixed_modes:
            fixed_summary: dict[str, Any] = {}
            for mode_name, mr in fixed_modes.items():
                if not isinstance(mr, dict):
                    continue
                # Convert fixed modes to the same Qlib-style structure.
                fixed_summary[mode_name] = {
                    "threshold_q": mr.get("threshold_q"),
                    "insample": {
                        "return": {
                            "mean": mr["is_metrics"]["mean_return"],
                            "std": mr["is_metrics"]["ann_vol"] / np.sqrt(252),
                            "annualized_return": mr["is_metrics"]["ann_return"],
                            "information_ratio": mr["is_metrics"]["information_ratio"],
                            "max_drawdown": mr["is_metrics"]["max_drawdown"],
                            "net_return": mr["is_metrics"]["net_return"],
                            "avg_holdings": mr["is_metrics"]["avg_holdings"],
                            "avg_turnover": mr["is_metrics"]["avg_turnover"],
                        },
                        "excess_return_without_cost": {
                            "mean": mr["is_excess_without_cost"]["mean_return"],
                            "std": mr["is_excess_without_cost"]["ann_vol"] / np.sqrt(252),
                            "annualized_return": mr["is_excess_without_cost"]["ann_return"],
                            "information_ratio": mr["is_excess_without_cost"]["information_ratio"],
                            "max_drawdown": mr["is_excess_without_cost"]["max_drawdown"],
                            "net_return": mr["is_excess_without_cost"]["net_return"],
                        },
                        "excess_return_with_cost": {
                            "mean": mr["is_excess_with_cost"]["mean_return"],
                            "std": mr["is_excess_with_cost"]["ann_vol"] / np.sqrt(252),
                            "annualized_return": mr["is_excess_with_cost"]["ann_return"],
                            "information_ratio": mr["is_excess_with_cost"]["information_ratio"],
                            "max_drawdown": mr["is_excess_with_cost"]["max_drawdown"],
                            "net_return": mr["is_excess_with_cost"]["net_return"],
                        },
                        "trade_metrics": {
                            "n_trades": mr["is_trade_metrics"]["n_trades"],
                            "win_rate": mr["is_trade_metrics"]["win_rate"],
                            "profit_factor": mr["is_trade_metrics"]["profit_factor"],
                            "avg_trade_return": mr["is_trade_metrics"]["avg_trade_return"],
                            "avg_holding_days": mr["is_trade_metrics"].get("avg_holding_days", 0.0),
                        },
                    },
                    "outsample": {
                        "return": {
                            "mean": mr["oos_metrics"]["mean_return"],
                            "std": mr["oos_metrics"]["ann_vol"] / np.sqrt(252),
                            "annualized_return": mr["oos_metrics"]["ann_return"],
                            "information_ratio": mr["oos_metrics"]["information_ratio"],
                            "max_drawdown": mr["oos_metrics"]["max_drawdown"],
                            "net_return": mr["oos_metrics"]["net_return"],
                            "avg_holdings": mr["oos_metrics"]["avg_holdings"],
                            "avg_turnover": mr["oos_metrics"]["avg_turnover"],
                        },
                        "excess_return_without_cost": {
                            "mean": mr["oos_excess_without_cost"]["mean_return"],
                            "std": mr["oos_excess_without_cost"]["ann_vol"] / np.sqrt(252),
                            "annualized_return": mr["oos_excess_without_cost"]["ann_return"],
                            "information_ratio": mr["oos_excess_without_cost"]["information_ratio"],
                            "max_drawdown": mr["oos_excess_without_cost"]["max_drawdown"],
                            "net_return": mr["oos_excess_without_cost"]["net_return"],
                        },
                        "excess_return_with_cost": {
                            "mean": mr["oos_excess_with_cost"]["mean_return"],
                            "std": mr["oos_excess_with_cost"]["ann_vol"] / np.sqrt(252),
                            "annualized_return": mr["oos_excess_with_cost"]["ann_return"],
                            "information_ratio": mr["oos_excess_with_cost"]["information_ratio"],
                            "max_drawdown": mr["oos_excess_with_cost"]["max_drawdown"],
                            "net_return": mr["oos_excess_with_cost"]["net_return"],
                        },
                        "trade_metrics": {
                            "n_trades": mr["oos_trade_metrics"]["n_trades"],
                            "win_rate": mr["oos_trade_metrics"]["win_rate"],
                            "profit_factor": mr["oos_trade_metrics"]["profit_factor"],
                            "avg_trade_return": mr["oos_trade_metrics"]["avg_trade_return"],
                            "avg_holding_days": mr["oos_trade_metrics"].get("avg_holding_days", 0.0),
                        },
                    },
                }
            combo_summary["fixed_modes"] = fixed_summary

        all_combinations_summary.append(combo_summary)

    # Summary (Qlib-style naming).
    evaluation_modes = []
    if cfg.stage4.enable_optuna:
        evaluation_modes.append("optuna")
    fixed_quantiles = list(getattr(cfg.stage4, "fixed_quantiles", []) or [])
    for q in fixed_quantiles:
        try:
            q = float(q)
        except Exception:
            continue
        if 0.0 < q < 1.0:
            evaluation_modes.append(f"fixed_q{int(round(q * 100)):02d}")

    # Get actual backtest mode used (all combinations use the same mode)
    actual_backtest_mode = all_combination_results[0].get("backtest_mode", "qlib_native") if all_combination_results else "unknown"

    summary = {
        "hypothesis_id": hypothesis_id,
        "backtest_mode": actual_backtest_mode,
        "evaluation_modes": evaluation_modes,
        "n_trials": cfg.stage4.n_trials,
        "horizon_days": cfg.stage4.horizon_days,
        "val_period": f"{cfg.data_split.val_start} ~ {cfg.data_split.val_end}",
        "test_period": f"{cfg.data_split.test_start} ~ {cfg.data_split.test_end}",
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
                "std": is_benchmark_metrics["ann_vol"] / np.sqrt(252),
                "annualized_return": is_benchmark_metrics["ann_return"],
                "max_drawdown": is_benchmark_metrics["max_drawdown"],
            },
            "outsample": {
                "information_ratio": oos_benchmark_metrics["information_ratio"],
                "net_return": oos_benchmark_metrics["net_return"],
                "mean": oos_benchmark_metrics["mean_return"],
                "std": oos_benchmark_metrics["ann_vol"] / np.sqrt(252),
                "annualized_return": oos_benchmark_metrics["ann_return"],
                "max_drawdown": oos_benchmark_metrics["max_drawdown"],
            },
        },
        "n_combinations_evaluated": len(all_combination_results),
        "all_combinations": all_combinations_summary,
    }

    # Report (includes all combinations).
    report_md = _generate_report_all_combinations(
        hypothesis_id=hypothesis_id,
        cfg=cfg,
        all_combinations=all_combinations_summary,
        benchmark=summary["benchmark"],
    )

    # Config dict for return value
    config_dict = {
        "train_start": cfg.data_split.train_start,
        "train_end": cfg.data_split.train_end,
        "val_start": cfg.data_split.val_start,
        "val_end": cfg.data_split.val_end,
        "test_start": cfg.data_split.test_start,
        "test_end": cfg.data_split.test_end,
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
        "backtest_mode": actual_backtest_mode,
    }

    # Save results
    if run_ctx is not None:
        # If outer_iter is provided, save per-iteration artifacts.
        if outer_iter is not None:
            run_ctx.save_json_with_iter("specs/stage4_summary.json", outer_iter, summary)
            run_ctx.save_text_with_iter("reports/stage4.md", outer_iter, report_md)
        else:
            run_ctx.save_json("specs/stage4_summary.json", summary)
            run_ctx.save_text("reports/stage4.md", report_md)

        # ═══════════════════════════════════════════════════════════════════
        # Save Optuna trials (combine all combinations into a single parquet file).
        # ═══════════════════════════════════════════════════════════════════
        # Only save trials when Optuna is enabled.
        all_trials = []
        if cfg.stage4.enable_optuna:
            for r in all_combination_results:
                trials_df = r.get("trials_df")
                if trials_df is None:
                    # Backward-compatible fallback (older results may carry the Study object).
                    study = r.get("study")
                    if study is not None:
                        trials_df = study.trials_dataframe()
                if trials_df is None:
                    continue

                # Add combination metadata.
                trials_df = trials_df.copy()
                trials_df["hypothesis_id"] = hypothesis_id
                trials_df["combo_idx"] = r["combo_idx"]
                trials_df["formula_names"] = ",".join(r["formula_names"])

                all_trials.append(trials_df)

        if all_trials:
            combined_trials = pd.concat(all_trials, ignore_index=True)

            # Save as parquet.
            trials_path = run_ctx.root_dir / "optuna_studies" / "all_trials.parquet"
            trials_path.parent.mkdir(parents=True, exist_ok=True)
            combined_trials.to_parquet(trials_path, index=False)

            if verbose:
                print(f"[Stage4] Saved {len(combined_trials)} Optuna trials to {trials_path}")

        # ═══════════════════════════════════════════════════════════════════
        # Save Qlib-style artifacts.
        # ═══════════════════════════════════════════════════════════════════
        iter_prefix = f"iter_{outer_iter}" if outer_iter is not None else "iter_1"
        for r in all_combination_results:
            combo_idx = r["combo_idx"]
            base_dir = f"qlib_artifacts/{iter_prefix}/combo_{combo_idx}"

            # --- OOS (Out-of-Sample) ---
            oos_dir = f"{base_dir}/oos"

            # report_normal_1day.pkl: pd.DataFrame (index=datetime, cols=return,bench,cost,turnover)
            if r.get("oos_report_df") is not None:
                run_ctx.save_pickle(f"{oos_dir}/report_normal_1day.pkl", r["oos_report_df"])

            # positions_normal_1day.pkl: dict[datetime -> list of position records]
            oos_positions = r.get("oos_positions", [])
            if oos_positions:
                pos_dict = {}
                for pos in oos_positions:
                    date_key = pd.Timestamp(pos.get("date", ""))
                    if date_key not in pos_dict:
                        pos_dict[date_key] = []
                    pos_dict[date_key].append({
                        "instrument": pos.get("ticker"),
                        "amount": pos.get("amount", 0.0),
                        "price": pos.get("price", 0.0),
                        "value": pos.get("value", 0.0),
                    })
                run_ctx.save_pickle(f"{oos_dir}/positions_normal_1day.pkl", pos_dict)

            # port_analysis_1day.pkl: pd.DataFrame with MultiIndex (analysis_name, metric), col="risk"
            oos_analysis = _build_port_analysis_df(r.get("oos_report_df"), freq="1day")
            run_ctx.save_pickle(f"{oos_dir}/port_analysis_1day.pkl", oos_analysis)

            # trades.pkl: list of trade records
            oos_trades = [t for t in r.get("oos_trades", []) if isinstance(t, dict)]
            if oos_trades:
                run_ctx.save_pickle(f"{oos_dir}/trades.pkl", oos_trades)

            # --- IS (In-Sample) ---
            is_dir = f"{base_dir}/is"

            # report_normal_1day.pkl
            if r.get("is_report_df") is not None:
                run_ctx.save_pickle(f"{is_dir}/report_normal_1day.pkl", r["is_report_df"])

            # positions_normal_1day.pkl
            is_positions = r.get("is_positions", [])
            if is_positions:
                pos_dict = {}
                for pos in is_positions:
                    date_key = pd.Timestamp(pos.get("date", ""))
                    if date_key not in pos_dict:
                        pos_dict[date_key] = []
                    pos_dict[date_key].append({
                        "instrument": pos.get("ticker"),
                        "amount": pos.get("amount", 0.0),
                        "price": pos.get("price", 0.0),
                        "value": pos.get("value", 0.0),
                    })
                run_ctx.save_pickle(f"{is_dir}/positions_normal_1day.pkl", pos_dict)

            # port_analysis_1day.pkl
            is_analysis = _build_port_analysis_df(r.get("is_report_df"), freq="1day")
            run_ctx.save_pickle(f"{is_dir}/port_analysis_1day.pkl", is_analysis)

            # --- Fixed modes artifacts (optional) ---
            fixed_modes = r.get("fixed_modes") or {}
            if isinstance(fixed_modes, dict) and fixed_modes:
                for mode_name, mr in fixed_modes.items():
                    if not isinstance(mr, dict):
                        continue
                    mode_base = f"{base_dir}/{mode_name}"
                    mode_oos_dir = f"{mode_base}/oos"
                    mode_is_dir = f"{mode_base}/is"

                    if mr.get("oos_report_df") is not None:
                        run_ctx.save_pickle(f"{mode_oos_dir}/report_normal_1day.pkl", mr["oos_report_df"])
                    if mr.get("is_report_df") is not None:
                        run_ctx.save_pickle(f"{mode_is_dir}/report_normal_1day.pkl", mr["is_report_df"])

                    # positions
                    for split, split_dir in [("oos_positions", mode_oos_dir), ("is_positions", mode_is_dir)]:
                        positions = mr.get(split, []) or []
                        if positions:
                            pos_dict = {}
                            for pos in positions:
                                date_key = pd.Timestamp(pos.get("date", ""))
                                if date_key not in pos_dict:
                                    pos_dict[date_key] = []
                                pos_dict[date_key].append({
                                    "instrument": pos.get("ticker"),
                                    "amount": pos.get("amount", 0.0),
                                    "price": pos.get("price", 0.0),
                                    "value": pos.get("value", 0.0),
                                })
                            run_ctx.save_pickle(f"{split_dir}/positions_normal_1day.pkl", pos_dict)

                    # port_analysis
                    oos_report_dict = mr.get("oos_report_df")
                    oos_report_df = pd.DataFrame.from_dict(oos_report_dict) if oos_report_dict else None
                    oos_analysis = _build_port_analysis_df(oos_report_df, freq="1day")
                    run_ctx.save_pickle(f"{mode_oos_dir}/port_analysis_1day.pkl", oos_analysis)

                    is_report_dict = mr.get("is_report_df")
                    is_report_df = pd.DataFrame.from_dict(is_report_dict) if is_report_dict else None
                    is_analysis = _build_port_analysis_df(
                        is_report_df,
                        freq="1day",
                    )
                    run_ctx.save_pickle(f"{mode_is_dir}/port_analysis_1day.pkl", is_analysis)

                    # trades
                    oos_trades = [t for t in mr.get("oos_trades", []) if isinstance(t, dict)]
                    if oos_trades:
                        run_ctx.save_pickle(f"{mode_oos_dir}/trades.pkl", oos_trades)

        # qlib_res.csv (summary)
        summary_rows = []
        for r in all_combination_results:
            summary_rows.append({
                "combo_idx": r["combo_idx"],
                "formula_names": "_".join(r.get("formula_names", [])),
                # OOS
                "1day.excess_return_without_cost.information_ratio": r.get("oos_excess_without_cost", {}).get("information_ratio"),
                "1day.excess_return_without_cost.annualized_return": r.get("oos_excess_without_cost", {}).get("ann_return"),
                "1day.excess_return_without_cost.max_drawdown": r.get("oos_excess_without_cost", {}).get("max_drawdown"),
                "1day.excess_return_with_cost.information_ratio": r.get("oos_excess_with_cost", {}).get("information_ratio"),
                "1day.excess_return_with_cost.annualized_return": r.get("oos_excess_with_cost", {}).get("ann_return"),
                "1day.excess_return_with_cost.max_drawdown": r.get("oos_excess_with_cost", {}).get("max_drawdown"),
            })
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res.csv", pd.DataFrame(summary_rows))

        # Save MultiIndex CSV (Qlib-style structure).
        multiindex_df = _combinations_to_multiindex_df(all_combinations_summary)
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res_multiindex.csv", multiindex_df)

        if verbose:
            print(f"[Stage4] Artifacts saved to qlib_artifacts/{iter_prefix}/")

    if verbose:
        print(f"[Stage4] Complete!")

    # Empty daily panel (per-combination results are in summary["all_combinations"]).
    empty_panel = pl.DataFrame({"timestamp": [], "gross_return": [], "net_return": []})

    return Stage4Result(
        hypothesis_id=hypothesis_id,
        config=config_dict,
        summary=summary,
        report_md=report_md,
        is_daily_panel=empty_panel,
        oos_daily_panel=empty_panel,
    )


def _generate_report_all_combinations(
    *,
    hypothesis_id: str,
    cfg: RDConfig,
    all_combinations: List[Dict[str, Any]],
    benchmark: Dict[str, float],
) -> str:
    """Generate a Markdown report that includes results for all combinations."""
    lines = [
        "# Stage 4: Validation / Test Backtest Results",
        "",
        f"**Hypothesis ID**: {hypothesis_id}",
        f"**Combinations Evaluated**: {len(all_combinations)}",
        f"**Backtest Engine**: Qlib",
        "",
        "## 1. Configuration",
        "",
        f"- **Train Period**: {cfg.data_split.train_start} ~ {cfg.data_split.train_end}",
        f"- **Validation Period (Optuna)**: {cfg.data_split.val_start} ~ {cfg.data_split.val_end}",
        f"- **Test Period (Final Eval)**: {cfg.data_split.test_start} ~ {cfg.data_split.test_end}",
        f"- **Horizon Days**: {cfg.stage4.horizon_days}",
        f"- **Optuna Trials**: {cfg.stage4.n_trials}",
        f"- **Threshold Range**: [{cfg.stage4.threshold_min:.2f}, {cfg.stage4.threshold_max:.2f}]",
        f"- **Transaction Costs**: open={cfg.qlib.open_cost:.4%}, close={cfg.qlib.close_cost:.4%}",
        "",
        f"## 2. Benchmark ({cfg.qlib.benchmark})",
        "",
        f"- **IS IR**: {benchmark['insample']['information_ratio']:.4f}, Return: {benchmark['insample']['net_return']:.2%}",
        f"- **OOS IR**: {benchmark['outsample']['information_ratio']:.4f}, Return: {benchmark['outsample']['net_return']:.2%}",
        "",
        "## 3. All Combinations Results",
        "",
    ]

    for combo in all_combinations:
        lines.extend([
            f"### Combination {combo['combo_idx']}: {', '.join(combo['formula_names'])}",
            "",
            "**Thresholds:**",
        ])
        for fname, thr in combo["optimal_thresholds"].items():
            lines.append(f"- {fname}: {thr:.3f}")

        is_strategy = combo['insample']['return']
        oos_strategy = combo['outsample']['return']
        is_excess = combo['insample']['excess_return_without_cost']
        oos_excess = combo['outsample']['excess_return_without_cost']
        oos_trades = combo['outsample']['trade_metrics']

        lines.extend([
            "",
            "| Metric | In-Sample | Out-of-Sample |",
            "|--------|-----------|---------------|",
            f"| IR | {is_strategy['information_ratio']:.2f} | {oos_strategy['information_ratio']:.2f} |",
            f"| Net Return | {is_strategy['net_return']:.2%} | {oos_strategy['net_return']:.2%} |",
            f"| Ann. Return | {is_strategy['annualized_return']:.2%} | {oos_strategy['annualized_return']:.2%} |",
            f"| Max DD | {is_strategy['max_drawdown']:.2%} | {oos_strategy['max_drawdown']:.2%} |",
            f"| Avg Holdings | {is_strategy['avg_holdings']:.1f} | {oos_strategy['avg_holdings']:.1f} |",
            f"| Excess IR | {is_excess['information_ratio']:.2f} | {oos_excess['information_ratio']:.2f} |",
            "",
            f"**OOS Trade Stats**: Trades={oos_trades['n_trades']}, WinRate={oos_trades['win_rate']:.1%}, PF={oos_trades['profit_factor']:.2f}",
            "",
        ])

        fixed_modes = combo.get("fixed_modes") or {}
        if isinstance(fixed_modes, dict) and fixed_modes:
            lines.append("**Fixed Quantile Modes (OOS)**")
            for mode_name, mode_obj in sorted(fixed_modes.items()):
                try:
                    oos_fixed = mode_obj["outsample"]["return"]
                    lines.append(
                        f"- {mode_name}: IR={oos_fixed['information_ratio']:.2f}, "
                        f"NetReturn={oos_fixed['net_return']:.2%}, "
                        f"AvgTurnover={oos_fixed['avg_turnover']:.3f}"
                    )
                except Exception:
                    continue
            lines.append("")

    lines.extend([
        "## 4. Methodology",
        "",
        "- **Optimization**: Optuna TPESampler, objective=In-Sample IR (information_ratio)",
        "- **Fixed Modes**: Optional fixed quantile thresholds (e.g., fixed_q90) for comparison",
        "- **No Selection Bias**: All combinations evaluated independently, no best selection",
        "- **Look-ahead Bias**: Removed (Out-of-Sample not used in optimization)",
        "",
    ])

    return "\n".join(lines)
