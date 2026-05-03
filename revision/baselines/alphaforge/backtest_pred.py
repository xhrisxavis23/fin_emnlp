import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch

from alphagen_generic.features import vwap, close, target
from alphagen.utils.correlation import batch_pearsonr
from alphagen_qlib.stock_data import StockData
from gan.utils.data import get_data_by_year

# Optional in some AlphaForge copies.
try:
    from gan.utils.data import get_data_by_date_range  # type: ignore
except Exception:  # pragma: no cover - fallback for missing helper
    def get_data_by_date_range(
        *,
        train_start: str,
        train_end: str,
        valid_start: str,
        valid_end: str,
        test_start: str,
        test_end: str,
        instruments: str,
        target: Any,
        freq: str,
        qlib_path: Optional[Dict[str, str]] = None,
        device: str = "cpu",
    ):
        """
        Fallback loader when gan.utils.data.get_data_by_date_range is missing.
        Mirrors get_data_by_year behavior but with explicit date ranges.
        """
        qlib_uri = qlib_path
        if qlib_uri is None:
            qlib_uri = {freq: os.path.expanduser(os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data/cn_data"))}

        # For US datasets (e.g., sp500), $factor may be missing; avoid raw=True in that case.
        raw_flag = True
        try:
            p = str(qlib_uri.get(freq)) if isinstance(qlib_uri, dict) else str(qlib_uri)
            if "sp500" in p or "us" in p:
                raw_flag = False
        except Exception:
            pass

        data = StockData(instruments, train_start, train_end, raw=raw_flag, qlib_path=qlib_uri, freq=freq, device=device)
        data_valid = StockData(
            instruments, valid_start, valid_end, raw=raw_flag, qlib_path=qlib_uri, freq=freq, device=device
        )
        # add head for valid/test (2 years head, same as get_data_by_year)
        valid_head_start = f"{int(valid_start[:4]) - 2}-01-01"
        test_head_start = f"{int(test_start[:4]) - 2}-01-01"
        data_valid_withhead = StockData(
            instruments, valid_head_start, valid_end, raw=raw_flag, qlib_path=qlib_uri, freq=freq, device=device
        )
        data_test = StockData(instruments, test_start, test_end, raw=raw_flag, qlib_path=qlib_uri, freq=freq, device=device)
        data_test_withhead = StockData(
            instruments, test_head_start, test_end, raw=raw_flag, qlib_path=qlib_uri, freq=freq, device=device
        )
        data_all = StockData(instruments, train_start, test_end, raw=raw_flag, qlib_path=qlib_uri, freq=freq, device=device)
        name = f"{instruments}_pkl_{str(target).replace('/','_').replace(' ','')}_{freq}_{train_start}_{train_end}_{valid_start}_{valid_end}_{test_start}_{test_end}"
        return data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, name
try:
    from alphaforge_defaults import (
        DEFAULT_UNIVERSE,
        DEFAULT_BENCHMARK,
        DEFAULT_TRAIN_START,
        DEFAULT_TRAIN_END,
        DEFAULT_VALID_START,
        DEFAULT_VALID_END,
        DEFAULT_TEST_START,
        DEFAULT_TEST_END,
        DEFAULT_FREQ,
    )
except Exception:
    # Fallback defaults when alphaforge_defaults.py is missing in this repo.
    DEFAULT_UNIVERSE = "sp500"
    DEFAULT_BENCHMARK = "^GSPC"
    DEFAULT_TRAIN_START = "2015-01-01"
    DEFAULT_TRAIN_END = "2019-12-29"
    DEFAULT_VALID_START = "2020-01-01"
    DEFAULT_VALID_END = "2020-12-29"
    DEFAULT_TEST_START = "2021-01-01"
    DEFAULT_TEST_END = "2025-12-29"
    DEFAULT_FREQ = "day"


def _as_qlib_path(qlib_path: Optional[str], freq: str) -> Optional[Dict[str, str]]:
    if qlib_path is None:
        return None
    return {freq: os.path.expanduser(qlib_path)}


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    equity = np.asarray(equity, dtype=float)
    equity = np.nan_to_num(equity, nan=np.nan)
    if np.all(np.isnan(equity)):
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - (equity / peak)
    if np.all(np.isnan(drawdown)):
        return 0.0
    return float(np.nanmax(drawdown))


def _metrics(daily_ret: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    if daily_ret.size == 0:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "mdd": 0.0}
    daily_ret = np.asarray(daily_ret, dtype=float)
    daily_ret = np.nan_to_num(daily_ret, nan=0.0, posinf=0.0, neginf=0.0)
    equity = np.cumprod(1.0 + daily_ret)
    total_return = float(equity[-1] - 1.0)
    n = daily_ret.size
    cagr = float((1.0 + total_return) ** (periods_per_year / max(n, 1)) - 1.0)
    vol = float(np.std(daily_ret, ddof=1)) if n > 1 else 0.0
    sharpe = float(np.sqrt(periods_per_year) * np.mean(daily_ret) / vol) if vol > 0 else 0.0
    mdd = _max_drawdown(equity)
    return {"total_return": total_return, "cagr": cagr, "sharpe": sharpe, "mdd": mdd}


def _ic_metrics(signal: torch.Tensor, forward_ret: torch.Tensor) -> Dict[str, float]:
    ic_series = batch_pearsonr(signal, forward_ret).detach().cpu().numpy().astype(float)
    ic_series = np.nan_to_num(ic_series, nan=0.0, posinf=0.0, neginf=0.0)

    # Memory-safe daily Spearman (Rank IC): rank per day, no (stocks x stocks) tensor.
    sig_np = signal.detach().cpu().numpy().astype(float)
    ret_np = forward_ret.detach().cpu().numpy().astype(float)

    def rankdata_average(x: np.ndarray) -> np.ndarray:
        n = x.size
        if n == 0:
            return x.astype(float)
        sorter = np.argsort(x, kind="mergesort")
        inv = np.empty_like(sorter)
        inv[sorter] = np.arange(n)
        x_sorted = x[sorter]
        obs = np.r_[True, x_sorted[1:] != x_sorted[:-1]]
        starts = np.flatnonzero(obs)
        ends = np.r_[starts[1:], n]
        ranks_sorted = np.empty(n, dtype=float)
        for s, e in zip(starts, ends):
            ranks_sorted[s:e] = 0.5 * (s + e - 1)
        return ranks_sorted[inv]

    def pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
        if x.size <= 1:
            return 0.0
        xm = float(x.mean())
        ym = float(y.mean())
        xv = x - xm
        yv = y - ym
        denom = float(np.sqrt((xv * xv).sum() * (yv * yv).sum()))
        if not np.isfinite(denom) or denom <= 0.0:
            return 0.0
        return float((xv * yv).sum() / denom)

    ric_series = np.zeros(sig_np.shape[0], dtype=float)
    for i in range(sig_np.shape[0]):
        x = sig_np[i]
        y = ret_np[i]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() <= 1:
            continue
        rx = rankdata_average(x[valid])
        ry = rankdata_average(y[valid])
        ric_series[i] = pearson_1d(rx, ry)
    ric_series = np.nan_to_num(ric_series, nan=0.0, posinf=0.0, neginf=0.0)
    ic_mean = float(ic_series.mean()) if ic_series.size else 0.0
    ic_std = float(ic_series.std(ddof=1)) if ic_series.size > 1 else 0.0
    icir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
    ric_mean = float(ric_series.mean()) if ric_series.size else 0.0
    ric_std = float(ric_series.std(ddof=1)) if ric_series.size > 1 else 0.0
    ricir = float(ric_mean / ric_std) if ric_std > 0 else 0.0
    return {
        "ic": ic_mean,
        "icir": icir,
        "rank_ic": ric_mean,
        "rank_icir": ricir,
    }


def _equal_weight_benchmark_ret(prices: torch.Tensor, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D; got {prices.shape=}")
    if prices.shape[0] <= horizon:
        return np.zeros(0, dtype=float)

    # Buy-and-hold equal-weight benchmark:
    # Allocate 1/N on the first day (across valid stocks), then hold positions (no rebalancing).
    prices = prices.to(dtype=torch.float32)
    filled = prices.clone()
    for t in range(1, filled.shape[0]):
        cur = filled[t]
        prev = filled[t - 1]
        filled[t] = torch.where(torch.isfinite(cur), cur, prev)

    p0 = filled[0]
    valid0 = torch.isfinite(p0) & (p0 > 0)
    n0 = int(valid0.sum().item())
    if n0 <= 0:
        raise ValueError("equal-weight benchmark: no valid prices on the first day")

    w0 = torch.zeros_like(p0)
    w0[valid0] = 1.0 / float(n0)
    p0_safe = torch.where(valid0, p0, torch.ones_like(p0))
    rel = filled / p0_safe
    equity = (rel * w0).sum(dim=1)
    ret = equity[horizon:] / equity[:-horizon] - 1.0
    ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    return ret.detach().cpu().numpy().astype(float)


def _information_ratio(active_ret: np.ndarray, periods_per_year: int = 252) -> float:
    active_ret = np.asarray(active_ret, dtype=float)
    active_ret = np.nan_to_num(active_ret, nan=0.0, posinf=0.0, neginf=0.0)
    if active_ret.size <= 1:
        return 0.0
    std = float(active_ret.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * active_ret.mean() / std)


def _tracking_error(active_ret: np.ndarray, periods_per_year: int = 252) -> float:
    active_ret = np.asarray(active_ret, dtype=float)
    active_ret = np.nan_to_num(active_ret, nan=0.0, posinf=0.0, neginf=0.0)
    if active_ret.size <= 1:
        return 0.0
    std = float(active_ret.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * std)


def _equity_curve(daily_ret: np.ndarray) -> np.ndarray:
    daily_ret = np.asarray(daily_ret, dtype=float)
    daily_ret = np.nan_to_num(daily_ret, nan=0.0, posinf=0.0, neginf=0.0)
    if daily_ret.size == 0:
        return np.zeros(0, dtype=float)
    return np.cumprod(1.0 + daily_ret)


def _cum_return_curve(daily_ret: np.ndarray) -> np.ndarray:
    equity = _equity_curve(daily_ret)
    if equity.size == 0:
        return equity
    return equity - 1.0


def _plot_cum_returns(
    dates: pd.Index,
    strat_daily_ret: np.ndarray,
    bench_daily_ret: np.ndarray,
    *,
    out_path: str,
    title: str,
) -> None:
    # Headless-safe plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strat_cum = _cum_return_curve(strat_daily_ret)
    bench_cum = _cum_return_curve(bench_daily_ret)

    n = int(min(len(dates), strat_cum.size, bench_cum.size))
    if n <= 0:
        raise ValueError("no data to plot")
    x = pd.Index(dates[:n])

    fig = plt.figure(figsize=(10, 5), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, strat_cum[:n], label="strategy", linewidth=1.5)
    ax.plot(x, bench_cum[:n], label="benchmark", linewidth=1.2, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _relative_cagr(
    strategy_ret: np.ndarray, benchmark_ret: np.ndarray, periods_per_year: int = 252
) -> Tuple[float, float]:
    """
    Compute relative performance based on the ratio of equity curves:
      rel_equity = strat_equity / bench_equity
    Returns (rel_total_return, rel_cagr).
    """
    strat_eq = _equity_curve(strategy_ret)
    bench_eq = _equity_curve(benchmark_ret)
    n = int(min(strat_eq.size, bench_eq.size))
    if n <= 0:
        return 0.0, 0.0
    strat_eq = strat_eq[:n]
    bench_eq = bench_eq[:n]
    bench_eq = np.where(bench_eq == 0.0, np.nan, bench_eq)
    rel_eq = strat_eq / bench_eq
    rel_eq = np.nan_to_num(rel_eq, nan=1.0, posinf=1.0, neginf=1.0)
    rel_total = float(rel_eq[-1] - 1.0)
    rel_cagr = float((1.0 + rel_total) ** (periods_per_year / max(n, 1)) - 1.0)
    return rel_total, rel_cagr


@torch.no_grad()
def _backtest_from_signal(
    signal: torch.Tensor,
    forward_ret: torch.Tensor,
    *,
    topk: int = 50,
    long_short: bool = False,
    fee_buy_bps: float = 0.0,
    fee_sell_bps: float = 0.0,
    gross_exposure: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if signal.ndim != 2 or forward_ret.ndim != 2:
        raise ValueError(f"signal/forward_ret must be 2D; got {signal.shape=} {forward_ret.shape=}")
    if signal.shape != forward_ret.shape:
        raise ValueError(f"signal and forward_ret must match; got {signal.shape=} {forward_ret.shape=}")

    n_days, n_stocks = signal.shape
    if topk <= 0:
        raise ValueError("topk must be > 0")
    topk = min(topk, n_stocks)

    daily_ret = torch.zeros(n_days, device=signal.device, dtype=torch.float32)
    daily_turnover = torch.zeros(n_days, device=signal.device, dtype=torch.float32)
    prev_w: Optional[torch.Tensor] = None

    if long_short:
        per_side = (gross_exposure / 2.0) / topk
    else:
        per_side = gross_exposure / topk

    for t in range(n_days):
        sig = signal[t]
        ret = forward_ret[t]
        valid = torch.isfinite(sig) & torch.isfinite(ret)
        if valid.sum().item() < (2 * topk if long_short else topk):
            continue

        w = torch.zeros_like(sig, dtype=torch.float32)

        sig_long = sig.clone()
        sig_long[~valid] = -float("inf")
        long_idx = torch.topk(sig_long, k=topk, largest=True).indices
        w[long_idx] = per_side

        if long_short:
            sig_short = sig.clone()
            sig_short[~valid] = float("inf")
            short_idx = torch.topk(-sig_short, k=topk, largest=True).indices
            w[short_idx] = -per_side

        had_prev_w = prev_w is not None
        buy = torch.tensor(0.0, device=signal.device, dtype=torch.float32)
        sell = torch.tensor(0.0, device=signal.device, dtype=torch.float32)
        if had_prev_w:
            delta = w - prev_w
            buy = torch.clamp(delta, min=0.0).sum()
            sell = torch.clamp(-delta, min=0.0).sum()
            daily_turnover[t] = 0.5 * torch.sum(torch.abs(delta))
        prev_w = w

        # Important: torch treats 0 * NaN as NaN, so mask invalid returns to 0 before dot.
        ret_masked = torch.where(valid, ret, torch.zeros_like(ret))
        gross_ret = torch.sum(w * ret_masked)
        if not had_prev_w:
            cost = torch.tensor(0.0, device=signal.device, dtype=torch.float32)
        else:
            # Keep the same scaling as the original implementation when fee_buy_bps == fee_sell_bps:
            # turnover = 0.5 * sum(|delta|), cost = fee_bps/10000 * turnover.
            cost = 0.5 * ((fee_buy_bps / 10000.0) * buy + (fee_sell_bps / 10000.0) * sell)
        daily_ret[t] = gross_ret - cost

    return daily_ret.detach().cpu().numpy(), daily_turnover.detach().cpu().numpy()


def main(
    *,
    instruments: str = DEFAULT_UNIVERSE,
    benchmark: str = DEFAULT_BENCHMARK,
    benchmark_mode: str = "auto",
    train_end_year: int = -1,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    valid_start: str = DEFAULT_VALID_START,
    valid_end: str = DEFAULT_VALID_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
    seed: int = 0,
    save_name: str = "test",
    n_factors: int = 10,
    window: str = "inf",
    freq: str = DEFAULT_FREQ,
    qlib_path: Optional[str] = None,
    device: str = "cpu",
    horizon: int = 1,
    topk: int = 50,
    long_short: bool = False,
    fee_bps: float = 0.0,
    fee_buy_bps: Optional[float] = None,
    fee_sell_bps: Optional[float] = None,
    pred_path: Optional[str] = None,
    pred_valid_path: Optional[str] = None,
    plot: bool = True,
    plot_dir: Optional[str] = None,
    export_csv: bool = False,
    csv_dir: Optional[str] = None,
    export_pkl: bool = False,
    pkl_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")

    qlib_uri = _as_qlib_path(qlib_path, freq)
    if qlib_uri is None:
        qlib_uri = {freq: os.path.expanduser(os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data/cn_data"))}

    # If instruments="all", expand to explicit list from instruments/all.txt (if present).
    if isinstance(instruments, str) and instruments.lower() == "all":
        try:
            qlib_root = str(qlib_uri.get(freq)) if isinstance(qlib_uri, dict) else str(qlib_uri)
            inst_path = os.path.join(qlib_root, "instruments", "all.txt")
            if os.path.exists(inst_path):
                with open(inst_path, "r", encoding="utf-8", errors="ignore") as f:
                    syms = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # qlib instruments file format: "SYM\\tSTART\\tEND"
                        sym = line.split("\\t")[0].strip()
                        if sym:
                            syms.append(sym)
                if syms:
                    instruments = syms
        except Exception:
            pass

    if train_end_year > 0:
        train_end_tag = train_end_year
        data_all_start = "2010-01-01"
        data_all_end = f"{train_end_year + 2}-12-31"
        data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = get_data_by_year(
            train_start=2010,
            train_end=train_end_year,
            valid_year=train_end_year + 1,
            test_year=train_end_year + 2,
            instruments=instruments,
            target=target,
            freq=freq,
            qlib_path=qlib_uri,
            device=device,
        )
    else:
        train_end_tag = int(str(train_end).split("-")[0])
        data_all_start = train_start
        data_all_end = test_end
        data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = get_data_by_date_range(
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            instruments=instruments,
            target=target,
            freq=freq,
            qlib_path=qlib_uri,
            device=device,
        )
    _ = data, data_valid_withhead, data_test_withhead

    if pred_path is None:
        name = f"{train_end_tag}_{n_factors}_{window}_{seed}"
        pred_path = f"out/{save_name}_{instruments}_{train_end_tag}_{seed}/pred_{name}.pt"
    if pred_valid_path is None:
        name = f"{train_end_tag}_{n_factors}_{window}_{seed}"
        pred_valid_path = f"out/{save_name}_{instruments}_{train_end_tag}_{seed}/pred_valid_{name}.pt"

    pred_test = torch.load(pred_path, map_location=device).to(device)
    pred_valid = torch.load(pred_valid_path, map_location=device).to(device) if os.path.exists(pred_valid_path) else None

    default_plot_dir = plot_dir or (os.path.dirname(pred_path) if pred_path is not None else None) or "out"
    default_csv_dir = csv_dir or default_plot_dir
    default_pkl_dir = pkl_dir or default_plot_dir

    prices_all = vwap.evaluate(data_all).to(device)
    dates_all = data_all._dates[data_all.max_backtrack_days : -data_all.max_future_days]  # type: ignore[attr-defined]

    benchmark_str = str(benchmark) if benchmark is not None else ""
    benchmark_key = benchmark_str.strip().lower()
    benchmark_mode_key = str(benchmark_mode).strip().lower()
    if benchmark_mode_key not in {"auto", "equal_weight", "equal-weight", "instrument", "ticker"}:
        raise ValueError(
            "benchmark_mode must be one of: 'auto', 'equal_weight', 'instrument'. "
            f"got {benchmark_mode!r}"
        )

    # Benchmark selection:
    # - 'equal_weight': use an equal-weight benchmark derived from the universe (no external ticker needed)
    # - 'instrument': use the `benchmark` ticker as a Qlib instrument
    # - 'auto': prefer instrument benchmark when provided; otherwise fall back to equal-weight
    if benchmark_mode_key in {"equal_weight", "equal-weight"}:
        use_equal_weight_bench = True
    elif benchmark_mode_key in {"instrument", "ticker"}:
        use_equal_weight_bench = False
    else:
        use_equal_weight_bench = benchmark_key in {"equal_weight", "equal-weight", "ew", "none", "null", ""}

    if use_equal_weight_bench:
        # Use an equal-weight benchmark computed from the universe itself.
        bench_ret_all = _equal_weight_benchmark_ret(prices_all, horizon=horizon)
    else:
        try:
            # Qlib treats a string `instruments` as a market name (e.g. "csi500") and looks for
            # `instruments/<market>.txt`. For a single instrument code, pass a list.
            bench_data = StockData(
                [benchmark],
                data_all_start,
                data_all_end,
                raw=True,
                qlib_path=qlib_uri,
                freq=freq,
                device=torch.device(device),
            )
            bench_prices_all = close.evaluate(bench_data).to(device).squeeze(-1)
            bench_dates_all = bench_data._dates[
                bench_data.max_backtrack_days : -bench_data.max_future_days
            ]  # type: ignore[attr-defined]

            # Align benchmark prices to data_all dates (intersection).
            bench_price_s = pd.Series(bench_prices_all.detach().cpu().numpy(), index=pd.Index(bench_dates_all))
            bench_price_s = bench_price_s.reindex(pd.Index(dates_all)).astype(float)
            bench_price_np = bench_price_s.to_numpy()
            bench_ret_all = bench_price_np[horizon:] / bench_price_np[:-horizon] - 1.0
            bench_ret_all = np.nan_to_num(bench_ret_all, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            if benchmark_mode_key == "auto":
                print(
                    f"[WARN] failed to load benchmark={benchmark!r} from qlib ({type(e).__name__}: {e}); "
                    "falling back to equal-weight benchmark."
                )
                bench_ret_all = _equal_weight_benchmark_ret(prices_all, horizon=horizon)
                use_equal_weight_bench = True
            else:
                raise

    def run_one(
        pred: torch.Tensor,
        prices: torch.Tensor,
        dates: pd.Index,
        bench_ret: np.ndarray,
        *,
        split: str,
    ) -> Dict[str, Any]:
        if prices.ndim != 2:
            raise ValueError(f"prices must be 2D; got {prices.shape=}")
        if pred.ndim != 2:
            raise ValueError(f"pred must be 2D; got {pred.shape=}")
        if prices.shape != pred.shape:
            raise ValueError(
                "pred/prices shape mismatch. "
                "This usually means predictions were generated with a different qlib dataset/split/universe.\n"
                f"{pred.shape=} {prices.shape=}"
            )
        fwd_ret = prices[horizon:] / prices[:-horizon] - 1.0
        sig = pred[:-horizon]
        sig = sig.to(device)
        fwd_ret = fwd_ret.to(device)
        ic = _ic_metrics(sig, fwd_ret)

        # Default asymmetrical fees expressed in bps: buy 0 / sell 5.
        # If `fee_bps` is provided (non-zero), use it symmetrically for both sides unless side-specific bps are set.
        effective_fee_buy_bps = (
            float(fee_buy_bps) if fee_buy_bps is not None else float(fee_bps if fee_bps != 0.0 else 0.0)
        )
        effective_fee_sell_bps = (
            float(fee_sell_bps) if fee_sell_bps is not None else float(fee_bps if fee_bps != 0.0 else 5.0)
        )
        daily_ret, turnover = _backtest_from_signal(
            sig,
            fwd_ret,
            topk=topk,
            long_short=long_short,
            fee_buy_bps=effective_fee_buy_bps,
            fee_sell_bps=effective_fee_sell_bps,
            gross_exposure=1.0,
        )
        met = _metrics(daily_ret)
        bench_ret = bench_ret[: daily_ret.size]
        bench_met = _metrics(bench_ret)
        active = np.asarray(daily_ret, dtype=float) - np.asarray(bench_ret, dtype=float)
        te = _tracking_error(active)
        active_ann_mean = float(np.mean(active) * 252.0) if active.size else 0.0
        rel_total, rel_cagr = _relative_cagr(daily_ret, bench_ret)
        out: Dict[str, Any] = {
            "n_days": int(daily_ret.size),
            "mean_daily_ret": float(np.mean(daily_ret)) if daily_ret.size else 0.0,
            "turnover_mean": float(np.mean(turnover)) if turnover.size else 0.0,
            "total_return": met["total_return"],
            "annualized_return": met["cagr"],
            "benchmark_total_return": bench_met["total_return"],
            "benchmark_annualized_return": bench_met["cagr"],
            # Historical behavior: difference of CAGRs (not the same as relative CAGR).
            "excess_annualized_return": float(met["cagr"] - bench_met["cagr"]),
            # Relative (compounded) excess return based on equity ratio.
            "excess_total_return": rel_total,
            "excess_annualized_return_rel": rel_cagr,
            # IR uses active (strategy - benchmark) arithmetic returns:
            # IR = mean(active) / std(active) * sqrt(252).
            "information_ratio": _information_ratio(active),
            "tracking_error": te,
            "active_annualized_mean": active_ann_mean,
            "max_drawdown": met["mdd"],
            **ic,
        }

        ret_dates = pd.Index(dates[horizon : horizon + daily_ret.size])
        if export_csv:
            strat_eq = _equity_curve(daily_ret)
            bench_eq = _equity_curve(bench_ret)
            n = int(min(strat_eq.size, bench_eq.size, turnover.size, ret_dates.size))
            if n > 0:
                strat_eq = strat_eq[:n]
                bench_eq = bench_eq[:n]
                bench_eq_safe = np.where(bench_eq == 0.0, np.nan, bench_eq)
                rel_eq = np.nan_to_num(strat_eq / bench_eq_safe, nan=1.0, posinf=1.0, neginf=1.0)

                df = pd.DataFrame(
                    {
                        "strategy_ret": np.asarray(daily_ret[:n], dtype=float),
                        "benchmark_ret": np.asarray(bench_ret[:n], dtype=float),
                        "active_ret": np.asarray(active[:n], dtype=float),
                        "turnover": np.asarray(turnover[:n], dtype=float),
                        "strategy_cumret": strat_eq - 1.0,
                        "benchmark_cumret": bench_eq - 1.0,
                        "relative_cumret": rel_eq - 1.0,
                    },
                    index=pd.Index(ret_dates[:n], name="date"),
                )
                csv_name = f"timeseries_{split}_{train_end_tag}_{n_factors}_{window}_{seed}.csv"
                csv_path_out = os.path.join(default_csv_dir, csv_name)
                os.makedirs(os.path.dirname(csv_path_out) or ".", exist_ok=True)
                df.to_csv(csv_path_out)
                out["timeseries_csv"] = csv_path_out

        if export_pkl:
            # Save qlib-style report/port_analysis pickles for downstream tooling.
            # report_normal_1day.pkl expects columns: return/cost/bench with date index.
            # Here daily_ret is already net (after cost), so set cost=0 and return=daily_ret.
            try:
                n = int(min(ret_dates.size, daily_ret.size, bench_ret.size))
                if n > 0:
                    report_df = pd.DataFrame(
                        {
                            "return": np.asarray(daily_ret[:n], dtype=float),
                            "cost": np.zeros(n, dtype=float),
                            "bench": np.asarray(bench_ret[:n], dtype=float),
                        },
                        index=pd.Index(ret_dates[:n], name="date"),
                    )
                    pa_key = "excess_return_with_cost"
                    pa = pd.DataFrame(
                        {
                            "risk": [
                                float(np.mean(active[:n])) if n > 0 else 0.0,
                                float(np.std(active[:n], ddof=1)) if n > 1 else 0.0,
                                float(rel_cagr),
                                float(_information_ratio(active[:n])),
                                float(met["mdd"]),
                            ]
                        },
                        index=pd.MultiIndex.from_tuples(
                            [
                                (pa_key, "mean"),
                                (pa_key, "std"),
                                (pa_key, "annualized_return"),
                                (pa_key, "information_ratio"),
                                (pa_key, "max_drawdown"),
                            ],
                            names=["metric", "field"],
                        ),
                    )

                    pkl_root = os.path.join(default_pkl_dir, "portfolio_analysis")
                    os.makedirs(pkl_root, exist_ok=True)
                    report_path = os.path.join(pkl_root, f"report_normal_1day_{split}.pkl")
                    pa_path = os.path.join(pkl_root, f"port_analysis_1day_{split}.pkl")
                    report_df.to_pickle(report_path)
                    pa.to_pickle(pa_path)
                    out["report_pkl"] = report_path
                    out["port_analysis_pkl"] = pa_path
            except Exception as e:
                out["pkl_export_error"] = repr(e)

        if plot:
            plot_name = f"cumret_{split}_{train_end_tag}_{n_factors}_{window}_{seed}.png"
            plot_path_out = os.path.join(default_plot_dir, plot_name)
            try:
                _plot_cum_returns(
                    ret_dates,
                    daily_ret,
                    bench_ret,
                    out_path=plot_path_out,
                    title=f"{instruments} {split} cumulative return (topk={topk}, horizon={horizon})",
                )
                out["cumret_plot"] = plot_path_out
            except Exception as e:
                out["cumret_plot_error"] = repr(e)
        return out

    # IMPORTANT: `combine_AFF.py` generates predictions on `data_all`'s stock axis (union over the whole period),
    # while `data_test`/`data_valid` may have fewer stocks. Align by slicing `data_all` to the same day ranges.
    test_len = int(pred_test.shape[0])
    if test_len > prices_all.shape[0]:
        raise ValueError(f"pred_test longer than available prices; {test_len=} {prices_all.shape[0]=}")
    test_start = prices_all.shape[0] - test_len
    prices_test = prices_all[test_start : test_start + test_len]
    dates_test = pd.Index(dates_all[test_start : test_start + test_len])
    bench_test = bench_ret_all[test_start : test_start + test_len - horizon]

    result: Dict[str, Any] = {"test": run_one(pred_test, prices_test, dates_test, bench_test, split="test")}

    if pred_valid is not None:
        valid_len = int(pred_valid.shape[0])
        valid_start = test_start - valid_len
        if valid_start < 0:
            raise ValueError(
                f"pred_valid does not fit before test segment; {valid_len=} {test_start=} {prices_all.shape[0]=}"
            )
        prices_valid = prices_all[valid_start : valid_start + valid_len]
        dates_valid = pd.Index(dates_all[valid_start : valid_start + valid_len])
        bench_valid = bench_ret_all[valid_start : valid_start + valid_len - horizon]
        result["valid"] = run_one(pred_valid, prices_valid, dates_valid, bench_valid, split="valid")
    return result


if __name__ == "__main__":
    import fire

    fire.Fire(main)
