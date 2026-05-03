from __future__ import annotations

import os
from pprint import pprint
from typing import Dict

import pandas as pd

from qlib.backtest import backtest as normal_backtest
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D
from qlib.utils import fill_placeholder, flatten_dict, get_date_by_shift
from qlib.workflow.record_temp import PortAnaRecord


def _parse_benchmark_list(value: str) -> list[str]:
    tickers = [v.strip() for v in value.split(",")]
    return [t for t in tickers if t]


def _compute_equal_weight_buy_and_hold_bench_return(
    tickers: list[str], *, start_time: pd.Timestamp, end_time: pd.Timestamp, report_index: pd.DatetimeIndex
) -> pd.Series:
    if not tickers:
        return pd.Series(index=report_index, dtype="float64")

    # `get_date_by_shift` requires `start_time` to be a trading day, but configs may provide a non-trading day.
    fetch_start = start_time - pd.Timedelta(days=30)
    close = D.features(tickers, ["$close"], start_time=fetch_start, end_time=end_time, freq="day")
    close = close["$close"].unstack(level="instrument").sort_index()

    close = close.reindex(report_index.union(close.index)).sort_index().ffill()

    first_dt = report_index.min()
    prev_dt_candidates = close.index[close.index < first_dt]
    base_dt = prev_dt_candidates.max() if len(prev_dt_candidates) else first_dt

    init_prices = close.loc[base_dt].dropna()
    if init_prices.empty:
        return pd.Series(index=report_index, dtype="float64")

    close = close[init_prices.index]
    value = close.div(init_prices, axis=1).mean(axis=1)
    bench_return = value.pct_change().reindex(report_index).fillna(0.0)
    bench_return.name = "bench"
    return bench_return.astype("float64")


class BuyHoldEqualWeightPortAnaRecord(PortAnaRecord):
    """
    Portfolio analysis record with a strict equal-weight buy-and-hold benchmark (no daily rebalancing).
    """

    def _generate(self, **kwargs):
        pred = self.load("pred.pkl")

        placeholder_value = {"<PRED>": pred}
        for k in ("executor_config", "strategy_config"):
            setattr(self, k, fill_placeholder(getattr(self, k), placeholder_value))

        dt_values = pred.index.get_level_values("datetime")
        if self.backtest_config["start_time"] is None:
            self.backtest_config["start_time"] = dt_values.min()
        if self.backtest_config["end_time"] is None:
            self.backtest_config["end_time"] = get_date_by_shift(dt_values.max(), 1)

        artifact_objects = {}
        portfolio_metric_dict, indicator_dict = normal_backtest(
            executor=self.executor_config, strategy=self.strategy_config, **self.backtest_config
        )

        benchmark_list_env = os.getenv("benchmark_list", "")
        tickers = _parse_benchmark_list(benchmark_list_env) if benchmark_list_env else []

        for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
            artifact_objects.update({f"report_normal_{_freq}.pkl": report_normal})
            artifact_objects.update({f"positions_normal_{_freq}.pkl": positions_normal})

            report_index = pd.DatetimeIndex(report_normal.index)
            start_time = pd.to_datetime(self.backtest_config["start_time"])
            end_time = pd.to_datetime(self.backtest_config["end_time"])
            bench = _compute_equal_weight_buy_and_hold_bench_return(
                tickers, start_time=start_time, end_time=end_time, report_index=report_index
            )
            report_normal["bench"] = bench
            portfolio_metric_dict[_freq] = (report_normal, positions_normal)

        for _freq, indicators_normal in indicator_dict.items():
            artifact_objects.update({f"indicators_normal_{_freq}.pkl": indicators_normal[0]})
            artifact_objects.update({f"indicators_normal_{_freq}_obj.pkl": indicators_normal[1]})

        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq not in portfolio_metric_dict:
                continue

            report_normal, _ = portfolio_metric_dict.get(_analysis_freq)
            analysis = dict()
            analysis["excess_return_without_cost"] = risk_analysis(
                report_normal["return"] - report_normal["bench"], freq=_analysis_freq
            )
            analysis["excess_return_with_cost"] = risk_analysis(
                report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_analysis_freq
            )

            analysis_df = pd.concat(analysis)  # type: pd.DataFrame
            analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
            self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})

            artifact_objects.update({f"port_analysis_{_analysis_freq}.pkl": analysis_df})

            pprint(f"The following are analysis results of benchmark return({_analysis_freq}).")
            pprint(risk_analysis(report_normal["bench"], freq=_analysis_freq))
            pprint(f"The following are analysis results of the excess return without cost({_analysis_freq}).")
            pprint(analysis["excess_return_without_cost"])
            pprint(f"The following are analysis results of the excess return with cost({_analysis_freq}).")
            pprint(analysis["excess_return_with_cost"])

        for _analysis_freq in self.indicator_analysis_freq:
            if _analysis_freq not in indicator_dict:
                continue
            indicators_normal = indicator_dict.get(_analysis_freq)[0]
            from qlib.contrib.evaluate import indicator_analysis

            if self.indicator_analysis_method is None:
                analysis_df = indicator_analysis(indicators_normal)
            else:
                analysis_df = indicator_analysis(indicators_normal, method=self.indicator_analysis_method)
            analysis_dict: Dict[str, float] = analysis_df["value"].to_dict()
            self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})
            artifact_objects.update({f"indicator_analysis_{_analysis_freq}.pkl": analysis_df})
            pprint(f"The following are analysis results of indicators({_analysis_freq}).")
            pprint(analysis_df)

        return artifact_objects
