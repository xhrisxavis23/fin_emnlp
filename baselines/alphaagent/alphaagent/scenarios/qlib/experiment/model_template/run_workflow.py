import os
import sys
from pathlib import Path

from jinja2 import Template, meta
from ruamel.yaml import YAML

import qlib
from qlib.config import C
from qlib.model.trainer import task_train
from qlib.utils.data import update_config


def _get_path_list(path):
    if isinstance(path, str):
        return [path]
    return list(path)


def _sys_config(config: dict, config_path: str) -> None:
    sys_cfg = config.get("sys", {})
    for p in _get_path_list(sys_cfg.get("path", [])):
        sys.path.append(p)
    for p in _get_path_list(sys_cfg.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))


def _render_template(config_path: str) -> str:
    text = Path(config_path).read_text(encoding="utf-8")
    template = Template(text)
    env = template.environment
    parsed_content = env.parse(text)
    variables = meta.find_undeclared_variables(parsed_content)
    context = {var: os.getenv(var, "") for var in variables if var in os.environ}
    return template.render(context)


def _load_config(config_path: str) -> dict:
    rendered_yaml = _render_template(config_path)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered_yaml) or {}

    base_config_path = config.get("BASE_CONFIG_PATH", None)
    if base_config_path:
        base_config_path = Path(base_config_path)
        if not base_config_path.exists():
            base_config_path = Path(config_path).absolute().parent.joinpath(base_config_path)
        base_config = YAML(typ="safe", pure=True).load(base_config_path.read_text(encoding="utf-8")) or {}
        config = update_config(base_config, config)

    _sys_config(config, config_path)
    return config


def _compute_market_buy_and_hold_benchmark(
    market: str,
    start_time,
    end_time,
    freq: str = "day",
):
    from qlib.data import D

    instruments = D.instruments(market)
    if not instruments:
        raise ValueError(f'Benchmark market "{market}" has no instruments.')

    close_df = D.features(instruments, ["$close"], start_time, end_time, freq=freq, disk_cache=True)
    if close_df is None or len(close_df) == 0:
        raise ValueError(f'Benchmark market "{market}" has no $close data in the given period.')

    close_series = close_df[close_df.columns.tolist()[0]]
    prices = close_series.unstack(level="instrument").sort_index()
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError(f'Benchmark market "{market}" has no usable price data in the given period.')

    base_time = prices.index[0]
    base_prices = prices.loc[base_time]
    universe_cols = base_prices.notna()
    prices = prices.loc[:, universe_cols]
    if prices.empty:
        raise ValueError(f'Benchmark market "{market}" has no instruments with prices at {base_time}.')

    base_prices = base_prices.loc[universe_cols]
    rel_value = prices.divide(base_prices).mean(axis=1, skipna=True)
    bench_ret = rel_value.pct_change().fillna(0.0)
    bench_ret.name = "benchmark"
    return bench_ret


def _inject_benchmark_series_if_needed(config: dict) -> None:
    task = config.get("task") or {}
    records = task.get("record") or []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("class") != "PortAnaRecord":
            continue
        kwargs = rec.get("kwargs") or {}
        port_cfg = kwargs.get("config") or {}
        backtest_cfg = port_cfg.get("backtest") or {}
        benchmark = backtest_cfg.get("benchmark")
        if not (isinstance(benchmark, str) and benchmark.startswith("market:")):
            return

        market = benchmark[len("market:") :].strip()
        if not market:
            raise ValueError('Invalid benchmark "market:"; expected "market:<market_name>"')

        bench = _compute_market_buy_and_hold_benchmark(
            market=market,
            start_time=backtest_cfg.get("start_time"),
            end_time=backtest_cfg.get("end_time"),
            freq=(port_cfg.get("executor", {}) or {}).get("kwargs", {}).get("time_per_step", "day"),
        )
        backtest_cfg["benchmark"] = bench
        return


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "conf.yaml"
    experiment_name = os.getenv("QLIB_EXPERIMENT_NAME", "workflow")
    uri_folder = os.getenv("QLIB_URI_FOLDER", "mlruns")

    config = _load_config(config_path)

    qlib_init = config.get("qlib_init") or {}
    if "exp_manager" in qlib_init:
        qlib.init(**qlib_init)
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**qlib_init, exp_manager=exp_manager)

    _inject_benchmark_series_if_needed(config)

    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)


if __name__ == "__main__":
    main()

