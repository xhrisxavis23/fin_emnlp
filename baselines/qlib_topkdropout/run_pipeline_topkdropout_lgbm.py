#!/usr/bin/env python
"""
Stage1~3는 기존 파이프라인(run.main.run_pipeline)을 그대로 사용하고,
Stage4만 "AlphaAgent 스타일(LGBM + TopkDropoutStrategy)"로 대체하여 실행하는 엔트리포인트.

사용 예:
  MARKET=cn python run_pipeline_topkdropout_lgbm.py
  MARKET=us python run_pipeline_topkdropout_lgbm.py "Short-term Mean-Reversion"
  python run_pipeline_topkdropout_lgbm.py "Short-term Mean-Reversion" --topk 50 --n-drop 5 --max-combos 5
  python run_pipeline_topkdropout_lgbm.py --outer-loop 5 --max-combos 0 --combo-agg gmean --feature-mode combo_scores --topk 50 --n-drop 5
  python run_pipeline_topkdropout_lgbm.py --feature-mode obs_pool --topk 50 --n-drop 5
  python run_pipeline_topkdropout_lgbm.py --feature-mode obs_ensemble --topk 50 --n-drop 5
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple


def _ensure_conda_lib_in_ld_library_path() -> None:
    if os.environ.get("_FINAGENT_LD_LIBRARY_PATH_REEXEC") == "1":
        return

    env_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV") or sys.prefix
    if not env_prefix:
        return

    conda_lib = os.path.join(env_prefix, "lib")
    if not os.path.isdir(conda_lib):
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current_ld_path.split(":") if p] if current_ld_path else []
    if conda_lib in parts:
        return

    os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}" if current_ld_path else conda_lib
    os.environ["_FINAGENT_LD_LIBRARY_PATH_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


_ensure_conda_lib_in_ld_library_path()

# 프로젝트 루트를 PYTHONPATH에 추가 (run/*, util/*, qlib/*(vendored) import)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def _block_dask_for_lightgbm() -> None:
    """
    현재 환경에서 dask import가 pandas 버전 이슈로 깨져 LightGBM import가 실패할 수 있음.
    LightGBM은 dask 기능을 쓰지 않아도 되므로, dask import를 의도적으로 막아 ImportError로 처리되게 한다.
    """
    import importlib.abc

    class _BlockDaskFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any, target: Any = None):  # type: ignore[override]
            if fullname == "dask" or fullname.startswith("dask."):
                raise ModuleNotFoundError("dask blocked to keep lightgbm import working")
            return None

    # 중복 삽입 방지
    for f in sys.meta_path:
        if f.__class__.__name__ == "_BlockDaskFinder":
            return
    sys.meta_path.insert(0, _BlockDaskFinder())


def _cs_zscore(df, cols: List[str], by: str = "timestamp"):
    import numpy as np

    for c in cols:
        g = df.groupby(by)[c]
        mean = g.transform("mean")
        std = g.transform("std").replace(0.0, np.nan)
        df[c] = (df[c] - mean) / std
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _cs_percentile(df, cols: List[str], by: str = "timestamp"):
    """
    Cross-sectional percentile rank in [0,1] for each date.
    Higher values => larger percentile.
    """
    import numpy as np

    for c in cols:
        df[c] = (
            df.groupby(by)[c]
            .rank(method="average", pct=True, ascending=True)
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    return df


def _compute_label_close_t1_to_t2(df):
    """
    Qlib 템플릿(label: Ref($close, -2)/Ref($close, -1)-1)을 단순화해 pandas로 계산.
    - label[t] = close[t+2] / close[t+1] - 1
    """
    import pandas as pd

    df = df.sort_values(["ticker", "timestamp"], kind="mergesort")
    close_t1 = df.groupby("ticker")["close"].shift(-1)
    close_t2 = df.groupby("ticker")["close"].shift(-2)
    df["label"] = (close_t2 / close_t1) - 1.0
    # 안정성: inf, extreme 제거는 여기선 하지 않고 그대로 둔다(필요 시 stage4에서 클립 가능)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df


def _normalize_trade_dates(ts):
    """
    Normalize timestamps to midnight so they align with Qlib daily calendar.
    """
    import pandas as pd

    return pd.to_datetime(ts).dt.normalize()


def _to_jsonable(obj: Any) -> Any:
    """
    Make objects JSON-serializable for run_ctx.save_json.
    Qlib metrics may include pandas/numpy objects (e.g., Series).
    """
    import numpy as np
    import pandas as pd

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Index,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series,)):
        # Use dict for labeled series; drop NA for compactness.
        try:
            return {str(k): _to_jsonable(v) for k, v in obj.dropna().to_dict().items()}
        except Exception:
            return [_to_jsonable(v) for v in obj.to_list()]
    if isinstance(obj, (pd.DataFrame,)):
        try:
            return obj.to_dict(orient="list")
        except Exception:
            return obj.astype(object).to_dict()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _risk_analysis_to_metrics(risk_obj: Any) -> Dict[str, float]:
    """
    qlib.contrib.evaluate.risk_analysis() returns a pandas DataFrame whose index contains metric names
    and whose single column is typically named 'risk'. Convert it to a flat dict[str, float].
    """
    import numpy as np
    import pandas as pd

    if risk_obj is None:
        return {}

    if isinstance(risk_obj, dict):
        out: Dict[str, float] = {}
        for k, v in risk_obj.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    if isinstance(risk_obj, pd.Series):
        ser = risk_obj
    elif isinstance(risk_obj, pd.DataFrame):
        if "risk" in risk_obj.columns:
            ser = risk_obj["risk"]
        else:
            ser = risk_obj.iloc[:, 0] if risk_obj.shape[1] > 0 else pd.Series(dtype=float)
    else:
        return {}

    out = {}
    for k, v in ser.items():
        try:
            fv = float(v)
            if not np.isfinite(fv):
                continue
            out[str(k)] = fv
        except Exception:
            continue
    return out


def _calc_signal_metrics(
    *,
    label_df,
    signal_series,
    start_time,
    end_time,
) -> tuple[Dict[str, float], Any]:
    """
    Compute IC/ICIR/RankIC/RankICIR from a signal series and label panel for a given time window.

    label_df columns: timestamp, ticker, label
    signal_series index: (datetime, instrument) and values: score
    """
    import numpy as np
    import pandas as pd

    if label_df is None or signal_series is None:
        return {}, pd.DataFrame(columns=["timestamp", "IC", "Rank IC", "n"])

    start_ts = pd.to_datetime(start_time).normalize()
    end_ts = pd.to_datetime(end_time).normalize()

    lbl = label_df[["timestamp", "ticker", "label"]].copy()
    lbl["timestamp"] = _normalize_trade_dates(lbl["timestamp"])
    lbl = lbl[(lbl["timestamp"] >= start_ts) & (lbl["timestamp"] <= end_ts)]

    sig = signal_series.rename("score").reset_index()
    sig = sig.rename(columns={"datetime": "timestamp", "instrument": "ticker"})
    sig["timestamp"] = _normalize_trade_dates(sig["timestamp"])
    sig = sig[(sig["timestamp"] >= start_ts) & (sig["timestamp"] <= end_ts)]

    m = lbl.merge(sig, on=["timestamp", "ticker"], how="inner")
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "score"])
    if m.empty:
        return {}, pd.DataFrame(columns=["timestamp", "IC", "Rank IC", "n"])

    def _per_day(g: pd.DataFrame) -> pd.Series:
        n = int(len(g))
        if n < 5:
            return pd.Series({"IC": np.nan, "Rank IC": np.nan, "n": n})
        ic = g["score"].corr(g["label"], method="pearson")
        ric = g["score"].corr(g["label"], method="spearman")
        return pd.Series({"IC": ic, "Rank IC": ric, "n": n})

    ts = m.groupby("timestamp", sort=True).apply(_per_day).reset_index()
    ic = pd.to_numeric(ts["IC"], errors="coerce")
    ric = pd.to_numeric(ts["Rank IC"], errors="coerce")

    def _safe_mean(x):
        v = float(x.mean()) if x is not None and x.notna().any() else None
        return v

    def _safe_icir(x):
        if x is None or not x.notna().any():
            return None
        std = float(x.std())
        if std == 0.0 or not np.isfinite(std):
            return None
        return float(x.mean()) / std

    out: Dict[str, float] = {}
    ic_mean = _safe_mean(ic)
    ric_mean = _safe_mean(ric)
    if ic_mean is not None:
        out["IC"] = float(ic_mean)
    icir = _safe_icir(ic)
    if icir is not None:
        out["ICIR"] = float(icir)
    if ric_mean is not None:
        out["Rank IC"] = float(ric_mean)
    ricir = _safe_icir(ric)
    if ricir is not None:
        out["Rank ICIR"] = float(ricir)

    return out, ts


def _calc_report_metrics(
    *,
    report_df,
    freq: str = "day",
) -> Dict[str, Dict[str, float]]:
    """
    Compute Qlib-style risk_analysis metrics for:
    - benchmark (bench)
    - excess_return_without_cost (return - bench)
    - excess_return_with_cost (return - bench - cost)

    Returns:
      {
        "benchmark": {...},
        "excess_return_without_cost": {...},
        "excess_return_with_cost": {...},
      }
    """
    if report_df is None:
        return {"benchmark": {}, "excess_return_without_cost": {}, "excess_return_with_cost": {}}

    try:
        from qlib.contrib.evaluate import risk_analysis
    except Exception:
        return {"benchmark": {}, "excess_return_without_cost": {}, "excess_return_with_cost": {}}

    bench = report_df.get("bench") if hasattr(report_df, "get") else None
    ret = report_df.get("return") if hasattr(report_df, "get") else None
    cost = report_df.get("cost", 0.0) if hasattr(report_df, "get") else 0.0
    if bench is None or ret is None:
        return {"benchmark": {}, "excess_return_without_cost": {}, "excess_return_with_cost": {}}

    benchmark_m = _risk_analysis_to_metrics(risk_analysis(bench, freq=freq))
    ex0 = _risk_analysis_to_metrics(risk_analysis(ret - bench, freq=freq))
    ex1 = _risk_analysis_to_metrics(risk_analysis(ret - bench - cost, freq=freq))
    return {"benchmark": benchmark_m, "excess_return_without_cost": ex0, "excess_return_with_cost": ex1}


def _positions_to_normal_dict(positions: Any) -> dict:
    """
    Convert qlib backtest positions output into Qlib-like positions_normal_1day.pkl format:
      dict[datetime -> list[ {instrument, amount, price, value} ]]
    """
    import pandas as pd

    if not positions:
        return {}

    if isinstance(positions, dict):
        out: dict = {}
        for dt, pos in positions.items():
            try:
                date_key = pd.Timestamp(dt).normalize()
            except Exception:
                continue
            try:
                stock_list = pos.get_stock_list()
            except Exception:
                continue
            rows = []
            for stock_id in stock_list:
                try:
                    amount = pos.get_stock_amount(stock_id)
                    price = pos.get_stock_price(stock_id)
                    value = (amount or 0.0) * (price or 0.0)
                    rows.append(
                        {
                            "instrument": stock_id,
                            "amount": float(amount) if amount is not None else 0.0,
                            "price": float(price) if price is not None else 0.0,
                            "value": float(value),
                        }
                    )
                except Exception:
                    continue
            out[date_key] = rows
        return out

    return {}


def _save_qlib_artifacts(
    *,
    run_ctx,
    iter_prefix: str,
    model_tag: str,
    split: str,
    report_df,
    positions_raw: Any,
) -> None:
    """
    Save a minimal subset of Qlib-style artifacts under runs/<run_id>/qlib_artifacts/.
    """
    try:
        from run.pipeline.stage4 import _build_port_analysis_df  # reuse existing helper
    except Exception:
        _build_port_analysis_df = None  # type: ignore

    base_dir = f"qlib_artifacts/{iter_prefix}/{model_tag}/{split}"
    if report_df is not None:
        run_ctx.save_pickle(f"{base_dir}/report_normal_1day.pkl", report_df)
        if _build_port_analysis_df is not None:
            run_ctx.save_pickle(f"{base_dir}/port_analysis_1day.pkl", _build_port_analysis_df(report_df, freq="1day"))
    pos_dict = _positions_to_normal_dict(positions_raw)
    if pos_dict:
        run_ctx.save_pickle(f"{base_dir}/positions_normal_1day.pkl", pos_dict)


def _split_is_train_valid(
    df,
    *,
    is_start: str,
    is_end: str,
    valid_ratio: float = 0.2,
):
    import pandas as pd

    ts = pd.to_datetime(df["timestamp"])
    is_mask = (ts >= pd.to_datetime(is_start)) & (ts <= pd.to_datetime(is_end))
    df_is = df.loc[is_mask].copy()

    # 날짜 기반 순차 split
    unique_dates = pd.Index(sorted(df_is["timestamp"].unique()))
    if len(unique_dates) < 5:
        # 데이터가 너무 작으면 전부 train로
        return df_is, df_is.iloc[0:0]

    cut = int(len(unique_dates) * (1.0 - valid_ratio))
    cut = min(max(cut, 1), len(unique_dates) - 1)
    train_dates = set(unique_dates[:cut])
    valid_dates = set(unique_dates[cut:])

    df_train = df_is[df_is["timestamp"].isin(train_dates)].copy()
    df_valid = df_is[df_is["timestamp"].isin(valid_dates)].copy()
    return df_train, df_valid


def _ensure_qlib_end_time_has_future_day(
    *,
    start_time,
    end_time,
    run_ctx=None,
    label: str = "",
):
    """
    Qlib backtest calendar은 interval 정의를 위해 end_time 이후의 '다음' 캘린더 인덱스를 참조한다.
    데이터 캘린더가 end_time에서 끝나면 IndexError가 날 수 있어, 미래 캘린더가 없을 때만 end_time을
    직전 trading day로 당겨준다.
    """
    import pandas as pd

    try:
        from qlib.data import D
    except Exception:
        return pd.to_datetime(end_time)

    start_ts = pd.to_datetime(start_time)
    end_ts = pd.to_datetime(end_time)

    try:
        cal_ext = D.calendar(start_time=start_ts, end_time=end_ts + pd.Timedelta(days=30), freq="day")
    except Exception:
        return end_ts

    if cal_ext is None or len(cal_ext) == 0:
        return end_ts

    last_cal = pd.Timestamp(cal_ext[-1])
    if last_cal > end_ts:
        return end_ts

    if len(cal_ext) < 2:
        return end_ts

    adjusted = pd.Timestamp(cal_ext[-2])
    if run_ctx is not None:
        tag = f"{label} " if label else ""
        run_ctx.log(
            f"[Stage4-alpha] {tag}end_time adjusted for qlib calendar boundary: "
            f"{end_ts.date()} -> {adjusted.date()}"
        )
    return adjusted


def _select_combinations(
    *,
    passed_combinations: List[List[Dict[str, Any]]],
    combination_stats: Optional[Dict[Tuple[str, ...], Dict[str, Any]]],
    max_combos: int,
    criterion: str,
) -> List[List[Dict[str, Any]]]:
    if max_combos == 0:
        max_combos = len(passed_combinations)
    max_combos = max(1, min(max_combos, len(passed_combinations)))

    def _combo_key(combo: List[Dict[str, Any]]) -> Tuple[str, ...]:
        return tuple(sorted(str(f.get("name", "")) for f in combo if f.get("name")))

    def _score(combo: List[Dict[str, Any]]) -> float:
        if not combination_stats:
            return 0.0
        ck = _combo_key(combo)
        st = combination_stats.get(ck, {}) if isinstance(combination_stats, dict) else {}
        if criterion == "s2_improvement":
            return float(st.get("s2_ratio_improvement") or 0.0)
        if criterion == "pass_rate":
            return float(st.get("pass_rate") or 0.0)
        if criterion == "mean_return":
            return float(st.get("mean_return_improvement") or 0.0)
        return float(st.get("s2_ratio_improvement") or 0.0)

    ranked = sorted(passed_combinations, key=_score, reverse=True)
    return ranked[:max_combos]


def run_stage4_alphaagent_style(
    *,
    hypothesis_id: str,
    passed_combinations: List[List[Dict[str, Any]]],
    ohlcv_df,
    formula_df,
    passed_formulas: Optional[List[Dict[str, Any]]] = None,
    hypothesis: Optional[Dict[str, Any]] = None,
    cfg=None,
    run_ctx=None,
    verbose: bool = True,
    outer_iter: int | None = None,
    combination_stats: Optional[Dict[Tuple[str, ...], Dict[str, Any]]] = None,
):
    """
    Stage4 대체 구현:
    - Feature: Stage3에서 선택된 formula 조합(연속값)
    - Model: LightGBM regression (직접 학습)
    - Signal: 예측값(score)
    - Backtest: Qlib TopkDropoutStrategy + backtest_daily
    """
    from run.config import load_rd_config
    from run.pipeline.stage4 import Stage4Result  # 기존 dataclass 재사용

    import numpy as np
    import pandas as pd
    import polars as pl

    cfg = cfg or load_rd_config()
    assert run_ctx is not None, "run_ctx is required"

    is_start = cfg.data_split.in_sample_start
    is_end = cfg.data_split.in_sample_end
    oos_start = cfg.data_split.out_sample_start
    oos_end = cfg.data_split.out_sample_end

    # CLI override
    topk = int(os.environ.get("ALPHA_STAGE4_TOPK", str(_CLI_TOPK)))
    n_drop = int(os.environ.get("ALPHA_STAGE4_N_DROP", str(_CLI_N_DROP)))
    max_combos = int(os.environ.get("ALPHA_STAGE4_MAX_COMBOS", str(_CLI_MAX_COMBOS)))
    feature_mode = str(os.environ.get("ALPHA_STAGE4_FEATURE_MODE", str(_CLI_FEATURE_MODE))).strip().lower()
    combo_agg = str(os.environ.get("ALPHA_STAGE4_COMBO_AGG", str(_CLI_COMBO_AGG))).strip().lower()
    use_csz = bool(int(os.environ.get("ALPHA_STAGE4_CSZ", "1" if _CLI_CSZ else "0")))

    combos: List[List[Dict[str, Any]]] = []
    if feature_mode in {"combo", "union", "combo_scores"}:
        combos = _select_combinations(
            passed_combinations=passed_combinations,
            combination_stats=combination_stats,
            max_combos=max_combos if max_combos is not None else cfg.stage4.max_combinations_to_evaluate,
            criterion=str(getattr(cfg.stage4, "combination_selection_criterion", "s2_improvement") or "s2_improvement"),
        )

        if not combos:
            raise ValueError("No passed_combinations to evaluate in Stage4 (alphaagent style).")

    # Base panel: timestamp, ticker, close + formula columns(조합별 subset)
    base_cols = ["timestamp", "ticker", "close"]
    base_pd = ohlcv_df.select([c for c in base_cols if c in ohlcv_df.columns]).to_pandas()
    base_pd["timestamp"] = _normalize_trade_dates(base_pd["timestamp"])

    # 미리 label 계산을 위해 close 포함
    # formula_df는 timestamp/ticker + formula cols
    all_formula_cols = set(formula_df.columns) - {"timestamp", "ticker"}
    # 합집합만 로딩(필요 없는 컬럼 merge 방지)
    need_cols: set[str] = set()
    if feature_mode in {"obs_pool", "obs_ensemble"}:
        if not passed_formulas:
            raise ValueError("feature-mode=obs_pool/obs_ensemble requires `passed_formulas` (e.g., Stage2 passed_formulas).")
        for f in passed_formulas:
            n = str(f.get("name") or "")
            if n and n in all_formula_cols:
                need_cols.add(n)
    elif feature_mode == "all":
        # LightGBM input = formula_df columns (max). Avoid restricting to Stage3 combos.
        need_cols = set(all_formula_cols)
    else:
        for combo in combos:
            for f in combo:
                n = f.get("name")
                if n in all_formula_cols:
                    need_cols.add(n)

    f_pd = formula_df.select(["timestamp", "ticker", *sorted(need_cols)]).to_pandas()
    f_pd["timestamp"] = _normalize_trade_dates(f_pd["timestamp"])

    panel = base_pd.merge(f_pd, on=["timestamp", "ticker"], how="inner")
    panel = panel.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(drop=True)
    panel = _compute_label_close_t1_to_t2(panel)
    label_df = panel[["timestamp", "ticker", "label"]].copy()

    # OOS backtest를 위해 prediction은 OOS 구간 전일까지도 필요(TopkDropoutStrategy가 shift=1 사용)
    # 여기서는 panel 전체에서 예측 생성 후, backtest start/end로 자른다.

    _block_dask_for_lightgbm()
    import lightgbm as lgb

    lgb_params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_data_in_leaf": 200,
        "seed": 42,
        "verbose": -1,
    }

    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy

    exchange_kwargs = {
        "freq": "day",
        "limit_threshold": cfg.qlib.limit_threshold,
        "deal_price": cfg.qlib.deal_price,
        "open_cost": cfg.qlib.open_cost,
        "close_cost": cfg.qlib.close_cost,
        "min_cost": cfg.qlib.min_cost,
    }

    all_combo_results: List[Dict[str, Any]] = []
    best_combo_idx = None
    best_is_ir = -1e18
    best_is_panel_pl: pl.DataFrame | None = None
    best_oos_panel_pl: pl.DataFrame | None = None

    is_start_ts = pd.to_datetime(is_start)
    is_end_ts = _ensure_qlib_end_time_has_future_day(
        start_time=is_start_ts,
        end_time=pd.to_datetime(is_end),
        run_ctx=run_ctx,
        label="IS",
    )
    oos_start_ts = pd.to_datetime(oos_start)
    oos_end_ts = _ensure_qlib_end_time_has_future_day(
        start_time=oos_start_ts,
        end_time=pd.to_datetime(oos_end),
        run_ctx=run_ctx,
        label="OOS",
    )

    iter_prefix = f"iter_{outer_iter}" if outer_iter is not None else "iter_1"

    def _fit_predict_signal_series(
        *,
        model_key: str,
        feat_cols: List[str],
        polarity_by_name: Dict[str, str],
        source_panel,
    ):
        """
        Fit LGBM on IS(train/valid) and produce a signal Series(index=[datetime,instrument]) over the full panel.
        Returns: (signal_series, booster, df_used_for_pred(optional))
        """
        df = source_panel[["timestamp", "ticker", "close", "label", *feat_cols]].copy()

        # polarity 정렬: lower_is_more_true는 sign flip해서 "higher is better"로 통일
        for name, pol in polarity_by_name.items():
            if name in df.columns and str(pol).startswith("lower"):
                df[name] = -df[name]

        df = df.replace([np.inf, -np.inf], np.nan)
        df_train, df_valid = _split_is_train_valid(df, is_start=is_start, is_end=is_end, valid_ratio=0.2)
        df_train = df_train.dropna(subset=["label"])
        df_valid = df_valid.dropna(subset=["label"])
        if len(df_train) < 1000:
            raise RuntimeError("Not enough training rows after label drop.")

        if use_csz:
            df_train = _cs_zscore(df_train, feat_cols, by="timestamp")
            if len(df_valid) > 0:
                df_valid = _cs_zscore(df_valid, feat_cols, by="timestamp")

        X_train = df_train[feat_cols].to_numpy(dtype=np.float32, copy=False)
        y_train = df_train["label"].to_numpy(dtype=np.float32, copy=False)

        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = []
        if len(df_valid) > 0:
            X_valid = df_valid[feat_cols].to_numpy(dtype=np.float32, copy=False)
            y_valid = df_valid["label"].to_numpy(dtype=np.float32, copy=False)
            valid_set = lgb.Dataset(X_valid, label=y_valid, free_raw_data=True)
            valid_sets.append(valid_set)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

        booster = lgb.train(
            params=lgb_params,
            train_set=train_set,
            num_boost_round=500,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Predict over full panel (same behavior as before)
        pred_df = df[["timestamp", "ticker", *feat_cols]].copy()
        if use_csz:
            pred_df = _cs_zscore(pred_df, feat_cols, by="timestamp")

        X_all = pred_df[feat_cols].to_numpy(dtype=np.float32, copy=False)
        pred = booster.predict(X_all, num_iteration=booster.best_iteration or booster.current_iteration())

        signal_df = pred_df[["timestamp", "ticker"]].copy()
        signal_df["score"] = pred.astype(np.float64)
        signal_df["timestamp"] = _normalize_trade_dates(signal_df["timestamp"])

        signal_series = signal_df.set_index(["timestamp", "ticker"])["score"]
        signal_series.index.names = ["datetime", "instrument"]
        signal_series = signal_series[~signal_series.index.duplicated(keep="last")].sort_index()

        return signal_series, booster


    def _save_signal_metrics_csv(
        *,
        model_tag: str,
        is_ts_df,
        oos_ts_df,
    ) -> None:
        try:
            run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/{model_tag}/signal_metrics_is.csv", is_ts_df)
            run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/{model_tag}/signal_metrics_oos.csv", oos_ts_df)
        except Exception:
            pass


    def _backtest_from_signal(
        *,
        model_key: str,
        signal_series,
    ) -> Dict[str, Any]:
        """
        Run IS and OOS backtests from a given signal series.
        """
        strategy = TopkDropoutStrategy(
            signal=signal_series,
            topk=topk,
            n_drop=n_drop,
            risk_degree=0.95,
        )

        is_report_df, is_positions_raw = backtest_daily(
            start_time=is_start_ts,
            end_time=is_end_ts,
            strategy=strategy,
            account=cfg.qlib.init_cash,
            benchmark=cfg.qlib.benchmark,
            exchange_kwargs=exchange_kwargs,
        )
        is_daily_returns = is_report_df["return"] - is_report_df.get("cost", 0.0)
        is_metrics = _risk_analysis_to_metrics(risk_analysis(is_daily_returns, freq="day"))

        oos_report_df, oos_positions_raw = backtest_daily(
            start_time=oos_start_ts,
            end_time=oos_end_ts,
            strategy=strategy,
            account=cfg.qlib.init_cash,
            benchmark=cfg.qlib.benchmark,
            exchange_kwargs=exchange_kwargs,
        )
        oos_daily_returns = oos_report_df["return"] - oos_report_df.get("cost", 0.0)
        oos_metrics = _risk_analysis_to_metrics(risk_analysis(oos_daily_returns, freq="day"))

        is_turnover_sum = (
            float(getattr(is_report_df.get("turnover"), "fillna", lambda *_: 0.0)(0.0).sum())
            if "turnover" in is_report_df.columns
            else 0.0
        )
        oos_turnover_sum = (
            float(getattr(oos_report_df.get("turnover"), "fillna", lambda *_: 0.0)(0.0).sum())
            if "turnover" in oos_report_df.columns
            else 0.0
        )

        is_sig_m, is_sig_ts = _calc_signal_metrics(
            label_df=label_df,
            signal_series=signal_series,
            start_time=is_start_ts,
            end_time=is_end_ts,
        )
        oos_sig_m, oos_sig_ts = _calc_signal_metrics(
            label_df=label_df,
            signal_series=signal_series,
            start_time=oos_start_ts,
            end_time=oos_end_ts,
        )
        _save_signal_metrics_csv(model_tag=model_key, is_ts_df=is_sig_ts, oos_ts_df=oos_sig_ts)

        is_report_metrics = _calc_report_metrics(report_df=is_report_df, freq="day")
        oos_report_metrics = _calc_report_metrics(report_df=oos_report_df, freq="day")

        return {
            "model_key": model_key,
            "is_metrics": dict(is_metrics),
            "oos_metrics": dict(oos_metrics),
            "is_signal_metrics": dict(is_sig_m),
            "oos_signal_metrics": dict(oos_sig_m),
            "is_report_metrics": dict(is_report_metrics),
            "oos_report_metrics": dict(oos_report_metrics),
            "is_turnover_sum": is_turnover_sum,
            "oos_turnover_sum": oos_turnover_sum,
            "is_report_df": is_report_df,
            "is_positions_raw": is_positions_raw,
            "oos_report_df": oos_report_df,
            "oos_positions_raw": oos_positions_raw,
        }


    def _run_one_model(
        *,
        model_key: str,
        feat_cols: List[str],
        polarity_by_name: Dict[str, str],
        source_panel,
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper: fit -> predict -> backtest.
        """
        sig, _booster = _fit_predict_signal_series(
            model_key=model_key,
            feat_cols=feat_cols,
            polarity_by_name=polarity_by_name,
            source_panel=source_panel,
        )
        bt = _backtest_from_signal(model_key=model_key, signal_series=sig)
        bt["n_feat"] = len(feat_cols)
        bt["feat_cols"] = list(feat_cols)
        bt["signal_series"] = sig
        return bt


    if feature_mode not in {"combo", "union", "all", "combo_scores", "obs_pool", "obs_ensemble"}:
        raise ValueError(f"Unknown --feature-mode {feature_mode!r} (expected: combo|union|all|combo_scores|obs_pool|obs_ensemble)")

# combo: Stage3에서 통과한 각 조합을 하나씩 따로 돌림. 조합 A면 그 조합의 수식들만 feature로 LGBM 학습 → 예측 → TopkDropout 백테스트. (조합별로 모델이 N개)
# union: Stage3에서 선택된 상위 --max-combos개 조합에 포함된 수식들을 전부 합집합으로 묶어서 한 번에 LGBM 학습/예측/백테스트. (모델 1개, feature는 “조합 풀”)
# all: formula_df에 있는 수식 컬럼(가능한 것)을 전부 feature로 한 번에 LGBM 학습/예측/백테스트. (모델 1개, feature 최대)
# combo_scores: “조합(AND) 의미”를 살리려고, 각 조합을 1개의 조합 점수 feature로 만들어서(combo_score_000, combo_score_001 …) 그걸로 LGBM을 한 번 학습/예측/백테스트. 조합 점수는 --combo-agg min|gmean으로 만듦.
# obs_pool: observation_id별 PASS 수식 “풀”을 전부 feature로 넣어서(그룹별 1개 선택 조합 X) LGBM이 자동으로 중요도를 학습하게 함.
# obs_ensemble: obs별로 (PASS 수식 풀)만으로 각각 LGBM을 1개씩 학습/예측해서 p_obs를 만들고,
#               최종 score = agg(p_obs1,p_obs2,...) 로 결합 후 TopkDropout 백테스트. (모델은 obs 개수만큼)


    if combo_agg not in {"min", "gmean"}:
        raise ValueError(f"Unknown --combo-agg {combo_agg!r} (expected: min|gmean)")


    if feature_mode == "obs_pool":
        # Use Stage2 passed_formulas (grouped by observation_id) as features, without 1-per-observation combinations.
        assert passed_formulas is not None

        polarity_by_name: Dict[str, str] = {}
        obs_to_names: Dict[str, List[str]] = {}
        for f in passed_formulas:
            name = str(f.get("name") or "")
            if not name or name not in panel.columns:
                continue
            obs_id = str(f.get("observation_id") or f.get("obs_id") or "UNKNOWN_OBS")
            obs_to_names.setdefault(obs_id, []).append(name)
            polarity_by_name.setdefault(name, str(f.get("polarity") or ""))

        # Flatten preserving a stable order (obs_id then name)
        feat_cols: List[str] = []
        for obs_id in sorted(obs_to_names.keys()):
            for name in sorted(set(obs_to_names[obs_id])):
                feat_cols.append(name)

        if not feat_cols:
            raise RuntimeError("feature-mode=obs_pool produced zero features (no passed formulas found in panel).")

        res = _run_one_model(
            model_key="obs_pool",
            feat_cols=feat_cols,
            polarity_by_name=polarity_by_name,
            source_panel=panel,
        )
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag="obs_pool",
            split="is",
            report_df=res.get("is_report_df"),
            positions_raw=res.get("is_positions_raw"),
        )
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag="obs_pool",
            split="oos",
            report_df=res.get("oos_report_df"),
            positions_raw=res.get("oos_positions_raw"),
        )
        is_ir = float(res["is_metrics"].get("information_ratio") or 0.0)
        best_is_ir = is_ir
        best_combo_idx = None
        is_pd = res["is_report_df"].reset_index().rename(columns={"index": "date"})
        oos_pd = res["oos_report_df"].reset_index().rename(columns={"index": "date"})
        best_is_panel_pl = pl.from_pandas(is_pd)
        best_oos_panel_pl = pl.from_pandas(oos_pd)
        all_combo_results.append(
            {
                "combo_idx": None,
                "combo_key": ("OBS_POOL",),
                "formula_names": feat_cols,
                "topk": topk,
                "n_drop": n_drop,
                "use_cszscore": use_csz,
                "lgb_params": lgb_params,
                "is_metrics": _to_jsonable(dict(res["is_metrics"])),
                "oos_metrics": _to_jsonable(dict(res["oos_metrics"])),
                "is_signal_metrics": _to_jsonable(dict(res.get("is_signal_metrics") or {})),
                "oos_signal_metrics": _to_jsonable(dict(res.get("oos_signal_metrics") or {})),
                "is_report_metrics": _to_jsonable(dict(res.get("is_report_metrics") or {})),
                "oos_report_metrics": _to_jsonable(dict(res.get("oos_report_metrics") or {})),
                "data_split": {
                    "insample": {"strategy": _to_jsonable(dict(res["is_metrics"]))},
                    "outsample": {"strategy": _to_jsonable(dict(res["oos_metrics"]))},
                },
                "is_turnover_sum": float(res["is_turnover_sum"]),
                "oos_turnover_sum": float(res["oos_turnover_sum"]),
                "obs_pool": {k: len(set(v)) for k, v in obs_to_names.items()},
            }
        )
        if verbose:
            run_ctx.log(
                f"[Stage4-alpha] feature-mode=obs_pool (k={len(feat_cols)}, obs={len(obs_to_names)}): "
                f"IS_IR={is_ir:.6f}, OOS_IR={float(res['oos_metrics'].get('information_ratio') or 0.0):.6f} "
                f"(turnover IS={float(res['is_turnover_sum']):.3f}, OOS={float(res['oos_turnover_sum']):.3f})"
            )

    elif feature_mode == "obs_ensemble":
        # obs별 PASS 수식 풀로 각각 LGBM 학습 -> p_obs 생성 -> 결합 -> 단일 backtest
        assert passed_formulas is not None

        obs_agg = str(os.environ.get("ALPHA_STAGE4_OBS_AGG", "wsum")).strip().lower()
        obs_eps = float(os.environ.get("ALPHA_STAGE4_OBS_EPS", "0.1"))
        obs_weights_raw = str(os.environ.get("ALPHA_STAGE4_OBS_WEIGHTS", "")).strip()

        if obs_agg not in {"wsum", "min", "gmean"}:
            raise ValueError(f"Unknown ALPHA_STAGE4_OBS_AGG={obs_agg!r} (expected: wsum|min|gmean)")

        polarity_by_name: Dict[str, str] = {}
        obs_to_names: Dict[str, List[str]] = {}
        for f in passed_formulas:
            name = str(f.get("name") or "")
            if not name or name not in panel.columns:
                continue
            obs_id = str(f.get("observation_id") or f.get("obs_id") or "UNKNOWN_OBS")
            obs_to_names.setdefault(obs_id, []).append(name)
            polarity_by_name.setdefault(name, str(f.get("polarity") or ""))

        # obs별 feature list
        obs_ids = sorted(obs_to_names.keys())
        obs_feat_cols: Dict[str, List[str]] = {}
        for obs_id in obs_ids:
            cols = sorted(set(obs_to_names.get(obs_id, [])))
            cols = [c for c in cols if c in panel.columns]
            if cols:
                obs_feat_cols[obs_id] = cols

        if not obs_feat_cols:
            raise RuntimeError("feature-mode=obs_ensemble produced zero obs feature groups (no passed formulas in panel).")

        # obs별 signal 예측
        import pandas as pd
        sig_frames = []
        used_obs = []
        for obs_id in obs_ids:
            feat_cols_i = obs_feat_cols.get(obs_id) or []
            if not feat_cols_i:
                continue

            # obs별 polarity map만 적용해도 되지만, polarity_by_name 전체를 줘도 안전(존재 컬럼만 처리)
            sig_i, _ = _fit_predict_signal_series(
                model_key=f"obs_{obs_id}",
                feat_cols=feat_cols_i,
                polarity_by_name=polarity_by_name,
                source_panel=panel,
            )

            df_i = sig_i.to_frame(name=f"p_{obs_id}")
            sig_frames.append(df_i)
            used_obs.append(obs_id)

        if not sig_frames:
            raise RuntimeError("obs_ensemble: all obs groups were empty or failed to produce signals.")

        # (datetime,instrument) 정렬된 wide DF
        pred_wide = pd.concat(sig_frames, axis=1).sort_index()
        pred_wide = pred_wide.fillna(0.0)

        # 스케일 맞추기: use_csz True면 obs별 예측도 날짜별 CS zscore로 정규화 (권장)
        if use_csz:
            tmp = pred_wide.reset_index().rename(columns={"datetime": "timestamp", "instrument": "ticker"})
            score_cols = [c for c in tmp.columns if c.startswith("p_")]
            tmp = _cs_zscore(tmp, score_cols, by="timestamp")
            pred_wide = tmp.set_index(["timestamp", "ticker"])[score_cols]
            pred_wide.index.names = ["datetime", "instrument"]
            pred_wide = pred_wide.sort_index()

        # 결합
        score_cols = [f"p_{oid}" for oid in used_obs if f"p_{oid}" in pred_wide.columns]
        if not score_cols:
            raise RuntimeError("obs_ensemble: no prediction columns to aggregate.")

        if obs_agg == "min":
            final_score = pred_wide[score_cols].min(axis=1)
            agg_meta = {"obs_agg": "min"}
        elif obs_agg == "gmean":
            # 안정적으로 하려면 percentile로 바꾼 후 gmean (0..1)
            tmp = pred_wide[score_cols].copy()
            tmp = tmp.reset_index()
            # percentile normalize each p_ column per day
            for c in score_cols:
                tmp[c] = tmp.groupby("datetime")[c].rank(method="average", pct=True, ascending=True).astype(float).fillna(0.0)
            # gmean
            import numpy as np
            eps = 1e-12
            vals = np.clip(tmp[score_cols].to_numpy(dtype=np.float64, copy=False), eps, 1.0)
            g = np.exp(np.mean(np.log(vals), axis=1))
            final_score = pd.Series(g, index=pred_wide.index, name="score")
            agg_meta = {"obs_agg": "gmean", "note": "gmean over CS-percentile of per-obs predictions"}
        else:
            # wsum with floor epsilon constraint
            import numpy as np

            def _project_to_floor_simplex(weights: List[float], floor: float) -> List[float]:
                """
                Project weights onto {w_i >= floor, sum(w)=1}.
                """
                w0 = np.asarray(weights, dtype=np.float64)
                n = int(w0.size)
                if n <= 0:
                    raise ValueError("obs_ensemble: empty weights.")
                if not np.isfinite(floor) or floor < 0.0:
                    raise ValueError(f"obs_ensemble: invalid obs_eps={floor!r} (must be >= 0).")
                if floor * n >= 1.0:
                    raise ValueError(f"obs_ensemble: obs_eps too large (eps*n_obs must be < 1). eps={floor}, n_obs={n}")

                # Standard simplex projection helper: argmin ||x-v|| s.t. x>=0, sum(x)=z
                def _proj_simplex(v: np.ndarray, z: float) -> np.ndarray:
                    if z <= 0.0:
                        return np.zeros_like(v)
                    u = np.sort(v)[::-1]
                    cssv = np.cumsum(u)
                    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - z))[0]
                    if rho.size == 0:
                        return np.full_like(v, z / n)
                    rho_i = int(rho[-1])
                    theta = (cssv[rho_i] - z) / float(rho_i + 1)
                    return np.maximum(v - theta, 0.0)

                w0 = np.nan_to_num(w0, nan=0.0, posinf=0.0, neginf=0.0)
                w0 = np.clip(w0, 0.0, None)
                if float(w0.sum()) <= 0.0:
                    w0 = np.full((n,), 1.0 / n, dtype=np.float64)
                else:
                    w0 = w0 / float(w0.sum())

                z = 1.0 - floor * n
                u0 = w0 - floor
                u = _proj_simplex(u0, z=z)
                w = u + floor
                # numeric safety
                w = np.clip(w, floor, None)
                w = w / float(w.sum())
                return [float(x) for x in w.tolist()]

            if obs_weights_raw:
                if "=" in obs_weights_raw:
                    # Support mapping form: "obs_id=0.2,obs_other=0.8" (also accepts "p_obs_id=0.2")
                    raw_map: Dict[str, float] = {}
                    for part in [p.strip() for p in obs_weights_raw.split(",") if p.strip()]:
                        if "=" not in part:
                            continue
                        k, v = part.split("=", 1)
                        raw_map[k.strip()] = float(v.strip())
                    w_in = []
                    for oid in used_obs:
                        if oid in raw_map:
                            w_in.append(float(raw_map[oid]))
                            continue
                        k2 = f"p_{oid}"
                        if k2 in raw_map:
                            w_in.append(float(raw_map[k2]))
                            continue
                        raise ValueError(
                            f"ALPHA_STAGE4_OBS_WEIGHTS is missing weight for obs_id={oid!r}. "
                            f"Provide '{oid}=<w>' or '{k2}=<w>'."
                        )
                else:
                    parts = [p.strip() for p in obs_weights_raw.split(",") if p.strip()]
                    w_in = [float(x) for x in parts]
                    if len(w_in) != len(score_cols):
                        raise ValueError(
                            f"ALPHA_STAGE4_OBS_WEIGHTS has {len(w_in)} weights but need {len(score_cols)} (score_cols={score_cols})"
                        )
            else:
                w_in = [1.0] * len(score_cols)

            w = _project_to_floor_simplex(w_in, floor=float(obs_eps))

            final_score = 0.0
            for wi, c in zip(w, score_cols):
                final_score = final_score + wi * pred_wide[c]
            agg_meta = {"obs_agg": "wsum", "obs_eps": obs_eps, "weights": dict(zip(score_cols, w))}

        # 최종 signal series
        signal_series = final_score.astype(float)
        signal_series.name = "score"
        signal_series = signal_series.sort_index()

        # 백테스트 1회(IS/OOS)
        bt = _backtest_from_signal(model_key="obs_ensemble", signal_series=signal_series)

        # artifacts 저장
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag="obs_ensemble",
            split="is",
            report_df=bt.get("is_report_df"),
            positions_raw=bt.get("is_positions_raw"),
        )
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag="obs_ensemble",
            split="oos",
            report_df=bt.get("oos_report_df"),
            positions_raw=bt.get("oos_positions_raw"),
        )

        is_ir = float(bt["is_metrics"].get("information_ratio") or 0.0)
        best_is_ir = is_ir
        best_combo_idx = None

        is_pd = bt["is_report_df"].reset_index().rename(columns={"index": "date"})
        oos_pd = bt["oos_report_df"].reset_index().rename(columns={"index": "date"})
        best_is_panel_pl = pl.from_pandas(is_pd)
        best_oos_panel_pl = pl.from_pandas(oos_pd)

        all_combo_results.append(
            {
                "combo_idx": None,
                "combo_key": ("OBS_ENSEMBLE", obs_agg),
                "formula_names": [],  # ensemble이라 "단일 formula list"가 애매해서 비움(원하면 obs별로 따로 기록 가능)
                "n_feat": int(sum(len(obs_feat_cols.get(k, [])) for k in used_obs)),
                "topk": topk,
                "n_drop": n_drop,
                "use_cszscore": use_csz,
                "lgb_params": lgb_params,
                "is_metrics": _to_jsonable(dict(bt["is_metrics"])),
                "oos_metrics": _to_jsonable(dict(bt["oos_metrics"])),
                "is_signal_metrics": _to_jsonable(dict(bt.get("is_signal_metrics") or {})),
                "oos_signal_metrics": _to_jsonable(dict(bt.get("oos_signal_metrics") or {})),
                "is_report_metrics": _to_jsonable(dict(bt.get("is_report_metrics") or {})),
                "oos_report_metrics": _to_jsonable(dict(bt.get("oos_report_metrics") or {})),
                "data_split": {
                    "insample": {"strategy": _to_jsonable(dict(bt["is_metrics"]))},
                    "outsample": {"strategy": _to_jsonable(dict(bt["oos_metrics"]))},
                },
                "is_turnover_sum": float(bt["is_turnover_sum"]),
                "oos_turnover_sum": float(bt["oos_turnover_sum"]),
                "obs_pool": {k: len(set(v)) for k, v in obs_to_names.items()},
                "obs_ensemble": {
                    "used_obs": used_obs,
                    "agg": agg_meta,
                    "n_feat_by_obs": {k: len(set(v)) for k, v in obs_to_names.items()},
                },
            }
        )

        if verbose:
            run_ctx.log(
                f"[Stage4-alpha] feature-mode=obs_ensemble (obs={len(used_obs)}, agg={obs_agg}): "
                f"IS_IR={is_ir:.6f}, OOS_IR={float(bt['oos_metrics'].get('information_ratio') or 0.0):.6f} "
                f"(turnover IS={float(bt['is_turnover_sum']):.3f}, OOS={float(bt['oos_turnover_sum']):.3f})"
            )




    elif feature_mode == "all":
        feat_cols = [c for c in panel.columns if c not in {"timestamp", "ticker", "close", "label"}]
        if not feat_cols:
            raise RuntimeError("feature-mode=all produced zero features (no formula columns in panel).")
        # polarity unknown for all; don't flip
        res = _run_one_model(model_key="all_formulas", feat_cols=feat_cols, polarity_by_name={}, source_panel=panel)
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag="combo_all",
            split="is",
            report_df=res.get("is_report_df"),
            positions_raw=res.get("is_positions_raw"),
        )
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag="combo_all",
            split="oos",
            report_df=res.get("oos_report_df"),
            positions_raw=res.get("oos_positions_raw"),
        )
        is_ir = float(res["is_metrics"].get("information_ratio") or 0.0)
        best_is_ir = is_ir
        best_combo_idx = None
        is_pd = res["is_report_df"].reset_index().rename(columns={"index": "date"})
        oos_pd = res["oos_report_df"].reset_index().rename(columns={"index": "date"})
        best_is_panel_pl = pl.from_pandas(is_pd)
        best_oos_panel_pl = pl.from_pandas(oos_pd)
        all_combo_results.append(
            {
                "combo_idx": None,
                "combo_key": ("ALL",),
                "formula_names": feat_cols,
                "topk": topk,
                "n_drop": n_drop,
                "use_cszscore": use_csz,
                "lgb_params": lgb_params,
                "is_metrics": _to_jsonable(dict(res["is_metrics"])),
                "oos_metrics": _to_jsonable(dict(res["oos_metrics"])),
                "is_signal_metrics": _to_jsonable(dict(res.get("is_signal_metrics") or {})),
                "oos_signal_metrics": _to_jsonable(dict(res.get("oos_signal_metrics") or {})),
                "is_report_metrics": _to_jsonable(dict(res.get("is_report_metrics") or {})),
                "oos_report_metrics": _to_jsonable(dict(res.get("oos_report_metrics") or {})),
                # Compatibility with outer-loop/refinement_4to1 expectations
                "data_split": {
                    "insample": {"strategy": _to_jsonable(dict(res["is_metrics"]))},
                    "outsample": {"strategy": _to_jsonable(dict(res["oos_metrics"]))},
                },
                "is_turnover_sum": float(res["is_turnover_sum"]),
                "oos_turnover_sum": float(res["oos_turnover_sum"]),
            }
        )
        if verbose:
            run_ctx.log(
                f"[Stage4-alpha] feature-mode=all (k={len(feat_cols)}): "
                f"IS_IR={is_ir:.6f}, OOS_IR={float(res['oos_metrics'].get('information_ratio') or 0.0):.6f} "
                f"(turnover IS={float(res['is_turnover_sum']):.3f}, OOS={float(res['oos_turnover_sum']):.3f})"
            )

    elif feature_mode == "union":
        union_cols: set[str] = set()
        polarity_by_name: Dict[str, str] = {}
        for combo in combos:
            for f in combo:
                n = str(f.get("name") or "")
                if n and n in panel.columns:
                    union_cols.add(n)
                    polarity_by_name.setdefault(n, str(f.get("polarity") or ""))
        feat_cols = sorted(union_cols)
        if not feat_cols:
            raise RuntimeError("feature-mode=union produced zero features.")
        res = _run_one_model(
            model_key="union_top_combos",
            feat_cols=feat_cols,
            polarity_by_name=polarity_by_name,
            source_panel=panel,
        )
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag="combo_union",
            split="is",
            report_df=res.get("is_report_df"),
            positions_raw=res.get("is_positions_raw"),
        )
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag="combo_union",
            split="oos",
            report_df=res.get("oos_report_df"),
            positions_raw=res.get("oos_positions_raw"),
        )
        is_ir = float(res["is_metrics"].get("information_ratio") or 0.0)
        best_is_ir = is_ir
        best_combo_idx = None
        is_pd = res["is_report_df"].reset_index().rename(columns={"index": "date"})
        oos_pd = res["oos_report_df"].reset_index().rename(columns={"index": "date"})
        best_is_panel_pl = pl.from_pandas(is_pd)
        best_oos_panel_pl = pl.from_pandas(oos_pd)
        all_combo_results.append(
            {
                "combo_idx": None,
                "combo_key": ("UNION",),
                "formula_names": feat_cols,
                "topk": topk,
                "n_drop": n_drop,
                "use_cszscore": use_csz,
                "lgb_params": lgb_params,
                "is_metrics": _to_jsonable(dict(res["is_metrics"])),
                "oos_metrics": _to_jsonable(dict(res["oos_metrics"])),
                "is_signal_metrics": _to_jsonable(dict(res.get("is_signal_metrics") or {})),
                "oos_signal_metrics": _to_jsonable(dict(res.get("oos_signal_metrics") or {})),
                "is_report_metrics": _to_jsonable(dict(res.get("is_report_metrics") or {})),
                "oos_report_metrics": _to_jsonable(dict(res.get("oos_report_metrics") or {})),
                "data_split": {
                    "insample": {"strategy": _to_jsonable(dict(res["is_metrics"]))},
                    "outsample": {"strategy": _to_jsonable(dict(res["oos_metrics"]))},
                },
                "is_turnover_sum": float(res["is_turnover_sum"]),
                "oos_turnover_sum": float(res["oos_turnover_sum"]),
            }
        )
        if verbose:
            run_ctx.log(
                f"[Stage4-alpha] feature-mode=union (k={len(feat_cols)}): "
                f"IS_IR={is_ir:.6f}, OOS_IR={float(res['oos_metrics'].get('information_ratio') or 0.0):.6f} "
                f"(turnover IS={float(res['is_turnover_sum']):.3f}, OOS={float(res['oos_turnover_sum']):.3f})"
            )

    elif feature_mode == "combo_scores":
        # Build "combo score" features that preserve AND-like meaning for each combination.
        # - Normalize each formula to cross-sectional percentile (0..1) per day (after polarity flip).
        # - Aggregate within a combo: min(p_i) (soft-AND bottleneck) or gmean(p_i) (smooth AND).
        import numpy as np

        # Collect needed formula columns and polarity
        need_formula: set[str] = set()
        polarity_by_name: Dict[str, str] = {}
        combo_to_names: List[List[str]] = []
        for combo in combos:
            names = []
            for f in combo:
                n = str(f.get("name") or "")
                if n and n in panel.columns:
                    names.append(n)
                    need_formula.add(n)
                    polarity_by_name.setdefault(n, str(f.get("polarity") or ""))
            combo_to_names.append(names)

        need_cols_sorted = sorted(need_formula)
        if not need_cols_sorted:
            raise RuntimeError("feature-mode=combo_scores produced zero base formula columns.")

        # Build a working frame with only required columns
        work = panel[["timestamp", "ticker", "close", "label", *need_cols_sorted]].copy()

        # Polarity flip before percentile so that "higher is more true" => higher percentile
        for name, pol in polarity_by_name.items():
            if name in work.columns and str(pol).startswith("lower"):
                work[name] = -work[name]

        # Percentile normalize per day for all required formulas once
        work = _cs_percentile(work, need_cols_sorted, by="timestamp")

        # Create combo score columns
        combo_feat_cols: List[str] = []
        for i, names in enumerate(combo_to_names):
            if not names:
                continue
            col_name = f"combo_score_{i:03d}"
            vals = work[names].to_numpy(dtype=np.float64, copy=False)
            if combo_agg == "min":
                score = np.nanmin(vals, axis=1)
            else:
                # geometric mean with epsilon to avoid log(0)
                eps = 1e-12
                score = np.exp(np.nanmean(np.log(np.clip(vals, eps, 1.0)), axis=1))
            work[col_name] = score.astype(np.float32)
            combo_feat_cols.append(col_name)

        if not combo_feat_cols:
            raise RuntimeError("feature-mode=combo_scores produced zero combo features.")

        res = _run_one_model(
            model_key=f"combo_scores_{combo_agg}",
            feat_cols=combo_feat_cols,
            polarity_by_name={},  # already encoded in combo scores
            source_panel=work,
        )
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag=f"combo_scores_{combo_agg}",
            split="is",
            report_df=res.get("is_report_df"),
            positions_raw=res.get("is_positions_raw"),
        )
        _save_qlib_artifacts(
            run_ctx=run_ctx,
            iter_prefix=iter_prefix,
            model_tag=f"combo_scores_{combo_agg}",
            split="oos",
            report_df=res.get("oos_report_df"),
            positions_raw=res.get("oos_positions_raw"),
        )
        is_ir = float(res["is_metrics"].get("information_ratio") or 0.0)
        best_is_ir = is_ir
        best_combo_idx = None
        is_pd = res["is_report_df"].reset_index().rename(columns={"index": "date"})
        oos_pd = res["oos_report_df"].reset_index().rename(columns={"index": "date"})
        best_is_panel_pl = pl.from_pandas(is_pd)
        best_oos_panel_pl = pl.from_pandas(oos_pd)

        all_combo_results.append(
            {
                "combo_idx": None,
                "combo_key": ("COMBO_SCORES", combo_agg),
                "formula_names": combo_feat_cols,
                "topk": topk,
                "n_drop": n_drop,
                "use_cszscore": use_csz,
                "lgb_params": lgb_params,
                "is_metrics": _to_jsonable(dict(res["is_metrics"])),
                "oos_metrics": _to_jsonable(dict(res["oos_metrics"])),
                "is_signal_metrics": _to_jsonable(dict(res.get("is_signal_metrics") or {})),
                "oos_signal_metrics": _to_jsonable(dict(res.get("oos_signal_metrics") or {})),
                "is_report_metrics": _to_jsonable(dict(res.get("is_report_metrics") or {})),
                "oos_report_metrics": _to_jsonable(dict(res.get("oos_report_metrics") or {})),
                "data_split": {
                    "insample": {"strategy": _to_jsonable(dict(res["is_metrics"]))},
                    "outsample": {"strategy": _to_jsonable(dict(res["oos_metrics"]))},
                },
                "is_turnover_sum": float(res["is_turnover_sum"]),
                "oos_turnover_sum": float(res["oos_turnover_sum"]),
            }
        )
        if verbose:
            run_ctx.log(
                f"[Stage4-alpha] feature-mode=combo_scores/{combo_agg} (k={len(combo_feat_cols)}): "
                f"IS_IR={is_ir:.6f}, OOS_IR={float(res['oos_metrics'].get('information_ratio') or 0.0):.6f} "
                f"(turnover IS={float(res['is_turnover_sum']):.3f}, OOS={float(res['oos_turnover_sum']):.3f})"
            )

    else:
        for combo_idx, combo in enumerate(combos):
            feat_cols = [str(f.get("name")) for f in combo if f.get("name") in panel.columns]
            if not feat_cols:
                continue

            polarity_by_name = {str(f.get("name")): str(f.get("polarity") or "") for f in combo if f.get("name")}
            try:
                res = _run_one_model(
                    model_key=f"combo_{combo_idx}",
                    feat_cols=feat_cols,
                    polarity_by_name=polarity_by_name,
                    source_panel=panel,
                )
            except Exception:
                continue
            _save_qlib_artifacts(
                run_ctx=run_ctx,
                iter_prefix=iter_prefix,
                model_tag=f"combo_{combo_idx}",
                split="is",
                report_df=res.get("is_report_df"),
                positions_raw=res.get("is_positions_raw"),
            )
            _save_qlib_artifacts(
                run_ctx=run_ctx,
                iter_prefix=iter_prefix,
                model_tag=f"combo_{combo_idx}",
                split="oos",
                report_df=res.get("oos_report_df"),
                positions_raw=res.get("oos_positions_raw"),
            )

            is_metrics = res["is_metrics"]
            oos_metrics = res["oos_metrics"]
            is_turnover_sum = float(res["is_turnover_sum"])
            oos_turnover_sum = float(res["oos_turnover_sum"])

            is_ir = float(is_metrics.get("information_ratio") or 0.0)
            if is_ir > best_is_ir:
                best_is_ir = is_ir
                best_combo_idx = combo_idx
                is_pd = res["is_report_df"].reset_index().rename(columns={"index": "date"})
                oos_pd = res["oos_report_df"].reset_index().rename(columns={"index": "date"})
                best_is_panel_pl = pl.from_pandas(is_pd)
                best_oos_panel_pl = pl.from_pandas(oos_pd)

            combo_key = tuple(sorted(str(f.get("name", "")) for f in combo if f.get("name")))
            all_combo_results.append(
                {
                    "combo_idx": combo_idx,
                    "combo_key": combo_key,
                    "formula_names": feat_cols,
                    "topk": topk,
                    "n_drop": n_drop,
                    "use_cszscore": use_csz,
                    "lgb_params": lgb_params,
                    "is_metrics": _to_jsonable(dict(is_metrics)),
                    "oos_metrics": _to_jsonable(dict(oos_metrics)),
                    "is_signal_metrics": _to_jsonable(dict(res.get("is_signal_metrics") or {})),
                    "oos_signal_metrics": _to_jsonable(dict(res.get("oos_signal_metrics") or {})),
                    "is_report_metrics": _to_jsonable(dict(res.get("is_report_metrics") or {})),
                    "oos_report_metrics": _to_jsonable(dict(res.get("oos_report_metrics") or {})),
                    "data_split": {
                        "insample": {"strategy": _to_jsonable(dict(is_metrics))},
                        "outsample": {"strategy": _to_jsonable(dict(oos_metrics))},
                    },
                    "is_turnover_sum": is_turnover_sum,
                    "oos_turnover_sum": oos_turnover_sum,
                }
            )

            if verbose:
                run_ctx.log(
                    f"[Stage4-alpha] combo={combo_idx+1}/{len(combos)} "
                    f"(k={len(feat_cols)}): IS_IR={is_ir:.6f}, OOS_IR={float(oos_metrics.get('information_ratio') or 0.0):.6f} "
                    f"(turnover IS={is_turnover_sum:.3f}, OOS={oos_turnover_sum:.3f})"
                )

    if best_is_panel_pl is None or best_oos_panel_pl is None:
        raise RuntimeError("Stage4-alpha produced no backtest results (all combos skipped).")

    summary = {
        "stage4_mode": "alphaagent_lgbm_topkdropout",
        "feature_mode": feature_mode,
        "combo_agg": combo_agg,
        "hypothesis_id": hypothesis_id,
        "market": os.getenv("MARKET", "cn").lower(),
        "data_split": {
            "in_sample_start": is_start,
            "in_sample_end": is_end,
            "out_sample_start": oos_start,
            "out_sample_end": oos_end,
        },
        "selection": {
            "criterion": "is_information_ratio",
            "best_combo_idx": best_combo_idx,
            "best_is_information_ratio": best_is_ir,
            "evaluated_combos": len(all_combo_results),
        },
        "all_combinations": all_combo_results,
    }

    lines = [
        "# Stage 4 (AlphaAgent-style): LGBM + TopkDropoutStrategy",
        "",
        f"- Market: `{summary['market']}`",
        f"- IS: `{is_start} ~ {is_end}` / OOS: `{oos_start} ~ {oos_end}`",
        f"- topk: `{topk}`, n_drop: `{n_drop}`",
        f"- evaluated combos: `{len(all_combo_results)}` (best by IS IR: `{best_combo_idx}`)",
        "",
        "## Top combinations (by IS IR)",
        "",
        "| combo_idx | n_feat | IS_IR | OOS_IR |",
        "|---:|---:|---:|---:|",
    ]
    # sort for reporting
    ranked = sorted(
        all_combo_results,
        key=lambda r: float((r.get("is_metrics") or {}).get("information_ratio") or 0.0),
        reverse=True,
    )[:10]
    for r in ranked:
        is_ir = float((r.get("is_metrics") or {}).get("information_ratio") or 0.0)
        oos_ir = float((r.get("oos_metrics") or {}).get("information_ratio") or 0.0)
        lines.append(f"| {r.get('combo_idx')} | {len(r.get('formula_names') or [])} | {is_ir:.3f} | {oos_ir:.3f} |")
    report_md = "\n".join(lines) + "\n"

    # 저장 (Stage4 전용 아티팩트)
    # Qlib-style summary CSVs (same location/filename pattern as stage4.py)
    try:
        import pandas as pd

        rows = []
        for r in all_combo_results:
            is_m = r.get("is_metrics") or {}
            oos_m = r.get("oos_metrics") or {}
            is_sig = r.get("is_signal_metrics") or {}
            oos_sig = r.get("oos_signal_metrics") or {}
            is_rm = (r.get("is_report_metrics") or {}) if isinstance(r.get("is_report_metrics"), dict) else {}
            oos_rm = (r.get("oos_report_metrics") or {}) if isinstance(r.get("oos_report_metrics"), dict) else {}

            def _infer_n_feat(rr: dict) -> int:
                try:
                    nf = rr.get("n_feat")
                    if nf is not None:
                        return int(nf)
                except Exception:
                    pass
                try:
                    oe = rr.get("obs_ensemble") or {}
                    if isinstance(oe, dict):
                        nf_by_obs = oe.get("n_feat_by_obs") or {}
                        if isinstance(nf_by_obs, dict) and nf_by_obs:
                            return int(sum(int(v) for v in nf_by_obs.values()))
                except Exception:
                    pass
                try:
                    op = rr.get("obs_pool") or {}
                    if isinstance(op, dict) and op:
                        return int(sum(int(v) for v in op.values()))
                except Exception:
                    pass
                return len(rr.get("formula_names") or [])

            def _sig_get(d: dict, key: str):
                if not isinstance(d, dict):
                    return None
                return d.get(key)

            def _rm_get(rm: dict, block: str, key: str):
                if not isinstance(rm, dict):
                    return None
                blk = rm.get(block) or {}
                return blk.get(key) if isinstance(blk, dict) else None

            rows.append(
                {
                    "combo_idx": r.get("combo_idx"),
                    "combo_key": "|".join(map(str, r.get("combo_key") or [])),
                    "feature_mode": feature_mode,
                    "n_feat": _infer_n_feat(r if isinstance(r, dict) else {}),
                    "is.information_ratio": (is_m.get("information_ratio") if isinstance(is_m, dict) else None),
                    "oos.information_ratio": (oos_m.get("information_ratio") if isinstance(oos_m, dict) else None),
                    "is.annualized_return": (is_m.get("annualized_return") if isinstance(is_m, dict) else None),
                    "oos.annualized_return": (oos_m.get("annualized_return") if isinstance(oos_m, dict) else None),
                    "is.max_drawdown": (is_m.get("max_drawdown") if isinstance(is_m, dict) else None),
                    "oos.max_drawdown": (oos_m.get("max_drawdown") if isinstance(oos_m, dict) else None),

                    # --- OOS-only flat columns (Qlib-style names expected by downstream) ---
                    "IC": _sig_get(oos_sig, "IC"),
                    "ICIR": _sig_get(oos_sig, "ICIR"),
                    "Rank_IC": _sig_get(oos_sig, "Rank IC"),
                    "Rank_ICIR": _sig_get(oos_sig, "Rank ICIR"),
                    "benchmark_mean": _rm_get(oos_rm, "benchmark", "mean"),
                    "benchmark_std": _rm_get(oos_rm, "benchmark", "std"),
                    "benchmark_annualized_return": _rm_get(oos_rm, "benchmark", "annualized_return"),
                    "benchmark_information_ratio": _rm_get(oos_rm, "benchmark", "information_ratio"),
                    "benchmark_max_drawdown": _rm_get(oos_rm, "benchmark", "max_drawdown"),
                    "excess_return_without_cost_mean": _rm_get(oos_rm, "excess_return_without_cost", "mean"),
                    "excess_return_without_cost_std": _rm_get(oos_rm, "excess_return_without_cost", "std"),
                    "excess_return_without_cost_annualized_return": _rm_get(oos_rm, "excess_return_without_cost", "annualized_return"),
                    "excess_return_without_cost_information_ratio": _rm_get(oos_rm, "excess_return_without_cost", "information_ratio"),
                    "excess_return_without_cost_max_drawdown": _rm_get(oos_rm, "excess_return_without_cost", "max_drawdown"),
                    "excess_return_with_cost_mean": _rm_get(oos_rm, "excess_return_with_cost", "mean"),
                    "excess_return_with_cost_std": _rm_get(oos_rm, "excess_return_with_cost", "std"),
                    "excess_return_with_cost_annualized_return": _rm_get(oos_rm, "excess_return_with_cost", "annualized_return"),
                    "excess_return_with_cost_information_ratio": _rm_get(oos_rm, "excess_return_with_cost", "information_ratio"),
                    "excess_return_with_cost_max_drawdown": _rm_get(oos_rm, "excess_return_with_cost", "max_drawdown"),

                    # --- IS-prefixed versions (kept for debugging/analysis) ---
                    "is_IC": _sig_get(is_sig, "IC"),
                    "is_ICIR": _sig_get(is_sig, "ICIR"),
                    "is_Rank_IC": _sig_get(is_sig, "Rank IC"),
                    "is_Rank_ICIR": _sig_get(is_sig, "Rank ICIR"),
                    "is_benchmark_mean": _rm_get(is_rm, "benchmark", "mean"),
                    "is_benchmark_std": _rm_get(is_rm, "benchmark", "std"),
                    "is_benchmark_annualized_return": _rm_get(is_rm, "benchmark", "annualized_return"),
                    "is_benchmark_information_ratio": _rm_get(is_rm, "benchmark", "information_ratio"),
                    "is_benchmark_max_drawdown": _rm_get(is_rm, "benchmark", "max_drawdown"),
                    "is_excess_return_without_cost_mean": _rm_get(is_rm, "excess_return_without_cost", "mean"),
                    "is_excess_return_without_cost_std": _rm_get(is_rm, "excess_return_without_cost", "std"),
                    "is_excess_return_without_cost_annualized_return": _rm_get(is_rm, "excess_return_without_cost", "annualized_return"),
                    "is_excess_return_without_cost_information_ratio": _rm_get(is_rm, "excess_return_without_cost", "information_ratio"),
                    "is_excess_return_without_cost_max_drawdown": _rm_get(is_rm, "excess_return_without_cost", "max_drawdown"),
                    "is_excess_return_with_cost_mean": _rm_get(is_rm, "excess_return_with_cost", "mean"),
                    "is_excess_return_with_cost_std": _rm_get(is_rm, "excess_return_with_cost", "std"),
                    "is_excess_return_with_cost_annualized_return": _rm_get(is_rm, "excess_return_with_cost", "annualized_return"),
                    "is_excess_return_with_cost_information_ratio": _rm_get(is_rm, "excess_return_with_cost", "information_ratio"),
                    "is_excess_return_with_cost_max_drawdown": _rm_get(is_rm, "excess_return_with_cost", "max_drawdown"),
                }
            )
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res.csv", pd.DataFrame(rows))
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res_compare.csv", pd.DataFrame(rows))
    except Exception:
        pass

    if outer_iter is not None:
        run_ctx.save_json_with_iter("specs/stage4_summary.json", outer_iter, summary)
        run_ctx.save_parquet_with_iter("data/stage4_is_daily.parquet", outer_iter, best_is_panel_pl)
        run_ctx.save_parquet_with_iter("data/stage4_oos_daily.parquet", outer_iter, best_oos_panel_pl)
        run_ctx.save_text_with_iter("reports/stage4.md", outer_iter, report_md)
    else:
        run_ctx.save_json("specs/stage4_summary.json", summary)
        run_ctx.save_parquet("data/stage4_is_daily.parquet", best_is_panel_pl)
        run_ctx.save_parquet("data/stage4_oos_daily.parquet", best_oos_panel_pl)
        run_ctx.save_text("reports/stage4.md", report_md)

    return Stage4Result(
        hypothesis_id=hypothesis_id,
        config={
            "stage4_mode": "alphaagent_lgbm_topkdropout",
            "topk": topk,
            "n_drop": n_drop,
            "use_cszscore": use_csz,
            "lgb_params": lgb_params,
        },
        summary=summary,
        result=None,
        report_md=report_md,
        is_daily_panel=best_is_panel_pl,
        oos_daily_panel=best_oos_panel_pl,
    )


def _parse_args(argv: List[str]) -> Dict[str, Any]:
    concept = "Mean Reversion after Panic Selling"
    use_outer_loop = False
    max_outer_iterations = None
    skip_stage3 = False

    # Stage4 alpha overrides (script-local)
    topk = 50
    n_drop = 5
    max_combos = 10
    feature_mode = "combo"
    combo_agg = "min"
    csz = True

    args = list(argv)
    if args and not args[0].startswith("--"):
        concept = args[0]
        args = args[1:]

    def _pop_int(flag: str) -> Optional[int]:
        if flag not in args:
            return None
        i = args.index(flag)
        if i + 1 < len(args) and args[i + 1].lstrip("-").isdigit():
            v = int(args[i + 1])
            del args[i : i + 2]
            return v
        del args[i]
        return None

    if "--outer-loop" in args:
        use_outer_loop = True
        i = args.index("--outer-loop")
        if i + 1 < len(args) and args[i + 1].isdigit():
            max_outer_iterations = int(args[i + 1])
            del args[i : i + 2]
        else:
            del args[i]

    v = _pop_int("--topk")
    if v is not None:
        topk = v
    v = _pop_int("--n-drop")
    if v is not None:
        n_drop = v
    v = _pop_int("--max-combos")
    if v is not None:
        max_combos = v

    if "--feature-mode" in args:
        i = args.index("--feature-mode")
        if i + 1 >= len(args):
            raise SystemExit("--feature-mode requires a value: combo|union|all|combo_scores|obs_pool|obs_ensemble")
        feature_mode = str(args[i + 1]).strip().lower()
        del args[i : i + 2]

    if "--combo-agg" in args:
        i = args.index("--combo-agg")
        if i + 1 >= len(args):
            raise SystemExit("--combo-agg requires a value: min|gmean")
        combo_agg = str(args[i + 1]).strip().lower()
        del args[i : i + 2]

    if "--no-csz" in args:
        csz = False
        args.remove("--no-csz")
    if "--skip-stage3" in args:
        skip_stage3 = True
        args.remove("--skip-stage3")
    if args:
        raise SystemExit(f"Unknown args: {args}")

    return {
        "concept": concept,
        "use_outer_loop": use_outer_loop,
        "max_outer_iterations": max_outer_iterations,
        "skip_stage3": skip_stage3,
        "topk": topk,
        "n_drop": n_drop,
        "max_combos": max_combos,
        "feature_mode": feature_mode,
        "combo_agg": combo_agg,
        "csz": csz,
    }


_CLI_TOPK = 50
_CLI_N_DROP = 5
_CLI_MAX_COMBOS = 10
_CLI_FEATURE_MODE = "combo"
_CLI_COMBO_AGG = "min"
_CLI_CSZ = True


def main() -> None:
    global _CLI_TOPK, _CLI_N_DROP, _CLI_MAX_COMBOS, _CLI_FEATURE_MODE, _CLI_COMBO_AGG, _CLI_CSZ
    parsed = _parse_args(sys.argv[1:])

    _CLI_TOPK = parsed["topk"]
    _CLI_N_DROP = parsed["n_drop"]
    _CLI_MAX_COMBOS = parsed["max_combos"]
    _CLI_FEATURE_MODE = parsed["feature_mode"]
    _CLI_COMBO_AGG = parsed["combo_agg"]
    _CLI_CSZ = parsed["csz"]

    # 런타임 패치: run.main의 Stage4만 교체(Stage1~3 동일)
    import run.main as main_mod

    main_mod.run_stage4 = run_stage4_alphaagent_style  # type: ignore[assignment]

    if parsed.get("skip_stage3"):
        # Skip-stage3 is only meaningful for Stage4 modes that don't depend on passed_combinations.
        if _CLI_FEATURE_MODE not in {"obs_pool", "obs_ensemble", "all"}:
            raise SystemExit("--skip-stage3 is only supported with --feature-mode obs_pool|obs_ensemble|all")

        from run.pipeline.stage3 import Stage3Result

        def _run_stage3_skipped(**kwargs):  # type: ignore[no-untyped-def]
            hypothesis = kwargs.get("hypothesis") or {}
            hypothesis_id = kwargs.get("hypothesis_id")
            if not hypothesis_id:
                try:
                    hyp_list = hypothesis.get("hypotheses", [])
                    hyp_obj = hyp_list[0] if isinstance(hyp_list, list) and hyp_list else hypothesis
                    hypothesis_id = (hyp_obj or {}).get("hypothesis_id") or "UNKNOWN"
                except Exception:
                    hypothesis_id = "UNKNOWN"
            report_md = "# Stage 3: Skipped\n\n- Reason: `--skip-stage3` flag\n"
            result = {"overall_verdict": "PASS", "pass_rate": 1.0, "n_passed_combinations": 0}
            return Stage3Result(
                hypothesis_id=str(hypothesis_id),
                result=result,
                report_md=report_md,
                ticker_results={},
                aggregated_result=result,
                passed_combinations=[],
                combination_stats={},
            )

        main_mod.run_stage3 = _run_stage3_skipped  # type: ignore[assignment]

    # Outer-loop feedback(hypothesis_memory['feedback'])를 다음 가설 생성 프롬프트의 Iteration Feedback으로 주입
    from agent.hypothesis_agent import HypothesisAgent

    _orig_purpose_hypothesis = HypothesisAgent.purpose_hypothesis

    def _patched_purpose_hypothesis(
        self: HypothesisAgent,
        concept: str,
        metadata: list = None,
        hypothesis_memory: list = None,
        knowledge: str = "",
        feedback: str = "",
    ):
        try:
            fb_parts = []
            if hypothesis_memory:
                for item in hypothesis_memory:
                    if isinstance(item, dict) and item.get("feedback"):
                        fb_parts.append(str(item.get("feedback")))
            if fb_parts:
                injected = "\n\n".join(fb_parts[-3:])
                if not feedback or feedback == "None":
                    feedback = injected
                else:
                    feedback = f"{feedback}\n\n{injected}"
        except Exception:
            pass
        return _orig_purpose_hypothesis(
            self,
            concept=concept,
            metadata=metadata,
            hypothesis_memory=hypothesis_memory,
            knowledge=knowledge,
            feedback=feedback,
        )

    HypothesisAgent.purpose_hypothesis = _patched_purpose_hypothesis  # type: ignore[assignment]

    from run.main import run_pipeline, run_outer_loop
    from util.run_context import RunContext

    # 이 엔트리포인트는 실행 결과/로그를 runs/lgbm 아래에 저장한다.
    run_ctx = RunContext.create(base_dir=os.path.join(PROJECT_ROOT, "runs", "lgbm"))

    if parsed["use_outer_loop"]:
        run_outer_loop(
            concept=parsed["concept"],
            max_outer_iterations=parsed["max_outer_iterations"],
            run_ctx=run_ctx,
        )
    else:
        run_pipeline(concept=parsed["concept"], run_ctx=run_ctx)


if __name__ == "__main__":
    main()
