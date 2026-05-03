#!/usr/bin/env python
"""
Stage1~3는 기존 파이프라인(run.main.run_pipeline)을 그대로 사용하고,
Stage4만 "Obs-branch MLP 앙상블 + TopkDropoutStrategy"로 대체하여 실행하는 엔트리포인트.

핵심 아이디어:
- Stage2 PASS 수식들을 observation_id별로 그룹핑해서 obs별 입력(feature)을 분리
- obs마다 별도 MLP를 학습해 pred_obs_i 생성
- 최종 score = Σ w_i * pred_obs_i, 단 w_i >= epsilon (hard lower-bound)로 "obs 무시" 방지

사용 예:
  MARKET=cn python run_pipeline_topkdropout_mlp.py
  python run_pipeline_topkdropout_mlp.py "Short-term Mean-Reversion" --topk 50 --n-drop 5 --eps 0.1
  python run_pipeline_topkdropout_mlp.py --outer-loop 5 --topk 50 --n-drop 5 --eps 0.1
"""

from __future__ import annotations

import os
import sys
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


def _cs_zscore(df, cols: List[str], by: str = "timestamp"):
    import numpy as np

    for c in cols:
        g = df.groupby(by)[c]
        mean = g.transform("mean")
        std = g.transform("std").replace(0.0, np.nan)
        df[c] = (df[c] - mean) / std
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df


def _normalize_trade_dates(ts):
    import pandas as pd

    return pd.to_datetime(ts).dt.normalize()


def _to_jsonable(obj: Any) -> Any:
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


def _calc_report_metrics(*, report_df, freq: str = "day") -> Dict[str, Dict[str, float]]:
    """
    Build Qlib-style benchmark/excess-return metrics from a backtest report_df.
    Returns a nested dict:
      {
        "benchmark": {...},
        "excess_return_without_cost": {...},
        "excess_return_with_cost": {...},
      }
    """
    from qlib.contrib.evaluate import risk_analysis

    if report_df is None:
        return {}
    if not hasattr(report_df, "__getitem__"):
        return {}

    try:
        ret = report_df["return"]
    except Exception:
        return {}

    bench = report_df["bench"] if "bench" in getattr(report_df, "columns", []) else 0.0
    cost = report_df["cost"] if "cost" in getattr(report_df, "columns", []) else 0.0

    bm = _risk_analysis_to_metrics(risk_analysis(bench, freq=freq))
    ex_wo = _risk_analysis_to_metrics(risk_analysis(ret - bench, freq=freq))
    ex_w = _risk_analysis_to_metrics(risk_analysis(ret - bench - cost, freq=freq))

    return {
        "benchmark": bm,
        "excess_return_without_cost": ex_wo,
        "excess_return_with_cost": ex_w,
    }


def _positions_to_normal_dict(positions: Any) -> dict:
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

    unique_dates = pd.Index(sorted(df_is["timestamp"].unique()))
    if len(unique_dates) < 5:
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
            f"[Stage4-mlp] {tag}end_time adjusted for qlib calendar boundary: {end_ts.date()} -> {adjusted.date()}"
        )
    return adjusted


def _build_mlp(input_dim: int, hidden_dim: int, depth: int, dropout: float):
    import torch.nn as nn

    if depth <= 1:
        return nn.Sequential(nn.Linear(input_dim, 1))

    layers: list[nn.Module] = []
    d = input_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _sample_rows(X, y, max_n: int, seed: int = 42):
    import numpy as np

    n = int(X.shape[0])
    if n <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return X[idx], y[idx]


def _train_mlp_regressor(
    *,
    X_train,
    y_train,
    X_valid,
    y_valid,
    device: str,
    hidden_dim: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_steps: int,
    eval_every: int,
    seed: int,
):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_mlp(int(X_train.shape[1]), hidden_dim=hidden_dim, depth=depth, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    X_tr = torch.from_numpy(X_train).to(device)
    y_tr = torch.from_numpy(y_train).to(device).view(-1, 1)
    ds = TensorDataset(X_tr, y_tr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if X_valid is not None and X_valid.shape[0] > 0:
        X_va_s, y_va_s = _sample_rows(X_valid, y_valid, max_n=min(50_000, int(X_valid.shape[0])), seed=seed)
        X_va = torch.from_numpy(X_va_s).to(device)
        y_va = torch.from_numpy(y_va_s).to(device).view(-1, 1)
    else:
        X_va = None
        y_va = None

    best_state = None
    best_val = float("inf")
    steps_no_improve = 0
    patience = max(5, int(max_steps // max(eval_every, 1) // 5))

    global_step = 0
    model.train()
    while global_step < max_steps:
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step >= max_steps:
                break

            if X_va is not None and (global_step % eval_every == 0):
                model.eval()
                with torch.no_grad():
                    vpred = model(X_va)
                    vloss = float(loss_fn(vpred, y_va).item())
                model.train()

                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1
                    if steps_no_improve >= patience:
                        global_step = max_steps
                        break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_valid_mse": best_val if best_state is not None else None}


def _predict(model, X, device: str):
    import torch

    if X is None or X.shape[0] == 0:
        return None
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        pred = model(X_t).view(-1).detach().cpu().numpy()
    return pred


def _calc_signal_metrics(
    *,
    label_df,
    signal_series,
    start_time,
    end_time,
):
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
    lbl["timestamp"] = pd.to_datetime(lbl["timestamp"]).dt.normalize()
    lbl = lbl[(lbl["timestamp"] >= start_ts) & (lbl["timestamp"] <= end_ts)]

    sig = signal_series.rename("score").reset_index()
    sig = sig.rename(columns={"datetime": "timestamp", "instrument": "ticker"})
    sig["timestamp"] = pd.to_datetime(sig["timestamp"]).dt.normalize()
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
        return float(x.mean()) if x is not None and x.notna().any() else None

    def _safe_icir(x):
        if x is None or not x.notna().any():
            return None
        std = float(x.std())
        if std == 0.0 or not np.isfinite(std):
            return None
        return float(x.mean()) / std

    out = {}
    ic_mean = _safe_mean(ic)
    ric_mean = _safe_mean(ric)
    if ic_mean is not None:
        out["IC"] = ic_mean
    icir = _safe_icir(ic)
    if icir is not None:
        out["ICIR"] = icir
    if ric_mean is not None:
        out["Rank IC"] = ric_mean
    ricir = _safe_icir(ric)
    if ricir is not None:
        out["Rank ICIR"] = ricir

    return out, ts


def _fit_constrained_weights(
    *,
    preds_by_obs: List,
    y: Any,
    eps: float,
    device: str,
    lr: float,
    steps: int,
    weight_decay: float,
    seed: int,
):
    import numpy as np
    import torch
    import torch.nn as nn

    n_obs = len(preds_by_obs)
    if n_obs <= 0:
        raise ValueError("No obs predictions to fit weights.")
    if eps < 0:
        raise ValueError("--eps must be >= 0")
    if eps * n_obs >= 1.0:
        raise ValueError(f"--eps too large: eps*n_obs must be < 1 (eps={eps}, n_obs={n_obs})")

    rng = np.random.default_rng(seed)
    preds_mat = np.stack(preds_by_obs, axis=1).astype(np.float32)  # (N, n_obs)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    max_n = min(int(len(y)), 200_000)
    if len(y) > max_n:
        idx = rng.choice(len(y), size=max_n, replace=False)
        preds_mat = preds_mat[idx]
        y = y[idx]

    P = torch.from_numpy(preds_mat).to(device)
    Y = torch.from_numpy(y).to(device)

    a = torch.zeros((n_obs,), device=device, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(max(1, steps)):
        opt.zero_grad(set_to_none=True)
        soft = torch.softmax(a, dim=0)
        w = eps + (1.0 - eps * n_obs) * soft
        yhat = (P * w.view(1, -1)).sum(dim=1)
        loss = loss_fn(yhat, Y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        soft = torch.softmax(a, dim=0)
        w = eps + (1.0 - eps * n_obs) * soft
        w_np = w.detach().cpu().numpy().astype(float).tolist()
    return w_np


def run_stage4_alphaagent_mlp_obs_branch(
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
    **_kwargs,
):
    """
    Stage4 대체 구현 (Obs-branch MLP):
    - obs별 수식 풀로 분리 → obs별 MLP 학습
    - 앙상블 가중치 w_i >= eps로 강제 → "obs 무시" 방지
    - 최종 score를 Qlib TopkDropoutStrategy로 백테스트
    """
    from run.config import load_rd_config
    from run.pipeline.stage4 import Stage4Result

    import numpy as np
    import pandas as pd
    import polars as pl
    import torch

    cfg = cfg or load_rd_config()
    assert run_ctx is not None, "run_ctx is required"

    # CLI/env override
    topk = int(os.environ.get("ALPHA_STAGE4_TOPK", str(_CLI_TOPK)))
    n_drop = int(os.environ.get("ALPHA_STAGE4_N_DROP", str(_CLI_N_DROP)))
    eps = float(os.environ.get("ALPHA_STAGE4_OBS_EPS", str(_CLI_EPS)))
    use_csz = bool(int(os.environ.get("ALPHA_STAGE4_CSZ", "1" if _CLI_CSZ else "0")))

    hidden_dim = int(os.environ.get("ALPHA_STAGE4_MLP_HIDDEN", str(_CLI_HIDDEN)))
    depth = int(os.environ.get("ALPHA_STAGE4_MLP_DEPTH", str(_CLI_DEPTH)))
    dropout = float(os.environ.get("ALPHA_STAGE4_MLP_DROPOUT", str(_CLI_DROPOUT)))
    lr = float(os.environ.get("ALPHA_STAGE4_MLP_LR", str(_CLI_LR)))
    weight_decay = float(os.environ.get("ALPHA_STAGE4_MLP_WEIGHT_DECAY", str(_CLI_WEIGHT_DECAY)))
    batch_size = int(os.environ.get("ALPHA_STAGE4_MLP_BATCH_SIZE", str(_CLI_BATCH_SIZE)))
    max_steps = int(os.environ.get("ALPHA_STAGE4_MLP_STEPS", str(_CLI_STEPS)))
    eval_every = int(os.environ.get("ALPHA_STAGE4_MLP_EVAL_EVERY", str(_CLI_EVAL_EVERY)))
    weight_lr = float(os.environ.get("ALPHA_STAGE4_OBS_W_LR", str(_CLI_W_LR)))
    weight_steps = int(os.environ.get("ALPHA_STAGE4_OBS_W_STEPS", str(_CLI_W_STEPS)))

    device = str(os.environ.get("ALPHA_STAGE4_MLP_DEVICE", str(_CLI_DEVICE))).strip().lower()
    if device in {"auto", ""}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    seed = int(os.environ.get("ALPHA_STAGE4_MLP_SEED", str(_CLI_SEED)))

    is_start = cfg.data_split.in_sample_start
    is_end = cfg.data_split.in_sample_end
    oos_start = cfg.data_split.out_sample_start
    oos_end = cfg.data_split.out_sample_end

    if not passed_formulas:
        raise ValueError("Stage4-mlp requires `passed_formulas` (Stage2 passed_formulas).")

    # Group formulas by observation_id
    obs_to_formulas: Dict[str, List[Dict[str, Any]]] = {}
    for f in passed_formulas:
        obs_id = str(f.get("observation_id") or f.get("obs_id") or "UNKNOWN_OBS")
        obs_to_formulas.setdefault(obs_id, []).append(f)

    # Base panel: timestamp, ticker, close + required formula cols
    base_cols = ["timestamp", "ticker", "close"]
    base_pd = ohlcv_df.select([c for c in base_cols if c in ohlcv_df.columns]).to_pandas()
    base_pd["timestamp"] = _normalize_trade_dates(base_pd["timestamp"])

    all_formula_cols = set(formula_df.columns) - {"timestamp", "ticker"}
    need_cols: set[str] = set()
    polarity_by_name: Dict[str, str] = {}
    obs_to_names: Dict[str, List[str]] = {}
    for obs_id, fl in obs_to_formulas.items():
        names: List[str] = []
        for f in fl:
            n = str(f.get("name") or "")
            if n and n in all_formula_cols:
                need_cols.add(n)
                names.append(n)
                polarity_by_name.setdefault(n, str(f.get("polarity") or ""))
        if names:
            obs_to_names[obs_id] = sorted(set(names))

    obs_ids = sorted(obs_to_names.keys())
    if not obs_ids:
        raise RuntimeError("Stage4-mlp: no obs groups with usable formula columns.")

    f_pd = formula_df.select(["timestamp", "ticker", *sorted(need_cols)]).to_pandas()
    f_pd["timestamp"] = _normalize_trade_dates(f_pd["timestamp"])
    panel = base_pd.merge(f_pd, on=["timestamp", "ticker"], how="inner")
    panel = panel.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(drop=True)
    panel = _compute_label_close_t1_to_t2(panel)
    label_df = panel[["timestamp", "ticker", "label"]].copy()

    # Train/valid split (IS only)
    df = panel.replace([np.inf, -np.inf], np.nan)
    df_train, df_valid = _split_is_train_valid(df, is_start=is_start, is_end=is_end, valid_ratio=0.2)
    df_train = df_train.dropna(subset=["label"])
    df_valid = df_valid.dropna(subset=["label"])
    if len(df_train) < 1000:
        raise RuntimeError("Stage4-mlp: not enough training rows after label drop.")

    # Normalize/flip polarity per obs branch separately (but reuse same df slices)
    for name, pol in polarity_by_name.items():
        if name in df_train.columns and str(pol).startswith("lower"):
            df_train[name] = -df_train[name]
            if name in df_valid.columns:
                df_valid[name] = -df_valid[name]
            if name in df.columns:
                df[name] = -df[name]

    if use_csz:
        all_cols = sorted(need_cols)
        df_train = _cs_zscore(df_train, all_cols, by="timestamp")
        if len(df_valid) > 0:
            df_valid = _cs_zscore(df_valid, all_cols, by="timestamp")
        df = _cs_zscore(df, all_cols, by="timestamp")

    y_train = df_train["label"].to_numpy(dtype=np.float32, copy=False)
    y_valid = df_valid["label"].to_numpy(dtype=np.float32, copy=False) if len(df_valid) > 0 else None

    models = {}
    meta = {}
    valid_preds = []
    all_preds = []

    for obs_id in obs_ids:
        feat_cols = obs_to_names[obs_id]
        X_tr = df_train[feat_cols].to_numpy(dtype=np.float32, copy=False)
        X_va = df_valid[feat_cols].to_numpy(dtype=np.float32, copy=False) if len(df_valid) > 0 else None
        X_all = df[feat_cols].to_numpy(dtype=np.float32, copy=False)

        model, info = _train_mlp_regressor(
            X_train=X_tr,
            y_train=y_train,
            X_valid=X_va if X_va is not None else np.empty((0, len(feat_cols)), dtype=np.float32),
            y_valid=y_valid if y_valid is not None else np.empty((0,), dtype=np.float32),
            device=device,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_steps=max_steps,
            eval_every=max(1, eval_every),
            seed=seed,
        )
        models[obs_id] = model
        meta[obs_id] = {"n_feat": len(feat_cols), **info}

        p_all = _predict(model, X_all, device=device)
        all_preds.append(p_all.astype(np.float32))
        if len(df_valid) > 0:
            p_va = _predict(model, X_va, device=device)
            valid_preds.append(p_va.astype(np.float32))

    if len(df_valid) == 0:
        # fallback: fit weights on a subset of train
        Xw = df_train.sample(n=min(200_000, len(df_train)), random_state=seed) if len(df_train) > 0 else df_train
        y_w = Xw["label"].to_numpy(dtype=np.float32, copy=False)
        preds_w = []
        for obs_id in obs_ids:
            feat_cols = obs_to_names[obs_id]
            X = Xw[feat_cols].to_numpy(dtype=np.float32, copy=False)
            preds_w.append(_predict(models[obs_id], X, device=device).astype(np.float32))
        w = _fit_constrained_weights(
            preds_by_obs=preds_w,
            y=y_w,
            eps=eps,
            device=device,
            lr=weight_lr,
            steps=weight_steps,
            weight_decay=0.0,
            seed=seed,
        )
    else:
        w = _fit_constrained_weights(
            preds_by_obs=valid_preds,
            y=y_valid,
            eps=eps,
            device=device,
            lr=weight_lr,
            steps=weight_steps,
            weight_decay=0.0,
            seed=seed,
        )

    # Combine for all dates
    weights = np.asarray(w, dtype=np.float64)
    combined = np.zeros((len(df),), dtype=np.float64)
    for wi, pi in zip(weights, all_preds, strict=False):
        combined += float(wi) * pi.astype(np.float64)

    signal_df = df[["timestamp", "ticker"]].copy()
    signal_df["score"] = combined
    signal_df["timestamp"] = _normalize_trade_dates(signal_df["timestamp"])
    signal_series = signal_df.set_index(["timestamp", "ticker"])["score"]
    signal_series.index.names = ["datetime", "instrument"]
    signal_series = signal_series[~signal_series.index.duplicated(keep="last")].sort_index()

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

    is_start_ts = pd.to_datetime(is_start)
    is_end_ts = _ensure_qlib_end_time_has_future_day(
        start_time=is_start_ts, end_time=pd.to_datetime(is_end), run_ctx=run_ctx, label="IS"
    )
    oos_start_ts = pd.to_datetime(oos_start)
    oos_end_ts = _ensure_qlib_end_time_has_future_day(
        start_time=oos_start_ts, end_time=pd.to_datetime(oos_end), run_ctx=run_ctx, label="OOS"
    )

    strategy = TopkDropoutStrategy(signal=signal_series, topk=topk, n_drop=n_drop, risk_degree=0.95)

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

    is_report_metrics = _calc_report_metrics(report_df=is_report_df, freq="day")
    oos_report_metrics = _calc_report_metrics(report_df=oos_report_df, freq="day")

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

    # Signal metrics (IC/RankIC) for IS/OOS
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

    iter_prefix = f"iter_{outer_iter}" if outer_iter is not None else "iter_1"
    _save_qlib_artifacts(
        run_ctx=run_ctx,
        iter_prefix=iter_prefix,
        model_tag="obs_branch_mlp",
        split="is",
        report_df=is_report_df,
        positions_raw=is_positions_raw,
    )
    _save_qlib_artifacts(
        run_ctx=run_ctx,
        iter_prefix=iter_prefix,
        model_tag="obs_branch_mlp",
        split="oos",
        report_df=oos_report_df,
        positions_raw=oos_positions_raw,
    )

    # Save per-day IC timeseries CSVs
    try:
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/obs_branch_mlp/signal_metrics_is.csv", is_sig_ts)
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/obs_branch_mlp/signal_metrics_oos.csv", oos_sig_ts)
    except Exception:
        pass

    # Save a compact qlib_res.csv-like summary row
    try:
        import pandas as pd

        def _sig_get(d: dict, key: str):
            if not isinstance(d, dict):
                return None
            return d.get(key)

        def _rm_get(rm: dict, block: str, key: str):
            if not isinstance(rm, dict):
                return None
            blk = rm.get(block) or {}
            return blk.get(key) if isinstance(blk, dict) else None

        row = {
            "combo_idx": None,
            "combo_key": "OBS_BRANCH_MLP",
            "feature_mode": "obs_branch",
            "n_feat": int(sum(len(obs_to_names.get(k, [])) for k in obs_ids)),
            "is.information_ratio": float(is_metrics.get("information_ratio") or 0.0),
            "oos.information_ratio": float(oos_metrics.get("information_ratio") or 0.0),

            # --- OOS-only flat columns (Qlib-style names expected by downstream) ---
            "IC": _sig_get(oos_sig_m, "IC"),
            "ICIR": _sig_get(oos_sig_m, "ICIR"),
            "Rank_IC": _sig_get(oos_sig_m, "Rank IC"),
            "Rank_ICIR": _sig_get(oos_sig_m, "Rank ICIR"),
            "benchmark_mean": _rm_get(oos_report_metrics, "benchmark", "mean"),
            "benchmark_std": _rm_get(oos_report_metrics, "benchmark", "std"),
            "benchmark_annualized_return": _rm_get(oos_report_metrics, "benchmark", "annualized_return"),
            "benchmark_information_ratio": _rm_get(oos_report_metrics, "benchmark", "information_ratio"),
            "benchmark_max_drawdown": _rm_get(oos_report_metrics, "benchmark", "max_drawdown"),
            "excess_return_without_cost_mean": _rm_get(oos_report_metrics, "excess_return_without_cost", "mean"),
            "excess_return_without_cost_std": _rm_get(oos_report_metrics, "excess_return_without_cost", "std"),
            "excess_return_without_cost_annualized_return": _rm_get(
                oos_report_metrics, "excess_return_without_cost", "annualized_return"
            ),
            "excess_return_without_cost_information_ratio": _rm_get(
                oos_report_metrics, "excess_return_without_cost", "information_ratio"
            ),
            "excess_return_without_cost_max_drawdown": _rm_get(oos_report_metrics, "excess_return_without_cost", "max_drawdown"),
            "excess_return_with_cost_mean": _rm_get(oos_report_metrics, "excess_return_with_cost", "mean"),
            "excess_return_with_cost_std": _rm_get(oos_report_metrics, "excess_return_with_cost", "std"),
            "excess_return_with_cost_annualized_return": _rm_get(oos_report_metrics, "excess_return_with_cost", "annualized_return"),
            "excess_return_with_cost_information_ratio": _rm_get(oos_report_metrics, "excess_return_with_cost", "information_ratio"),
            "excess_return_with_cost_max_drawdown": _rm_get(oos_report_metrics, "excess_return_with_cost", "max_drawdown"),

            # --- IS-prefixed versions (kept for debugging/analysis) ---
            "is_IC": _sig_get(is_sig_m, "IC"),
            "is_ICIR": _sig_get(is_sig_m, "ICIR"),
            "is_Rank_IC": _sig_get(is_sig_m, "Rank IC"),
            "is_Rank_ICIR": _sig_get(is_sig_m, "Rank ICIR"),
            "is_benchmark_mean": _rm_get(is_report_metrics, "benchmark", "mean"),
            "is_benchmark_std": _rm_get(is_report_metrics, "benchmark", "std"),
            "is_benchmark_annualized_return": _rm_get(is_report_metrics, "benchmark", "annualized_return"),
            "is_benchmark_information_ratio": _rm_get(is_report_metrics, "benchmark", "information_ratio"),
            "is_benchmark_max_drawdown": _rm_get(is_report_metrics, "benchmark", "max_drawdown"),
            "is_excess_return_without_cost_mean": _rm_get(is_report_metrics, "excess_return_without_cost", "mean"),
            "is_excess_return_without_cost_std": _rm_get(is_report_metrics, "excess_return_without_cost", "std"),
            "is_excess_return_without_cost_annualized_return": _rm_get(
                is_report_metrics, "excess_return_without_cost", "annualized_return"
            ),
            "is_excess_return_without_cost_information_ratio": _rm_get(
                is_report_metrics, "excess_return_without_cost", "information_ratio"
            ),
            "is_excess_return_without_cost_max_drawdown": _rm_get(is_report_metrics, "excess_return_without_cost", "max_drawdown"),
            "is_excess_return_with_cost_mean": _rm_get(is_report_metrics, "excess_return_with_cost", "mean"),
            "is_excess_return_with_cost_std": _rm_get(is_report_metrics, "excess_return_with_cost", "std"),
            "is_excess_return_with_cost_annualized_return": _rm_get(
                is_report_metrics, "excess_return_with_cost", "annualized_return"
            ),
            "is_excess_return_with_cost_information_ratio": _rm_get(
                is_report_metrics, "excess_return_with_cost", "information_ratio"
            ),
            "is_excess_return_with_cost_max_drawdown": _rm_get(is_report_metrics, "excess_return_with_cost", "max_drawdown"),
        }
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res.csv", pd.DataFrame([row]))
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/qlib_res_compare.csv", pd.DataFrame([row]))
    except Exception:
        pass

    summary = {
        "stage4_mode": "alphaagent_mlp_obs_branch",
        "hypothesis_id": hypothesis_id,
        "market": os.getenv("MARKET", "cn").lower(),
        "data_split": {
            "in_sample_start": is_start,
            "in_sample_end": is_end,
            "out_sample_start": oos_start,
            "out_sample_end": oos_end,
        },
        "topk": topk,
        "n_drop": n_drop,
        "obs_eps": eps,
        "device": device,
        "use_cszscore": use_csz,
        "mlp_params": {
            "hidden_dim": hidden_dim,
            "depth": depth,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "eval_every": eval_every,
            "seed": seed,
        },
        "obs_ids": obs_ids,
        "obs_formula_counts": {k: len(v) for k, v in obs_to_names.items()},
        "obs_model_meta": _to_jsonable(meta),
        "ensemble_weights": {obs_id: float(wi) for obs_id, wi in zip(obs_ids, weights.tolist(), strict=False)},
        "is_signal_metrics": _to_jsonable(dict(is_sig_m)),
        "oos_signal_metrics": _to_jsonable(dict(oos_sig_m)),
        "is_report_metrics": _to_jsonable(dict(is_report_metrics)),
        "oos_report_metrics": _to_jsonable(dict(oos_report_metrics)),
        "is_metrics": _to_jsonable(dict(is_metrics)),
        "oos_metrics": _to_jsonable(dict(oos_metrics)),
        "data_split_metrics": {
            "insample": {"strategy": _to_jsonable(dict(is_metrics))},
            "outsample": {"strategy": _to_jsonable(dict(oos_metrics))},
        },
        "is_turnover_sum": float(is_turnover_sum),
        "oos_turnover_sum": float(oos_turnover_sum),
    }

    lines = [
        "# Stage 4 (AlphaAgent-style): Obs-branch MLP + TopkDropoutStrategy",
        "",
        f"- Market: `{summary['market']}`",
        f"- IS: `{is_start} ~ {is_end}` / OOS: `{oos_start} ~ {oos_end}`",
        f"- topk: `{topk}`, n_drop: `{n_drop}`",
        f"- obs: `{len(obs_ids)}`, eps: `{eps}`",
        f"- weights: `{summary['ensemble_weights']}`",
        "",
        "## Metrics",
        "",
        f"- IS_IR: `{float(is_metrics.get('information_ratio') or 0.0):.6f}`",
        f"- OOS_IR: `{float(oos_metrics.get('information_ratio') or 0.0):.6f}`",
    ]
    report_md = "\n".join(lines) + "\n"

    is_pd = is_report_df.reset_index().rename(columns={"index": "date"})
    oos_pd = oos_report_df.reset_index().rename(columns={"index": "date"})
    is_panel_pl = pl.from_pandas(is_pd)
    oos_panel_pl = pl.from_pandas(oos_pd)

    if outer_iter is not None:
        run_ctx.save_json_with_iter("specs/stage4_summary.json", outer_iter, summary)
        run_ctx.save_parquet_with_iter("data/stage4_is_daily.parquet", outer_iter, is_panel_pl)
        run_ctx.save_parquet_with_iter("data/stage4_oos_daily.parquet", outer_iter, oos_panel_pl)
        run_ctx.save_text_with_iter("reports/stage4.md", outer_iter, report_md)
    else:
        run_ctx.save_json("specs/stage4_summary.json", summary)
        run_ctx.save_parquet("data/stage4_is_daily.parquet", is_panel_pl)
        run_ctx.save_parquet("data/stage4_oos_daily.parquet", oos_panel_pl)
        run_ctx.save_text("reports/stage4.md", report_md)

    if verbose:
        run_ctx.log(
            f"[Stage4-mlp] obs_branch_mlp (obs={len(obs_ids)}, eps={eps}, device={device}): "
            f"IS_IR={float(is_metrics.get('information_ratio') or 0.0):.6f}, "
            f"OOS_IR={float(oos_metrics.get('information_ratio') or 0.0):.6f}"
        )

    return Stage4Result(
        hypothesis_id=hypothesis_id,
        config={"stage4_mode": "alphaagent_mlp_obs_branch"},
        summary=summary,
        result=None,
        report_md=report_md,
        is_daily_panel=is_panel_pl,
        oos_daily_panel=oos_panel_pl,
    )


def _parse_args(argv: List[str]) -> Dict[str, Any]:
    concept = "Mean Reversion after Panic Selling"
    use_outer_loop = False
    max_outer_iterations = None
    skip_stage3 = False

    topk = 50
    n_drop = 5
    eps = 0.1
    csz = True

    hidden = 128
    depth = 3
    dropout = 0.0
    lr = 2e-3
    weight_decay = 2e-4
    batch_size = 8192
    steps = 2000
    eval_every = 50
    device = "auto"
    seed = 42
    w_lr = 5e-2
    w_steps = 200

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

    def _pop_float(flag: str) -> Optional[float]:
        if flag not in args:
            return None
        i = args.index(flag)
        if i + 1 < len(args):
            try:
                v = float(args[i + 1])
                del args[i : i + 2]
                return v
            except Exception:
                pass
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

    v = _pop_float("--eps")
    if v is not None:
        eps = v

    v = _pop_int("--hidden")
    if v is not None:
        hidden = v
    v = _pop_int("--depth")
    if v is not None:
        depth = v
    v = _pop_float("--dropout")
    if v is not None:
        dropout = v
    v = _pop_float("--lr")
    if v is not None:
        lr = v
    v = _pop_float("--weight-decay")
    if v is not None:
        weight_decay = v
    v = _pop_int("--batch-size")
    if v is not None:
        batch_size = v
    v = _pop_int("--steps")
    if v is not None:
        steps = v
    v = _pop_int("--eval-every")
    if v is not None:
        eval_every = v
    v = _pop_int("--seed")
    if v is not None:
        seed = v
    v = _pop_float("--w-lr")
    if v is not None:
        w_lr = v
    v = _pop_int("--w-steps")
    if v is not None:
        w_steps = v

    if "--device" in args:
        i = args.index("--device")
        if i + 1 >= len(args):
            raise SystemExit("--device requires a value: auto|cpu|cuda")
        device = str(args[i + 1]).strip().lower()
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
        "eps": eps,
        "csz": csz,
        "hidden": hidden,
        "depth": depth,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "steps": steps,
        "eval_every": eval_every,
        "device": device,
        "seed": seed,
        "w_lr": w_lr,
        "w_steps": w_steps,
    }


_CLI_TOPK = 50
_CLI_N_DROP = 5
_CLI_EPS = 0.1
_CLI_CSZ = True
_CLI_HIDDEN = 128
_CLI_DEPTH = 3
_CLI_DROPOUT = 0.0
_CLI_LR = 2e-3
_CLI_WEIGHT_DECAY = 2e-4
_CLI_BATCH_SIZE = 8192
_CLI_STEPS = 2000
_CLI_EVAL_EVERY = 50
_CLI_DEVICE = "auto"
_CLI_SEED = 42
_CLI_W_LR = 5e-2
_CLI_W_STEPS = 200


def main() -> None:
    global _CLI_TOPK, _CLI_N_DROP, _CLI_EPS, _CLI_CSZ
    global _CLI_HIDDEN, _CLI_DEPTH, _CLI_DROPOUT, _CLI_LR, _CLI_WEIGHT_DECAY, _CLI_BATCH_SIZE, _CLI_STEPS, _CLI_EVAL_EVERY
    global _CLI_DEVICE, _CLI_SEED, _CLI_W_LR, _CLI_W_STEPS

    parsed = _parse_args(sys.argv[1:])
    _CLI_TOPK = parsed["topk"]
    _CLI_N_DROP = parsed["n_drop"]
    _CLI_EPS = parsed["eps"]
    _CLI_CSZ = parsed["csz"]
    _CLI_HIDDEN = parsed["hidden"]
    _CLI_DEPTH = parsed["depth"]
    _CLI_DROPOUT = parsed["dropout"]
    _CLI_LR = parsed["lr"]
    _CLI_WEIGHT_DECAY = parsed["weight_decay"]
    _CLI_BATCH_SIZE = parsed["batch_size"]
    _CLI_STEPS = parsed["steps"]
    _CLI_EVAL_EVERY = parsed["eval_every"]
    _CLI_DEVICE = parsed["device"]
    _CLI_SEED = parsed["seed"]
    _CLI_W_LR = parsed["w_lr"]
    _CLI_W_STEPS = parsed["w_steps"]

    # 런타임 패치: run.main의 Stage4만 교체(Stage1~3 동일)
    import run.main as main_mod

    main_mod.run_stage4 = run_stage4_alphaagent_mlp_obs_branch  # type: ignore[assignment]

    if parsed.get("skip_stage3"):
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

    from run.main import run_pipeline, run_outer_loop
    from util.run_context import RunContext

    run_ctx = RunContext.create(base_dir=os.path.join(PROJECT_ROOT, "runs", "mlp"))

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
