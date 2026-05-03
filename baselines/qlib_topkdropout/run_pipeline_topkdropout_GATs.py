#!/usr/bin/env python
"""
Stage1~3는 기존 파이프라인(run.main.run_pipeline)을 그대로 사용하고,
Stage4만 "Obs-graph GAT + TopkDropoutStrategy"로 대체하여 실행하는 엔트리포인트.

핵심:
- Stage2 PASS 수식을 observation_id(=obs)별로 그룹핑해 obs별 입력을 분리한다.
- 각 obs 입력을 encoder로 embedding으로 바꾼 뒤, obs 노드들 사이에 GAT(fully-connected)를 적용한다.
- obs별 head 예측 p_i를 만들고, 최종 score를 w_i>=eps(하드 하한)로 결합해 "obs 무시 불가"를 보장한다.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch


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


def _normalize_trade_dates(ts):
    import pandas as pd

    return pd.to_datetime(ts).dt.normalize()


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


def _cs_zscore(df, cols: List[str], by: str = "timestamp"):
    import numpy as np

    for c in cols:
        g = df.groupby(by)[c]
        mean = g.transform("mean")
        std = g.transform("std").replace(0.0, np.nan)
        df[c] = (df[c] - mean) / std
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


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
            f"[Stage4-gats] {tag}end_time adjusted for qlib calendar boundary: {end_ts.date()} -> {adjusted.date()}"
        )
    return adjusted


def _floor_softmax_weights(logits, eps: float):
    """
    logits: (n_obs,)
    return weights: (n_obs,), s.t. weights>=eps and sum=1.
    """
    import torch

    n = int(logits.shape[0])
    if eps < 0.0:
        raise ValueError("--eps must be >= 0")
    if eps * n >= 1.0:
        raise ValueError(f"--eps too large: eps*n_obs must be < 1 (eps={eps}, n_obs={n})")
    soft = torch.softmax(logits, dim=0)
    return eps + (1.0 - eps * n) * soft


class ObsEncoderGAT(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dims: List[int],
        emb_dim: int,
        enc_hidden: int,
        enc_depth: int,
        enc_dropout: float,
        gat_heads: int,
        gat_layers: int,
        gat_dropout: float,
        eps: float,
    ) -> None:
        import torch
        import torch.nn as nn

        super().__init__()
        self.n_obs = int(len(input_dims))
        self.eps = float(eps)

        def _mlp(in_dim: int) -> nn.Module:
            if enc_depth <= 1:
                return nn.Sequential(nn.Linear(in_dim, emb_dim))
            layers: List[nn.Module] = []
            d = in_dim
            for _ in range(enc_depth - 1):
                layers.append(nn.Linear(d, enc_hidden))
                layers.append(nn.ReLU())
                if enc_dropout > 0:
                    layers.append(nn.Dropout(p=enc_dropout))
                d = enc_hidden
            layers.append(nn.Linear(d, emb_dim))
            return nn.Sequential(*layers)

        self.encoders = nn.ModuleList([_mlp(int(d)) for d in input_dims])
        self.gat = nn.ModuleList(
            [
                _ObsGATLayer(emb_dim=emb_dim, heads=gat_heads, dropout=gat_dropout)
                for _ in range(max(1, int(gat_layers)))
            ]
        )
        self.head = nn.Linear(emb_dim, 1)
        self.weight_logits = nn.Parameter(torch.zeros((self.n_obs,), dtype=torch.float32))

    def forward(self, xs: List):
        import torch

        if len(xs) != self.n_obs:
            raise ValueError(f"Expected {self.n_obs} obs inputs, got {len(xs)}")

        emb_list = []
        for x, enc in zip(xs, self.encoders):
            emb_list.append(enc(x))  # (B, emb)
        H = torch.stack(emb_list, dim=1)  # (B, n_obs, emb)
        for layer in self.gat:
            H = layer(H)
        p = self.head(H).squeeze(-1)  # (B, n_obs)

        w = _floor_softmax_weights(self.weight_logits, eps=self.eps)  # (n_obs,)
        score = (p * w.view(1, -1)).sum(dim=1)  # (B,)
        return score, p, w


class _ObsGATLayer(torch.nn.Module):
    """
    Minimal fully-connected GAT over obs dimension, per-sample.
    Input:  (B, n, emb)
    Output: (B, n, emb)
    """

    def __init__(self, *, emb_dim: int, heads: int, dropout: float) -> None:
        import torch
        import torch.nn as nn

        super().__init__()
        self.emb_dim = int(emb_dim)
        self.heads = max(1, int(heads))
        self.dropout = float(dropout)
        if self.emb_dim % self.heads != 0:
            raise ValueError(f"emb_dim must be divisible by heads (emb_dim={emb_dim}, heads={heads})")
        self.d_head = self.emb_dim // self.heads

        self.lin = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty((self.heads, self.d_head), dtype=torch.float32))
        self.a_dst = nn.Parameter(torch.empty((self.heads, self.d_head), dtype=torch.float32))
        self.out = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(self.emb_dim)
        self.drop = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x):
        import torch

        B, n, emb = x.shape
        if emb != self.emb_dim:
            raise ValueError(f"Unexpected emb dim: {emb} (expected {self.emb_dim})")

        h = self.lin(x)  # (B,n,emb)
        h = h.view(B, n, self.heads, self.d_head)  # (B,n,H,D)

        f = (h * self.a_src.view(1, 1, self.heads, self.d_head)).sum(dim=-1)  # (B,n,H)
        g = (h * self.a_dst.view(1, 1, self.heads, self.d_head)).sum(dim=-1)  # (B,n,H)
        e = self.leaky(f.unsqueeze(2) + g.unsqueeze(1))  # (B,n,n,H)
        alpha = torch.softmax(e, dim=2)  # over neighbor j
        alpha = self.drop(alpha)

        out = torch.einsum("b i j h, b j h d -> b i h d", alpha, h)  # (B,n,H,D)
        out = out.reshape(B, n, self.emb_dim)
        out = self.out(out)
        out = self.drop(out)
        return self.norm(x + out)


def _train_gats(
    *,
    train_xs: List,
    train_y,
    valid_xs: List,
    valid_y,
    model: ObsEncoderGAT,
    device: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_steps: int,
    eval_every: int,
    patience: int,
    seed: int,
):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    class _DS(Dataset):
        def __init__(self, xs: List[np.ndarray], y: np.ndarray):
            self.xs = xs
            self.y = y
            self.n = int(y.shape[0])

        def __len__(self):
            return self.n

        def __getitem__(self, idx: int):
            xs = [torch.from_numpy(np.asarray(x[idx], dtype=np.float32)) for x in self.xs]
            y = float(self.y[idx])
            return xs, y

    def _to_tensor_batch(batch):
        xs_list, yb = batch  # xs_list: List[Tensor[B, d_i]], yb: Tensor[B]
        xb = [x.to(device) for x in xs_list]
        yb_t = torch.as_tensor(yb, dtype=torch.float32, device=device).view(-1)
        return xb, yb_t

    train_y = np.asarray(train_y, dtype=np.float32).reshape(-1)
    valid_y = np.asarray(valid_y, dtype=np.float32).reshape(-1) if valid_y is not None else None

    ds_tr = _DS(train_xs, train_y)
    dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    if valid_xs and valid_y is not None and len(valid_y) > 0:
        max_v = min(len(valid_y), 50_000)
        if len(valid_y) > max_v:
            idx = rng.choice(len(valid_y), size=max_v, replace=False)
            valid_xs = [x[idx] for x in valid_xs]
            valid_y = valid_y[idx]
        ds_va = _DS(valid_xs, valid_y)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        dl_va = None

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    no_improve = 0

    global_step = 0
    model.train()
    while global_step < max_steps:
        for batch in dl:
            xb, yb = _to_tensor_batch(batch)
            opt.zero_grad(set_to_none=True)
            pred, _, _ = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step >= max_steps:
                break

            if dl_va is not None and (global_step % eval_every == 0):
                model.eval()
                vs = []
                with torch.no_grad():
                    for vb in dl_va:
                        vxb, vyb = _to_tensor_batch(vb)
                        vpred, _, _ = model(vxb)
                        vs.append(loss_fn(vpred, vyb).item())
                vloss = float(np.mean(vs)) if vs else float("inf")
                model.train()

                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        global_step = max_steps
                        break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_valid_mse": best_val if best_state is not None else None}


def _predict_gats(
    *,
    xs: List,
    model: ObsEncoderGAT,
    device: str,
    batch_size: int,
):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, xs: List[np.ndarray]):
            self.xs = xs
            self.n = int(xs[0].shape[0]) if xs else 0

        def __len__(self):
            return self.n

        def __getitem__(self, idx: int):
            return [torch.from_numpy(np.asarray(x[idx], dtype=np.float32)) for x in self.xs]

    ds = _DS(xs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    preds = []
    with torch.no_grad():
        for batch in dl:
            xb = [x.to(device) for x in batch]
            p, _, _ = model(xb)
            preds.append(p.detach().cpu().numpy())
    if not preds:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(preds, axis=0).astype(np.float64)


def run_stage4_alphaagent_gats_obs_branch(
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
    from run.config import load_rd_config
    from run.pipeline.stage4 import Stage4Result

    import numpy as np
    import pandas as pd
    import polars as pl
    import torch

    cfg = cfg or load_rd_config()
    assert run_ctx is not None, "run_ctx is required"

    # CLI/env
    topk = int(os.environ.get("ALPHA_STAGE4_TOPK", str(_CLI_TOPK)))
    n_drop = int(os.environ.get("ALPHA_STAGE4_N_DROP", str(_CLI_N_DROP)))
    eps = float(os.environ.get("ALPHA_STAGE4_OBS_EPS", str(_CLI_EPS)))
    use_csz = bool(int(os.environ.get("ALPHA_STAGE4_CSZ", "1" if _CLI_CSZ else "0")))

    emb_dim = int(os.environ.get("ALPHA_STAGE4_GATS_EMB_DIM", str(_CLI_EMB_DIM)))
    enc_hidden = int(os.environ.get("ALPHA_STAGE4_GATS_ENC_HIDDEN", str(_CLI_ENC_HIDDEN)))
    enc_depth = int(os.environ.get("ALPHA_STAGE4_GATS_ENC_DEPTH", str(_CLI_ENC_DEPTH)))
    enc_dropout = float(os.environ.get("ALPHA_STAGE4_GATS_ENC_DROPOUT", str(_CLI_ENC_DROPOUT)))
    gat_heads = int(os.environ.get("ALPHA_STAGE4_GATS_HEADS", str(_CLI_GAT_HEADS)))
    gat_layers = int(os.environ.get("ALPHA_STAGE4_GATS_LAYERS", str(_CLI_GAT_LAYERS)))
    gat_dropout = float(os.environ.get("ALPHA_STAGE4_GATS_DROPOUT", str(_CLI_GAT_DROPOUT)))

    lr = float(os.environ.get("ALPHA_STAGE4_GATS_LR", str(_CLI_LR)))
    weight_decay = float(os.environ.get("ALPHA_STAGE4_GATS_WEIGHT_DECAY", str(_CLI_WEIGHT_DECAY)))
    batch_size = int(os.environ.get("ALPHA_STAGE4_GATS_BATCH_SIZE", str(_CLI_BATCH_SIZE)))
    max_steps = int(os.environ.get("ALPHA_STAGE4_GATS_STEPS", str(_CLI_STEPS)))
    eval_every = int(os.environ.get("ALPHA_STAGE4_GATS_EVAL_EVERY", str(_CLI_EVAL_EVERY)))
    patience = int(os.environ.get("ALPHA_STAGE4_GATS_PATIENCE", str(_CLI_PATIENCE)))
    seed = int(os.environ.get("ALPHA_STAGE4_GATS_SEED", str(_CLI_SEED)))

    device = str(os.environ.get("ALPHA_STAGE4_GATS_DEVICE", str(_CLI_DEVICE))).strip().lower()
    if device in {"auto", ""}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    is_start = cfg.data_split.in_sample_start
    is_end = cfg.data_split.in_sample_end
    oos_start = cfg.data_split.out_sample_start
    oos_end = cfg.data_split.out_sample_end

    if not passed_formulas:
        raise ValueError("Stage4-gats requires `passed_formulas` (Stage2 passed_formulas).")

    # Group formulas by observation_id and collect required cols
    all_formula_cols = set(formula_df.columns) - {"timestamp", "ticker"}
    obs_to_names: Dict[str, List[str]] = {}
    polarity_by_name: Dict[str, str] = {}
    for f in passed_formulas:
        name = str(f.get("name") or "")
        if not name or name not in all_formula_cols:
            continue
        obs_id = str(f.get("observation_id") or f.get("obs_id") or "UNKNOWN_OBS")
        obs_to_names.setdefault(obs_id, []).append(name)
        polarity_by_name.setdefault(name, str(f.get("polarity") or ""))

    obs_ids = sorted(obs_to_names.keys())
    obs_feat_cols: Dict[str, List[str]] = {}
    for obs_id in obs_ids:
        cols = sorted(set(obs_to_names.get(obs_id, [])))
        cols = [c for c in cols if c in all_formula_cols]
        if cols:
            obs_feat_cols[obs_id] = cols
    obs_ids = sorted(obs_feat_cols.keys())
    if not obs_ids:
        raise RuntimeError("Stage4-gats produced zero obs groups with usable formula columns.")

    # Build panel: base + only needed columns
    need_cols = sorted({c for cols in obs_feat_cols.values() for c in cols})
    base_cols = ["timestamp", "ticker", "close"]
    base_pd = ohlcv_df.select([c for c in base_cols if c in ohlcv_df.columns]).to_pandas()
    base_pd["timestamp"] = _normalize_trade_dates(base_pd["timestamp"])
    f_pd = formula_df.select(["timestamp", "ticker", *need_cols]).to_pandas()
    f_pd["timestamp"] = _normalize_trade_dates(f_pd["timestamp"])
    panel = base_pd.merge(f_pd, on=["timestamp", "ticker"], how="inner")
    panel = panel.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(drop=True)
    panel = _compute_label_close_t1_to_t2(panel)
    label_df = panel[["timestamp", "ticker", "label"]].copy()

    # Apply polarity flip in-place (lower_is_more_true => sign flip)
    for name, pol in polarity_by_name.items():
        if name in panel.columns and str(pol).startswith("lower"):
            panel[name] = -panel[name]

    panel = panel.replace([np.inf, -np.inf], np.nan)

    # Split IS -> train/valid
    df_train, df_valid = _split_is_train_valid(panel, is_start=is_start, is_end=is_end, valid_ratio=0.2)
    df_train = df_train.dropna(subset=["label"])
    df_valid = df_valid.dropna(subset=["label"])
    if len(df_train) < 1000:
        raise RuntimeError("Stage4-gats: not enough training rows after label drop.")

    # Feature NaN은 0으로 채움(라벨은 dropna로 이미 제거)
    df_train[need_cols] = df_train[need_cols].fillna(0.0)
    if len(df_valid) > 0:
        df_valid[need_cols] = df_valid[need_cols].fillna(0.0)
    panel[need_cols] = panel[need_cols].fillna(0.0)

    if use_csz:
        df_train = _cs_zscore(df_train, need_cols, by="timestamp")
        if len(df_valid) > 0:
            df_valid = _cs_zscore(df_valid, need_cols, by="timestamp")
        panel = _cs_zscore(panel, need_cols, by="timestamp")

    def _build_xs(df) -> List[np.ndarray]:
        xs = []
        for obs_id in obs_ids:
            cols = obs_feat_cols[obs_id]
            x = df[cols].to_numpy(dtype=np.float32, copy=False)
            xs.append(x)
        return xs

    train_xs = _build_xs(df_train)
    train_y = df_train["label"].to_numpy(dtype=np.float32, copy=False)
    valid_xs = _build_xs(df_valid) if len(df_valid) > 0 else []
    valid_y = df_valid["label"].to_numpy(dtype=np.float32, copy=False) if len(df_valid) > 0 else None

    input_dims = [int(x.shape[1]) for x in train_xs]
    model = ObsEncoderGAT(
        input_dims=input_dims,
        emb_dim=emb_dim,
        enc_hidden=enc_hidden,
        enc_depth=enc_depth,
        enc_dropout=enc_dropout,
        gat_heads=gat_heads,
        gat_layers=gat_layers,
        gat_dropout=gat_dropout,
        eps=eps,
    )
    model, train_info = _train_gats(
        train_xs=train_xs,
        train_y=train_y,
        valid_xs=valid_xs,
        valid_y=valid_y,
        model=model,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_steps=max_steps,
        eval_every=max(1, eval_every),
        patience=max(3, patience),
        seed=seed,
    )

    # Predict over full panel for final signal
    all_xs = _build_xs(panel)
    pred = _predict_gats(xs=all_xs, model=model, device=device, batch_size=batch_size)
    signal_df = panel[["timestamp", "ticker"]].copy()
    signal_df["score"] = pred.astype(float)
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

    with torch.no_grad():
        w = _floor_softmax_weights(model.weight_logits.detach().to(device), eps=eps).detach().cpu().numpy().tolist()
    weights = {obs_id: float(wi) for obs_id, wi in zip(obs_ids, w, strict=False)}

    iter_prefix = f"iter_{outer_iter}" if outer_iter is not None else "iter_1"
    _save_qlib_artifacts(
        run_ctx=run_ctx,
        iter_prefix=iter_prefix,
        model_tag="obs_gats",
        split="is",
        report_df=is_report_df,
        positions_raw=is_positions_raw,
    )
    _save_qlib_artifacts(
        run_ctx=run_ctx,
        iter_prefix=iter_prefix,
        model_tag="obs_gats",
        split="oos",
        report_df=oos_report_df,
        positions_raw=oos_positions_raw,
    )

    try:
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/obs_gats/signal_metrics_is.csv", is_sig_ts)
        run_ctx.save_csv(f"qlib_artifacts/{iter_prefix}/obs_gats/signal_metrics_oos.csv", oos_sig_ts)
    except Exception:
        pass

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
            "combo_key": "OBS_GATS",
            "feature_mode": "obs_graph",
            "n_feat": int(sum(len(obs_feat_cols.get(k, [])) for k in obs_ids)),
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

    combo_entry = {
        "combo_idx": None,
        "combo_key": ("OBS_GATS",),
        "formula_names": [],
        "topk": topk,
        "n_drop": n_drop,
        "use_cszscore": use_csz,
        "gats_params": {
            "emb_dim": emb_dim,
            "enc_hidden": enc_hidden,
            "enc_depth": enc_depth,
            "enc_dropout": enc_dropout,
            "gat_heads": gat_heads,
            "gat_layers": gat_layers,
            "gat_dropout": gat_dropout,
            "eps": eps,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "eval_every": eval_every,
            "patience": patience,
            "seed": seed,
            "device": device,
        },
        "obs_ids": obs_ids,
        "obs_formula_counts": {k: len(v) for k, v in obs_feat_cols.items()},
        "obs_weights": weights,
        "train_info": _to_jsonable(train_info),
        "is_signal_metrics": _to_jsonable(dict(is_sig_m)),
        "oos_signal_metrics": _to_jsonable(dict(oos_sig_m)),
        "is_report_metrics": _to_jsonable(dict(is_report_metrics)),
        "oos_report_metrics": _to_jsonable(dict(oos_report_metrics)),
        "is_metrics": _to_jsonable(dict(is_metrics)),
        "oos_metrics": _to_jsonable(dict(oos_metrics)),
        "data_split": {
            "insample": {"strategy": _to_jsonable(dict(is_metrics))},
            "outsample": {"strategy": _to_jsonable(dict(oos_metrics))},
        },
        "is_turnover_sum": float(is_turnover_sum),
        "oos_turnover_sum": float(oos_turnover_sum),
    }

    summary = {
        "stage4_mode": "alphaagent_gats_obs_branch",
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
            "best_combo_idx": None,
            "best_is_information_ratio": float(is_metrics.get("information_ratio") or 0.0),
            "evaluated_combos": 1,
        },
        "all_combinations": [combo_entry],
    }

    lines = [
        "# Stage 4 (AlphaAgent-style): Obs-graph GAT + TopkDropoutStrategy",
        "",
        f"- Market: `{summary['market']}`",
        f"- IS: `{is_start} ~ {is_end}` / OOS: `{oos_start} ~ {oos_end}`",
        f"- topk: `{topk}`, n_drop: `{n_drop}`",
        f"- obs: `{len(obs_ids)}`, eps: `{eps}`",
        f"- weights: `{weights}`",
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
            f"[Stage4-gats] obs_gats (obs={len(obs_ids)}, eps={eps}, device={device}): "
            f"IS_IR={float(is_metrics.get('information_ratio') or 0.0):.6f}, "
            f"OOS_IR={float(oos_metrics.get('information_ratio') or 0.0):.6f}"
        )

    return Stage4Result(
        hypothesis_id=hypothesis_id,
        config={"stage4_mode": "alphaagent_gats_obs_branch"},
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

    topk = 50
    n_drop = 5
    eps = 0.1
    csz = True
    skip_stage3 = False

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
        "topk": topk,
        "n_drop": n_drop,
        "eps": eps,
        "csz": csz,
        "skip_stage3": skip_stage3,
    }


_CLI_TOPK = 50
_CLI_N_DROP = 5
_CLI_EPS = 0.1
_CLI_CSZ = True
_CLI_EMB_DIM = 32
_CLI_ENC_HIDDEN = 128
_CLI_ENC_DEPTH = 2
_CLI_ENC_DROPOUT = 0.0
_CLI_GAT_HEADS = 4
_CLI_GAT_LAYERS = 2
_CLI_GAT_DROPOUT = 0.0
_CLI_LR = 2e-3
_CLI_WEIGHT_DECAY = 2e-4
_CLI_BATCH_SIZE = 8192
_CLI_STEPS = 3000
_CLI_EVAL_EVERY = 50
_CLI_PATIENCE = 30
_CLI_SEED = 42
_CLI_DEVICE = "auto"


def main() -> None:
    global _CLI_TOPK, _CLI_N_DROP, _CLI_EPS, _CLI_CSZ
    parsed = _parse_args(sys.argv[1:])
    _CLI_TOPK = parsed["topk"]
    _CLI_N_DROP = parsed["n_drop"]
    _CLI_EPS = parsed["eps"]
    _CLI_CSZ = parsed["csz"]

    # 런타임 패치: run.main의 Stage4만 교체(Stage1~3 동일)
    import run.main as main_mod

    main_mod.run_stage4 = run_stage4_alphaagent_gats_obs_branch  # type: ignore[assignment]

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

    from run.main import run_outer_loop, run_pipeline
    from util.run_context import RunContext

    run_ctx = RunContext.create(base_dir=os.path.join(PROJECT_ROOT, "runs", "gats"))

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
