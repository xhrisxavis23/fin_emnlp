from __future__ import annotations

import polars as pl

from run.config import DataConfig


def standardize_price_columns(df: pl.DataFrame, cfg: DataConfig) -> pl.DataFrame:
    """
    Normalize an arbitrary price panel into the pipeline's canonical schema:
    - timestamp, ticker, open, high, low, close, volume

    If a column is already in canonical name, it is kept as-is.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a polars.DataFrame")

    rename_map: dict[str, str] = {}

    # time + asset identifiers
    if cfg.date_col in df.columns and cfg.date_col != "timestamp":
        rename_map[cfg.date_col] = "timestamp"
    if cfg.asset_col in df.columns and cfg.asset_col != "ticker":
        rename_map[cfg.asset_col] = "ticker"

    # OHLCV
    if cfg.open_col in df.columns and cfg.open_col != "open":
        rename_map[cfg.open_col] = "open"
    if cfg.high_col in df.columns and cfg.high_col != "high":
        rename_map[cfg.high_col] = "high"
    if cfg.low_col in df.columns and cfg.low_col != "low":
        rename_map[cfg.low_col] = "low"
    if cfg.price_col in df.columns and cfg.price_col != "close":
        rename_map[cfg.price_col] = "close"
    if cfg.volume_col in df.columns and cfg.volume_col != "volume":
        rename_map[cfg.volume_col] = "volume"

    out = df.rename(rename_map) if rename_map else df

    required = {"timestamp", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"price data missing required columns after standardization: {missing}")

    return out

