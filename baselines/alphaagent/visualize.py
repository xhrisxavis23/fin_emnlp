#!/usr/bin/env python3
"""
Visualize AlphaAgent/Qlib daily trade records saved as a pickle DataFrame.

Expected input
  - trade_records_normal_<freq>.pkl

This file is produced by our patched Qlib PortAnaRecord and contains per-instrument
buy/sell rows with at least:
  - status:  1=buy, -1=sell
  - datetime: either an index level named "datetime" or a column
  - instrument: either an index level named "instrument" or a column
Optional columns (used if present): amount, price, weight, cash.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def _require_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency 'pandas'.\n"
            "Run this script inside the same environment you use for AlphaAgent/Qlib (e.g., conda env)."
        ) from e


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency 'matplotlib'.\n"
            "Install it in your AlphaAgent/Qlib environment (e.g., `pip install matplotlib`)."
        ) from e


def _load_trade_df(path: Path):
    pd = _require_pandas()
    try:
        obj = pd.read_pickle(path)
    except Exception as e:
        raise SystemExit(f"Failed to read pickle: {path}\n{e}") from e

    if hasattr(pd, "DataFrame") and isinstance(obj, pd.DataFrame):
        return obj
    raise SystemExit(f"Unsupported pickle content type: {type(obj)} (expected pandas.DataFrame)")


def _extract_cols_and_index(df):
    pd = _require_pandas()

    if "datetime" in df.index.names:
        dt = df.index.get_level_values("datetime")
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    else:
        raise SystemExit("Trade records must have 'datetime' as an index level or a column.")

    if "instrument" in df.index.names:
        inst = df.index.get_level_values("instrument")
    elif "instrument" in df.columns:
        inst = df["instrument"].astype(str)
    else:
        # Still allow plotting without instrument-level details.
        inst = pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype="object")

    return dt, inst


def _build_daily_stats(df):
    pd = _require_pandas()
    import numpy as np  # type: ignore

    if "status" not in df.columns:
        raise SystemExit("Trade records must have a 'status' column (1=buy, -1=sell).")

    dt, inst = _extract_cols_and_index(df)
    day = pd.to_datetime(dt).dt.floor("D")

    work = df.copy()
    work["_day"] = day
    work["_instrument"] = inst.astype(str)
    work["_is_buy"] = work["status"] == 1
    work["_is_sell"] = work["status"] == -1

    if "amount" in work.columns and "price" in work.columns:
        amt = pd.to_numeric(work["amount"], errors="coerce").abs()
        px = pd.to_numeric(work["price"], errors="coerce")
        work["_trade_value"] = amt * px
    else:
        work["_trade_value"] = np.nan

    daily = (
        work.groupby("_day", sort=True)
        .agg(
            buy_count=("_is_buy", "sum"),
            sell_count=("_is_sell", "sum"),
            traded_instruments=("_instrument", "nunique"),
            buy_value=("_trade_value", lambda x: x[work.loc[x.index, "_is_buy"]].sum(skipna=True)),
            sell_value=("_trade_value", lambda x: x[work.loc[x.index, "_is_sell"]].sum(skipna=True)),
        )
        .reset_index()
        .rename(columns={"_day": "day"})
    )
    daily["net_count"] = daily["buy_count"] - daily["sell_count"]
    if not daily["buy_value"].isna().all() or not daily["sell_value"].isna().all():
        daily["net_value"] = daily["buy_value"].fillna(0.0) - daily["sell_value"].fillna(0.0)
    else:
        daily["net_value"] = np.nan

    top_instruments = (
        work.groupby("_instrument", sort=False)
        .agg(trade_days=("_day", "nunique"), trades=("status", "size"))
        .sort_values(["trades", "trade_days"], ascending=False)
        .reset_index()
        .rename(columns={"_instrument": "instrument"})
    )

    return daily, top_instruments


def visualize_trade_records(
    *,
    pkl_path: Path,
    out: Optional[Path],
    show: bool,
    topn: int,
) -> int:
    plt = _require_matplotlib()
    pd = _require_pandas()

    df = _load_trade_df(pkl_path)
    daily, top_instruments = _build_daily_stats(df)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Daily Trade Records: {pkl_path.name}")

    # 1) counts (sell as negative for readability)
    axes[0].bar(daily["day"], daily["buy_count"], label="BUY count")
    axes[0].bar(daily["day"], -daily["sell_count"], label="SELL count")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Trades (count)")
    axes[0].legend(loc="upper left")

    # 2) value (sell as negative); skip if NaN
    if not daily["net_value"].isna().all():
        axes[1].bar(daily["day"], daily["buy_value"].fillna(0.0), label="BUY value")
        axes[1].bar(daily["day"], -daily["sell_value"].fillna(0.0), label="SELL value")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Trade value (amount*price)")
        axes[1].legend(loc="upper left")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Trade value not available (missing 'amount' and/or 'price').",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
        axes[1].set_axis_off()

    # 3) unique instruments traded
    axes[2].plot(daily["day"], daily["traded_instruments"], label="Unique traded instruments", color="tab:purple")
    axes[2].set_ylabel("Unique instruments")
    axes[2].legend(loc="upper left")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)

        # Also write summary CSVs next to the figure for convenience.
        daily_csv = out.with_suffix(".daily.csv")
        top_csv = out.with_suffix(".top_instruments.csv")
        daily.to_csv(daily_csv, index=False)
        top_instruments.head(max(topn, 0)).to_csv(top_csv, index=False)

        print(f"Wrote figure: {out}")
        print(f"Wrote daily summary: {daily_csv}")
        print(f"Wrote top instruments: {top_csv}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Print a quick text summary.
    pd.set_option("display.width", 120)
    print("\nTop instruments:")
    print(top_instruments.head(max(topn, 0)).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize daily trade records saved as a pickle DataFrame.")
    ap.add_argument(
        "--pkl",
        required=True,
        help="Path to trade_records_normal_<freq>.pkl (e.g. .../portfolio_analysis/trade_records_normal_1day.pkl)",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output image path (png/pdf). If empty, no file is written (use --show to display).",
    )
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Show an interactive window.")
    ap.add_argument("--topn", type=int, default=20, help="How many instruments to print/save in the top list.")
    args = ap.parse_args()

    pkl_path = Path(args.pkl).expanduser()
    if not pkl_path.exists():
        raise SystemExit(f"pkl not found: {pkl_path}")

    out = Path(args.out).expanduser() if args.out else None
    return visualize_trade_records(pkl_path=pkl_path, out=out, show=bool(args.show), topn=int(args.topn))


if __name__ == "__main__":
    raise SystemExit(main())
