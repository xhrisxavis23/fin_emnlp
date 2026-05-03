from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    """
    Compatibility shim for reading pickles created in environments where NumPy's
    internal module path is `numpy._core.*` (NumPy 2.x).

    In NumPy 1.x that path doesn't exist, so we remap it to `numpy.core.*`
    during unpickling.
    """

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 - pickle API
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def read_pickle_compat(path: Path) -> Any:  # noqa: ANN401 - generic pickle loader
    with path.open("rb") as f:
        return _NumpyCoreCompatUnpickler(f).load()


def _ensure_datetime_ts(v: Any) -> pd.Timestamp:
    """
    positions_normal_1day.pkl keys can become non-portable across pandas/numpy
    versions. Converting via `str(v)` is the most robust normalization.
    """

    return pd.Timestamp(str(v))


def _standard_max_drawdown(daily_returns: pd.Series) -> float:
    """
    Standard (equity-curve) max drawdown based on compounded returns.
    Returns a negative number in [-1, 0].
    """

    if daily_returns is None or len(daily_returns) == 0:
        return 0.0
    r = pd.to_numeric(daily_returns, errors="coerce").fillna(0.0)
    nav = (1.0 + r).cumprod()
    if nav.empty:
        return 0.0
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


@dataclass(frozen=True)
class Paths:
    run_dir: Path
    iter_n: int
    combo_idx: int
    split: str
    mode: str

    def qlib_dir(self) -> Path:
        base = self.run_dir / "qlib_artifacts" / f"iter_{self.iter_n}" / f"combo_{self.combo_idx}"
        if self.mode == "normal":
            return base / self.split
        return base / self.mode / self.split

    def report_pkl(self) -> Path:
        return self.qlib_dir() / "report_normal_1day.pkl"

    def positions_pkl(self) -> Path:
        return self.qlib_dir() / "positions_normal_1day.pkl"

    def price_parquet(self) -> Path:
        return self.run_dir / "data" / f"price_with_formulas_iter_{self.iter_n}.parquet"

    def out_dir(self) -> Path:
        return (
            self.run_dir
            / "reports"
            / "diagnostics"
            / f"iter_{self.iter_n}"
            / f"combo_{self.combo_idx}"
            / self.mode
            / self.split
        )


def _iter_held_positions(pos_dict: dict[Any, Any]) -> Iterable[tuple[pd.Timestamp, str]]:
    for k, positions in pos_dict.items():
        ts = _ensure_datetime_ts(k)
        if not isinstance(positions, list):
            continue
        for rec in positions:
            if not isinstance(rec, dict):
                continue
            inst = rec.get("instrument")
            if inst is None:
                continue
            yield ts, str(inst)


def _compute_ret1_for_slice(price: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker next-trading-day returns (ret1) with a calendar guard to avoid
    treating long gaps as "1-day" returns.
    """

    out = price.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["ticker", "timestamp"], kind="mergesort")

    cal = pd.Index(sorted(out["timestamp"].unique()))
    next_map = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}

    out["next_close"] = out.groupby("ticker")["close"].shift(-1)
    out["next_ts"] = out.groupby("ticker")["timestamp"].shift(-1)
    out["expected_next_ts"] = out["timestamp"].map(next_map)

    consec = out["next_ts"].eq(out["expected_next_ts"])
    out["ret1"] = np.where(consec, out["next_close"] / out["close"] - 1.0, np.nan)
    return out[["timestamp", "ticker", "ret1"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose MDD vs. equity MDD, and held per-ticker returns.")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--iter", dest="iter_n", required=True, type=int)
    ap.add_argument("--combo", dest="combo_idx", required=True, type=int)
    ap.add_argument("--split", choices=["is", "oos"], required=True)
    ap.add_argument("--mode", default="normal", help="normal or fixed_qXX (e.g., fixed_q90)")
    ap.add_argument("--threshold", type=float, default=-0.05, help="Flag held ret1 <= threshold")
    ap.add_argument("--topk", type=int, default=10, help="Show/plot worst N ticker-days")
    args = ap.parse_args()

    paths = Paths(
        run_dir=args.run_dir,
        iter_n=int(args.iter_n),
        combo_idx=int(args.combo_idx),
        split=str(args.split),
        mode=str(args.mode),
    )

    out_dir = paths.out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = paths.report_pkl()
    if not report_path.exists():
        raise SystemExit(f"Missing report: {report_path}")

    report_df = read_pickle_compat(report_path)
    if not isinstance(report_df, pd.DataFrame) or "return" not in report_df.columns:
        raise SystemExit(f"Unexpected report format: {report_path}")

    net = pd.to_numeric(report_df["return"], errors="coerce").fillna(0.0) - pd.to_numeric(
        report_df.get("cost", 0.0), errors="coerce"
    ).fillna(0.0)
    nav = (1.0 + net).cumprod()
    net_return = float(nav.iloc[-1] - 1.0) if len(nav) else 0.0

    mdd_equity = _standard_max_drawdown(net)
    try:
        from qlib.contrib.evaluate import risk_analysis

        mdd_qlib_sum = float(risk_analysis(net, freq="day").loc["max_drawdown", "risk"])
    except Exception:
        mdd_qlib_sum = float("nan")

    # Held ret1
    held_stats = {
        "held_pairs": 0,
        "matched_pairs": 0,
        "threshold": float(args.threshold),
        "count_le_threshold": 0,
        "min_ret1": None,
    }
    worst_rows: pd.DataFrame | None = None

    positions_path = paths.positions_pkl()
    if positions_path.exists():
        pos_obj = pd.read_pickle(positions_path)
        if isinstance(pos_obj, dict):
            held = pd.DataFrame(_iter_held_positions(pos_obj), columns=["timestamp", "ticker"])
        else:
            held = pd.DataFrame(columns=["timestamp", "ticker"])

        held_stats["held_pairs"] = int(held.shape[0])

        if not held.empty and paths.price_parquet().exists():
            start = held["timestamp"].min()
            end = held["timestamp"].max()

            price = pd.read_parquet(paths.price_parquet(), engine="pyarrow", columns=["ticker", "timestamp", "close"])
            price["timestamp"] = pd.to_datetime(price["timestamp"])
            price = price[(price["timestamp"] >= start) & (price["timestamp"] <= end)].copy()

            ret1_df = _compute_ret1_for_slice(price)
            merged = held.merge(ret1_df, on=["timestamp", "ticker"], how="left")

            avail = merged.dropna(subset=["ret1"]).copy()
            held_stats["matched_pairs"] = int(avail.shape[0])
            if not avail.empty:
                arr = avail["ret1"].to_numpy(dtype=float)
                held_stats["min_ret1"] = float(np.min(arr))
                held_stats["count_le_threshold"] = int(np.sum(arr <= float(args.threshold)))
                worst_rows = avail.nsmallest(int(args.topk), "ret1")

                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(12, 8))
                gs = fig.add_gridspec(2, 2)

                ax_nav = fig.add_subplot(gs[0, :])
                ax_hist = fig.add_subplot(gs[1, 0])
                ax_dd = fig.add_subplot(gs[1, 1])

                ax_nav.plot(nav.index, nav.values, lw=1.5, label="NAV (net)")
                ax_nav.plot(nav.index, nav.cummax().values, lw=1.0, ls="--", label="Running max")
                ax_nav.set_title("Equity Curve (Net)")
                ax_nav.legend(loc="best")

                ax_hist.hist(arr, bins=60, alpha=0.8)
                ax_hist.axvline(float(args.threshold), color="red", ls="--", lw=1.5, label=f"threshold={args.threshold:.2%}")
                ax_hist.set_title("Held Ticker-Day ret1 Distribution")
                ax_hist.legend(loc="best")

                dd = nav / nav.cummax() - 1.0
                ax_dd.plot(dd.index, dd.values, lw=1.0, color="tab:orange")
                ax_dd.set_title(f"Drawdown (equity) | min={mdd_equity:.2%}")

                fig.tight_layout()
                fig.savefig(out_dir / "mdd_and_held_ret1.png", dpi=160)
                plt.close(fig)

                worst_rows.to_csv(out_dir / "worst_held_ret1.csv", index=False)

    summary = {
        "run_dir": str(paths.run_dir),
        "iter": paths.iter_n,
        "combo": paths.combo_idx,
        "mode": paths.mode,
        "split": paths.split,
        "net_return": net_return,
        "mdd_equity": mdd_equity,
        "mdd_qlib_sum": mdd_qlib_sum,
        "min_daily_net": float(net.min()) if len(net) else 0.0,
        "max_daily_net": float(net.max()) if len(net) else 0.0,
        "held": held_stats,
    }
    (out_dir / "summary.json").write_text(pd.Series(summary).to_json(indent=2, force_ascii=False), encoding="utf-8")

    print(f"[Report] net_return={net_return:.2%} | mdd_equity={mdd_equity:.2%} | mdd_qlib_sum={mdd_qlib_sum:.2%}")
    if held_stats["held_pairs"] > 0:
        print(
            f"[Held ret1] matched={held_stats['matched_pairs']}/{held_stats['held_pairs']} | "
            f"min={held_stats['min_ret1'] if held_stats['min_ret1'] is not None else 'NA'} | "
            f"count(ret1<={held_stats['threshold']})={held_stats['count_le_threshold']}"
        )
        if worst_rows is not None and not worst_rows.empty:
            print("[Worst held ret1]")
            for _, row in worst_rows.iterrows():
                ts = pd.Timestamp(row["timestamp"]).date()
                print(f"  {ts} {row['ticker']} ret1={float(row['ret1']):.2%}")

    print(f"[Saved] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

