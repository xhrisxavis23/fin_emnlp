from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def _project_root() -> Path:
    # .../01_15_new_qlib/analysis/export_qlib_report.py -> .../01_15_new_qlib
    return Path(__file__).resolve().parents[1]


def _artifact_dir(
    project_root: Path,
    run_id: str,
    iter_idx: int,
    combo_idx: int,
    split: str,
    variant: Optional[str],
) -> Path:
    base = (
        project_root
        / "runs"
        / run_id
        / "qlib_artifacts"
        / f"iter_{iter_idx}"
        / f"combo_{combo_idx}"
    )
    if variant:
        base = base / variant
    return base / split


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export Qlib HTML charts from saved Stage4 artifacts (report_normal_1day.pkl). "
            "This avoids re-running the backtest."
        )
    )
    parser.add_argument("--run-id", required=True, help="Run folder name under runs/, e.g. 20260127_181328")
    parser.add_argument("--iter", type=int, default=1, help="Outer loop iteration index (1-based). Default: 1")
    parser.add_argument("--combo", type=int, default=1, help="Combination index (1-based). Default: 1")
    parser.add_argument("--split", choices=["is", "oos"], default="oos", help="Which split to export. Default: oos")
    parser.add_argument(
        "--variant",
        default="",
        help="Optional variant folder under combo (e.g. fixed_q90). Default: none (use combo_*/is|oos).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: runs/<run-id>/reports/iter_X/combo_Y/<variant>/<split>/",
    )
    parser.add_argument(
        "--dump-csv",
        action="store_true",
        help="Also export report_normal_1day.pkl and port_analysis_1day.pkl to CSV.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export a simple cumulative-return PNG using matplotlib (no kaleido needed).",
    )

    args = parser.parse_args()

    project_root = _project_root()
    sys.path.insert(0, str(project_root))

    # Lazy import after sys.path update (repo-local qlib).
    import qlib.contrib.report.analysis_position as ap  # noqa: E402

    variant = (args.variant or "").strip()
    artifact_dir = _artifact_dir(project_root, args.run_id, args.iter, args.combo, args.split, variant or None)

    report_pkl = artifact_dir / "report_normal_1day.pkl"
    if not report_pkl.exists():
        raise FileNotFoundError(f"Missing artifact: {report_pkl}")

    report_df = pd.read_pickle(report_pkl)
    # Qlib report functions typically treat the index as the trading date.
    report_df.index = report_df.index.rename("date")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            project_root
            / "runs"
            / args.run_id
            / "reports"
            / f"iter_{args.iter}"
            / f"combo_{args.combo}"
            / (variant if variant else "default")
            / args.split
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) report_graph (cumulative return / benchmark / cost / turnover, etc.)
    report_figs = ap.report_graph(report_df, show_notebook=False)
    if isinstance(report_figs, (list, tuple)) and len(report_figs) > 0:
        report_figs[0].write_html(out_dir / "report_graph.html")

    # 2) risk_analysis_graph (monthly bars; needs only report_normal_df)
    risk_figs = ap.risk_analysis_graph(report_normal_df=report_df, show_notebook=False)
    for i, fig in enumerate(risk_figs):
        fig.write_html(out_dir / f"risk_analysis_graph_{i}.html")

    if args.png:
        # Replicate Qlib's report_graph core series (simple version).
        import matplotlib.pyplot as plt  # noqa: E402

        df = report_df.sort_index()
        cum_bench = df["bench"].cumsum() if "bench" in df.columns else None
        cum_return_wo_cost = df["return"].cumsum()
        cum_return_w_cost = (df["return"] - df.get("cost", 0.0)).cumsum()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cum_return_wo_cost.index, cum_return_wo_cost.values, label="cum_return_wo_cost")
        ax.plot(cum_return_w_cost.index, cum_return_w_cost.values, label="cum_return_w_cost")
        if cum_bench is not None:
            ax.plot(cum_bench.index, cum_bench.values, label="cum_bench")
        ax.set_title("Cumulative Return")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / "cumulative_return.png", dpi=150)
        plt.close(fig)

    if args.dump_csv:
        report_df.to_csv(out_dir / "report_normal_1day.csv")
        port_pkl = artifact_dir / "port_analysis_1day.pkl"
        if port_pkl.exists():
            port_df = pd.read_pickle(port_pkl)
            if hasattr(port_df, "to_csv"):
                port_df.to_csv(out_dir / "port_analysis_1day.csv")

    print(f"Saved charts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
