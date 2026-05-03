#!/usr/bin/env python3
"""
Extract Qlib metrics from an rdagent log directory.

Default input example:
  log/2026-01-22_20-23-03-119739

This script looks for pickled stdout strings under:
  Loop_*/running/Qlib_execute_log/*/*.pkl

and parses the following metrics (if present):
  IC, ICIR, Rank IC, Rank ICIR, Annualized Return, Information Ratio, Max Drawdown
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_TS_FMT = "%Y-%m-%d_%H-%M-%S-%f"


@dataclass(frozen=True)
class ExtractedRow:
    loop: Optional[int]
    timestamp: Optional[datetime]
    source_pkl: str
    ic: Optional[float]
    icir: Optional[float]
    rank_ic: Optional[float]
    rank_icir: Optional[float]
    annualized_return: Optional[float]
    information_ratio: Optional[float]
    max_drawdown: Optional[float]
    section: str


def _parse_timestamp_from_stem(stem: str) -> Optional[datetime]:
    try:
        return datetime.strptime(stem, _TS_FMT)
    except Exception:
        return None


def _loop_index_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.startswith("Loop_"):
            try:
                return int(part.split("_", 1)[1])
            except Exception:
                return None
    return None


def _read_pickle_string(path: Path) -> str:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, str):
        raise TypeError(f"Expected a pickled str in {path}, got {type(obj)}")
    return obj


def _extract_float(stdout: str, key: str) -> Optional[float]:
    # Supports both:
    #   'IC': np.float64(0.123)
    #   "IC": 0.123
    #   IC: 0.123
    pattern = re.compile(
        r"(?P<k>['\"]?%s['\"]?)\s*:\s*(?:np\.float64\()?\s*(?P<v>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)?"
        % re.escape(key),
        flags=re.MULTILINE,
    )
    m = pattern.search(stdout)
    if not m:
        return None
    try:
        return float(m.group("v"))
    except Exception:
        return None


def _slice_section(stdout: str, section_key: str) -> Optional[str]:
    section_markers = {
        "with_cost": "analysis results of the excess return with cost",
        "without_cost": "analysis results of the excess return without cost",
        "benchmark": "analysis results of benchmark return",
    }
    marker = section_markers.get(section_key)
    if not marker:
        return None

    idx = stdout.lower().find(marker.lower())
    if idx < 0:
        return None

    # Take a small window after the marker; the table is usually within ~10 lines.
    window = stdout[idx : idx + 2000]
    return window


def _extract_risk_table_metrics(section_text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    def pick(name: str) -> Optional[float]:
        m = re.search(
            r"^\s*%s\s+(?P<v>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$" % re.escape(name),
            section_text,
            flags=re.MULTILINE,
        )
        if not m:
            return None
        try:
            return float(m.group("v"))
        except Exception:
            return None

    return pick("annualized_return"), pick("information_ratio"), pick("max_drawdown")


def parse_metrics_from_stdout(stdout: str, *, section: str = "with_cost") -> Tuple[Dict[str, Optional[float]], str]:
    metrics: Dict[str, Optional[float]] = {
        "IC": _extract_float(stdout, "IC"),
        "ICIR": _extract_float(stdout, "ICIR"),
        "Rank IC": _extract_float(stdout, "Rank IC"),
        "Rank ICIR": _extract_float(stdout, "Rank ICIR"),
    }

    section_used = section
    section_text = _slice_section(stdout, section)
    if section_text is None:
        # Fall back: pick the first section we can find.
        for candidate in ("with_cost", "without_cost", "benchmark"):
            section_text = _slice_section(stdout, candidate)
            if section_text is not None:
                section_used = candidate
                break

    ar = ir = mdd = None
    if section_text is not None:
        ar, ir, mdd = _extract_risk_table_metrics(section_text)

    metrics.update(
        {
            "Annualized Return": ar,
            "Information Ratio": ir,
            "Max Drawdown": mdd,
        }
    )
    return metrics, section_used


def iter_qlib_execute_log_pkls(log_dir: Path) -> Iterable[Path]:
    # Match: **/Loop_*/running/Qlib_execute_log/*/*.pkl
    for p in log_dir.rglob("Qlib_execute_log"):
        if not p.is_dir():
            continue
        for pkl in p.rglob("*.pkl"):
            yield pkl


def pick_latest_per_loop(pkls: Iterable[Path]) -> List[Path]:
    best: Dict[Optional[int], Tuple[Optional[datetime], Path]] = {}
    for pkl in pkls:
        loop = _loop_index_from_path(pkl)
        ts = _parse_timestamp_from_stem(pkl.stem)
        prev = best.get(loop)
        if prev is None:
            best[loop] = (ts, pkl)
            continue
        prev_ts, _ = prev
        if prev_ts is None and ts is not None:
            best[loop] = (ts, pkl)
        elif prev_ts is not None and ts is not None and ts > prev_ts:
            best[loop] = (ts, pkl)
        elif prev_ts is None and ts is None:
            # Fall back to mtime comparison when filenames don't include timestamps.
            if pkl.stat().st_mtime > prev[1].stat().st_mtime:
                best[loop] = (ts, pkl)

    # Sort by loop index if present, otherwise by path.
    items = list(best.items())
    items.sort(key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 10**9, str(kv[1][1])))
    return [p for _, (_, p) in items]


def write_csv(rows: List[ExtractedRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "loop",
                "timestamp",
                "IC",
                "ICIR",
                "Rank IC",
                "Rank ICIR",
                "Annualized Return",
                "Information Ratio",
                "Max Drawdown",
                "section",
                "source_pkl",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "loop": "" if r.loop is None else r.loop,
                    "timestamp": "" if r.timestamp is None else r.timestamp.isoformat(),
                    "IC": "" if r.ic is None else r.ic,
                    "ICIR": "" if r.icir is None else r.icir,
                    "Rank IC": "" if r.rank_ic is None else r.rank_ic,
                    "Rank ICIR": "" if r.rank_icir is None else r.rank_icir,
                    "Annualized Return": "" if r.annualized_return is None else r.annualized_return,
                    "Information Ratio": "" if r.information_ratio is None else r.information_ratio,
                    "Max Drawdown": "" if r.max_drawdown is None else r.max_drawdown,
                    "section": r.section,
                    "source_pkl": r.source_pkl,
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="log/2026-01-22_20-23-03-119739", help="rdagent log directory")
    ap.add_argument(
        "--section",
        default="with_cost",
        choices=["with_cost", "without_cost", "benchmark"],
        help="Which performance section to use for Annualized Return / IR / Max Drawdown",
    )
    ap.add_argument("--out", default="", help="Write CSV output to this path (default: <log-dir>/metrics.csv)")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Export all Qlib_execute_log pkls (not just latest per loop)",
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log dir does not exist: {log_dir}")

    pkls = list(iter_qlib_execute_log_pkls(log_dir))
    if not pkls:
        raise SystemExit(f"No Qlib_execute_log pickles found under: {log_dir}")

    if not args.all:
        pkls = pick_latest_per_loop(pkls)

    rows: List[ExtractedRow] = []
    for pkl in pkls:
        stdout = _read_pickle_string(pkl)
        m, used_section = parse_metrics_from_stdout(stdout, section=args.section)
        rows.append(
            ExtractedRow(
                loop=_loop_index_from_path(pkl),
                timestamp=_parse_timestamp_from_stem(pkl.stem),
                source_pkl=str(pkl),
                ic=m["IC"],
                icir=m["ICIR"],
                rank_ic=m["Rank IC"],
                rank_icir=m["Rank ICIR"],
                annualized_return=m["Annualized Return"],
                information_ratio=m["Information Ratio"],
                max_drawdown=m["Max Drawdown"],
                section=used_section,
            )
        )

    out_path = Path(args.out) if args.out else (log_dir / "metrics.csv")
    write_csv(rows, out_path)

    # Minimal console output for quick confirmation.
    print(f"Wrote {len(rows)} row(s) to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

