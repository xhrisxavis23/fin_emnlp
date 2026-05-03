#!/usr/bin/env python3
"""
Extract Qlib metrics from an AlphaAgent/RD-Agent log directory.

Default input example:
  log/2026-01-22_20-23-03-119739

You can also pass the parent log directory (e.g. `log/`); in that case, this script
will auto-pick the latest timestamped run directory under it.

This script supports two log layouts:

1) Legacy layout (pickled stdout strings), typically under:
     **/Qlib_execute_log/**.pkl

2) Current AlphaAgent layout where the backtest metrics are logged into text files
   like:
     **/ef/**/common_logs.log
   and appear under a "Backtesting results:" block.

and parses the following metrics (if present):
  IC, ICIR, Rank IC, Rank ICIR, Cumulative Return, Annualized Return, Information Ratio, Max Drawdown
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_TS_FMT = "%Y-%m-%d_%H-%M-%S-%f"
_LOG_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
_LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\s+\|")
_RISK_N_DAY = 238.0  # qlib.contrib.evaluate.risk_analysis() scaler for daily frequency


def _has_any_metrics_artifacts(log_dir: Path) -> bool:
    if not log_dir.exists() or not log_dir.is_dir():
        return False
    try:
        next(log_dir.rglob("Qlib_execute_log"))
        return True
    except StopIteration:
        pass
    try:
        next(log_dir.rglob("common_logs.log"))
        return True
    except StopIteration:
        pass
    try:
        next(log_dir.rglob("Quantitative Backtesting Chart"))
        return True
    except StopIteration:
        pass
    return False


def _resolve_log_dir(log_dir: Path) -> Path:
    """
    Accept either:
      - a run directory like log/2026-01-22_20-23-03-119739, or
      - the parent log directory like log/ (auto-picks the latest run).
    """
    if not log_dir.exists() or not log_dir.is_dir():
        return log_dir

    # If the directory itself looks like a run directory (timestamped), treat it as such.
    if _parse_timestamp_from_stem(log_dir.name) is not None:
        return log_dir

    candidates: List[Tuple[bool, datetime, float, Path]] = []
    for child in log_dir.iterdir():
        if not child.is_dir():
            continue
        ts = _parse_timestamp_from_stem(child.name)
        if ts is None:
            continue
        try:
            mtime = child.stat().st_mtime
        except Exception:
            mtime = 0.0
        # sort key: timestamped first (False), newer ts/mtime last
        candidates.append((ts is None, ts or datetime.min, mtime, child))

    # Newest first.
    candidates.sort(key=lambda t: (t[0], t[1], t[2], str(t[3])), reverse=True)
    for _, _, _, child in candidates:
        if _has_any_metrics_artifacts(child):
            return child

    # Fallback: if the provided directory isn't timestamped but still contains artifacts directly.
    if _has_any_metrics_artifacts(log_dir):
        return log_dir

    return log_dir


@dataclass(frozen=True)
class ExtractedRow:
    loop: Optional[int]
    timestamp: Optional[datetime]
    source_pkl: str
    ic: Optional[float]
    icir: Optional[float]
    rank_ic: Optional[float]
    rank_icir: Optional[float]
    cumulative_return: Optional[float]
    annualized_return: Optional[float]
    information_ratio: Optional[float]
    max_drawdown: Optional[float]
    section: str


@dataclass(frozen=True)
class _MetricSource:
    loop: Optional[int]
    timestamp: Optional[datetime]
    source_path: Path
    text: str


@dataclass(frozen=True)
class _ChartRiskSource:
    timestamp: Optional[datetime]
    source_path: Path
    by_section: Dict[str, "_RiskMetrics"]


@dataclass(frozen=True)
class _RiskMetrics:
    cumulative_return: Optional[float]
    annualized_return: Optional[float]
    information_ratio: Optional[float]
    max_drawdown: Optional[float]


@dataclass(frozen=True)
class _ChartPklRef:
    timestamp: Optional[datetime]
    source_path: Path


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
    # Supports:
    #   'IC': np.float64(0.123)
    #   "IC": 0.123
    #   IC: 0.123
    #   IC                                                   0.123
    pattern = re.compile(
        r"^\s*(?P<k>['\"]?%s['\"]?)\s*(?::|\s+)\s*(?:np\.float64\()?\s*(?P<v>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)?\s*$"
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


def _extract_risk_kv_metrics(stdout: str, section_key: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    prefix_by_section = {
        "with_cost": "1day.excess_return_with_cost",
        "without_cost": "1day.excess_return_without_cost",
        "benchmark": "1day.benchmark_return",
    }
    prefix = prefix_by_section.get(section_key)
    if not prefix:
        return None, None, None
    ar = _extract_float(stdout, f"{prefix}.annualized_return")
    ir = _extract_float(stdout, f"{prefix}.information_ratio")
    mdd = _extract_float(stdout, f"{prefix}.max_drawdown")

    # Some runs only log mean/std/IR for the selected section (not annualized_return).
    # As a best-effort fallback, infer annualized_return using qlib's daily scaler (238).
    if ar is None:
        std = _extract_float(stdout, f"{prefix}.std")
        if ir is not None and std is not None:
            # information_ratio ~= annualized_return / (std * sqrt(N))
            ar = ir * std * math.sqrt(_RISK_N_DAY)
        else:
            mean = _extract_float(stdout, f"{prefix}.mean")
            if mean is not None:
                ar = mean * _RISK_N_DAY

    return ar, ir, mdd


def iter_quantitative_backtesting_chart_pkls(log_dir: Path) -> Iterable[Path]:
    for p in log_dir.rglob("Quantitative Backtesting Chart"):
        if not p.is_dir():
            continue
        for pkl in p.rglob("*.pkl"):
            yield pkl


def _iter_chart_pkl_refs(log_dir: Path) -> List[_ChartPklRef]:
    refs: List[_ChartPklRef] = []
    for p in iter_quantitative_backtesting_chart_pkls(log_dir):
        refs.append(_ChartPklRef(timestamp=_parse_timestamp_from_stem(p.stem), source_path=p))
    refs.sort(key=lambda r: (r.timestamp is None, r.timestamp or datetime.min, str(r.source_path)))
    return refs


def _pick_best_chart_pkl(ts: Optional[datetime], charts: Sequence[_ChartPklRef]) -> Optional[_ChartPklRef]:
    if not charts:
        return None
    if ts is None:
        return max(charts, key=lambda c: (c.timestamp is not None, c.timestamp or datetime.min, str(c.source_path)))

    timestamped = [c for c in charts if c.timestamp is not None]
    if not timestamped:
        return max(charts, key=lambda c: str(c.source_path))

    return min(timestamped, key=lambda c: abs((c.timestamp - ts).total_seconds()))


def _compute_risk_from_ret_df(ret_df, section_key: str) -> _RiskMetrics:
    """
    Compute (Cumulative Return, Annualized Return, Information Ratio, Max Drawdown) from Qlib's ret.pkl DataFrame.

    This follows qlib.contrib.evaluate.risk_analysis(mode="sum", freq="day") semantics:
      - cumulative_return prefers "account" based return (if available), i.e. account[-1] / account[0] - 1
      - annualized_return = mean(r) * 238
      - information_ratio = mean(r) / std(r, ddof=1) * sqrt(238)
      - max_drawdown = (cumsum(r) - cummax(cumsum(r))).min()
    """
    cols = getattr(ret_df, "columns", None)
    if cols is None:
        return _RiskMetrics(None, None, None, None)

    cumulative_return = None
    if section_key == "with_cost" and "account" in cols:
        try:
            start = float(ret_df["account"].iloc[0])
            end = float(ret_df["account"].iloc[-1])
            if start != 0.0:
                cumulative_return = end / start - 1.0
        except Exception:
            cumulative_return = None
    elif section_key == "benchmark" and "bench_account" in cols:
        try:
            start = float(ret_df["bench_account"].iloc[0])
            end = float(ret_df["bench_account"].iloc[-1])
            if start != 0.0:
                cumulative_return = end / start - 1.0
        except Exception:
            cumulative_return = None

    # Fallback: compute strategy/benchmark cumulative return from daily returns using "sum" mode.
    # This matches qlib's risk_analysis(mode="sum") interpretation.
    if cumulative_return is None:
        try:
            if section_key == "with_cost" and "return" in cols:
                cost = ret_df["cost"] if "cost" in cols else 0.0
                daily = ret_df["return"] - cost
                cumulative_return = float(daily.cumsum().iloc[-1])
            elif section_key == "without_cost" and "return" in cols:
                cumulative_return = float(ret_df["return"].cumsum().iloc[-1])
            elif section_key == "benchmark" and "bench" in cols:
                cumulative_return = float(ret_df["bench"].cumsum().iloc[-1])
        except Exception:
            cumulative_return = None

    if section_key == "benchmark":
        if "bench" not in cols:
            return _RiskMetrics(None, None, None, None)
        r = ret_df["bench"]
    else:
        if "return" not in cols or "bench" not in cols:
            return _RiskMetrics(None, None, None, None)
        cost = ret_df["cost"] if "cost" in cols else 0.0
        if section_key == "with_cost":
            r = ret_df["return"] - ret_df["bench"] - cost
        elif section_key == "without_cost":
            r = ret_df["return"] - ret_df["bench"]
        else:
            return _RiskMetrics(None, None, None, None)

    try:
        mean = float(r.mean())
    except Exception:
        return _RiskMetrics(None, None, None, None)

    try:
        std = float(r.std(ddof=1))
    except Exception:
        std = float("nan")

    annualized_return = mean * _RISK_N_DAY
    information_ratio = None
    if std and not math.isnan(std):
        try:
            information_ratio = mean / std * math.sqrt(_RISK_N_DAY)
        except Exception:
            information_ratio = None

    try:
        c = r.cumsum()
        max_drawdown = float((c - c.cummax()).min())
    except Exception:
        max_drawdown = None

    return _RiskMetrics(cumulative_return, annualized_return, information_ratio, max_drawdown)


def _load_chart_risk_source(pkl_path: Path) -> Optional[_ChartRiskSource]:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        ret_df = pd.read_pickle(pkl_path)
    except Exception:
        return None

    by_section: Dict[str, _RiskMetrics] = {}
    for section_key in ("with_cost", "without_cost", "benchmark"):
        by_section[section_key] = _compute_risk_from_ret_df(ret_df, section_key)

    return _ChartRiskSource(
        timestamp=_parse_timestamp_from_stem(pkl_path.stem),
        source_path=pkl_path,
        by_section=by_section,
    )


def _pick_best_chart_source(ts: Optional[datetime], charts: List[_ChartRiskSource]) -> Optional[_ChartRiskSource]:
    if not charts:
        return None
    if ts is None:
        # Prefer the most recent timestamped chart; otherwise fall back to path order.
        return max(charts, key=lambda c: (c.timestamp is not None, c.timestamp or datetime.min, str(c.source_path)))

    timestamped = [c for c in charts if c.timestamp is not None]
    if not timestamped:
        return max(charts, key=lambda c: str(c.source_path))

    return min(timestamped, key=lambda c: abs((c.timestamp - ts).total_seconds()))


def _compute_cumulative_excess_return_timeseries(ret_df):
    """
    Compute time series columns for plotting cumulative excess return curves.

    "Default" cumulative excess return:
      - Prefer account-based: (account/account0 - 1) - (bench_account/bench_account0 - 1)
      - Otherwise fall back to sum-mode: (return - bench - cost).cumsum()
    """
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required to export time-series curves") from e

    cols = getattr(ret_df, "columns", None)
    if cols is None:
        raise ValueError("ret_df has no columns")

    out = pd.DataFrame(index=ret_df.index.copy())
    if "return" in cols:
        out["return"] = ret_df["return"]
    if "bench" in cols:
        out["bench"] = ret_df["bench"]
    out["cost"] = ret_df["cost"] if "cost" in cols else 0.0

    if "return" in out.columns and "bench" in out.columns:
        out["excess_return_without_cost"] = out["return"] - out["bench"]
        out["excess_return_with_cost"] = out["return"] - out["bench"] - out["cost"]
        out["cum_excess_return_without_cost_sum"] = out["excess_return_without_cost"].cumsum()
        out["cum_excess_return_with_cost_sum"] = out["excess_return_with_cost"].cumsum()
        out["cum_benchmark_return_sum"] = out["bench"].cumsum()
    else:
        out["excess_return_without_cost"] = None
        out["excess_return_with_cost"] = None
        out["cum_excess_return_without_cost_sum"] = None
        out["cum_excess_return_with_cost_sum"] = None
        out["cum_benchmark_return_sum"] = None

    out["cum_excess_return_with_cost"] = out["cum_excess_return_with_cost_sum"]

    if "account" in cols and "bench_account" in cols:
        try:
            a0 = float(ret_df["account"].iloc[0])
            b0 = float(ret_df["bench_account"].iloc[0])
            if a0 != 0.0 and b0 != 0.0:
                strat_curve = ret_df["account"] / a0 - 1.0
                bench_curve = ret_df["bench_account"] / b0 - 1.0
                out["cum_strategy_return_account"] = strat_curve
                out["cum_benchmark_return_account"] = bench_curve
                out["cum_excess_return_with_cost_account"] = strat_curve - bench_curve
                out["cum_excess_return_with_cost"] = out["cum_excess_return_with_cost_account"]
        except Exception:
            pass

    # Always surface the index as a dedicated column for plotting (works even if index is unnamed).
    out = out.copy()
    out.insert(0, "date", out.index)
    out = out.reset_index(drop=True)
    return out


def _write_series_csv(*, log_dir: Path, out_path: Path) -> int:
    """
    Export a time-series CSV to support plotting cumulative excess return curves.

    - If "Backtesting results:" blocks exist, exports one series per block (nearest chart pkl by timestamp).
    - Otherwise, exports one series per chart pkl.
    """
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:
        raise SystemExit(
            "pandas is required to export time-series curves.\n"
            "Install dependencies (e.g. `pip install -r requirements.txt`) and retry."
        ) from e

    sources = list(iter_backtesting_result_texts(log_dir))
    chart_refs = _iter_chart_pkl_refs(log_dir)
    if not chart_refs:
        raise SystemExit(
            f"No chart pickles found under: {log_dir}\n"
            "Expected **/Quantitative Backtesting Chart/**.pkl"
        )

    selected: List[Tuple[Optional[int], Optional[datetime], Path]] = []
    if sources:
        sources.sort(
            key=lambda s: (
                s.timestamp is None,
                s.timestamp or datetime.min,
                str(s.source_path),
                s.loop is None,
                s.loop or 10**9,
            )
        )
        for src in sources:
            best = _pick_best_chart_pkl(src.timestamp, chart_refs)
            if best is None:
                continue
            selected.append((src.loop, src.timestamp, best.source_path))
    else:
        for ref in chart_refs:
            selected.append((None, ref.timestamp, ref.source_path))

    frames: List["pd.DataFrame"] = []
    for loop, ts, pkl_path in selected:
        ret_df = pd.read_pickle(pkl_path)
        series_df = _compute_cumulative_excess_return_timeseries(ret_df)
        series_df.insert(0, "loop", "" if loop is None else loop)
        series_df.insert(1, "timestamp", "" if ts is None else ts.isoformat())
        series_df.insert(2, "source_pkl", str(pkl_path))
        frames.append(series_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(out_path, index=False)
    return len(frames)


def _choose_best_risk_metrics(stdout: str, preferred_section: str) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Pick a single section for (Annualized Return / IR / Max Drawdown), preferring the requested one.

    If the preferred section is missing some metrics (common in AlphaAgent's key/value dumps),
    we fall back to another section that has more complete coverage, and return the section used.
    """
    candidates = [preferred_section] + [s for s in ("with_cost", "without_cost", "benchmark") if s != preferred_section]

    scored: List[Tuple[int, str, Tuple[Optional[float], Optional[float], Optional[float]]]] = []
    for section_key in candidates:
        # Try table-style section first.
        section_text = _slice_section(stdout, section_key)
        if section_text is not None:
            ar, ir, mdd = _extract_risk_table_metrics(section_text)
        else:
            ar, ir, mdd = _extract_risk_kv_metrics(stdout, section_key)
        score = sum(v is not None for v in (ar, ir, mdd))
        scored.append((score, section_key, (ar, ir, mdd)))
        if score == 3:
            return ar, ir, mdd, section_key

    # Prefer the highest score; tie-break by candidate order (preferred first).
    scored.sort(key=lambda t: (-t[0], candidates.index(t[1])))
    best_score, best_section, (ar, ir, mdd) = scored[0]
    _ = best_score  # score is used only for selection.
    return ar, ir, mdd, best_section


def parse_metrics_from_stdout(stdout: str, *, section: str = "with_cost") -> Tuple[Dict[str, Optional[float]], str]:
    metrics: Dict[str, Optional[float]] = {
        "IC": _extract_float(stdout, "IC"),
        "ICIR": _extract_float(stdout, "ICIR"),
        "Rank IC": _extract_float(stdout, "Rank IC"),
        "Rank ICIR": _extract_float(stdout, "Rank ICIR"),
        "Cumulative Return": None,
    }

    # Risk/performance metrics can appear either as:
    # - Qlib table sections (legacy stdout), or
    # - AlphaAgent "Backtesting results:" key/value dumps (common_logs.log).
    #
    # Some runs only contain (annualized_return, information_ratio, max_drawdown) for without_cost,
    # so we pick the most complete section available while keeping metrics consistent (from one section).
    ar, ir, mdd, section_used = _choose_best_risk_metrics(stdout, section)

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


def _parse_backtesting_blocks_from_common_log(path: Path) -> List[_MetricSource]:
    """
    Extract "Backtesting results:" blocks from a common_logs.log file.

    The block format typically looks like:
      2026-01-23 10:29:49.774 | INFO ... - Backtesting results:
      Rank IC                                              0.011314
      ...
      Name: 0, dtype: float64
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    lines = text.splitlines()
    sources: List[_MetricSource] = []

    local_idx = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Backtesting results:" not in line:
            i += 1
            continue

        ts = None
        try:
            ts_str = line.split(" |", 1)[0].strip()
            ts = datetime.strptime(ts_str, _LOG_TS_FMT)
        except Exception:
            ts = None

        block_lines: List[str] = []
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            # Next log record begins with a timestamp prefix.
            if _LOG_PREFIX_RE.match(nxt):
                break
            block_lines.append(nxt)
            # Common terminator in the dumped Series.
            if nxt.strip().startswith("Name:"):
                j += 1
                break
            j += 1

        block = "\n".join(block_lines).strip()
        if block:
            sources.append(
                _MetricSource(
                    loop=local_idx,
                    timestamp=ts,
                    source_path=path,
                    text=block,
                )
            )
            local_idx += 1

        i = j

    return sources


def iter_backtesting_result_texts(log_dir: Path) -> Iterable[_MetricSource]:
    """
    Find backtesting metric blocks in the current AlphaAgent log layout.

    We prefer ef/**/common_logs.log, but fall back to any common_logs.log under log_dir.
    """
    candidates = list(log_dir.rglob("common_logs.log"))

    def score(p: Path) -> int:
        # Prefer ef/ scope since that's where factor_backtest logs are written.
        return 0 if "ef" in p.parts else 1

    candidates.sort(key=lambda p: (score(p), str(p)))

    for path in candidates:
        for src in _parse_backtesting_blocks_from_common_log(path):
            yield src


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
                "Cumulative Return",
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
                    "Cumulative Return": "" if r.cumulative_return is None else r.cumulative_return,
                    "Annualized Return": "" if r.annualized_return is None else r.annualized_return,
                    "Information Ratio": "" if r.information_ratio is None else r.information_ratio,
                    "Max Drawdown": "" if r.max_drawdown is None else r.max_drawdown,
                    "section": r.section,
                    "source_pkl": r.source_pkl,
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-dir",
        default="log",
        help="AlphaAgent run dir (e.g. log/<timestamp>) or parent log dir (e.g. log/) to pick the latest run",
    )
    ap.add_argument(
        "--section",
        default="with_cost",
        choices=["with_cost", "without_cost", "benchmark"],
        help="Which performance section to use for Annualized Return / IR / Max Drawdown",
    )
    ap.add_argument("--out", default="", help="Write CSV output to this path (default: <log-dir>/metrics.csv)")
    ap.add_argument("--series", action="store_true", help="Also write <log-dir>/series.csv for curve plotting")
    ap.add_argument("--series-out", default="", help="Write time-series CSV (cumulative excess return curve) to this path")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Export all Qlib_execute_log pkls (not just latest per loop)",
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log dir does not exist: {log_dir}")

    resolved = _resolve_log_dir(log_dir)
    if resolved != log_dir:
        print(f"Resolved --log-dir to: {resolved}")
    log_dir = resolved

    chart_pkls = list(iter_quantitative_backtesting_chart_pkls(log_dir))
    chart_sources: List[_ChartRiskSource] = []
    for pkl_path in chart_pkls:
        cs = _load_chart_risk_source(pkl_path)
        if cs is not None:
            chart_sources.append(cs)

    pkls = list(iter_qlib_execute_log_pkls(log_dir))
    rows: List[ExtractedRow] = []
    if pkls:
        if not args.all:
            pkls = pick_latest_per_loop(pkls)
        for pkl in pkls:
            stdout = _read_pickle_string(pkl)
            m, used_section = parse_metrics_from_stdout(stdout, section=args.section)
            cum = m["Cumulative Return"]
            ar = m["Annualized Return"]
            ir = m["Information Ratio"]
            mdd = m["Max Drawdown"]

            if chart_sources and (cum is None or ar is None or ir is None or mdd is None):
                best = _pick_best_chart_source(_parse_timestamp_from_stem(pkl.stem), chart_sources)
                if best is not None:
                    rm = best.by_section.get(used_section)
                    if rm is not None:
                        cum = cum if cum is not None else rm.cumulative_return
                        ar = ar if ar is not None else rm.annualized_return
                        ir = ir if ir is not None else rm.information_ratio
                        mdd = mdd if mdd is not None else rm.max_drawdown

            rows.append(
                ExtractedRow(
                    loop=_loop_index_from_path(pkl),
                    timestamp=_parse_timestamp_from_stem(pkl.stem),
                    source_pkl=str(pkl),
                    ic=m["IC"],
                    icir=m["ICIR"],
                    rank_ic=m["Rank IC"],
                    rank_icir=m["Rank ICIR"],
                    cumulative_return=cum,
                    annualized_return=ar,
                    information_ratio=ir,
                    max_drawdown=mdd,
                    section=used_section,
                )
            )
    else:
        sources = list(iter_backtesting_result_texts(log_dir))

        if not sources and not chart_sources:
            raise SystemExit(
                "No backtest metrics found under: "
                f"{log_dir}\n"
                "Tried legacy pickles (**/Qlib_execute_log/**.pkl), current logs (**/ef/**/common_logs.log), "
                "and chart pickles (**/Quantitative Backtesting Chart/**.pkl)."
            )

        # Keep stable ordering: by timestamp, then by path, then by loop index within the file.
        sources.sort(
            key=lambda s: (
                s.timestamp is None,
                s.timestamp or datetime.min,
                str(s.source_path),
                s.loop is None,
                s.loop or 10**9,
            )
        )

        if sources:
            for src in sources:
                m, used_section = parse_metrics_from_stdout(src.text, section=args.section)
                cum = m["Cumulative Return"]
                ar = m["Annualized Return"]
                ir = m["Information Ratio"]
                mdd = m["Max Drawdown"]
                src_path = src.source_path

                if (cum is None or ar is None or ir is None or mdd is None) and chart_sources:
                    best = _pick_best_chart_source(src.timestamp, chart_sources)
                    if best is not None:
                        rm = best.by_section.get(used_section)
                        if rm is not None:
                            cum = cum if cum is not None else rm.cumulative_return
                            ar = ar if ar is not None else rm.annualized_return
                            ir = ir if ir is not None else rm.information_ratio
                            mdd = mdd if mdd is not None else rm.max_drawdown
                        src_path = best.source_path

                rows.append(
                    ExtractedRow(
                        loop=src.loop,
                        timestamp=src.timestamp,
                        source_pkl=str(src_path),
                        ic=m["IC"],
                        icir=m["ICIR"],
                        rank_ic=m["Rank IC"],
                        rank_icir=m["Rank ICIR"],
                        cumulative_return=cum,
                        annualized_return=ar,
                        information_ratio=ir,
                        max_drawdown=mdd,
                        section=used_section,
                    )
                )
        else:
            # Chart-only export: no IC/ICIR, but we can still export risk metrics.
            chart_sources.sort(
                key=lambda c: (
                    c.timestamp is None,
                    c.timestamp or datetime.min,
                    str(c.source_path),
                )
            )
            for i, cs in enumerate(chart_sources):
                rm = cs.by_section.get(args.section, _RiskMetrics(None, None, None, None))
                rows.append(
                    ExtractedRow(
                        loop=i,
                        timestamp=cs.timestamp,
                        source_pkl=str(cs.source_path),
                        ic=None,
                        icir=None,
                        rank_ic=None,
                        rank_icir=None,
                        cumulative_return=rm.cumulative_return,
                        annualized_return=rm.annualized_return,
                        information_ratio=rm.information_ratio,
                        max_drawdown=rm.max_drawdown,
                        section=args.section,
                    )
                )

    out_path = Path(args.out) if args.out else (log_dir / "metrics.csv")
    write_csv(rows, out_path)

    if args.series or args.series_out:
        series_out_path = Path(args.series_out) if args.series_out else (log_dir / "series.csv")
        n = _write_series_csv(log_dir=log_dir, out_path=series_out_path)
        print(f"Wrote {n} series block(s) to: {series_out_path}")

    # Minimal console output for quick confirmation.
    print(f"Wrote {len(rows)} row(s) to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
