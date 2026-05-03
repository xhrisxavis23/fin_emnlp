#!/usr/bin/env python3
"""
Stage 3: Strictness Monotonicity Validation (single factor / single score)

This stage is NOT a strategy evaluation. PASS/FAIL must not depend on alpha metrics.

Goal
  For a single factor score, validate the hypothesis *structure* via strictness monotonicity:
    strictness ↑ (i.e., selecting more extreme tail by top-p%)  =>  false-positive rate ↓

Key constraints (from todo.md)
  - Use split separation to avoid "plausible overfitting":
      train : define/fix (raw-consistency rule + strictness ladder)
      valid : decide PASS/FAIL monotonicity
      test  : sealed (optional reporting only; never used for PASS/FAIL)
  - False-positive definition must not be a tautological re-statement of the factor itself.
    => We define FP using a *pre-fixed composite* raw-consistency score derived from raw OHLCV features.

Inputs
  - factor workspace folder (git_ignore_folder/RD-Agent_workspace/<uuid>) containing:
      - result.h5 (factor score; Series over (datetime,instrument))
      - daily_pv.h5 (raw OHLCV; DataFrame over (datetime,instrument))
      - optional stage2/stage2_summary.json (expectations: mag/dir/vol/pos direction)
  - a Qlib-style conf yaml (to parse train/valid/test segments) without requiring PyYAML.

Outputs (default under /home/dgu/fin/AlphaAgent/results/<run>/stage3/<factor>/)
  - stage3_table.csv
  - stage3_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

_NO_IMPORTS_MODE = any(a in ("-h", "--help", "--dry-run") for a in sys.argv)
try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as e:
    if not _NO_IMPORTS_MODE:
        missing = getattr(e, "name", "a required dependency")
        raise SystemExit(
            f"Missing dependency '{missing}'.\n"
            "Stage3 requires pandas+numpy. Run this script inside the same environment you use for AlphaAgent "
            "(e.g., your conda env `alphaagent`)."
        ) from e


Split = Literal["train", "valid", "test"]
Direction = Literal["up", "down", "any"]

RAW_FEATURES = ("mag", "dir", "vol", "pos")


@dataclass(frozen=True)
class Segment:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class StrictnessRow:
    split: str
    p: float
    event_count: int
    total_count: int
    event_rate: float
    fp_count: int
    fp_rate: Optional[float]
    consistency_mean: Optional[float]


@dataclass(frozen=True)
class MonotonicityResult:
    expected: str
    step_consistency: Optional[float]
    spearman_rho: Optional[float]
    passed: bool


@dataclass(frozen=True)
class Stage3Summary:
    factor_ws: str
    split_for_decision: str
    segments: Dict[str, Tuple[str, str]]
    strictness_ladder_p: List[float]
    quantile_mode: str
    expectations: Dict[str, str]
    consistency_threshold: float
    min_events: int
    eps: float
    valid_monotonicity: MonotonicityResult
    passed: bool
    notes: List[str]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_TS_FMT = "%Y-%m-%d_%H-%M-%S-%f"

_DEFAULT_RESULTS_ROOT = Path("/home/dgu/fin/AlphaAgent/results")


def _results_root() -> Path:
    root = _DEFAULT_RESULTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "").strip("_")
    return s or "unnamed"


def _parse_timestamp_from_stem(stem: str) -> Optional[datetime]:
    try:
        return datetime.strptime(stem, _TS_FMT)
    except Exception:
        return None


def _has_any_run_logs(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    try:
        next(run_dir.rglob("common_logs.log"))
        return True
    except StopIteration:
        return False


def _resolve_log_dir(log_dir: Path) -> Path:
    """
    Accept either:
      - a run directory like log/2026-01-28_09-36-06-094990, or
      - the parent log directory like log/ (auto-picks the latest run).
    """
    if not log_dir.exists() or not log_dir.is_dir():
        return log_dir
    if _parse_timestamp_from_stem(log_dir.name) is not None:
        return log_dir

    candidates: List[Tuple[datetime, float, Path]] = []
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
        candidates.append((ts, mtime, child))

    candidates.sort(key=lambda t: (t[0], t[1], str(t[2])), reverse=True)
    for _, _, child in candidates:
        if _has_any_run_logs(child):
            return child
    return log_dir


def _extract_factor_workspaces_from_run(run_dir: Path) -> Dict[str, Path]:
    """
    Parse AlphaAgent debug logs under:
      log/<run>/d/**/common_logs.log
    for lines like:
      evolving code workspace: File Factor[FactorName]: /abs/path/to/RD-Agent_workspace/<uuid>

    Returns mapping {factor_name -> workspace_path} (first occurrence wins).
    """
    mapping: Dict[str, Path] = {}
    pattern = re.compile(r"File Factor\[(?P<name>[^]]+)\]\s*:\s*(?P<path>/\S+)")

    all_logs = list(run_dir.rglob("common_logs.log"))
    candidates = [p for p in all_logs if "d" in p.parts]
    if not candidates:
        candidates = all_logs
    candidates.sort(key=lambda p: str(p))

    for log_path in candidates:
        try:
            for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = pattern.search(line)
                if not m:
                    continue
                name = m.group("name").strip()
                ws = Path(m.group("path")).expanduser()
                mapping.setdefault(name, ws)
        except Exception:
            continue
    return mapping


def _parse_segments_from_conf(conf_path: Path) -> Dict[str, Segment]:
    text = _read_text(conf_path)

    def find_seg(key: str) -> Optional[Segment]:
        m = re.search(
            r"^\s*%s\s*:\s*\[\s*(\d{4}-\d{2}-\d{2})\s*,\s*(\d{4}-\d{2}-\d{2})\s*\]\s*$"
            % re.escape(key),
            text,
            flags=re.MULTILINE,
        )
        if not m:
            return None
        return Segment(start=pd.Timestamp(m.group(1)), end=pd.Timestamp(m.group(2)))

    segs: Dict[str, Segment] = {}
    for k in ("train", "valid", "test"):
        s = find_seg(k)
        if s is not None:
            segs[k] = s
    return segs


def _safe_read_hdf(path: Path, *, key: str):
    return pd.read_hdf(path, key=key)


def _ensure_multiindex(df_or_s: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    idx = df_or_s.index
    if isinstance(idx, pd.MultiIndex) and {"datetime", "instrument"} <= set(idx.names):
        return df_or_s
    if isinstance(idx, pd.MultiIndex) and len(idx.levels) >= 2:
        names = list(idx.names)
        if names[0] is None:
            names[0] = "datetime"
        if len(names) > 1 and names[1] is None:
            names[1] = "instrument"
        df_or_s.index = idx.set_names(names)
        return df_or_s
    raise ValueError("Expected a MultiIndex with ('datetime','instrument') levels in HDF data.")


def _filter_by_date(df_or_s: pd.DataFrame | pd.Series, seg: Segment) -> pd.DataFrame | pd.Series:
    if not isinstance(df_or_s.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex to filter by date.")
    dt = df_or_s.index.get_level_values("datetime")
    mask = (dt >= seg.start) & (dt <= seg.end)
    return df_or_s[mask]


def _resolve_ohlcv_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    candidates = {
        "open": ["$open", "open", "OPEN"],
        "high": ["$high", "high", "HIGH"],
        "low": ["$low", "low", "LOW"],
        "close": ["$close", "close", "CLOSE"],
        "volume": ["$volume", "volume", "VOL", "VOLUME"],
    }
    out: Dict[str, str] = {}
    for k, opts in candidates.items():
        for opt in opts:
            if opt in cols:
                out[k] = opt
                break
    missing = [k for k in candidates if k not in out]
    if missing:
        raise KeyError(f"Missing OHLCV columns in daily_pv.h5: {missing}. Found columns: {cols[:20]} ...")
    return out


def _compute_raw_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    ohlcv = _ensure_multiindex(ohlcv)  # type: ignore[assignment]
    colmap = _resolve_ohlcv_columns(ohlcv)
    o = ohlcv[colmap["open"]].astype(float)
    h = ohlcv[colmap["high"]].astype(float)
    l = ohlcv[colmap["low"]].astype(float)
    c = ohlcv[colmap["close"]].astype(float)
    v = ohlcv[colmap["volume"]].astype(float)

    eps = 1e-12
    mag = (h - l).astype(float)
    dir_ = (c - o).astype(float)
    pos = ((c - l) / (mag + eps)).astype(float)
    vol = v.astype(float)

    return pd.DataFrame(
        {
            "mag": mag,
            "dir": dir_,
            "vol": vol,
            "pos": pos,
        },
        index=ohlcv.index,
    )


def _parse_expectations_from_stage2(stage2_summary_path: Path) -> Dict[str, Direction]:
    obj = json.loads(_read_text(stage2_summary_path))
    exp = obj.get("expectations", {}) if isinstance(obj, dict) else {}
    out: Dict[str, Direction] = {f: "any" for f in RAW_FEATURES}
    for k in RAW_FEATURES:
        v = exp.get(k)
        if isinstance(v, str) and v.lower() in ("up", "down", "any"):
            out[k] = v.lower()  # type: ignore[assignment]
    return out


def _parse_expectations_from_cli(items: List[str]) -> Dict[str, Direction]:
    out: Dict[str, Direction] = {f: "any" for f in RAW_FEATURES}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"Invalid --expect '{item}'. Use form feature:up|down|any")
        k, v = item.split(":", 1)
        k = k.strip()
        v = v.strip().lower()
        if k not in RAW_FEATURES:
            raise SystemExit(f"Invalid feature in --expect: {k}. Allowed: {', '.join(RAW_FEATURES)}")
        if v not in ("up", "down", "any"):
            raise SystemExit(f"Invalid direction in --expect: {v}. Use up|down|any")
        out[k] = v  # type: ignore[assignment]
    return out


def _consistency_score(raw: pd.DataFrame, expectations: Dict[str, Direction]) -> pd.Series:
    """
    Pre-fixed composite rule (tautology-resistant):
      - per date cross-sectional percentile rank for each raw feature
      - map rank to [-0.5, +0.5] by (pct_rank - 0.5)
      - apply expectation sign (up:+, down:-, any:ignore)
      - sum across features

    Higher score = more consistent with expectation.
    """
    signs: Dict[str, float] = {}
    for k in RAW_FEATURES:
        v = expectations.get(k, "any")
        if v == "up":
            signs[k] = 1.0
        elif v == "down":
            signs[k] = -1.0
        else:
            signs[k] = 0.0

    use_feats = [k for k, s in signs.items() if s != 0.0]
    if not use_feats:
        # If user didn't specify expectations, fall back to a neutral 0 score.
        return pd.Series(index=raw.index, data=0.0, dtype="float64")

    def per_date(df: pd.DataFrame) -> pd.Series:
        parts = []
        for k in use_feats:
            s = df[k].astype(float)
            pr = s.rank(pct=True, method="average")
            parts.append((pr - 0.5) * signs[k])
        out = sum(parts)
        return out.astype("float64")

    return raw.groupby(level="datetime", group_keys=False).apply(per_date)


def _event_mask_top_p(score: pd.Series, p: float) -> pd.Series:
    """
    Event occurs if the score is within top-p fraction per date (cross-sectional).
    """
    if p <= 0 or p > 1:
        raise ValueError("p must be in (0,1].")

    def per_date(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if x.empty:
            return pd.Series(index=x.index, dtype="bool")
        n = len(x)
        k = max(1, int(math.ceil(p * n)))
        # highest score = smallest rank when sorting descending
        r = x.rank(method="first", ascending=False)
        return (r <= k).astype(bool)

    m = score.groupby(level="datetime", group_keys=False).apply(per_date)
    # Reindex back to full index; non-computed rows -> False
    return m.reindex(score.index, fill_value=False).astype(bool)


def _spearman_rho(x: Iterable[float], y: Iterable[float]) -> Optional[float]:
    try:
        x = np.asarray(list(x), dtype=float)
        y = np.asarray(list(y), dtype=float)
        if len(x) != len(y) or len(x) < 2:
            return None
        if np.all(np.isnan(y)):
            return None
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        if np.std(xr) == 0 or np.std(yr) == 0:
            return None
        return float(np.corrcoef(xr, yr)[0, 1])
    except Exception:
        return None


def _step_consistency(values: List[Optional[float]], *, eps: float = 0.0) -> Optional[float]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2:
        return None
    ok = 0
    total = 0
    for a, b in zip(v, v[1:]):
        total += 1
        ok += 1 if b <= a + eps else 0
    return ok / total if total else None


def _evaluate_split(
    *,
    split: str,
    score: pd.Series,
    raw: pd.DataFrame,
    expectations: Dict[str, Direction],
    ladder_p: List[float],
    consistency_threshold: float,
    min_events: int,
) -> Tuple[List[StrictnessRow], List[str]]:
    notes: List[str] = []
    cscore = _consistency_score(raw, expectations)
    # Align
    cscore = cscore.reindex(score.index)
    raw_ok = cscore > consistency_threshold

    total = int(score.notna().sum())
    rows: List[StrictnessRow] = []
    for p in ladder_p:
        event = _event_mask_top_p(score, p)
        event = event & score.notna()
        event_count = int(event.sum())
        fp = event & (~raw_ok)
        fp_count = int(fp.sum())
        fp_rate = (fp_count / event_count) if event_count > 0 else None
        if event_count < min_events:
            notes.append(f"{split}: p={p:.4f} has too few events: {event_count} < min_events={min_events}.")
        cm = None
        try:
            if event_count > 0:
                cm = float(cscore[event].mean())
        except Exception:
            cm = None
        rows.append(
            StrictnessRow(
                split=split,
                p=float(p),
                event_count=event_count,
                total_count=total,
                event_rate=(event_count / total) if total else 0.0,
                fp_count=fp_count,
                fp_rate=fp_rate,
                consistency_mean=cm,
            )
        )

    return rows, notes


def _monotonicity_on_valid(rows: List[StrictnessRow], *, eps: float, min_events: int) -> MonotonicityResult:
    valid_rows = [r for r in rows if r.split == "valid"]
    # Keep only ladder points with sufficient event count and defined fp_rate
    filt = [r for r in valid_rows if (r.fp_rate is not None and r.event_count >= min_events)]
    if len(filt) < 2:
        return MonotonicityResult(
            expected="fp_rate should be non-increasing as p decreases (strictness increases)",
            step_consistency=None,
            spearman_rho=None,
            passed=False,
        )

    # We expect fp_rate to decrease when strictness increases.
    # strictness increases as p decreases, so define strictness index as the order of decreasing p.
    ordered = sorted(filt, key=lambda r: r.p, reverse=True)  # loose -> strict
    fp_rates = [r.fp_rate for r in ordered]  # type: ignore[list-item]
    sc = _step_consistency(fp_rates, eps=eps)
    rho = _spearman_rho(range(len(fp_rates)), fp_rates)
    passed = (sc is not None and sc >= 0.8) and (rho is None or rho <= 0.0)
    return MonotonicityResult(
        expected="fp_rate should be non-increasing as p decreases (strictness increases)",
        step_consistency=sc,
        spearman_rho=rho,
        passed=passed,
    )


def _stage2_passed(stage2_summary_path: Path) -> Optional[bool]:
    try:
        obj = json.loads(_read_text(stage2_summary_path))
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("passed"), bool):
        return bool(obj["passed"])
    return None


def _resolve_stage2_summary_path(
    *, factor_ws: Path, display_name: str, run_dir: Optional[Path], args: argparse.Namespace
) -> Path:
    if args.stage2_summary:
        return Path(args.stage2_summary)

    safe = _safe_name(display_name or factor_ws.name)
    candidates: List[Path] = [
        factor_ws / "stage2" / "stage2_summary.json",
    ]
    if run_dir is not None:
        candidates.append(_results_root() / run_dir.name / "stage2" / safe / "stage2_summary.json")
    candidates.append(_results_root() / "manual" / "stage2" / safe / "stage2_summary.json")

    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _run_stage3_for_workspace(
    *, factor_ws: Path, display_name: str, args: argparse.Namespace, run_dir: Optional[Path]
) -> Tuple[dict, bool]:
    """
    Run Stage3 for a single factor workspace.
    Returns (index_row, passed_bool). Raises on unrecoverable errors.
    """
    factor_ws = factor_ws.expanduser()
    if not factor_ws.exists():
        raise FileNotFoundError(f"factor workspace does not exist: {factor_ws}")

    if args.out_dir:
        base = Path(args.out_dir)
    else:
        run_tag = run_dir.name if run_dir is not None else "manual"
        base = _results_root() / run_tag / "stage3"
    out_dir = base / _safe_name(display_name or factor_ws.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_path = Path(args.conf)
    if not conf_path.exists():
        raise FileNotFoundError(f"Config yaml does not exist: {conf_path}")
    segments = _parse_segments_from_conf(conf_path)
    if "train" not in segments or "valid" not in segments:
        raise SystemExit(f"Failed to parse train/valid segments from conf: {conf_path}")

    stage2_path = _resolve_stage2_summary_path(factor_ws=factor_ws, display_name=display_name, run_dir=run_dir, args=args)
    if stage2_path.exists() and not args.include_failed_stage2:
        s2 = _stage2_passed(stage2_path)
        if s2 is False:
            return (
                {
                    "factor_name": display_name or factor_ws.name,
                    "workspace": str(factor_ws),
                    "skipped": True,
                    "reason": f"Stage2 failed ({stage2_path})",
                    "passed": False,
                    "out_dir": str(out_dir),
                },
                False,
            )

    expectations: Dict[str, Direction]
    notes: List[str] = []
    if args.expect:
        expectations = _parse_expectations_from_cli(args.expect)
        notes.append("Expectations loaded from CLI (--expect).")
    elif stage2_path.exists():
        expectations = _parse_expectations_from_stage2(stage2_path)
        notes.append(f"Expectations loaded from stage2 summary: {stage2_path}")
    else:
        expectations = {f: "any" for f in RAW_FEATURES}
        notes.append("No expectations provided; using 'any' for all raw features (consistency score becomes 0).")

    score_path = factor_ws / "result.h5"
    ohlcv_path = factor_ws / "daily_pv.h5"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing score file result.h5 in factor workspace: {score_path}")
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"Missing daily_pv.h5 in factor workspace: {ohlcv_path}")

    score = _safe_read_hdf(score_path, key="data")
    if isinstance(score, pd.DataFrame):
        if score.shape[1] == 1:
            score = score.iloc[:, 0]
        else:
            raise SystemExit(f"Unexpected DataFrame in result.h5 (expected Series or 1-col DF): {score_path}")
    score = _ensure_multiindex(score)  # type: ignore[assignment]

    ohlcv = _safe_read_hdf(ohlcv_path, key="data")
    ohlcv = _ensure_multiindex(ohlcv)  # type: ignore[assignment]

    common_idx = score.index.intersection(ohlcv.index)
    score = score.reindex(common_idx)
    ohlcv = ohlcv.reindex(common_idx)
    raw = _compute_raw_features(ohlcv)

    ladder_p: List[float] = []
    for part in str(args.strictness_p).split(","):
        part = part.strip()
        if not part:
            continue
        p = float(part)
        if p <= 0 or p > 1:
            raise SystemExit(f"Invalid strictness p: {p}. Must be in (0,1].")
        ladder_p.append(p)
    if not ladder_p:
        raise SystemExit("Empty strictness ladder.")

    all_rows: List[StrictnessRow] = []

    train_rows, n1 = _evaluate_split(
        split="train",
        score=_filter_by_date(score, segments["train"]),  # type: ignore[arg-type]
        raw=_filter_by_date(raw, segments["train"]),  # type: ignore[arg-type]
        expectations=expectations,
        ladder_p=ladder_p,
        consistency_threshold=float(args.consistency_threshold),
        min_events=int(args.min_events),
    )
    all_rows.extend(train_rows)
    notes.extend(n1)

    valid_rows, n2 = _evaluate_split(
        split="valid",
        score=_filter_by_date(score, segments["valid"]),  # type: ignore[arg-type]
        raw=_filter_by_date(raw, segments["valid"]),  # type: ignore[arg-type]
        expectations=expectations,
        ladder_p=ladder_p,
        consistency_threshold=float(args.consistency_threshold),
        min_events=int(args.min_events),
    )
    all_rows.extend(valid_rows)
    notes.extend(n2)

    if args.include_test:
        if "test" not in segments:
            notes.append("include_test requested but no test segment parsed from conf.")
        else:
            test_rows, n3 = _evaluate_split(
                split="test",
                score=_filter_by_date(score, segments["test"]),  # type: ignore[arg-type]
                raw=_filter_by_date(raw, segments["test"]),  # type: ignore[arg-type]
                expectations=expectations,
                ladder_p=ladder_p,
                consistency_threshold=float(args.consistency_threshold),
                min_events=int(args.min_events),
            )
            all_rows.extend(test_rows)
            notes.extend(n3)
            notes.append("Test split computed for reporting only; PASS/FAIL still decided on valid.")
    else:
        notes.append("Test split is sealed (not computed). Use --include-test for reporting only.")

    mono = _monotonicity_on_valid(all_rows, eps=float(args.eps), min_events=int(args.min_events))
    passed = mono.passed

    table_path = out_dir / "stage3_table.csv"
    pd.DataFrame([asdict(r) for r in all_rows]).to_csv(table_path, index=False)

    summary = Stage3Summary(
        factor_ws=str(factor_ws),
        split_for_decision="valid",
        segments={k: (str(v.start.date()), str(v.end.date())) for k, v in segments.items()},
        strictness_ladder_p=[float(p) for p in ladder_p],
        quantile_mode="cross_sectional_top_p",
        expectations={k: str(v) for k, v in expectations.items()},
        consistency_threshold=float(args.consistency_threshold),
        min_events=int(args.min_events),
        eps=float(args.eps),
        valid_monotonicity=mono,
        passed=passed,
        notes=notes,
    )

    summary_path = out_dir / "stage3_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(summary),
                "valid_monotonicity": asdict(mono),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[Stage3] factor={display_name or factor_ws.name} split=valid passed={passed}")
    print(f"[Stage3] wrote: {table_path}")
    print(f"[Stage3] wrote: {summary_path}")

    row = {
        "factor_name": display_name or factor_ws.name,
        "workspace": str(factor_ws),
        "skipped": False,
        "passed": bool(passed),
        "valid_step_consistency": mono.step_consistency,
        "valid_spearman_rho": mono.spearman_rho,
        "out_dir": str(out_dir),
    }
    return row, bool(passed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-ws", default="", help="Factor workspace dir (RD-Agent_workspace/<uuid>)")
    ap.add_argument(
        "--log-dir",
        default="log",
        help="AlphaAgent run dir (e.g. log/<timestamp>) or parent log dir (e.g. log/) to pick the latest run",
    )
    ap.add_argument(
        "--factor-name",
        action="append",
        default=[],
        help="When using --log-dir mode: only process factors with these names (repeatable). Default: all found.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List resolved run dir and discovered factor workspaces, then exit.",
    )
    ap.add_argument(
        "--conf",
        default="alphaagent/scenarios/qlib/experiment/factor_template/conf_cn_combined_kdd_ver.yaml",
        help="Config yaml to read segments from (train/valid/test). Parsed via regex (no PyYAML).",
    )
    ap.add_argument(
        "--stage2-summary",
        default="",
        help="Path to stage2_summary.json to load expectations from. Default auto-searches under factor workspace and /home/dgu/fin/AlphaAgent/results/<run>/stage2/<factor>/",
    )
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        help="Expectation per raw feature, overrides stage2_summary if provided. Example: --expect dir:down --expect pos:down",
    )
    ap.add_argument(
        "--strictness-p",
        default="0.5,0.3,0.2,0.1,0.05,0.02",
        help="Comma-separated top-p ladder (loose->strict via decreasing p). Example: 0.5,0.3,0.2,0.1,0.05,0.02",
    )
    ap.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.0,
        help="Raw-consistency threshold. Event is false-positive if consistency_score <= threshold.",
    )
    ap.add_argument(
        "--min-events",
        type=int,
        default=200,
        help="Minimum number of events required for a ladder point to be used in monotonicity decision.",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Numerical tolerance for monotonicity (allows small increases within eps).",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory base. Default: /home/dgu/fin/AlphaAgent/results/<run>/stage3/<factor>/",
    )
    ap.add_argument(
        "--include-failed-stage2",
        action="store_true",
        help="Also run Stage3 even if stage2_summary.json exists and says passed=false (default: skip).",
    )
    ap.add_argument(
        "--include-test",
        action="store_true",
        help="Compute strictness table for test split too (PASS/FAIL still decided on valid only).",
    )
    args = ap.parse_args()

    run_dir: Optional[Path] = None
    targets: List[Tuple[str, Path]] = []
    single_mode = False

    if args.factor_ws:
        ws = Path(args.factor_ws).expanduser()
        targets = [(ws.name, ws)]
        single_mode = True
    else:
        log_dir = Path(args.log_dir).expanduser()
        if not log_dir.exists():
            raise SystemExit(f"log dir does not exist: {log_dir}")
        run_dir = _resolve_log_dir(log_dir)
        mapping = _extract_factor_workspaces_from_run(run_dir)
        if args.factor_name:
            wanted = set(args.factor_name)
            mapping = {k: v for k, v in mapping.items() if k in wanted}
        targets = list(mapping.items())

        if args.dry_run:
            print(f"Resolved --log-dir to: {run_dir}")
            if not targets:
                print("No factor workspaces found.")
            else:
                for name, ws in targets:
                    print(f"- {name}: {ws}")
            return 0

        if not targets:
            raise SystemExit(
                f"No factor workspaces found under run logs: {run_dir}\n"
                "Tip: pass --factor-ws directly, or ensure log/<run>/d/**/common_logs.log exists."
            )

    index_rows: List[dict] = []
    failed: List[str] = []
    passed_flags: List[bool] = []
    for display_name, ws in targets:
        try:
            row, ok = _run_stage3_for_workspace(factor_ws=ws, display_name=display_name, args=args, run_dir=run_dir)
            index_rows.append(row)
            passed_flags.append(ok)
        except Exception as e:
            failed.append(f"{display_name}: {e}")
            passed_flags.append(False)

    if run_dir is not None:
        index_base = _results_root() / run_dir.name
        index_base.mkdir(parents=True, exist_ok=True)
        index_csv = index_base / "stage3_index.csv"
        index_json = index_base / "stage3_index.json"
        try:
            pd.DataFrame(index_rows).to_csv(index_csv, index=False)
            with index_json.open("w", encoding="utf-8") as f:
                json.dump(index_rows, f, ensure_ascii=False, indent=2)
            print(f"[Stage3] wrote run index: {index_csv}")
        except Exception:
            pass

    if failed:
        print("[Stage3] Some factors were skipped/failed:")
        for item in failed:
            print(f"- {item}")

    # Exit code: in single-factor mode preserve 0/2 behavior; in multi-mode, return 2 if any failed/failed-pass.
    if single_mode:
        return 0 if (passed_flags[0] if passed_flags else False) else 2
    return 0 if all(passed_flags) else 2


if __name__ == "__main__":
    raise SystemExit(main())
