#!/usr/bin/env python3
"""
Stage 2: Observation Formula Validation (NO alpha / NO IC / NO Sharpe)

Goal
  Verify only this:
    "Does this factor score actually implement the intended observation (obs) condition
     in real data?"

What it does
  - Uses IS only (train/valid/train_valid); test is intentionally not used.
  - Buckets samples by factor score quantiles (recommended: per-date cross-sectional).
  - Observes how RAW OHLCV-derived distributions shift across buckets.
  - Checks monotonicity (directional) against an expectation spec -> PASS/FAIL.

Inputs
  - factor workspace folder (usually under git_ignore_folder/RD-Agent_workspace/<uuid>)
    containing:
      - factor.py (for expr/name reference)
      - result.h5 (factor values; output of factor.py)
      - daily_pv.h5 symlink (raw OHLCV source)
  - a Qlib-like conf yaml (to get train/valid/test segments) without requiring pyyaml.

Outputs (default under /home/dgu/fin/AlphaAgent/results/<run>/stage2/<factor>/)
  - stage2_summary.json
  - stage2_distributions.csv
  - stage2_<stat>.png (if matplotlib available) or stage2_<stat>.svg
  - stage2_evidence.json (LLM evidence packet)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

_NO_IMPORTS_MODE = any(a in ("-h", "--help", "--dry-run") for a in sys.argv)
try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as e:
    if not _NO_IMPORTS_MODE:
        missing = getattr(e, "name", "a required dependency")
        raise SystemExit(
            f"Missing dependency '{missing}'.\n"
            "Stage2 requires pandas+numpy. Run this script inside the same environment you use for AlphaAgent "
            "(e.g., your conda env `alphaagent`)."
        ) from e

Split = Literal["train", "valid", "train_valid"]
Direction = Literal["up", "down", "any"]
Polarity = Literal["higher_is_more_true", "lower_is_more_true"]


RAW_FEATURES = ("mag", "dir", "vol", "pos")
DIST_STATS = ("mean", "median", "std", "q10", "q25", "q75", "q90", "skewness", "kurtosis")


def _parse_stats_csv_arg(raw: str) -> Tuple[str, ...]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return tuple()
    bad = [p for p in parts if p not in DIST_STATS]
    if bad:
        raise SystemExit(f"Invalid stats: {bad}. Allowed: {', '.join(DIST_STATS)}")
    # Preserve order, dedupe.
    seen = set()
    out: List[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return tuple(out)


@dataclass(frozen=True)
class Segment:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class Monotonicity:
    feature: str
    expectation: str
    n_buckets: int
    bucket_counts: List[int]
    bucket_median: List[Optional[float]]
    step_consistency: Optional[float]
    spearman_rho: Optional[float]
    passed: bool


@dataclass(frozen=True)
class Stage2Summary:
    factor_ws: str
    factor_name: str
    factor_expr: str
    definition: str
    polarity: str
    obs_id: str
    obs_description: str
    score_source: str
    ohlcv_source: str
    split: str
    segments: Dict[str, Tuple[str, str]]
    n_quantiles: int
    quantile_mode: str
    expectations: Dict[str, str]
    min_bucket_frac: float
    min_pass_features: int
    pass_features: List[str]
    fail_features: List[str]
    passed: bool
    pass_source: str
    heuristic_passed: bool
    notes: List[str]
    monotonicity: List[Monotonicity]
    plot_path: str
    evidence_json_path: str
    plot_paths: List[str] = field(default_factory=list)
    llm_judgment: Optional[Dict[str, Any]] = None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _load_obs_meta_from_workspace(factor_ws: Path) -> Dict[str, str]:
    """
    Best-effort load of observation metadata for non-log-dir sources.

    Supported workspace files:
      - obs_description.txt (formula_packager)
      - formula_spec.json (formula_packager)
      - hypothesis.txt / gpt_factor.json (example_gpt_agent)
      - alpha101_spec.json (+ alpha101.py comment) (alpha101_packager)

    Returns dict with optional keys: obs_id, obs_description, polarity, definition.
    """
    out: Dict[str, str] = {}

    # 1) Our formula workspaces
    spec = factor_ws / "formula_spec.json"
    if spec.exists():
        try:
            obj = json.loads(spec.read_text(encoding="utf-8"))
            formula = obj.get("formula") if isinstance(obj, dict) else None
            if isinstance(formula, dict):
                out["obs_id"] = str(formula.get("obs_id") or "").strip()
                out["obs_description"] = str(formula.get("obs_description") or "").strip()
                out["polarity"] = str(formula.get("polarity") or "").strip()
                out["definition"] = str(formula.get("definition") or "").strip()
        except Exception:
            pass

    # If a plain text description exists, prefer it (often cleaner).
    txt = _read_optional_text(factor_ws / "obs_description.txt")
    if txt:
        out["obs_description"] = txt

    # 2) GPT workspaces
    hyp_txt = _read_optional_text(factor_ws / "hypothesis.txt")
    if hyp_txt and not out.get("obs_description"):
        out["obs_description"] = hyp_txt
    gpt_json = factor_ws / "gpt_factor.json"
    if gpt_json.exists() and not out.get("obs_description"):
        try:
            obj = json.loads(gpt_json.read_text(encoding="utf-8"))
            hyp = str(obj.get("hypothesis") or "").strip()
            rat = str(obj.get("rationale") or "").strip()
            joined = "\n\n".join([x for x in [hyp, rat] if x]).strip()
            if joined:
                out["obs_description"] = joined
        except Exception:
            pass

    # 3) Alpha101 workspaces
    a101 = factor_ws / "alpha101_spec.json"
    if a101.exists():
        try:
            obj = json.loads(a101.read_text(encoding="utf-8"))
            alpha = str(obj.get("alpha") or "").strip()
            if alpha:
                out.setdefault("obs_id", alpha)
                # Try to extract the original alpha expression from alpha101.py comments.
                try:
                    n = int(alpha.replace("alpha", "").lstrip("0") or "0")
                except Exception:
                    n = 0
                alpha_expr = ""
                if n > 0:
                    try:
                        alpha101_py = Path(__file__).resolve().parent / "alpha101.py"
                        text = _read_text(alpha101_py)
                        # e.g. "# Alpha#2\t (-1 * correlation(...))"
                        m = re.search(rf"^\\s*#\\s*Alpha#{n}\\s*\\(?\\s*(?P<expr>[^\\n]+?)\\s*\\)?\\s*$", text, flags=re.MULTILINE)
                        if m:
                            alpha_expr = m.group("expr").strip()
                    except Exception:
                        alpha_expr = ""
                if alpha_expr:
                    out.setdefault("definition", alpha_expr)
                    out.setdefault("obs_description", f"Alpha101 baseline {alpha}: {alpha_expr}")
                else:
                    out.setdefault("obs_description", f"Alpha101 baseline {alpha} as implemented in alpha101.py.")
        except Exception:
            pass

    return {k: v for k, v in out.items() if v}


def _fallback_llm_judgment(
    *,
    error: Exception,
    heuristic_passed: bool,
    evidence_json: Dict[str, Any],
    distribution_summary: str,
) -> Dict[str, Any]:
    """
    If --use-llm is enabled but the LLM call fails, still populate llm_judgment
    so downstream consumers always see verdict+reasoning.
    """
    verdict = "PASS" if heuristic_passed else "FAIL"
    bins = evidence_json.get("bins", [])
    features = evidence_json.get("features", {}) if isinstance(evidence_json.get("features", {}), dict) else {}

    # Pick up to 2 best (feature, stat) pairs by step_consistency to cite.
    candidates: List[Tuple[float, str, str, List[Any]]] = []
    for feat, meta in features.items():
        if not isinstance(meta, dict):
            continue
        stats = meta.get("stats", {})
        mono = meta.get("monotonicity", {}).get("by_stat", {}) if isinstance(meta.get("monotonicity", {}), dict) else {}
        if not isinstance(stats, dict) or not isinstance(mono, dict):
            continue
        for st, arr in stats.items():
            try:
                sc = mono.get(st, {}).get("step_consistency", None)
                scv = float(sc) if sc is not None else -1.0
            except Exception:
                scv = -1.0
            if not isinstance(arr, list):
                continue
            candidates.append((scv, feat, str(st), arr))
    candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    chosen = candidates[:2]

    primary_evidence: List[Dict[str, Any]] = []
    for _, feat, st, arr in chosen:
        nums = []
        for v in arr:
            try:
                if v is None:
                    nums.append(None)
                else:
                    nums.append(float(v))
            except Exception:
                nums.append(None)
        finite = [x for x in nums if isinstance(x, (int, float)) and math.isfinite(float(x))]
        pattern = "increasing"
        if len(finite) >= 2 and finite[-1] < finite[0]:
            pattern = "decreasing"
        primary_evidence.append(
            {
                "feature": feat,
                "stat": st,
                "pattern": pattern,
                "bins": bins,
                "numbers": nums,
            }
        )

    # Per-feature analysis (short, consistent, numeric).
    feature_analysis: Dict[str, str] = {}
    for feat, meta in features.items():
        if not isinstance(meta, dict):
            continue
        stats = meta.get("stats", {})
        if not isinstance(stats, dict):
            continue
        # Prefer mean + q90 if present; else first two stats.
        preferred = []
        for k in ("mean", "q90"):
            if k in stats and isinstance(stats.get(k), list):
                preferred.append(k)
        if len(preferred) < 2:
            for k, v in stats.items():
                if k in preferred:
                    continue
                if isinstance(v, list):
                    preferred.append(str(k))
                if len(preferred) >= 2:
                    break
        if len(preferred) < 2:
            continue
        a0 = stats.get(preferred[0], [])
        a1 = stats.get(preferred[1], [])
        if not isinstance(a0, list) or not isinstance(a1, list) or not bins:
            continue
        def endpts(arr):
            try:
                return arr[0], arr[-1]
            except Exception:
                return None, None
        v0s, v0e = endpts(a0)
        v1s, v1e = endpts(a1)
        feature_analysis[str(feat)] = (
            f"{feat}: {preferred[0]} {bins[0]}={v0s} → {bins[-1]}={v0e}; "
            f"{preferred[1]} {bins[0]}={v1s} → {bins[-1]}={v1e}."
        )

    # Keep it short but useful.
    reason_bits = []
    reason_bits.append(f"LLM judgment failed ({type(error).__name__}); fell back to heuristic verdict={verdict}.")
    if primary_evidence:
        ev = primary_evidence[0]
        if ev.get("bins") and isinstance(ev.get("numbers"), list) and len(ev["numbers"]) >= 2:
            b0 = ev["bins"][0]
            bk = ev["bins"][-1]
            n0 = ev["numbers"][0]
            nk = ev["numbers"][-1]
            reason_bits.append(f"Top signal: {ev['feature']} {ev['stat']} changes from {b0}={n0} to {bk}={nk} ({ev['pattern']}).")
    reason_bits.append("Heuristic summary: " + (distribution_summary.splitlines()[0] if distribution_summary else "n/a"))

    return {
        "verdict": verdict,
        "reasoning": " ".join(reason_bits).strip(),
        "primary_evidence": primary_evidence,
        "feature_analysis": feature_analysis,
        "raw_response": json.dumps({"error": str(error)}, ensure_ascii=False),
    }

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

    # Prefer d/ scope; fall back to any common_logs.log if needed.
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


def _extract_from_factor_py(factor_py: Path) -> Tuple[str, str]:
    """
    Best-effort parse of:
      expr = "..."
      name = "..."
    """
    text = _read_text(factor_py)
    # Keep it simple: look for the last assignment in the file.
    name_m = list(re.finditer(r'^\s*name\s*=\s*"([^"]+)"', text, flags=re.MULTILINE))
    expr_m = list(re.finditer(r'^\s*expr\s*=\s*"([^"]+)"', text, flags=re.MULTILINE))
    factor_name = name_m[-1].group(1) if name_m else ""
    factor_expr = expr_m[-1].group(1) if expr_m else ""
    return factor_name, factor_expr


def _iter_role_contents(text: str, role: str) -> List[str]:
    """
    Parse llm_messages/common_logs.log style blocks:
      Role:<role>
      Content: ...
    """
    pattern = re.compile(
        rf"^Role:{re.escape(role)}\s*$\n^Content:\s*(?P<content>.*?)(?=^\d{{4}}-\d{{2}}-\d{{2}}|^Role:|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    return [m.group("content").strip() for m in pattern.finditer(text)]


def _extract_obs_context_from_run(*, run_dir: Path, factor_name: str) -> Tuple[str, str]:
    """
    Best-effort extraction of observation description + formulation from:
      log/<run>/llm_messages/**/common_logs.log
    """
    if not run_dir.exists():
        return "", ""
    logs = sorted(run_dir.rglob("llm_messages/**/common_logs.log"), key=lambda p: str(p))
    for p in logs:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if factor_name not in text:
            continue

        for content in _iter_role_contents(text, "user"):
            if "Tasks and Factors" not in content or factor_name not in content:
                continue

            hyp = ""
            mh = re.search(r"Target hypothesis:\s*(?P<h>.*?)(?:\n\s*Tasks and Factors:)", content, flags=re.DOTALL)
            if mh:
                hyp = mh.group("h").strip().replace("\n", " ").strip()

            mf = re.search(
                rf"^\s*-\s*{re.escape(factor_name)}\s*:\s*(?P<desc>.*)$",
                content,
                flags=re.MULTILINE,
            )
            desc = mf.group("desc").strip() if mf else ""

            form = ""
            mform = re.search(r"Factor Formulation:\s*(?P<f>.*)$", content, flags=re.MULTILINE)
            if mform:
                form = mform.group("f").strip()

            obs_desc = " ".join([x for x in [hyp, desc] if x]).strip()
            return obs_desc, form

    return "", ""


def _parse_segments_from_conf(conf_path: Path) -> Dict[str, Segment]:
    """
    Parse segments.* from a Qlib-style yaml without PyYAML.
    Expected forms in our repo:
      train: [2015-01-01, 2018-12-31]
      valid: [2019-01-01, 2020-12-31]
      test:  [2021-01-01, 2025-12-26]
    """
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


def _pick_split_range(segments: Dict[str, Segment], split: Split) -> Segment:
    if split == "train":
        return segments["train"]
    if split == "valid":
        return segments["valid"]
    # train_valid
    tr = segments["train"]
    va = segments["valid"]
    return Segment(start=min(tr.start, va.start), end=max(tr.end, va.end))


def _safe_read_hdf(path: Path, *, key: str, columns: Optional[List[str]] = None):
    try:
        if columns is None:
            return pd.read_hdf(path, key=key)
        return pd.read_hdf(path, key=key, columns=columns)
    except Exception:
        # Some HDF stores don't support column selection; fall back to full read.
        return pd.read_hdf(path, key=key)


def _ensure_multiindex(df_or_s: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    idx = df_or_s.index
    if isinstance(idx, pd.MultiIndex) and {"datetime", "instrument"} <= set(idx.names):
        return df_or_s

    # Try common alternatives
    if isinstance(idx, pd.MultiIndex) and len(idx.levels) >= 2:
        names = list(idx.names)
        if names[0] is None:
            names[0] = "datetime"
        if len(names) > 1 and names[1] is None:
            names[1] = "instrument"
        df_or_s.index = idx.set_names(names)
        return df_or_s

    raise ValueError("Expected a MultiIndex with ('datetime','instrument') levels in HDF data.")


def _filter_by_date(
    df_or_s: pd.DataFrame | pd.Series, seg: Segment, *, datetime_level: str = "datetime"
) -> pd.DataFrame | pd.Series:
    idx = df_or_s.index
    if not isinstance(idx, pd.MultiIndex):
        raise ValueError("Expected MultiIndex to filter by date.")
    dt = idx.get_level_values(datetime_level)
    mask = (dt >= seg.start) & (dt <= seg.end)
    return df_or_s[mask]


def _resolve_ohlcv_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    # Qlib-style columns: $open, $high, $low, $close, $volume
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


def _assign_quantiles_cross_sectional(score: pd.Series, n_quantiles: int) -> pd.Series:
    """
    Assign buckets 1..n_quantiles per date (cross-sectional), using rank-based split.
    Bucket n_quantiles corresponds to the highest score group.
    """

    def _per_date(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if x.empty:
            return pd.Series(index=x.index, dtype="float64")
        n = len(x)
        if n < n_quantiles:
            return pd.Series(index=x.index, data=np.nan, dtype="float64")
        # Break ties deterministically.
        r = x.rank(method="first")
        bucket = ((r - 1) / n * n_quantiles).astype(int) + 1
        return bucket.astype("float64")

    return score.groupby(level="datetime", group_keys=False).apply(_per_date)


def _assign_quantiles_global(score: pd.Series, n_quantiles: int) -> pd.Series:
    s = score.copy()
    s = s.astype(float)
    if s.notna().sum() < n_quantiles:
        return pd.Series(index=s.index, data=np.nan, dtype="float64")
    r = s.rank(method="first", na_option="keep")
    n = int(s.notna().sum())
    bucket = ((r - 1) / n * n_quantiles).astype("float64") + 1.0
    return bucket


def _distribution_table(df: pd.DataFrame, bucket: pd.Series, n_quantiles: int) -> pd.DataFrame:
    # Align and filter NaN buckets
    bucket = bucket.reindex(df.index)
    mask = bucket.notna()
    df = df.loc[mask]
    bucket = bucket.loc[mask].astype(int)

    rows = []
    for q in range(1, n_quantiles + 1):
        sub = df.loc[bucket == q]
        row = {"bucket": q, "count": int(len(sub))}
        for f in RAW_FEATURES:
            s = sub[f].astype(float)
            if s.empty:
                for k in DIST_STATS:
                    row[f"{f}_{k}"] = np.nan
                continue
            row[f"{f}_mean"] = float(s.mean())
            row[f"{f}_median"] = float(s.median())
            row[f"{f}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            row[f"{f}_q10"] = float(s.quantile(0.10))
            row[f"{f}_q25"] = float(s.quantile(0.25))
            row[f"{f}_q75"] = float(s.quantile(0.75))
            row[f"{f}_q90"] = float(s.quantile(0.90))
            row[f"{f}_skewness"] = float(s.skew()) if len(s) > 2 else np.nan
            row[f"{f}_kurtosis"] = float(s.kurt()) if len(s) > 3 else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("bucket")


def _to_num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    except Exception:
        return None


def _build_evidence_json(
    dist: pd.DataFrame,
    *,
    expectations: Dict[str, Direction],
    n_quantiles: int,
    notes: List[str],
    stats_to_emit: Tuple[str, ...],
) -> Dict[str, Any]:
    bins = [f"Q{i}" for i in range(1, n_quantiles + 1)]
    bin_counts = [int(dist.loc[i, "count"]) for i in range(1, n_quantiles + 1)]

    feature_map = {"mag": "MAG", "dir": "DIR", "vol": "VOL", "pos": "POS"}
    for st in stats_to_emit:
        if st not in DIST_STATS:
            raise ValueError(f"Invalid stats_to_emit item: {st}. Allowed: {', '.join(DIST_STATS)}")

    features: Dict[str, Any] = {}
    for feat in RAW_FEATURES:
        out_name = feature_map[feat]
        exp = expectations.get(feat, "any")
        stats: Dict[str, List[Optional[float]]] = {}
        by_stat: Dict[str, Any] = {}
        sc_scores: List[float] = []
        for st in stats_to_emit:
            arr = [_to_num(dist.loc[i, f"{feat}_{st}"]) for i in range(1, n_quantiles + 1)]
            stats[st] = arr
            sc = _step_consistency(arr, exp)
            rho = _spearman_rho(range(1, n_quantiles + 1), [np.nan if v is None else v for v in arr])
            by_stat[st] = {"step_consistency": sc, "spearman_rho": rho}
            if sc is not None:
                sc_scores.append(float(sc))
        features[out_name] = {
            "expectation": exp,
            "stats": stats,
            "monotonicity": {"score": (max(sc_scores) if sc_scores else None), "by_stat": by_stat},
        }

    return {
        "bins": bins,
        "bin_counts": bin_counts,
        "warnings": list(notes),
        "features": features,
    }


def _build_distribution_summary(
    *,
    expectations: Dict[str, Direction],
    passed: bool,
    pass_feats: List[str],
    fail_feats: List[str],
    notes: List[str],
    n_quantiles: int,
) -> str:
    exp_str = ", ".join(f"{k}:{v}" for k, v in expectations.items())
    parts = [
        f"n_quantiles={n_quantiles}",
        f"expectations={exp_str}",
        f"heuristic_passed={passed}",
        f"pass_features={pass_feats}",
        f"fail_features={fail_feats}",
    ]
    if notes:
        parts.append("notes=" + "; ".join(notes))
    return "\n".join(parts)


def _spearman_rho(x: Iterable[float], y: Iterable[float]) -> Optional[float]:
    try:
        x = np.asarray(list(x), dtype=float)
        y = np.asarray(list(y), dtype=float)
        if len(x) != len(y) or len(x) < 2:
            return None
        if np.all(np.isnan(y)):
            return None
        # rank transform
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        # corr
        if np.std(xr) == 0 or np.std(yr) == 0:
            return None
        return float(np.corrcoef(xr, yr)[0, 1])
    except Exception:
        return None


def _step_consistency(values: List[Optional[float]], direction: Direction) -> Optional[float]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2:
        return None

    def score_for(dir_: Literal["up", "down"]) -> float:
        ok = 0
        total = 0
        for a, b in zip(v, v[1:]):
            total += 1
            if dir_ == "up":
                ok += 1 if b >= a else 0
            else:
                ok += 1 if b <= a else 0
        return ok / total if total else 0.0

    if direction == "any":
        return max(score_for("up"), score_for("down"))
    return score_for(direction)


def _monotonicity_check(
    dist: pd.DataFrame,
    *,
    expectations: Dict[str, Direction],
    n_quantiles: int,
    min_bucket_frac: float,
    min_pass_features: int,
) -> Tuple[bool, List[Monotonicity], List[str], List[str], List[str]]:
    notes: List[str] = []

    total = int(dist["count"].sum())
    counts = [int(dist.loc[i, "count"]) for i in range(1, n_quantiles + 1)]
    if total == 0:
        return False, [], [], list(RAW_FEATURES), ["No samples after bucketing."]

    min_count = min(counts) if counts else 0
    if min_count / total < min_bucket_frac:
        notes.append(
            f"Bucket imbalance: min_bucket_frac={min_bucket_frac} but min bucket has {min_count}/{total} samples."
        )

    mono: List[Monotonicity] = []
    pass_feats: List[str] = []
    fail_feats: List[str] = []

    for feat in RAW_FEATURES:
        exp = expectations.get(feat, "any")
        med = []
        for q in range(1, n_quantiles + 1):
            v = dist.loc[q, f"{feat}_median"]
            med.append(None if (isinstance(v, float) and math.isnan(v)) else float(v))
        sc = _step_consistency(med, exp)
        rho = _spearman_rho(range(1, n_quantiles + 1), [np.nan if v is None else v for v in med])
        # Feature pass rule: strong step-consistency and bucket coverage OK.
        feat_pass = (sc is not None and sc >= 0.8) and (min_count / total >= min_bucket_frac)
        # If expectation is directional, also require rho sign if rho exists.
        if feat_pass and exp in ("up", "down") and rho is not None:
            if exp == "up" and rho < 0:
                feat_pass = False
            if exp == "down" and rho > 0:
                feat_pass = False

        mono.append(
            Monotonicity(
                feature=feat,
                expectation=str(exp),
                n_buckets=n_quantiles,
                bucket_counts=counts,
                bucket_median=med,
                step_consistency=sc,
                spearman_rho=rho,
                passed=feat_pass,
            )
        )
        (pass_feats if feat_pass else fail_feats).append(feat)

    overall_pass = len(pass_feats) >= min_pass_features and (min_count / total >= min_bucket_frac)
    if len(pass_feats) < min_pass_features:
        notes.append(f"Not enough monotonic features: {len(pass_feats)} < min_pass_features={min_pass_features}.")
    if min_count / total < min_bucket_frac:
        notes.append("FAIL due to bucket imbalance/insufficient samples in extreme buckets.")

    return overall_pass, mono, pass_feats, fail_feats, notes


def _plot_distributions(
    *,
    dist: pd.DataFrame,
    expectations: Dict[str, Direction],
    n_quantiles: int,
    title: str,
    out_path: Path,
    stat: str,
    raw: Optional[pd.DataFrame] = None,
    bucket: Optional[pd.Series] = None,
    scatter_max_points_per_bucket: int = 600,
    scatter_alpha: float = 0.08,
) -> Optional[str]:
    """
    Plot ONE distribution statistic per figure (one image per stat), across quantile buckets.

    Writes PNG. Values are min-max normalized to [0, 1] for readability/comparability.
    Prefer matplotlib if available; otherwise use PIL.
    Returns written path as string, or None if plotting fails.
    """
    if stat not in DIST_STATS:
        raise ValueError(f"Invalid stat: {stat}. Allowed: {', '.join(DIST_STATS)}")

    out_path_4x1 = out_path.with_name(f"{out_path.stem}_4x1{out_path.suffix}")
    out_path_1x4 = out_path.with_name(f"{out_path.stem}_1x4{out_path.suffix}")

    # Add a little breathing room on the scaled axis for visualization.
    y_pad = 0.1
    y_vmin, y_vmax = -y_pad, 1.0 + y_pad

    # Robust scaling ranges for visualization: [1%, 99%] (clipped).
    robust_range: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if raw is not None and bucket is not None and scatter_max_points_per_bucket > 0:
        b = bucket.reindex(raw.index)
        mask = b.notna()
        raw2 = raw.loc[mask]
        b2 = b.loc[mask].astype(int)
        for feat in RAW_FEATURES:
            s = raw2[feat].astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            vmin = float(s.quantile(0.01))
            vmax = float(s.quantile(0.99))
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
                vmin = float(s.min())
                vmax = float(s.max())
            robust_range[feat] = (vmin, vmax)

    def _normalize_0_1(values: List[Optional[float]]) -> Tuple[List[Optional[float]], Optional[float], Optional[float]]:
        finite = [v for v in values if v is not None and math.isfinite(v)]
        if not finite:
            return [None for _ in values], None, None
        vmin = float(min(finite))
        vmax = float(max(finite))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            out = []
            for v in values:
                out.append(None if v is None or not math.isfinite(v) else 0.5)
            return out, vmin, vmax
        denom = vmax - vmin
        out = []
        for v in values:
            if v is None or not math.isfinite(v):
                out.append(None)
            else:
                out.append((float(v) - vmin) / denom)
        return out, vmin, vmax

    def _scale_raw(v: Optional[float], feat: str) -> Optional[float]:
        if v is None or not math.isfinite(v):
            return None
        vmin, vmax = robust_range.get(feat, (None, None))
        if vmin is None or vmax is None or vmax <= vmin:
            return None
        x = (float(v) - float(vmin)) / (float(vmax) - float(vmin))
        # Clip for robust display.
        return float(min(1.0, max(0.0, x)))

    def _write_png_pil(png_path: Path, *, layout: str) -> str:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        def _font(size: int):
            for fp in (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ):
                try:
                    return ImageFont.truetype(fp, size=size)
                except Exception:
                    continue
            return ImageFont.load_default()

        # Match the matplotlib output geometry:
        # - 2x2: 15x8 inches @ 160 DPI -> 2400x1280
        # - 4x1: 15x16 inches @ 160 DPI -> 2400x2560
        # - 1x4: 30x4 inches @ 160 DPI -> 4800x640
        if layout == "4x1":
            width, height = 2400, 2560
            cols, rows = 1, 4
        elif layout == "1x4":
            width, height = 4800, 640
            cols, rows = 4, 1
        else:
            width, height = 2400, 1280
            cols, rows = 2, 2
        pad = 20
        title_h = 0
        legend_h = 0

        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font_title = _font(20)
        font_med = _font(18)
        font_small = _font(17)

        grid_top = title_h + 10
        grid_left = pad
        grid_right = pad
        grid_bottom = pad + legend_h

        cell_w = (width - grid_left - grid_right) / cols
        cell_h = (height - grid_top - grid_bottom) / rows

        buckets = list(range(1, n_quantiles + 1))
        feature_titles = {"mag": "MAG", "dir": "DIR", "vol": "VOL", "pos": "POS"}

        def panel_rect(i: int):
            r = i // cols
            c = i % cols
            x0 = grid_left + c * cell_w
            y0 = grid_top + r * cell_h
            return x0, y0, cell_w, cell_h

        def scale_x(x0: float, w: float, v: float, vmin: float, vmax: float) -> float:
            left = x0 + 55
            right = x0 + w - 15
            if vmax <= vmin:
                return (left + right) / 2
            return left + (v - vmin) * (right - left) / (vmax - vmin)

        def scale_y(y0: float, h: float, v: float, vmin: float, vmax: float) -> float:
            top = y0 + 45
            bottom = y0 + h - 35
            if vmax <= vmin:
                return (top + bottom) / 2
            return bottom - (v - vmin) * (bottom - top) / (vmax - vmin)

        for i, feat in enumerate(RAW_FEATURES):
            x0, y0, w, h = panel_rect(i)
            exp = expectations.get(feat, "any")

            raw_values = [_to_num(dist.loc[q, f"{feat}_{stat}"]) for q in buckets]
            # If we have robust raw ranges, scale stats to the same raw-feature scale so scatter/mean/median align.
            if feat in robust_range and stat in ("mean", "median", "std", "q10", "q25", "q75", "q90"):
                values = [_scale_raw(v, feat) for v in raw_values]
                raw_min, raw_max = robust_range.get(feat, (None, None))
            else:
                values, raw_min, raw_max = _normalize_0_1(raw_values)

            sc = _step_consistency(values, exp)
            rho = _spearman_rho(range(1, n_quantiles + 1), [np.nan if v is None else v for v in values])
            status = "PASS" if (sc is not None and sc >= 0.8) else "FAIL"

            draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(221, 221, 221), width=1)
            draw.text((x0 + 10, y0 + 8), feature_titles.get(feat, feat), fill=(0, 0, 0), font=font_small)

            ax_left = x0 + 50
            ax_right = x0 + w - 20
            ax_top = y0 + 35
            ax_bottom = y0 + h - 30
            draw.line((ax_left, ax_top, ax_left, ax_bottom), fill=(153, 153, 153), width=1)
            draw.line((ax_left, ax_bottom, ax_right, ax_bottom), fill=(153, 153, 153), width=1)

            # x-axis: bucket index (Q1..Qn)
            tick_step = 1 if len(buckets) <= 10 else 2
            grid_col = (205, 205, 205)
            # Draw gridlines for all bucket positions; label a subset for readability.
            for q in buckets:
                x = scale_x(x0, w, float(q), 1.0, float(n_quantiles))
                draw.line((x, ax_top, x, ax_bottom), fill=grid_col, width=1)
            for q in buckets[::tick_step]:
                x = scale_x(x0, w, float(q), 1.0, float(n_quantiles))
                draw.line((x, ax_bottom, x, ax_bottom + 4), fill=(153, 153, 153), width=1)
                draw.text((x - 10, ax_bottom + 7), f"Q{q}", fill=(85, 85, 85), font=font_small)

            # y-axis: scaled value (0..1)
            for tval in (0.0, 0.5, 1.0):
                y = scale_y(y0, h, float(tval), float(y_vmin), float(y_vmax))
                draw.line((ax_left, y, ax_right, y), fill=grid_col, width=1)
                draw.line((ax_left - 4, y, ax_left, y), fill=(153, 153, 153), width=1)
                draw.text((x0 + 6, y - 7), f"{tval:.3g}", fill=(85, 85, 85), font=font_small)

            pts: List[Tuple[float, float]] = []
            for q in buckets:
                v = values[q - 1]
                if v is None or not math.isfinite(v):
                    continue
                pts.append(
                    (
                        scale_x(x0, w, float(q), 1.0, float(n_quantiles)),
                        scale_y(y0, h, float(v), float(y_vmin), float(y_vmax)),
                    )
                )
            if len(pts) >= 2:
                draw.line(pts, fill=(76, 120, 168), width=5)
            for x, y in pts:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(76, 120, 168))

            # Legend: inside upper-right of each panel.
            sw = 34
            lx = int(x0 + w - 12 - 160)
            ly = int(y0 + 14)
            draw.rectangle((lx - 10, ly - 8, lx + 150, ly + 24), fill=(255, 255, 255), outline=(220, 220, 220), width=1)
            draw.line((lx, ly + 8, lx + sw, ly + 8), fill=(76, 120, 168), width=15)
            draw.text((lx + sw + 10, ly - 2), stat, fill=(0, 0, 0), font=font_med)

        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(png_path, format="PNG")
        return str(png_path)

    try:
        import matplotlib  # type: ignore

        # Force a headless backend so this works reliably on servers/CI.
        matplotlib.use("Agg")  # type: ignore[attr-defined]
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        out_main = _write_png_pil(out_path, layout="2x2")
        _write_png_pil(out_path_4x1, layout="4x1")
        _write_png_pil(out_path_1x4, layout="1x4")
        return out_main

    buckets = list(range(1, n_quantiles + 1))
    feature_titles = {"mag": "MAG", "dir": "DIR", "vol": "VOL", "pos": "POS"}

    def _write_png_matplotlib(png_path: Path, *, layout: str) -> str:
        if layout == "4x1":
            # 15x16 inches @ 160 DPI -> 2400x2560
            fig, axes = plt.subplots(4, 1, figsize=(15, 16), constrained_layout=True)
            ax_list = axes.ravel().tolist() if isinstance(axes, np.ndarray) else [axes]
        elif layout == "1x4":
            # 30x4 inches @ 160 DPI -> 4800x640
            fig, axes = plt.subplots(1, 4, figsize=(30, 4), constrained_layout=True)
            ax_list = axes.ravel().tolist() if isinstance(axes, np.ndarray) else [axes]
        else:
            # 15x8 inches @ 160 DPI -> 2400x1280
            fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
            ax_list = axes.ravel().tolist()

        for ax, feat in zip(ax_list, RAW_FEATURES):
            raw_values = [_to_num(dist.loc[q, f"{feat}_{stat}"]) for q in buckets]
            # If we have robust raw ranges, scale stats to the same raw-feature scale.
            if feat in robust_range and stat in ("mean", "median", "std", "q10", "q25", "q75", "q90"):
                values = [_scale_raw(v, feat) for v in raw_values]
            else:
                values, _raw_min, _raw_max = _normalize_0_1(raw_values)

            y = [np.nan if v is None else float(v) for v in values]
            ax.plot(buckets, y, marker="o", markersize=4.5, linewidth=3.0, label=stat)

            ax.set_xlim(0.5, n_quantiles + 0.5)
            ax.set_xticks(buckets)
            ax.set_xticklabels(
                [f"Q{b}" for b in buckets],
                rotation=(45 if len(buckets) > 10 else 0),
                ha=("right" if len(buckets) > 10 else "center"),
            )
            ax.set_ylim(float(y_vmin), float(y_vmax))
            ax.set_yticks([0.0, 0.5, 1.0])

            ax.set_title(feature_titles.get(feat, feat))
            ax.set_xlabel("bucket (Q1..Qn)", fontsize=11)
            ax.set_ylabel("scaled (0-1)", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(True, which="major", axis="both", alpha=0.38, linewidth=1.0)
            ax.legend(loc="upper right", frameon=False, fontsize=11, handlelength=2.2)

        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        return str(png_path)

    out_main = _write_png_matplotlib(out_path, layout="2x2")
    _write_png_matplotlib(out_path_4x1, layout="4x1")
    _write_png_matplotlib(out_path_1x4, layout="1x4")
    return out_main


def _plot_feature_all_stats(
    *,
    dist: pd.DataFrame,
    n_quantiles: int,
    out_path: Path,
    feature: str,
    stats: List[str],
    raw: Optional[pd.DataFrame] = None,
    bucket: Optional[pd.Series] = None,
) -> Optional[str]:
    """
    Plot ONE raw feature (mag/dir/vol/pos) per figure, overlaying multiple statistics as separate lines.
    Values are scaled to [0,1] per stat (robust raw-range scaling for raw-compatible stats when possible).
    """
    if feature not in RAW_FEATURES:
        raise ValueError(f"Invalid feature: {feature}. Allowed: {', '.join(RAW_FEATURES)}")
    stats = [s for s in stats if s in DIST_STATS]
    if not stats:
        return None

    robust_range: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if raw is not None and bucket is not None:
        b = bucket.reindex(raw.index)
        mask = b.notna()
        raw2 = raw.loc[mask]
        s = raw2[feature].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if not s.empty:
            vmin = float(s.quantile(0.01))
            vmax = float(s.quantile(0.99))
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
                vmin = float(s.min())
                vmax = float(s.max())
            robust_range[feature] = (vmin, vmax)

    def _normalize_0_1(values: List[Optional[float]]) -> List[Optional[float]]:
        finite = [v for v in values if v is not None and math.isfinite(v)]
        if not finite:
            return [None for _ in values]
        vmin = float(min(finite))
        vmax = float(max(finite))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            return [None if v is None or not math.isfinite(v) else 0.5 for v in values]
        denom = vmax - vmin
        out: List[Optional[float]] = []
        for v in values:
            if v is None or not math.isfinite(v):
                out.append(None)
            else:
                out.append((float(v) - vmin) / denom)
        return out

    def _scale_raw(v: Optional[float]) -> Optional[float]:
        if v is None or not math.isfinite(v):
            return None
        vmin, vmax = robust_range.get(feature, (None, None))
        if vmin is None or vmax is None or vmax <= vmin:
            return None
        x = (float(v) - float(vmin)) / (float(vmax) - float(vmin))
        return float(min(1.0, max(0.0, x)))

    y_pad = 0.05
    y_vmin, y_vmax = -y_pad, 1.0 + y_pad
    buckets = list(range(1, n_quantiles + 1))
    title = feature.upper()
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#FF9DA6"]

    series: List[Tuple[str, str, List[Optional[float]]]] = []
    for i, st in enumerate(stats):
        col = f"{feature}_{st}"
        if col not in dist.columns:
            continue
        raw_values = [_to_num(dist.loc[q, col]) for q in buckets]
        if feature in robust_range and st in ("mean", "median", "std", "q10", "q25", "q75", "q90"):
            values = [_scale_raw(v) for v in raw_values]
            if all(v is None for v in values):
                values = _normalize_0_1(raw_values)
        else:
            values = _normalize_0_1(raw_values)
        series.append((st, palette[i % len(palette)], values))

    if not series:
        return None

    # Prefer matplotlib; fall back to PIL.
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")  # type: ignore[attr-defined]
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore[assignment]

    if plt is not None:
        # Match Stage2 plot style (same fonts/grid/line weight); single-panel figure.
        # Keep the same canvas size as stage2_<stat>.png (15x8 inches @ 160 DPI).
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), constrained_layout=False)
        for st, _color, vals in series:
            y = [np.nan if v is None else float(v) for v in vals]
            ax.plot(buckets, y, marker="o", markersize=4.5, linewidth=3.0, label=st)
        ax.set_xlim(0.5, n_quantiles + 0.5)
        ax.set_xticks(buckets)
        ax.set_xticklabels([f"Q{b}" for b in buckets], rotation=(45 if len(buckets) > 10 else 0), ha=("right" if len(buckets) > 10 else "center"))
        ax.set_ylim(float(y_vmin), float(y_vmax))
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlabel("bucket (Q1..Qn)", fontsize=11)
        ax.set_ylabel("scaled (0-1)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(True, which="major", axis="both", alpha=0.38, linewidth=1.0)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = min(len(labels), 5)
            nrows = int(math.ceil(len(labels) / ncol)) if ncol else 1
            # Reserve more top padding when the legend wraps to multiple rows.
            top = 0.82 - 0.05 * max(0, nrows - 1)
            top = float(min(0.82, max(0.68, top)))
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=ncol,
                frameon=False,
                fontsize=11,
                handlelength=2.2,
                columnspacing=1.8,
                labelspacing=0.9,
                handletextpad=0.7,
                borderaxespad=0.0,
            )
        # Use the former title area for legend.
        fig.subplots_adjust(left=0.07, right=0.985, top=(top if handles else 0.82), bottom=0.12)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return str(out_path)

    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    def _font(size: int):
        for fp in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _parse_color(c: str) -> Tuple[int, int, int]:
        c = (c or "").strip()
        if c.startswith("#") and len(c) == 7:
            return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
        return (0, 0, 0)

    # Match stage2_<stat>.png canvas size: 15x8 inches @ 160 DPI -> 2400x1280.
    width, height = 2400, 1280
    pad = 28
    x0 = pad
    y0 = pad
    w = width - 2 * pad
    h = height - 2 * pad

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Match Stage2 PIL styling.
    font_title = _font(20)
    font_med = _font(18)
    font_small = _font(17)

    def sx(q: int) -> float:
        if len(buckets) <= 1:
            return x0 + w / 2
        i = buckets.index(q)
        return x0 + i * (w / (len(buckets) - 1))

    def sy(v: float) -> float:
        return y0 + h - (float(v) - float(y_vmin)) * (h / (float(y_vmax) - float(y_vmin)))

    # panel frame
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(221, 221, 221), width=1)

    # Legend in the title area (top of panel), wrapped.
    leg_x0 = int(x0 + 10)
    leg_y0 = int(y0 + 8)
    leg_x1 = int(x0 + w - 10)
    line_h = 36

    def _text_w(text: str, font) -> int:
        try:
            box = draw.textbbox((0, 0), text, font=font)
            return int(box[2] - box[0])
        except Exception:
            try:
                w0, _h0 = draw.textsize(text, font=font)
                return int(w0)
            except Exception:
                return int(len(text) * 12)

    items: List[Tuple[int, int, str, str]] = []  # x, y, label, color
    available_w = max(1, int(leg_x1 - leg_x0))
    item_ws = [int(42 + 10 + _text_w(st, font_med) + 30) for st, _c, _v in series]
    max_item_w = max(item_ws) if item_ws else 1
    max_cols = 5
    ncol = min(len(series), max_cols, max(1, int(available_w // max_item_w)))
    col_w = float(available_w) / float(ncol) if ncol else float(available_w)
    for i, (st, color, _vals) in enumerate(series):
        row = int(i // ncol) if ncol else 0
        col = int(i % ncol) if ncol else 0
        x = int(leg_x0 + col * col_w)
        y = int(leg_y0 + row * line_h)
        items.append((x, y, st, color))

    legend_bottom = leg_y0
    if items:
        box_left = int(leg_x0 - 10)
        box_top = int(leg_y0 - 8)
        box_right = int(leg_x1)
        nrows = int(math.ceil(len(items) / ncol)) if ncol else 1
        box_bottom = int(leg_y0 + nrows * line_h + 6)
        legend_bottom = box_bottom
        draw.rectangle((box_left, box_top, box_right, box_bottom), fill=(255, 255, 255), outline=(220, 220, 220), width=1)
        for x, y, st, color in items:
            draw.line((x, y + 14, x + 42, y + 14), fill=_parse_color(color), width=5)
            draw.text((x + 52, y + 2), st, fill=(0, 0, 0), font=font_med)

    # axes geometry (match Stage2 panel proportions; push down if legend wraps)
    ax_left = x0 + 50
    ax_right = x0 + w - 20
    ax_top = max(y0 + 35, float(legend_bottom + 12))
    ax_bottom = y0 + h - 30
    ax_top = min(ax_top, ax_bottom - 120)

    # grid + axes
    grid_col = (205, 205, 205)
    ax_col = (153, 153, 153)
    draw.line((ax_left, ax_top, ax_left, ax_bottom), fill=ax_col, width=1)
    draw.line((ax_left, ax_bottom, ax_right, ax_bottom), fill=ax_col, width=1)

    tick_step = 1 if len(buckets) <= 10 else 2
    for q in buckets:
        x = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
        draw.line((x, ax_top, x, ax_bottom), fill=grid_col, width=1)
    for q in buckets[::tick_step]:
        x = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
        draw.line((x, ax_bottom, x, ax_bottom + 4), fill=ax_col, width=1)
        draw.text((x - 10, ax_bottom + 7), f"Q{q}", fill=(85, 85, 85), font=font_small)

    for tval in (0.0, 0.5, 1.0):
        y = ax_bottom - (float(tval) - float(y_vmin)) * ((ax_bottom - ax_top) / (float(y_vmax) - float(y_vmin)))
        draw.line((ax_left, y, ax_right, y), fill=grid_col, width=1)
        draw.line((ax_left - 4, y, ax_left, y), fill=ax_col, width=1)
        draw.text((x0 + 6, y - 7), f"{tval:.3g}", fill=(85, 85, 85), font=font_small)

    # lines
    for st, color, vals in series:
        pts: List[Tuple[float, float]] = []
        for i, q in enumerate(buckets):
            v = vals[i] if i < len(vals) else None
            if v is None or not math.isfinite(float(v)):
                continue
            x = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
            y = ax_bottom - (float(v) - float(y_vmin)) * ((ax_bottom - ax_top) / (float(y_vmax) - float(y_vmin)))
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=_parse_color(color), width=5)
        for x, y in pts:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=_parse_color(color))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return str(out_path)


def _plot_all_features_all_stats_2x2(
    *,
    dist: pd.DataFrame,
    n_quantiles: int,
    out_path: Path,
    stats: List[str],
    raw: Optional[pd.DataFrame] = None,
    bucket: Optional[pd.Series] = None,
) -> Optional[str]:
    """
    Plot a 2x2 grid (MAG/DIR/VOL/POS), overlaying multiple statistics as separate lines per feature.
    Legend is unified at the top (title area). Values are scaled to [0,1] for readability.
    """
    stats = [s for s in stats if s in DIST_STATS]
    if not stats:
        return None

    # Robust scaling ranges for raw-comparable stats: [1%, 99%] (clipped).
    robust_range: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if raw is not None and bucket is not None:
        b = bucket.reindex(raw.index)
        mask = b.notna()
        raw2 = raw.loc[mask]
        for feat in RAW_FEATURES:
            s = raw2[feat].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            vmin = float(s.quantile(0.01))
            vmax = float(s.quantile(0.99))
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
                vmin = float(s.min())
                vmax = float(s.max())
            robust_range[feat] = (vmin, vmax)

    def _normalize_0_1(values: List[Optional[float]]) -> List[Optional[float]]:
        finite = [v for v in values if v is not None and math.isfinite(v)]
        if not finite:
            return [None for _ in values]
        vmin = float(min(finite))
        vmax = float(max(finite))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            return [None if v is None or not math.isfinite(v) else 0.5 for v in values]
        denom = vmax - vmin
        out: List[Optional[float]] = []
        for v in values:
            if v is None or not math.isfinite(v):
                out.append(None)
            else:
                out.append((float(v) - vmin) / denom)
        return out

    def _scale_raw(v: Optional[float], feat: str) -> Optional[float]:
        if v is None or not math.isfinite(v):
            return None
        vmin, vmax = robust_range.get(feat, (None, None))
        if vmin is None or vmax is None or vmax <= vmin:
            return None
        x = (float(v) - float(vmin)) / (float(vmax) - float(vmin))
        return float(min(1.0, max(0.0, x)))

    y_pad = 0.05
    y_vmin, y_vmax = -y_pad, 1.0 + y_pad
    buckets = list(range(1, n_quantiles + 1))
    feature_titles = {"mag": "MAG", "dir": "DIR", "vol": "VOL", "pos": "POS"}
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#FF9DA6"]
    stat_colors = {st: palette[i % len(palette)] for i, st in enumerate(stats)}

    per_feat: Dict[str, List[Tuple[str, str, List[Optional[float]]]]] = {}
    for feat in RAW_FEATURES:
        sers: List[Tuple[str, str, List[Optional[float]]]] = []
        for st in stats:
            col = f"{feat}_{st}"
            if col not in dist.columns:
                continue
            raw_values = [_to_num(dist.loc[q, col]) for q in buckets]
            if feat in robust_range and st in ("mean", "median", "std", "q10", "q25", "q75", "q90"):
                values = [_scale_raw(v, feat) for v in raw_values]
                if all(v is None for v in values):
                    values = _normalize_0_1(raw_values)
            else:
                values = _normalize_0_1(raw_values)
            sers.append((st, stat_colors[st], values))
        per_feat[feat] = sers

    if not any(per_feat.get(f) for f in RAW_FEATURES):
        return None

    # Prefer matplotlib; fall back to PIL.
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")  # type: ignore[attr-defined]
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore[assignment]

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=False)
        ax_list = axes.ravel().tolist()
        for ax, feat in zip(ax_list, RAW_FEATURES):
            for st, color, vals in per_feat.get(feat, []):
                y = [np.nan if v is None else float(v) for v in vals]
                ax.plot(buckets, y, marker="o", markersize=4.5, linewidth=3.0, label=st, color=color)

            ax.set_xlim(0.5, n_quantiles + 0.5)
            ax.set_xticks(buckets)
            ax.set_xticklabels(
                [f"Q{b}" for b in buckets],
                rotation=(45 if len(buckets) > 10 else 0),
                ha=("right" if len(buckets) > 10 else "center"),
            )
            ax.set_ylim(float(y_vmin), float(y_vmax))
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_title(feature_titles.get(feat, feat))
            ax.set_xlabel("bucket (Q1..Qn)", fontsize=11)
            ax.set_ylabel("scaled (0-1)", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(True, which="major", axis="both", alpha=0.38, linewidth=1.0)

        # Unified legend (stats) in the former title area.
        uniq: Dict[str, object] = {}
        for ax in ax_list:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll and ll not in uniq:
                    uniq[ll] = hh
        labels = [st for st in stats if st in uniq]
        handles = [uniq[st] for st in labels]
        if handles:
            ncol = min(len(labels), 5)
            nrows = int(math.ceil(len(labels) / ncol)) if ncol else 1
            top = 0.84 - 0.05 * max(0, nrows - 1)
            top = float(min(0.84, max(0.70, top)))
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=ncol,
                frameon=False,
                fontsize=11,
                handlelength=2.2,
                columnspacing=1.8,
                labelspacing=0.9,
                handletextpad=0.7,
                borderaxespad=0.0,
            )
        else:
            top = 0.84

        fig.subplots_adjust(left=0.07, right=0.985, top=top, bottom=0.10, wspace=0.18, hspace=0.32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return str(out_path)

    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    def _font(size: int):
        for fp in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _parse_color(c: str) -> Tuple[int, int, int]:
        c = (c or "").strip()
        if c.startswith("#") and len(c) == 7:
            return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
        return (0, 0, 0)

    def _text_w(draw_: ImageDraw.ImageDraw, text: str, font) -> int:
        try:
            box = draw_.textbbox((0, 0), text, font=font)
            return int(box[2] - box[0])
        except Exception:
            try:
                w0, _h0 = draw_.textsize(text, font=font)
                return int(w0)
            except Exception:
                return int(len(text) * 12)

    # Match stage2_<stat>.png canvas size: 15x8 inches @ 160 DPI -> 2400x1280.
    width, height = 2400, 1280
    pad = 28
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_med = _font(18)
    font_small = _font(17)

    # Unified legend at the top (wrapped into columns).
    leg_x0 = pad + 10
    leg_y0 = pad + 8
    leg_x1 = width - pad - 10
    available_w = max(1, int(leg_x1 - leg_x0))
    item_ws = [int(42 + 10 + _text_w(draw, st, font_med) + 30) for st in stats]
    max_item_w = max(item_ws) if item_ws else 1
    max_cols = 5
    ncol = min(len(stats), max_cols, max(1, int(available_w // max_item_w)))
    col_w = float(available_w) / float(ncol) if ncol else float(available_w)
    line_h = 36
    items: List[Tuple[int, int, str, str]] = []
    for i, st in enumerate(stats):
        row = int(i // ncol) if ncol else 0
        col = int(i % ncol) if ncol else 0
        x = int(leg_x0 + col * col_w)
        y = int(leg_y0 + row * line_h)
        items.append((x, y, st, stat_colors[st]))

    legend_bottom = leg_y0
    if items:
        nrows = int(math.ceil(len(items) / ncol)) if ncol else 1
        box_left = int(leg_x0 - 10)
        box_top = int(leg_y0 - 8)
        box_right = int(leg_x1)
        box_bottom = int(leg_y0 + nrows * line_h + 6)
        legend_bottom = box_bottom
        draw.rectangle((box_left, box_top, box_right, box_bottom), fill=(255, 255, 255), outline=(220, 220, 220), width=1)
        for x, y, st, color in items:
            draw.line((x, y + 14, x + 42, y + 14), fill=_parse_color(color), width=5)
            draw.text((x + 52, y + 2), st, fill=(0, 0, 0), font=font_med)

    # Panel grid
    grid_gap = 18
    top = int(legend_bottom + 18)
    panel_w = int((width - 2 * pad - grid_gap) / 2)
    panel_h = int((height - pad - top - grid_gap) / 2)

    # Colors/axes styles
    grid_col = (205, 205, 205)
    ax_col = (153, 153, 153)
    tick_col = (85, 85, 85)

    def _draw_panel(*, x0: int, y0: int, w: int, h: int, feat: str, series: List[Tuple[str, str, List[Optional[float]]]]):
        # frame
        draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(221, 221, 221), width=1)
        # subtitle
        draw.text((x0 + 10, y0 + 8), feature_titles.get(feat, feat), fill=(0, 0, 0), font=font_med)

        ax_left = x0 + 58
        ax_right = x0 + w - 18
        ax_top = y0 + 42
        ax_bottom = y0 + h - 34
        if ax_bottom - ax_top < 120 or ax_right - ax_left < 200:
            return

        # axes
        draw.line((ax_left, ax_top, ax_left, ax_bottom), fill=ax_col, width=1)
        draw.line((ax_left, ax_bottom, ax_right, ax_bottom), fill=ax_col, width=1)

        # grid
        tick_step = 1 if len(buckets) <= 10 else 2
        for q in buckets:
            xi = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
            draw.line((xi, ax_top, xi, ax_bottom), fill=grid_col, width=1)
        for q in buckets[::tick_step]:
            xi = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
            draw.line((xi, ax_bottom, xi, ax_bottom + 4), fill=ax_col, width=1)
            draw.text((xi - 10, ax_bottom + 7), f"Q{q}", fill=tick_col, font=font_small)

        for tval in (0.0, 0.5, 1.0):
            yi = ax_bottom - (float(tval) - float(y_vmin)) * ((ax_bottom - ax_top) / (float(y_vmax) - float(y_vmin)))
            draw.line((ax_left, yi, ax_right, yi), fill=grid_col, width=1)
            draw.line((ax_left - 4, yi, ax_left, yi), fill=ax_col, width=1)

        # lines
        for _st, color, vals in series:
            pts: List[Tuple[float, float]] = []
            for i, q in enumerate(buckets):
                v = vals[i] if i < len(vals) else None
                if v is None or not math.isfinite(float(v)):
                    continue
                xi = ax_left + (buckets.index(q) * ((ax_right - ax_left) / (len(buckets) - 1)) if len(buckets) > 1 else (ax_right - ax_left) / 2)
                yi = ax_bottom - (float(v) - float(y_vmin)) * ((ax_bottom - ax_top) / (float(y_vmax) - float(y_vmin)))
                pts.append((xi, yi))
            if len(pts) >= 2:
                draw.line(pts, fill=_parse_color(color), width=5)
            for xi, yi in pts:
                draw.ellipse((xi - 3.5, yi - 3.5, xi + 3.5, yi + 3.5), fill=_parse_color(color))

    coords = [
        (pad, top, panel_w, panel_h),  # MAG
        (pad + panel_w + grid_gap, top, panel_w, panel_h),  # DIR
        (pad, top + panel_h + grid_gap, panel_w, panel_h),  # VOL
        (pad + panel_w + grid_gap, top + panel_h + grid_gap, panel_w, panel_h),  # POS
    ]
    for (x0, y0, w, h), feat in zip(coords, RAW_FEATURES):
        _draw_panel(x0=x0, y0=y0, w=w, h=h, feat=feat, series=per_feat.get(feat, []))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return str(out_path)


def _run_factor_py_if_missing(factor_ws: Path) -> None:
    factor_py = factor_ws / "factor.py"
    if not factor_py.exists():
        raise FileNotFoundError(f"Missing factor.py: {factor_py}")
    subprocess.check_call([sys.executable, str(factor_py.name)], cwd=str(factor_ws))


def _run_stage2_for_workspace(
    *, factor_ws: Path, display_name: str, args: argparse.Namespace, run_dir: Optional[Path]
) -> dict:
    """
    Run Stage2 for a single factor workspace and return a compact index row.
    Raises on unrecoverable errors.
    """
    factor_ws = factor_ws.expanduser()
    if not factor_ws.exists():
        raise FileNotFoundError(f"factor workspace does not exist: {factor_ws}")

    if args.out_dir:
        base = Path(args.out_dir)
    else:
        run_tag = run_dir.name if run_dir is not None else "manual"
        base = _results_root() / run_tag / "stage2"
    out_dir = base / _safe_name(display_name or factor_ws.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_py = factor_ws / "factor.py"
    factor_name, factor_expr = ("", "")
    if factor_py.exists():
        factor_name, factor_expr = _extract_from_factor_py(factor_py)
    if not factor_name:
        factor_name = display_name or factor_ws.name
    obs_desc, formulation = ("", "")
    if run_dir is not None:
        obs_desc, formulation = _extract_obs_context_from_run(run_dir=run_dir, factor_name=factor_name)
    definition = formulation or factor_expr or ""

    # Fill observation metadata for non-log-dir sources (OURS/GPT/alpha101 workspaces).
    obs_id = factor_name
    meta = _load_obs_meta_from_workspace(factor_ws)
    if meta.get("obs_id"):
        obs_id = meta["obs_id"]
    if not obs_desc:
        obs_desc = meta.get("obs_description", "")
    if not definition:
        definition = meta.get("definition", "")
    if not obs_desc:
        obs_desc = definition or f"Observation: {factor_name}"

    score_path = factor_ws / "result.h5"
    if not score_path.exists() and args.run_factor_if_missing:
        _run_factor_py_if_missing(factor_ws)
    if not score_path.exists():
        raise FileNotFoundError(f"Missing score file result.h5 in factor workspace: {score_path}")

    score = _safe_read_hdf(score_path, key="data")
    if isinstance(score, pd.DataFrame):
        if score.shape[1] == 1:
            score = score.iloc[:, 0]
        else:
            raise SystemExit(f"Unexpected DataFrame in result.h5 (expected Series or 1-col DF): {score_path}")
    score = _ensure_multiindex(score)  # type: ignore[assignment]

    ohlcv_path = factor_ws / "daily_pv.h5"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"Missing daily_pv.h5 in factor workspace: {ohlcv_path}")
    ohlcv = _safe_read_hdf(ohlcv_path, key="data")
    ohlcv = _ensure_multiindex(ohlcv)  # type: ignore[assignment]

    conf_path = Path(args.conf)
    if not conf_path.exists():
        raise FileNotFoundError(f"Config yaml does not exist: {conf_path}")
    segments = _parse_segments_from_conf(conf_path)
    if "train" not in segments or "valid" not in segments:
        raise SystemExit(f"Failed to parse train/valid segments from conf: {conf_path}")

    seg = _pick_split_range(segments, args.split)
    score_is = _filter_by_date(score, seg)  # type: ignore[arg-type]
    ohlcv_is = _filter_by_date(ohlcv, seg)  # type: ignore[arg-type]

    common_idx = score_is.index.intersection(ohlcv_is.index)
    score_is = score_is.reindex(common_idx)
    ohlcv_is = ohlcv_is.reindex(common_idx)

    raw = _compute_raw_features(ohlcv_is)

    expectations: Dict[str, Direction] = {f: "any" for f in RAW_FEATURES}
    for item in args.expect:
        if ":" not in item:
            raise SystemExit(f"Invalid --expect '{item}'. Use form feature:up|down|any")
        k, v = item.split(":", 1)
        k = k.strip()
        v = v.strip().lower()
        if k not in RAW_FEATURES:
            raise SystemExit(f"Invalid feature in --expect: {k}. Allowed: {', '.join(RAW_FEATURES)}")
        if v not in ("up", "down", "any"):
            raise SystemExit(f"Invalid direction in --expect: {v}. Use up|down|any")
        expectations[k] = v  # type: ignore[assignment]

    if args.quantile_mode == "cross_sectional":
        bucket = _assign_quantiles_cross_sectional(score_is, args.n_quantiles)
    else:
        bucket = _assign_quantiles_global(score_is, args.n_quantiles)

    dist = _distribution_table(raw, bucket, args.n_quantiles)
    dist_path = out_dir / "stage2_distributions.csv"
    dist.to_csv(dist_path)

    heuristic_passed, mono, pass_feats, fail_feats, notes = _monotonicity_check(
        dist,
        expectations=expectations,
        n_quantiles=args.n_quantiles,
        min_bucket_frac=args.min_bucket_frac,
        min_pass_features=args.min_pass_features,
    )

    plot_path_str = ""
    plot_paths: List[str] = []
    if args.plot or (args.plot_on_fail and not heuristic_passed):
        title = f"Stage2 distributions | factor={factor_name} | split={args.split} | quantile_mode={args.quantile_mode}"
        plot_stats = _parse_stats_csv_arg(args.plot_stats)
        for st in plot_stats:
            out_path = out_dir / f"stage2_{st}.png"
            try:
                maybe = _plot_distributions(
                    dist=dist,
                    expectations=expectations,
                    n_quantiles=args.n_quantiles,
                    title=title,
                    out_path=out_path,
                    stat=st,
                    raw=raw,
                    bucket=bucket,
                )
            except Exception as e:  # noqa: BLE001
                maybe = None
                notes.append(f"Plotting failed for stat={st}; skipping. error={e}")

            if maybe is None:
                continue
            plot_paths.append(maybe)
            alt = str(Path(maybe).with_name(f"{Path(maybe).stem}_4x1{Path(maybe).suffix}"))
            if Path(alt).exists():
                plot_paths.append(alt)
            alt2 = str(Path(maybe).with_name(f"{Path(maybe).stem}_1x4{Path(maybe).suffix}"))
            if Path(alt2).exists():
                plot_paths.append(alt2)
            if not plot_path_str:
                plot_path_str = maybe

        # Also emit per-feature plots overlaid with all stats.
        for feat in RAW_FEATURES:
            out_path = out_dir / f"stage2_{feat}_all_stats.png"
            try:
                maybe = _plot_feature_all_stats(
                    dist=dist,
                    n_quantiles=args.n_quantiles,
                    out_path=out_path,
                    feature=feat,
                    stats=plot_stats,
                    raw=raw,
                    bucket=bucket,
                )
            except Exception as e:  # noqa: BLE001
                maybe = None
                notes.append(f"Feature all-stats plotting failed for feature={feat}; skipping. error={e}")

            if maybe is not None:
                plot_paths.append(maybe)

        # A Figure-5-friendly plot: 2x2 (MAG/DIR/VOL/POS) with multiple stats overlaid per feature.
        fig5_stats = ["mean", "q10", "q90", "kurtosis", "skewness"]
        out_path = out_dir / "stage2_all_features_all_stats.png"
        try:
            maybe = _plot_all_features_all_stats_2x2(
                dist=dist,
                n_quantiles=args.n_quantiles,
                out_path=out_path,
                stats=fig5_stats,
                raw=raw,
                bucket=bucket,
            )
        except Exception as e:  # noqa: BLE001
            maybe = None
            notes.append(f"All-features all-stats plotting failed; skipping. error={e}")
        if maybe is not None:
            plot_paths.append(maybe)

    evidence_stats = _parse_stats_csv_arg(args.evidence_stats)
    evidence = _build_evidence_json(
        dist,
        expectations=expectations,
        n_quantiles=args.n_quantiles,
        notes=notes,
        stats_to_emit=evidence_stats,
    )
    evidence_path = out_dir / "stage2_evidence.json"
    with evidence_path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    llm_judgment: Optional[Dict[str, Any]] = None
    passed = heuristic_passed
    pass_source = "heuristic"
    if args.use_llm:
        try:
            from validation_agent import run_stage2_llm_judgment

            dist_summary = _build_distribution_summary(
                expectations=expectations,
                passed=heuristic_passed,
                pass_feats=pass_feats,
                fail_feats=fail_feats,
                notes=notes,
                n_quantiles=args.n_quantiles,
            )
            llm_judgment = run_stage2_llm_judgment(
                model=args.llm_model,
                formula_name=factor_name,
                definition=definition,
                polarity=args.polarity,
                obs_id=obs_id,
                obs_description=obs_desc or definition or f"Observation: {factor_name}",
                evidence_json=evidence,
                distribution_summary=dist_summary,
            )
            passed = str(llm_judgment.get("verdict", "")).upper() == "PASS"
            pass_source = "llm"
        except Exception as e:
            notes.append(f"LLM judgment failed; falling back to heuristic. error={e}")
            llm_judgment = _fallback_llm_judgment(
                error=e, heuristic_passed=heuristic_passed, evidence_json=evidence, distribution_summary=dist_summary
            )

    summary = Stage2Summary(
        factor_ws=str(factor_ws),
        factor_name=factor_name,
        factor_expr=factor_expr,
        definition=definition,
        polarity=args.polarity,
        obs_id=obs_id,
        obs_description=obs_desc,
        score_source=str(score_path),
        ohlcv_source=str(ohlcv_path),
        split=args.split,
        segments={k: (str(v.start.date()), str(v.end.date())) for k, v in segments.items()},
        n_quantiles=args.n_quantiles,
        quantile_mode=args.quantile_mode,
        expectations={k: str(v) for k, v in expectations.items()},
        min_bucket_frac=args.min_bucket_frac,
        min_pass_features=args.min_pass_features,
        pass_features=pass_feats,
        fail_features=fail_feats,
        passed=passed,
        pass_source=pass_source,
        heuristic_passed=heuristic_passed,
        notes=notes,
        monotonicity=mono,
        plot_path=plot_path_str,
        plot_paths=plot_paths,
        evidence_json_path=str(evidence_path),
        llm_judgment=llm_judgment,
    )

    summary_path = out_dir / "stage2_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(summary),
                "monotonicity": [asdict(m) for m in mono],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[Stage2] factor={factor_name} split={args.split} passed={passed} (source={pass_source})")
    print(f"[Stage2] wrote: {dist_path}")
    print(f"[Stage2] wrote: {summary_path}")
    print(f"[Stage2] wrote: {evidence_path}")
    if notes:
        for n in notes:
            print(f"[Stage2][note] {n}")

    return {
        "factor_name": factor_name,
        "workspace": str(factor_ws),
        "passed": bool(passed),
        "pass_source": pass_source,
        "llm_verdict": (str(llm_judgment.get("verdict")) if llm_judgment else ""),
        "pass_features": ",".join(pass_feats),
        "fail_features": ",".join(fail_feats),
        "out_dir": str(out_dir),
        "plot_path": plot_path_str,
    }


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
    ap.add_argument("--split", choices=["train", "valid", "train_valid"], default="train", help="Which IS split to use")
    ap.add_argument("--n-quantiles", type=int, default=10, help="Number of quantile buckets")
    ap.add_argument(
        "--quantile-mode",
        choices=["cross_sectional", "global"],
        default="cross_sectional",
        help="Bucket assignment mode: per-date cross-sectional (recommended) vs global",
    )
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        help="Expectation per raw feature. Repeatable. Example: --expect dir:down --expect pos:down --expect vol:up --expect mag:any",
    )
    ap.add_argument(
        "--min-bucket-frac",
        type=float,
        default=0.02,
        help="Minimum fraction of total samples required in each bucket (to avoid tiny tails)",
    )
    ap.add_argument(
        "--min-pass-features",
        type=int,
        default=2,
        help="Minimum number of raw features that must pass monotonicity for overall PASS",
    )
    ap.add_argument(
        "--polarity",
        choices=["higher_is_more_true", "lower_is_more_true"],
        default="higher_is_more_true",
        help="Polarity for Stage2 LLM judgment: does stronger observation correspond to higher or lower factor values?",
    )
    ap.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LLM to judge PASS/FAIL based on evidence_json (default: on). Disable with --no-use-llm.",
    )
    ap.add_argument(
        "--llm-model",
        default="",
        help="LLM model name (default: LLM_SETTINGS.chat_model when using AlphaAgent backend).",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory base. Default: /home/dgu/fin/AlphaAgent/results/<run>/stage2/<factor>/",
    )
    ap.add_argument(
        "--run-factor-if-missing",
        action="store_true",
        help="If result.h5 is missing, run factor.py inside factor workspace to generate it.",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Write plots visualizing per-bucket distribution stats (one image per stat; PNG).",
    )
    ap.add_argument(
        "--plot-stats",
        default="mean,std,q90,skewness,kurtosis",
        help=f"Comma-separated stats to plot (one image per stat). Allowed: {', '.join(DIST_STATS)}",
    )
    ap.add_argument(
        "--plot-on-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, write plots automatically when the heuristic monotonicity check fails (default: on).",
    )
    ap.add_argument(
        "--evidence-stats",
        default="mean,median,std,q90",
        help=f"Comma-separated stats to include in stage2_evidence.json. Allowed: {', '.join(DIST_STATS)}",
    )
    args = ap.parse_args()
    if not args.llm_model:
        try:
            from alphaagent.oai.llm_conf import LLM_SETTINGS  # type: ignore

            args.llm_model = LLM_SETTINGS.chat_model
        except Exception:
            args.llm_model = "gpt-4o-mini"

    run_dir: Optional[Path] = None
    targets: List[Tuple[str, Path]] = []

    if args.factor_ws:
        ws = Path(args.factor_ws).expanduser()
        targets = [(ws.name, ws)]
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
    for display_name, ws in targets:
        try:
            index_rows.append(
                _run_stage2_for_workspace(factor_ws=ws, display_name=display_name, args=args, run_dir=run_dir)
            )
        except Exception as e:
            # Record failures so we don't end up with:
            # - an empty out_dir folder, and
            # - a missing row in stage2_index.{csv,json}.
            if args.out_dir:
                base = Path(args.out_dir)
            else:
                run_tag = run_dir.name if run_dir is not None else "manual"
                base = _results_root() / run_tag / "stage2"

            out_dir = base / _safe_name(display_name or ws.name)
            out_dir.mkdir(parents=True, exist_ok=True)

            err = {
                "factor_name": display_name or ws.name,
                "workspace": str(Path(ws).expanduser()),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            try:
                with (out_dir / "stage2_error.json").open("w", encoding="utf-8") as f:
                    json.dump(err, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            index_rows.append(
                {
                    "factor_name": display_name or ws.name,
                    "workspace": str(Path(ws).expanduser()),
                    "passed": False,
                    "pass_source": "error",
                    "llm_verdict": "",
                    "pass_features": "",
                    "fail_features": "",
                    "out_dir": str(out_dir),
                    "plot_path": "",
                    "error": str(e),
                }
            )
            failed.append(f"{display_name}: {e}")

    if run_dir is not None:
        index_base = _results_root() / run_dir.name
        index_base.mkdir(parents=True, exist_ok=True)
        index_csv = index_base / "stage2_index.csv"
        index_json = index_base / "stage2_index.json"
        try:
            pd.DataFrame(index_rows).to_csv(index_csv, index=False)
            with index_json.open("w", encoding="utf-8") as f:
                json.dump(index_rows, f, ensure_ascii=False, indent=2)
            print(f"[Stage2] wrote run index: {index_csv}")
        except Exception:
            pass

    if failed:
        print("[Stage2] Some factors were skipped/failed:")
        for item in failed:
            print(f"- {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
