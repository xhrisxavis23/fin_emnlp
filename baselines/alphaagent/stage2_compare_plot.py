#!/usr/bin/env python3
"""
Overlay Stage2 distribution curves from multiple sources onto each OURS (formulas) plot.

This is designed as a post-processing step after `run_stage2_4ways.sh` finishes.
It finds OURS factor outputs under `results/formulas/**/stage2_distributions.csv` and
writes comparison PNGs next to them (one PNG per feature, per requested stat).

Overlay semantics
  - OURS: per-formula/per-workspace curve (the specific output directory).
  - Other sources: aggregated mean curve across all their available Stage2 outputs
    (one curve per source, per feature, per stat, per bucket).

Outputs
  - stage2_compare_<feature>_<stat>.png written into each OURS output directory that has a stage2_distributions.csv.
    (One PNG per feature: MAG/DIR/VOL/POS. Overlaid source curves; no median markers.)
  - stage2_compare_all_<stat>.png written into each OURS output directory (a 2x2 stitched view of MAG/DIR/VOL/POS).
  - stage2_compare_<feature>_<stat>.analysis.json written next to the outputs (heuristic interpretation; per feature+stat).
  - stage2_compare_<stat>.summary.json written per OURS directory (per-source analysis across MAG/DIR/VOL/POS).

Axes (compare PNGs)
  - X axis: quantile bucket index Q1..Qn of the factor score (Q1=lowest score, Qn=highest score).
    Interpretation depends on polarity:
      - higher_is_more_true: moving Q1→Qn corresponds to obs truth weak→strong.
      - lower_is_more_true: moving Q1→Qn corresponds to obs truth strong→weak.
  - Y axis: the requested distribution statistic (mean/std/q90/...) scaled to [0,1].
    - default (--scale-mode common): uses a shared range across sources, per feature+stat.
      For raw-compatible stats (mean/median/std/q10/q25/q75/q90), it prefers a feature-level range derived from q10/q90 columns.
      Otherwise it falls back to a shared robust range (default q05..q95) computed from the stat values.
    - legacy (--scale-mode per_series): min-max scales each source curve independently.

Paper mode (--paper)
  - Prefers Matplotlib and applies fixed ACM/KDD-ish typography via rcParams.
  - If Matplotlib is unavailable, falls back to PIL but still writes a high-res PNG (300 DPI metadata).
  - Also writes a PDF:
      - vector PDF when Matplotlib is available
      - raster PDF when falling back to PIL
"""

from __future__ import annotations

import argparse
import csv
import math
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FEATURES = ("MAG", "DIR", "VOL", "POS")
FEATURE_KEYS = {"MAG": "mag", "DIR": "dir", "VOL": "vol", "POS": "pos"}
DEFAULT_POLARITY = "higher_is_more_true"
COMPARE_MONO_PASS_THRESHOLD = 0.8
DEFAULT_SCALE_MODE = "common"  # common | per_series
DEFAULT_SCALE_Q_LOW = 0.05
DEFAULT_SCALE_Q_HIGH = 0.95
RAW_COMPATIBLE_STATS = ("mean", "median", "std", "q10", "q25", "q75", "q90")

PAPER_RCPARAMS = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "lines.markersize": 3.2,
}

# Single-column-ish figure size (inches).
PAPER_SINGLE_FIGSIZE = (3.35, 2.6)


@dataclass(frozen=True)
class DistTable:
    buckets: List[int]
    cols: Dict[str, List[Optional[float]]]


def _is_finite(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _find_latest_under(root: Path, filename: str) -> Optional[Path]:
    if not root.exists():
        return None
    direct = root / filename
    if direct.exists():
        return direct
    best: Tuple[Optional[Path], float] = (None, -1.0)
    for p in root.rglob(filename):
        try:
            mt = p.stat().st_mtime
        except Exception:
            continue
        if mt > best[1]:
            best = (p, mt)
    return best[0]


def _load_stage2_summary(root: Path) -> Dict[str, Any]:
    """
    Load the most relevant stage2_summary.json under a root.
    - If root/stage2_summary.json exists, use it.
    - Else pick the newest stage2_summary.json under root.
    Returns empty dict if not found / unreadable.
    """
    p = _find_latest_under(root, "stage2_summary.json")
    if not p:
        return {}
    obj = _read_json(p)
    if not isinstance(obj, dict):
        return {}
    obj["_stage2_summary_path"] = str(p)
    return obj


def _extract_expr_from_summary(summary: Dict[str, Any]) -> str:
    for k in ("factor_expr", "expr", "factor_expression"):
        v = summary.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_obs_description(summary: Dict[str, Any]) -> str:
    for k in ("obs_description", "hypothesis", "description"):
        v = summary.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_definition(summary: Dict[str, Any]) -> str:
    v = summary.get("definition")
    return v.strip() if isinstance(v, str) else ""


def _extract_polarity(summary: Dict[str, Any]) -> str:
    v = summary.get("polarity")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return DEFAULT_POLARITY


def _alpha101_expr_from_repo(alpha_id: int) -> str:
    """
    Best-effort extraction of the original Alpha101 formula string from `alpha101.py` comments.
    Example line:
      # Alpha#2     (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    """
    try:
        root = Path(__file__).resolve().parent
        p = root / "alpha101.py"
        if not p.exists():
            return ""
        pat = re.compile(rf"^\s*#\s*Alpha#{alpha_id}\b.*?\((.+)\)\s*$")
        for line in p.read_text(encoding="utf-8").splitlines():
            m = pat.match(line)
            if m:
                return m.group(1).strip()
    except Exception:
        return ""
    return ""


def _extract_expectations(summary: Dict[str, Any]) -> Dict[str, str]:
    """
    Stage2Summary typically stores expectations with lowercase keys: mag/dir/vol/pos.
    Values: "increasing" | "decreasing" | "any" | (sometimes other).
    """
    exp = summary.get("expectations")
    if not isinstance(exp, dict):
        return {}
    out: Dict[str, str] = {}
    for feat, key in FEATURE_KEYS.items():
        v = exp.get(key)
        if isinstance(v, str) and v.strip():
            out[feat] = v.strip().lower()
    return out


def _infer_focus_features(expr: str) -> Dict[str, Any]:
    """
    Heuristic "formula analysis": which raw inputs the expression touches.
    This is NOT a semantic proof; it's a best-effort signal for reporting.
    """
    e = expr.lower()
    has = {
        "close": "close" in e or "$close" in e,
        "open": "open" in e or "$open" in e,
        "high": "high" in e or "$high" in e,
        "low": "low" in e or "$low" in e,
        "volume": "volume" in e or "$volume" in e,
    }
    focuses: List[str] = []
    if has["volume"]:
        focuses.append("VOL")
    if has["open"]:
        focuses.append("DIR")
    if has["high"] or has["low"]:
        # high/low participates in both MAG and POS definitions.
        focuses.extend(["MAG", "POS"])
    # If it only touches close, it is likely a return/momentum-type factor.
    if not focuses and has["close"]:
        focuses.append("DIR")
    # Extract common window sizes (ints).
    windows = sorted({int(m.group(0)) for m in re.finditer(r"\b\d{1,3}\b", expr) if 1 <= int(m.group(0)) <= 252})
    # Flag common operators.
    ops = {
        "uses_delay": "delay(" in e,
        "uses_delta": "delta(" in e,
        "uses_ema": "ema(" in e,
        "uses_ts_mean": "ts_mean(" in e,
        "uses_ts_std": "ts_std(" in e,
        "uses_ts_max": "ts_max(" in e,
        "uses_ts_min": "ts_min(" in e,
        "uses_rank": "rank(" in e,
        "uses_abs": "abs(" in e,
    }
    return {
        "touched_inputs": has,
        "focus_features": list(dict.fromkeys(focuses)),
        "windows": windows,
        "operators": ops,
    }


def _describe_formula(expr: str) -> str:
    """
    Best-effort natural language description from the expression string.
    This is intentionally heuristic (for reporting), not a formal parser.
    """
    e = expr.lower()
    if not e.strip():
        return ""
    def _has(*ks: str) -> bool:
        return all(k in e for k in ks)

    if _has("abs(", "ema(", "ts_std("):
        return "normalized absolute deviation from EMA (|close-EMA| / TS_STD)"
    if _has("ts_zscore(", "ema("):
        return "z-scored deviation from EMA"
    if _has("ts_rank("):
        return "rolling rank of an input signal"
    if _has("rank(", "delta("):
        return "ranked short-horizon change (delta) signal"
    if _has("close", "low") and ("close - low" in e or "$close - $low" in e):
        return "distance of close above low (intraday recovery proxy)"
    if _has("high", "low") and ("high - low" in e or "$high - $low" in e):
        return "intraday range (high-low) based signal"
    if _has("volume", "ts_mean("):
        return "volume anomaly relative to its rolling mean"
    return "factor expression over OHLCV inputs"


def _leading_sign(expr: str) -> int:
    """
    Very rough sign heuristic:
    - returns -1 if expression clearly starts with a negative factor (e.g. '-1 *', '-(')
    - otherwise returns +1
    """
    e = expr.strip()
    if not e:
        return 1
    if e.startswith("-1") or e.startswith("-(") or e.startswith("- "):
        return -1
    if re.match(r"^\-\s*1\s*\*", e):
        return -1
    return 1


def _opp_dir(d: str) -> str:
    if d == "increasing":
        return "decreasing"
    if d == "decreasing":
        return "increasing"
    return d


def _expected_by_feature(
    *,
    expr: str,
    obs_description: str,
    polarity: str,
    expectations: Dict[str, str],
) -> Dict[str, Any]:
    """
    Infer expected movement of MAG/DIR/VOL/POS across factor-score quantiles.

    Returns per-feature expectations in the plot direction Q1->Qk (increasing quantile index).
    For "higher_is_more_true", Qk corresponds to "more true"; for "lower_is_more_true", invert.
    """
    sign = _leading_sign(expr)
    focus = _infer_focus_features(expr)
    text = f"{obs_description}\n{expr}".lower()

    # Base: use explicit expectations from Stage2Summary if present.
    out: Dict[str, Any] = {}
    for feat in FEATURES:
        base = expectations.get(feat, "any")
        out[feat] = {
            "direction_q1_to_qk": base if base in ("increasing", "decreasing", "any") else "any",
            "direction_qk_to_q1": _opp_dir(base) if base in ("increasing", "decreasing") else base,
            "confidence": 0.2 if base != "any" else 0.0,
            "rationale": (["Stage2Summary.expectations"] if base != "any" else []),
            "should_separate": base in ("increasing", "decreasing"),
            "pattern_note": "",
        }

    # Keyword heuristics from obs_description (only apply if not already specified).
    def maybe_set(feat: str, dir_q1_to_qk: str, why: str, conf: float):
        if out[feat]["direction_q1_to_qk"] != "any":
            return
        out[feat]["direction_q1_to_qk"] = dir_q1_to_qk
        out[feat]["direction_qk_to_q1"] = _opp_dir(dir_q1_to_qk)
        out[feat]["confidence"] = max(float(out[feat]["confidence"]), conf)
        out[feat]["rationale"].append(why)
        out[feat]["should_separate"] = True

    def maybe_mark_uncertain(feat: str, why: str, conf: float):
        if out[feat]["direction_q1_to_qk"] != "any":
            return
        out[feat]["direction_q1_to_qk"] = "uncertain"
        out[feat]["direction_qk_to_q1"] = "uncertain"
        out[feat]["confidence"] = max(float(out[feat]["confidence"]), conf)
        out[feat]["rationale"].append(why)
        out[feat]["should_separate"] = True

    # Stabilization / range narrowing -> MAG down, POS up, DIR up (usually).
    if any(k in text for k in ("stabilization", "stabilise", "stabilize", "narrowing", "range contraction", "reduction in selling pressure")):
        maybe_set("MAG", "decreasing", "obs_description suggests range narrowing/stabilization → lower MAG in high-score bins", 0.55)
        maybe_set("POS", "increasing", "obs_description suggests closes shift upward within range → higher POS in high-score bins", 0.45)
        maybe_set("DIR", "increasing", "obs_description suggests less negative / more positive intraday direction", 0.35)

    # Intraday recovery / close near high / upper end.
    if any(k in text for k in ("intraday recovery", "upper end", "close near", "closing strength", "close near the upper", "close near high")):
        maybe_set("POS", "increasing", "obs_description suggests close near high/upper range → POS increases with score", 0.6)
        maybe_set("DIR", "increasing", "obs_description suggests upward intraday move → DIR increases with score", 0.45)

    # Volume language.
    if any(k in text for k in ("volume spike", "high volume", "surge in volume", "volume surge")):
        maybe_set("VOL", "increasing", "obs_description suggests higher volume in high-score bins → VOL increases with score", 0.6)
    if any(k in text for k in ("low volume", "thin volume", "volume dries", "dry up")):
        maybe_set("VOL", "decreasing", "obs_description suggests lower volume in high-score bins → VOL decreases with score", 0.55)

    # If the formula directly touches a raw feature strongly, at least expect separation.
    touched = focus.get("touched_inputs", {})
    if touched.get("volume") and out["VOL"]["direction_q1_to_qk"] == "any":
        # If leading sign is negative, invert the natural assumption.
        dir0 = "decreasing" if sign < 0 else "increasing"
        maybe_set("VOL", dir0, "factor_expr directly uses volume → VOL should separate monotonically across score bins", 0.45)
    if (touched.get("high") or touched.get("low")) and out["MAG"]["direction_q1_to_qk"] == "any":
        # high/low could affect range (MAG) and position (POS). Direction depends on whether it's in numerator/denominator.
        if " / (ts_max(" in text or "/(ts_max(" in text or "/ (high - low" in text or "/(high - low" in text:
            maybe_set("MAG", "decreasing", "factor_expr uses high/low in denominator → larger ranges reduce score → MAG expected lower in high-score bins", 0.35)
        elif "high - low" in text or "$high - $low" in text:
            dir0 = "decreasing" if sign < 0 else "increasing"
            maybe_set("MAG", dir0, "factor_expr directly uses (high-low) → MAG should separate monotonically across score bins", 0.35)
        else:
            maybe_mark_uncertain("MAG", "factor_expr touches high/low → expect MAG separation but direction is ambiguous", 0.25)
    if touched.get("open") and out["DIR"]["direction_q1_to_qk"] == "any":
        dir0 = "decreasing" if sign < 0 else "increasing"
        maybe_set("DIR", dir0, "factor_expr directly uses open/close relationship → DIR likely separates monotonically across score bins", 0.35)

    # More specific structure cues.
    if out["POS"]["direction_q1_to_qk"] == "any" and ("(close - low)" in text or "close - low" in text or "$close - $low" in text):
        dir0 = "decreasing" if sign < 0 else "increasing"
        maybe_set("POS", dir0, "factor_expr uses (close-low) → higher score implies close higher within range → POS should increase with score", 0.45)
    if out["DIR"]["direction_q1_to_qk"] == "any" and ("close - open" in text or "$close - $open" in text):
        dir0 = "decreasing" if sign < 0 else "increasing"
        maybe_set("DIR", dir0, "factor_expr uses (close-open) → DIR should increase with score", 0.5)
    if out["DIR"]["direction_q1_to_qk"] == "any" and ("delta(" in text or "sign(" in text):
        maybe_mark_uncertain("DIR", "factor_expr contains delta/sign of returns → expect DIR-related separation but direction can be model-dependent", 0.25)

    # Polarity inversion: if lower_is_more_true, swap expected directions in plot direction.
    if polarity.strip() == "lower_is_more_true":
        for feat in FEATURES:
            dq = out[feat]["direction_q1_to_qk"]
            if dq in ("increasing", "decreasing"):
                out[feat]["direction_q1_to_qk"] = _opp_dir(dq)
                out[feat]["direction_qk_to_q1"] = _opp_dir(out[feat]["direction_q1_to_qk"])
                out[feat]["rationale"].append("polarity=lower_is_more_true → invert expected direction")
            # Keep 'uncertain' as uncertain.

    return {
        "by_feature": out,
        "analysis_inputs": {
            "leading_sign": sign,
            "focus": focus,
            "formula_description": _describe_formula(expr),
        },
    }


def _alignment_grade(expected_dir: str, observed_best_dir: Optional[str], observed_passed: Optional[bool]) -> str:
    if expected_dir in ("any", "uncertain", ""):
        return "unknown"
    if not observed_best_dir:
        return "unknown"
    if expected_dir != observed_best_dir:
        return "mismatch"
    # same direction
    if observed_passed is True:
        return "match_strong"
    return "match_weak"


def _series_minmax(arr: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in arr if _is_finite(v)]
    if not vals:
        return None, None
    mn = float(min(vals))
    mx = float(max(vals))
    if not math.isfinite(mn) or not math.isfinite(mx):
        return None, None
    return mn, mx


def _normalize_0_1(
    arr: List[Optional[float]],
    *,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    clip: bool = False,
) -> Tuple[List[Optional[float]], Optional[float], Optional[float]]:
    mn, mx = (scale_min, scale_max) if (scale_min is not None and scale_max is not None) else _series_minmax(arr)
    if mn is None or mx is None:
        return [None for _ in arr], None, None
    if mx <= mn:
        out = [0.5 if _is_finite(v) else None for v in arr]
        return out, mn, mx
    out: List[Optional[float]] = []
    denom = mx - mn
    for v in arr:
        if not _is_finite(v):
            out.append(None)
        else:
            x = (float(v) - mn) / denom
            if clip:
                x = float(min(1.0, max(0.0, x)))
            out.append(x)
    return out, mn, mx


def _quantile_sorted(vals_sorted: List[float], q: float) -> Optional[float]:
    if not vals_sorted:
        return None
    if q <= 0:
        return float(vals_sorted[0])
    if q >= 1:
        return float(vals_sorted[-1])
    pos = float(q) * (len(vals_sorted) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals_sorted[lo])
    w = pos - lo
    return float(vals_sorted[lo] * (1.0 - w) + vals_sorted[hi] * w)


def _robust_minmax(values: Iterable[Optional[float]], q_low: float, q_high: float) -> Tuple[Optional[float], Optional[float]]:
    vals = sorted(float(v) for v in values if _is_finite(v))
    if not vals:
        return None, None
    lo = _quantile_sorted(vals, q_low)
    hi = _quantile_sorted(vals, q_high)
    if lo is None or hi is None:
        return None, None
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
        return None, None
    if float(hi) <= float(lo):
        return float(vals[0]), float(vals[-1])
    return float(lo), float(hi)


def _mean_abs_diff(a: List[Optional[float]], b: List[Optional[float]]) -> Optional[float]:
    diffs: List[float] = []
    for x, y in zip(a, b):
        if _is_finite(x) and _is_finite(y):
            diffs.append(abs(float(x) - float(y)))
    if not diffs:
        return None
    return float(sum(diffs) / len(diffs))


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return float(math.sqrt(v))


def _delta_std(arr: List[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in arr if _is_finite(v)]
    if len(finite) < 3:
        return None
    deltas = [b - a for a, b in zip(finite, finite[1:])]
    return _std(deltas)


def _spearman_rho(a: List[Optional[float]], b: List[Optional[float]]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for x, y in zip(a, b):
        if _is_finite(x) and _is_finite(y):
            pairs.append((float(x), float(y)))
    if len(pairs) < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def rank(vals: List[float]) -> List[float]:
        # Average rank for ties (1-based ranks).
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    xr = rank(xs)
    yr = rank(ys)

    mx = sum(xr) / len(xr)
    my = sum(yr) / len(yr)
    vx = sum((x - mx) ** 2 for x in xr)
    vy = sum((y - my) ** 2 for y in yr)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xr, yr))
    return float(cov / math.sqrt(vx * vy))


def _pattern(arr: List[Optional[float]]) -> str:
    finite = [float(v) for v in arr if _is_finite(v)]
    if len(finite) < 2:
        return "flat"
    # Use endpoints for a coarse direction.
    if finite[-1] > finite[0]:
        direction = "increasing"
    elif finite[-1] < finite[0]:
        direction = "decreasing"
    else:
        direction = "flat"
    # Check step consistency.
    ok = 0
    total = 0
    for a, b in zip(finite, finite[1:]):
        total += 1
        if direction == "increasing":
            ok += 1 if b >= a else 0
        elif direction == "decreasing":
            ok += 1 if b <= a else 0
    if total == 0:
        return direction
    sc = ok / total
    if direction in ("increasing", "decreasing") and sc < 0.75:
        return "non-monotonic"
    return direction


def _monotonic_metrics(arr: List[Optional[float]], *, pass_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Compute simple monotonicity diagnostics on an ordered bucket sequence.

    - direction: inferred from endpoints on the finite subsequence
    - step_consistency: fraction of consecutive steps consistent with direction
    - spearman_bucket: Spearman rho between bucket index and values
    - passed: step_consistency >= pass_threshold (and direction not flat)
    """
    finite_idx: List[int] = []
    finite_vals: List[float] = []
    for i, v in enumerate(arr):
        if _is_finite(v):
            finite_idx.append(i + 1)
            finite_vals.append(float(v))  # type: ignore[arg-type]
    if len(finite_vals) < 2:
        return {
            "direction": "flat",
            "step_consistency": None,
            "spearman_bucket": None,
            "passed": False,
            "pass_threshold": pass_threshold,
        }

    first = finite_vals[0]
    last = finite_vals[-1]
    if last > first:
        direction = "increasing"
    elif last < first:
        direction = "decreasing"
    else:
        direction = "flat"

    ok = 0
    total = 0
    if direction in ("increasing", "decreasing"):
        for a, b in zip(finite_vals, finite_vals[1:]):
            total += 1
            if direction == "increasing":
                ok += 1 if b >= a else 0
            else:
                ok += 1 if b <= a else 0
    step_consistency = (ok / total) if total else None
    rho = _spearman_rho([float(x) for x in finite_idx], finite_vals)
    passed = bool(step_consistency is not None and step_consistency >= pass_threshold and direction != "flat")
    return {
        "direction": direction,
        "step_consistency": step_consistency,
        "spearman_bucket": rho,
        "passed": passed,
        "pass_threshold": pass_threshold,
    }


def _step_consistency_any(arr: List[Optional[float]]) -> Dict[str, Any]:
    """
    Monotonicity score without assuming direction from endpoints.
    Returns the better of (increasing, decreasing) by step_consistency.
    """
    finite = [float(v) for v in arr if _is_finite(v)]
    if len(finite) < 2:
        return {
            "best_direction": "flat",
            "best_step_consistency": None,
            "step_consistency_increasing": None,
            "step_consistency_decreasing": None,
        }

    def sc_for(dir_: str) -> float:
        ok = 0
        total = 0
        for a, b in zip(finite, finite[1:]):
            total += 1
            if dir_ == "increasing":
                ok += 1 if b >= a else 0
            else:
                ok += 1 if b <= a else 0
        return ok / total if total else 0.0

    sc_up = sc_for("increasing")
    sc_dn = sc_for("decreasing")
    if sc_up > sc_dn:
        return {
            "best_direction": "increasing",
            "best_step_consistency": sc_up,
            "step_consistency_increasing": sc_up,
            "step_consistency_decreasing": sc_dn,
        }
    if sc_dn > sc_up:
        return {
            "best_direction": "decreasing",
            "best_step_consistency": sc_dn,
            "step_consistency_increasing": sc_up,
            "step_consistency_decreasing": sc_dn,
        }
    # Tie: call it flat-ish; keep a direction for reporting.
    return {
        "best_direction": "increasing",
        "best_step_consistency": sc_up,
        "step_consistency_increasing": sc_up,
        "step_consistency_decreasing": sc_dn,
    }


def _shape_class_from_diffs(diffs: List[float]) -> str:
    nz = [d for d in diffs if d != 0]
    if not nz:
        return "flat"
    signs = [1 if d > 0 else -1 for d in nz]
    sign_changes = 0
    for a, b in zip(signs, signs[1:]):
        if a != b:
            sign_changes += 1
    if sign_changes == 0:
        return "monotonic_increasing" if signs[0] > 0 else "monotonic_decreasing"
    if sign_changes == 1:
        return "u_shape" if (signs[0] < 0 and signs[-1] > 0) else "inverted_u"
    if sign_changes == 2:
        return "w_shape_or_m_shape"
    return "zigzag"


def _curve_diagnostics(
    raw: List[Optional[float]],
    *,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    scale_basis: str = "per_series",
    scale_clip: bool = False,
) -> Dict[str, Any]:
    """
    Rich diagnostics for a bucket curve.
    Includes both raw and scaled (0-1) summaries.
    """
    raw_min, raw_max = _series_minmax(raw)
    scaled, used_min, used_max = _normalize_0_1(raw, scale_min=scale_min, scale_max=scale_max, clip=scale_clip)

    finite_scaled = [float(v) for v in scaled if _is_finite(v)]
    sign_changes = 0
    max_abs_step = None
    shape_class = "flat"
    diffs_scaled: List[float] = []
    if len(finite_scaled) >= 3:
        diffs_scaled = [b - a for a, b in zip(finite_scaled, finite_scaled[1:])]
        max_abs_step = max(abs(d) for d in diffs_scaled) if diffs_scaled else None
        # Count sign changes in diffs (ignoring zeros).
        nz = [d for d in diffs_scaled if d != 0]
        for a, b in zip(nz, nz[1:]):
            if (a > 0 and b < 0) or (a < 0 and b > 0):
                sign_changes += 1
        shape_class = _shape_class_from_diffs(diffs_scaled)
    elif len(finite_scaled) >= 2:
        diffs_scaled = [b - a for a, b in zip(finite_scaled, finite_scaled[1:])]
        shape_class = _shape_class_from_diffs(diffs_scaled)

    mono_ep = _monotonic_metrics(scaled, pass_threshold=COMPARE_MONO_PASS_THRESHOLD)
    mono_best = _step_consistency_any(scaled)
    # Prefer best-direction monotonicity for compare-plot evaluation, but keep endpoint direction too.
    best_dir = mono_best.get("best_direction")
    best_sc = mono_best.get("best_step_consistency")
    passed_best = bool(best_sc is not None and float(best_sc) >= COMPARE_MONO_PASS_THRESHOLD and best_dir != "flat")

    # Endpoints in both spaces (for citations).
    def endpoints(arr: List[Optional[float]]) -> Dict[str, Optional[float]]:
        finite = [float(v) for v in arr if _is_finite(v)]
        if len(finite) < 2:
            return {"q1": None, "qk": None}
        return {"q1": finite[0], "qk": finite[-1]}

    ep_raw = endpoints(raw)
    ep_scaled = endpoints(scaled)
    delta_scaled = None
    if ep_scaled.get("q1") is not None and ep_scaled.get("qk") is not None:
        delta_scaled = float(ep_scaled["qk"]) - float(ep_scaled["q1"])
    delta_raw = None
    if ep_raw.get("q1") is not None and ep_raw.get("qk") is not None:
        delta_raw = float(ep_raw["qk"]) - float(ep_raw["q1"])

    return {
        "raw": raw,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_endpoints": ep_raw,
        "raw_endpoint_delta": delta_raw,
        "scale_min": used_min,
        "scale_max": used_max,
        "scale_basis": scale_basis,
        "scale_clip": bool(scale_clip),
        "scaled_0_1": scaled,
        "scaled_endpoints": ep_scaled,
        "scaled_endpoint_delta": delta_scaled,
        "scaled_diffs": diffs_scaled,
        "pattern_scaled": _pattern(scaled),
        "shape_class_scaled": shape_class,
        "turbulence_scaled": _delta_std(scaled),
        "max_abs_step_scaled": max_abs_step,
        "sign_changes_scaled": sign_changes,
        "monotonicity_scaled": {
            "endpoint_rule": mono_ep,
            "best_direction": best_dir,
            "best_step_consistency": best_sc,
            "step_consistency_increasing": mono_best.get("step_consistency_increasing"),
            "step_consistency_decreasing": mono_best.get("step_consistency_decreasing"),
            "passed_best": passed_best,
            "pass_threshold": COMPARE_MONO_PASS_THRESHOLD,
        },
    }


def _compact_formula_summary(meta: Dict[str, Any]) -> str:
    name = _safe_str(meta.get("factor_name") or meta.get("obs_id") or "")
    expr = _safe_str(meta.get("factor_expr") or "")
    if expr:
        expr = re.sub(r"\s+", " ", expr).strip()
        if len(expr) > 160:
            expr = expr[:157] + "..."
    pol = _safe_str(meta.get("polarity") or DEFAULT_POLARITY)
    if name and expr:
        return f"{name} | polarity={pol} | expr={expr}"
    if expr:
        return f"polarity={pol} | expr={expr}"
    return f"{name} | polarity={pol}".strip(" |")


def _build_stat_context(
    *,
    ours_dir: Path,
    buckets: List[int],
    stat: str,
    ours_tab: DistTable,
    other_tabs: Dict[str, DistTable],
    source_roots: Dict[str, Path],
    scale_mode: str,
    scale_q_low: float,
    scale_q_high: float,
) -> Dict[str, Any]:
    """
    Build the full analysis context the user asked for:
      (1) per-source formula analysis
      (2) expected MAG/DIR/VOL/POS behavior across quantiles
      (3) observed curve direction/monotonicity per source
      (4) synthesis + alignment to expectations
    """
    sources_tabs: Dict[str, DistTable] = {"ours": ours_tab, **other_tabs}

    scaling: Dict[str, Any] = {
        "mode": scale_mode,
        "q_low": scale_q_low,
        "q_high": scale_q_high,
        "per_feature": {},
    }
    if scale_mode == "common":
        for feat in FEATURES:
            mn: Optional[float] = None
            mx: Optional[float] = None
            basis = ""

            # Stage2-style scaling: for raw-compatible stats, use a feature-level robust range
            # (independent of stat), then clip to [0,1]. We approximate the raw-feature range
            # using per-bucket q10/q90 columns when available.
            if stat in RAW_COMPATIBLE_STATS:
                lo_col = f"{feat.lower()}_q10"
                hi_col = f"{feat.lower()}_q90"
                lo_vals: List[float] = []
                hi_vals: List[float] = []
                for _src, tab in sources_tabs.items():
                    a0 = tab.cols.get(lo_col)
                    a1 = tab.cols.get(hi_col)
                    if isinstance(a0, list):
                        lo_vals.extend(float(v) for v in a0 if _is_finite(v))
                    if isinstance(a1, list):
                        hi_vals.extend(float(v) for v in a1 if _is_finite(v))
                if lo_vals and hi_vals:
                    mn = float(min(lo_vals))
                    mx = float(max(hi_vals))
                    basis = "feature_range_from_q10_q90"

            # Fallback: derive a shared robust range from the stat values directly.
            if mn is None or mx is None or not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
                col = f"{feat.lower()}_{stat}"
                vals: List[Optional[float]] = []
                for _src, tab in sources_tabs.items():
                    arr = tab.cols.get(col)
                    if isinstance(arr, list):
                        vals.extend(arr)
                mn, mx = _robust_minmax(vals, scale_q_low, scale_q_high)
                basis = f"common_range(q{int(scale_q_low*100):02d},q{int(scale_q_high*100):02d})"

            scaling["per_feature"][feat] = {"min": mn, "max": mx, "basis": basis, "clip": True}

    sources: Dict[str, Any] = {}
    for src, tab in sources_tabs.items():
        root = ours_dir if src == "ours" else source_roots.get(src, Path("."))
        s2 = _load_stage2_summary(root)

        expr = _extract_expr_from_summary(s2)
        obs = _extract_obs_description(s2)
        definition = _extract_definition(s2)
        polarity = _extract_polarity(s2)
        expectations = _extract_expectations(s2)
        factor_name = _safe_str(s2.get("factor_name") or "")

        if not expr and factor_name.startswith("Alpha101_alpha"):
            m = re.search(r"Alpha101_alpha(\d{3})", factor_name)
            if m:
                try:
                    alpha_id = int(m.group(1))
                except Exception:
                    alpha_id = -1
                if alpha_id >= 0:
                    expr = _alpha101_expr_from_repo(alpha_id)

        formula_analysis = _infer_focus_features(expr) if expr else {"touched_inputs": {}, "focus_features": [], "windows": [], "operators": {}}
        expected_pack = _expected_by_feature(expr=expr, obs_description=obs, polarity=polarity, expectations=expectations)
        expected = expected_pack["by_feature"]

        # Observed per-feature diagnostics (this stat).
        features: Dict[str, Any] = {}
        alignment: Dict[str, Any] = {}
        pass_feats: List[str] = []
        for feat in FEATURES:
            col = f"{feat.lower()}_{stat}"
            raw = tab.cols.get(col)
            if raw is None:
                features[feat] = {"missing": True}
                alignment[feat] = {"alignment": "unknown", "missing": True}
                continue
            srange = scaling.get("per_feature", {}).get(feat, {}) if isinstance(scaling.get("per_feature"), dict) else {}
            smin = srange.get("min") if isinstance(srange, dict) else None
            smax = srange.get("max") if isinstance(srange, dict) else None
            sbasis = srange.get("basis") if isinstance(srange, dict) else None
            if scale_mode == "common" and isinstance(smin, (int, float)) and isinstance(smax, (int, float)):
                diag = _curve_diagnostics(
                    raw, scale_min=float(smin), scale_max=float(smax), scale_basis=str(sbasis or "common"), scale_clip=True
                )
            else:
                diag = _curve_diagnostics(raw, scale_basis="per_series")
            diag["missing"] = False
            features[feat] = diag

            mono = diag.get("monotonicity_scaled", {}) if isinstance(diag, dict) else {}
            obs_best = mono.get("best_direction")
            obs_sc = mono.get("best_step_consistency")
            obs_pass = mono.get("passed_best")
            if obs_pass is True:
                pass_feats.append(feat)

            exp_dir = _safe_str(expected.get(feat, {}).get("direction_q1_to_qk") or "any")
            grade = _alignment_grade(exp_dir, obs_best, obs_pass)
            alignment[feat] = {
                "expected_direction_q1_to_qk": exp_dir,
                "expected_direction_qk_to_q1": _safe_str(expected.get(feat, {}).get("direction_qk_to_q1") or exp_dir),
                "observed_best_direction_q1_to_qk": obs_best,
                "observed_best_direction_qk_to_q1": _opp_dir(obs_best) if isinstance(obs_best, str) else None,
                "observed_step_consistency": obs_sc,
                "observed_passed_monotonic": obs_pass,
                "alignment": grade,
            }

        sources[src] = {
            "meta": {
                "factor_name": factor_name,
                "factor_expr": expr,
                "definition": definition,
                "polarity": polarity,
                "obs_description": obs,
                "expectations": expectations,
                "stage2_summary_path": _safe_str(s2.get("_stage2_summary_path")),
            },
            "formula_analysis": formula_analysis,
            "expected_movements": expected,
            "expected_analysis_inputs": expected_pack.get("analysis_inputs", {}),
            "features": features,  # observed
            "alignment": alignment,
            "summary": {
                "passed_monotonic_features": pass_feats,
                "n_passed_monotonic_features": len(pass_feats),
            },
        }

    # Pairwise vs OURS (shape similarity in this stat).
    pairwise: Dict[str, Any] = {}
    ours_feats = sources.get("ours", {}).get("features", {})
    for src, smeta in sources.items():
        if src == "ours":
            continue
        pf: Dict[str, Any] = {}
        spears: List[float] = []
        mads: List[float] = []
        for feat in FEATURES:
            a = ours_feats.get(feat, {}).get("scaled_0_1") if isinstance(ours_feats.get(feat), dict) else None
            b = smeta.get("features", {}).get(feat, {}).get("scaled_0_1") if isinstance(smeta.get("features", {}).get(feat), dict) else None
            if not isinstance(a, list) or not isinstance(b, list):
                continue
            rho = _spearman_rho(a, b)
            mad = _mean_abs_diff(a, b)
            pf[feat] = {"spearman_scaled": rho, "mean_abs_diff_scaled": mad}
            if rho is not None:
                spears.append(float(rho))
            if mad is not None:
                mads.append(float(mad))
        pairwise[src] = {
            "per_feature": pf,
            "avg_spearman_scaled": (sum(spears) / len(spears)) if spears else None,
            "avg_mean_abs_diff_scaled": (sum(mads) / len(mads)) if mads else None,
        }

    # Synthesis: per-source alignment counts and turbulence ranking per feature.
    synthesis: Dict[str, Any] = {"per_source": {}, "per_feature": {}}
    for src, s in sources.items():
        aln = s.get("alignment", {})
        match_strong = sum(1 for v in aln.values() if isinstance(v, dict) and v.get("alignment") == "match_strong")
        match_weak = sum(1 for v in aln.values() if isinstance(v, dict) and v.get("alignment") == "match_weak")
        mismatch = sum(1 for v in aln.values() if isinstance(v, dict) and v.get("alignment") == "mismatch")
        known = sum(1 for v in aln.values() if isinstance(v, dict) and v.get("alignment") in ("match_strong", "match_weak", "mismatch"))
        synthesis["per_source"][src] = {
            "alignment_counts": {
                "match_strong": match_strong,
                "match_weak": match_weak,
                "mismatch": mismatch,
                "known": known,
            }
        }

    for feat in FEATURES:
        turb_list: List[Tuple[str, float]] = []
        for src, s in sources.items():
            t = s.get("features", {}).get(feat, {}).get("turbulence_scaled")
            if t is not None:
                try:
                    turb_list.append((src, float(t)))
                except Exception:
                    pass
        turb_list.sort(key=lambda x: x[1], reverse=True)
        synthesis["per_feature"][feat] = {"turbulence_rank": turb_list}

    # Build structured reasoning sections (as strings) to make reading easier.
    formula_lines: List[str] = []
    expected_lines: List[str] = []
    observed_lines: List[str] = []
    synth_lines: List[str] = []

    formula_lines.append(f"[1/4] Formula analysis (stat={stat}, buckets={len(buckets)})")
    for src, s in sources.items():
        meta = s.get("meta", {})
        fa = s.get("formula_analysis", {})
        desc = _safe_str((s.get("expected_analysis_inputs", {}) or {}).get("formula_description"))
        formula_lines.append(
            f"- {src}: {_compact_formula_summary(meta)} | focus={fa.get('focus_features')} | windows={fa.get('windows')} | desc={desc or 'n/a'}"
        )

    expected_lines.append("[2/4] Expected movements (interpreting 'quantile decreases' as Qk→Q1)")
    expected_lines.append(
        "- Note: buckets are ordered Q1→Qk on the x-axis (Q1 lowest factor score, Qk highest); 'Qk→Q1' directions are the inverse."
    )
    expected_lines.append("- Convention: a trailing '*' means we expect monotonic separation in that feature, even if direction is uncertain.")
    for src, s in sources.items():
        em = s.get("expected_movements", {})
        parts = []
        for feat in FEATURES:
            d = em.get(feat, {})
            tag = d.get("direction_qk_to_q1", "any")
            sep = d.get("should_separate", False)
            parts.append(f"{feat}={tag}{'*' if sep else ''}")
        expected_lines.append(f"- {src}: " + ", ".join(parts))

    if scale_mode == "common":
        observed_lines.append("[3/4] Observed curve movements (from plotted curves; shared-range scaled 0–1)")
        observed_lines.append(
            "- Axes: x=bucket(Q1..Qk; quantiles of the factor score), y=scaled value(0..1; common robust range shared across sources per feature+stat)."
        )
    else:
        observed_lines.append("[3/4] Observed curve movements (from plotted curves; per-series scaled 0–1)")
        observed_lines.append(
            "- Axes: x=bucket(Q1..Qk; quantiles of the factor score), y=scaled value(0..1; min-max per series across buckets)."
        )
    for feat in FEATURES:
        parts = []
        for src, s in sources.items():
            fm = s.get("features", {}).get(feat, {})
            if not isinstance(fm, dict) or fm.get("missing") is True:
                parts.append(f"{src}=n/a")
                continue
            mono = fm.get("monotonicity_scaled", {}) or {}
            parts.append(
                f"{src}:{mono.get('best_direction')}, sc={None if mono.get('best_step_consistency') is None else round(float(mono.get('best_step_consistency')),3)}, "
                f"pass={bool(mono.get('passed_best'))}, shape={fm.get('shape_class_scaled')}, turb={None if fm.get('turbulence_scaled') is None else round(float(fm.get('turbulence_scaled')),3)}"
            )
        observed_lines.append(f"- {feat}: " + " | ".join(parts))

    synth_lines.append("[4/4] Synthesis (expectation vs observation + similarity to OURS)")
    for src in (k for k in sources.keys() if k != "ours"):
        aln = synthesis["per_source"].get(src, {}).get("alignment_counts", {})
        sim = pairwise.get(src, {})
        synth_lines.append(
            f"- {src}: alignment(match_strong={aln.get('match_strong',0)}, match_weak={aln.get('match_weak',0)}, mismatch={aln.get('mismatch',0)}, known={aln.get('known',0)}); "
            f"avg_spearman={None if sim.get('avg_spearman_scaled') is None else round(float(sim.get('avg_spearman_scaled')),3)}, "
            f"avg_mean_abs_diff={None if sim.get('avg_mean_abs_diff_scaled') is None else round(float(sim.get('avg_mean_abs_diff_scaled')),3)}."
        )

    reasoning_sections = {
        "formula_analysis": "\n".join(formula_lines),
        "expected_movements": "\n".join(expected_lines),
        "observed_movements": "\n".join(observed_lines),
        "synthesis": "\n".join(synth_lines),
    }

    return {
        "stat": stat,
        "buckets": buckets,
        "scaling": scaling,
        "sources": sources,
        "pairwise_vs_ours": pairwise,
        "synthesis": synthesis,
        "reasoning_sections": reasoning_sections,
        "reasoning": "\n\n".join(reasoning_sections.values()),
        "monotonicity_policy": {
            "basis": (
                "scaled_0_1 shared_across_sources_per_feature (prefer feature_range_from_q10_q90; fallback common_range)"
                if scale_mode == "common"
                else "scaled_0_1 per_series_minmax"
            ),
            "direction_rule": "best_of_increasing_or_decreasing_by_step_consistency",
            "score": "step_consistency",
            "pass_threshold": COMPARE_MONO_PASS_THRESHOLD,
            "note": "Compare-plot heuristic; Stage2 PASS/FAIL uses median-based monotonicity in stage2.py.",
        },
    }


def _write_compare_analysis_json(
    *,
    out_path: Path,
    buckets: List[int],
    feature: str,
    stat: str,
    series: List[Tuple[str, str, float, List[Optional[float]]]],
    stat_context: Dict[str, Any],
) -> str:
    """
    Write a lightweight interpretation of the compare plot (heuristic).
    Uses the same scaling policy as the compare plot (see stat_context['scaling']).
    """
    rows: Dict[str, Dict[str, Any]] = {}
    sources_ctx = stat_context.get("sources", {}) if isinstance(stat_context, dict) else {}
    scaling = stat_context.get("scaling", {}) if isinstance(stat_context, dict) else {}
    per_feat = scaling.get("per_feature", {}) if isinstance(scaling, dict) else {}
    srange = per_feat.get(feature, {}) if isinstance(per_feat, dict) else {}
    smin = srange.get("min") if isinstance(srange, dict) else None
    smax = srange.get("max") if isinstance(srange, dict) else None
    sbasis = srange.get("basis") if isinstance(srange, dict) else None
    for name, _color, stroke, raw in series:
        d = (
            (((sources_ctx.get(name, {}) or {}).get("features", {}) or {}).get(feature))
            if isinstance(sources_ctx.get(name, {}), dict)
            else None
        )
        if not isinstance(d, dict) or d.get("missing") is True:
            if isinstance(smin, (int, float)) and isinstance(smax, (int, float)):
                d = _curve_diagnostics(
                    raw, scale_min=float(smin), scale_max=float(smax), scale_basis=str(sbasis or "common"), scale_clip=True
                )
            else:
                d = _curve_diagnostics(raw, scale_basis="per_series")
            d["missing"] = False
        d["stroke_width"] = stroke
        rows[name] = d

    ours = rows.get("ours")
    alpha101 = rows.get("alpha101")
    pairs: Dict[str, Dict[str, Any]] = {}
    if ours:
        for name, meta in rows.items():
            if name == "ours":
                continue
            pairs[name] = {
                "spearman_scaled": _spearman_rho(ours["scaled_0_1"], meta["scaled_0_1"]),
                "mean_abs_diff_scaled": _mean_abs_diff(ours["scaled_0_1"], meta["scaled_0_1"]),
            }

    def r3(x: Any) -> Any:
        if x is None:
            return None
        try:
            return round(float(x), 3)
        except Exception:
            return x

    # Build the 4-part reasoning the user asked for (for this feature + stat).
    sources_ctx = stat_context.get("sources", {}) if isinstance(stat_context, dict) else {}

    formula_lines: List[str] = []
    formula_lines.append(f"[1/4] Formula analysis by source (feature={feature}, stat={stat})")
    for name in [s[0] for s in series]:
        meta = (sources_ctx.get(name, {}) or {}).get("meta", {}) if isinstance(sources_ctx.get(name, {}), dict) else {}
        fa = (sources_ctx.get(name, {}) or {}).get("formula_analysis", {}) if isinstance(sources_ctx.get(name, {}), dict) else {}
        desc = _safe_str(((sources_ctx.get(name, {}) or {}).get("expected_analysis_inputs", {}) or {}).get("formula_description"))
        formula_lines.append(f"- {name}: {_compact_formula_summary(meta)} | focus={fa.get('focus_features')} | windows={fa.get('windows')} | desc={desc or 'n/a'}")

    expected_lines: List[str] = []
    expected_lines.append("[2/4] Expected feature movement (Qk→Q1, i.e., 'quantile decreases')")
    expected_lines.append(
        "- Note: buckets are ordered Q1→Qk on the x-axis (Q1 lowest factor score, Qk highest). Expected directions below are expressed as Qk→Q1."
    )
    expected_lines.append("- Convention: trailing '*' means we expect monotonic separation in that feature, even if direction is uncertain.")
    for name in [s[0] for s in series]:
        em = (sources_ctx.get(name, {}) or {}).get("expected_movements", {}) if isinstance(sources_ctx.get(name, {}), dict) else {}
        parts = []
        for feat in FEATURES:
            d = em.get(feat, {}) if isinstance(em, dict) else {}
            tag = d.get("direction_qk_to_q1", "any")
            sep = d.get("should_separate", False)
            parts.append(f"{feat}={tag}{'*' if sep else ''}")
        expected_lines.append(f"- {name}: expected(Qk→Q1): " + ", ".join(parts))
        d0 = em.get(feature, {}) if isinstance(em, dict) else {}
        expected_lines.append(f"  ↳ {feature}({stat}) rationale={d0.get('rationale',[])}")

    observed_lines: List[str] = []
    observed_lines.append("[3/4] Observed curve movement (from the plotted curve)")
    scale_mode = str((scaling.get("mode") if isinstance(scaling, dict) else "") or "per_series")
    if scale_mode == "common":
        observed_lines.append(
            "- Axes: x=bucket(Q1..Qk; quantiles of the factor score), y=scaled value(0..1; shared range across sources per feature+stat)."
        )
        observed_lines.append("- Curves are scaled by a shared robust range (not per-series); monotonicity uses best-of(↑,↓) step-consistency.")
    else:
        observed_lines.append(
            "- Axes: x=bucket(Q1..Qk; quantiles of the factor score), y=scaled value(0..1; min-max per series across buckets)."
        )
        observed_lines.append("- Curves are min-max scaled per-series on the y-axis to compare shapes; monotonicity uses best-of(↑,↓) step-consistency.")
    # Cross-feature snapshot (so one file contains MAG/DIR/VOL/POS movement context).
    for feat in FEATURES:
        parts = []
        for name in [s[0] for s in series]:
            fm = (sources_ctx.get(name, {}) or {}).get("features", {}) if isinstance(sources_ctx.get(name, {}), dict) else {}
            f0 = fm.get(feat, {}) if isinstance(fm, dict) else {}
            if not isinstance(f0, dict) or f0.get("missing") is True:
                parts.append(f"{name}=n/a")
                continue
            mono = f0.get("monotonicity_scaled", {}) or {}
            parts.append(
                f"{name}:{mono.get('best_direction')}, sc={None if mono.get('best_step_consistency') is None else round(float(mono.get('best_step_consistency')),3)}, pass={bool(mono.get('passed_best'))}"
            )
        observed_lines.append(f"- {feat}: " + " | ".join(parts))
    observed_lines.append(f"- Detail for current feature={feature}:")
    ours_t = ours.get("turbulence_scaled") if ours else None
    for name in [s[0] for s in series]:
        meta = rows.get(name, {})
        ep = meta.get("scaled_endpoints", {})
        mono = meta.get("monotonicity_scaled", {})
        best_dir = mono.get("best_direction")
        best_sc = mono.get("best_step_consistency")
        passed = mono.get("passed_best")
        turb = meta.get("turbulence_scaled")
        shape = meta.get("shape_class_scaled")
        delta = meta.get("scaled_endpoint_delta")
        max_step = meta.get("max_abs_step_scaled")
        rel = None
        if ours_t is not None and turb is not None and float(ours_t) > 0:
            rel = float(turb) / float(ours_t)
        observed_lines.append(
            f"- {name}: endpoints(Q1→Qk)={r3(ep.get('q1'))}→{r3(ep.get('qk'))} (Δ={r3(delta)}), "
            f"shape={shape}, sign_changes={meta.get('sign_changes_scaled')}, "
            f"best_dir(Q1→Qk)={best_dir} / (Qk→Q1)={_opp_dir(best_dir) if isinstance(best_dir,str) else None}, "
            f"sc={r3(best_sc)}, pass={bool(passed)}, turb={r3(turb)}{'' if rel is None else f' (x{r3(rel)} vs ours)'}"
            f"{'' if max_step is None else f', max_step={r3(max_step)}'}."
        )
        if ours and name != "ours":
            sim = pairs.get(name, {}).get("spearman_scaled")
            mad = pairs.get(name, {}).get("mean_abs_diff_scaled")
            observed_lines.append(f"  ↳ similarity to ours: spearman={r3(sim)}, mean_abs_diff={r3(mad)} (scaled).")

    synth_lines: List[str] = []
    synth_lines.append("[4/4] Synthesis (expectation vs observation)")
    # Overall per-source alignment counts (across MAG/DIR/VOL/POS).
    syn = stat_context.get("synthesis", {}) if isinstance(stat_context, dict) else {}
    per_src = syn.get("per_source", {}) if isinstance(syn, dict) else {}
    for name in [s[0] for s in series]:
        ac = (per_src.get(name, {}) or {}).get("alignment_counts", {}) if isinstance(per_src.get(name, {}), dict) else {}
        if ac:
            synth_lines.append(
                f"- {name}: overall_alignment(match_strong={ac.get('match_strong',0)}, match_weak={ac.get('match_weak',0)}, mismatch={ac.get('mismatch',0)}, known={ac.get('known',0)})"
            )
    for name in [s[0] for s in series]:
        aln = (sources_ctx.get(name, {}) or {}).get("alignment", {}) if isinstance(sources_ctx.get(name, {}), dict) else {}
        a = aln.get(feature, {}) if isinstance(aln, dict) else {}
        synth_lines.append(
            f"- {name}: alignment={a.get('alignment','unknown')} "
            f"(expected Qk→Q1={a.get('expected_direction_qk_to_q1')}, observed Qk→Q1={a.get('observed_best_direction_qk_to_q1')}, "
            f"monotonic_pass={a.get('observed_passed_monotonic')})."
        )

    payload = {
        "feature": feature,
        "stat": stat,
        "buckets": buckets,
        "series": rows,
        "pairwise_vs_ours": pairs,
        "sources": sources_ctx,
        "reasoning_sections": {
            "formula_analysis": "\n".join(formula_lines),
            "expected_movements": "\n".join(expected_lines),
            "observed_movements": "\n".join(observed_lines),
            "synthesis": "\n".join(synth_lines),
        },
        "reasoning": "\n\n".join(["\n".join(formula_lines), "\n".join(expected_lines), "\n".join(observed_lines), "\n".join(synth_lines)]),
        "monotonicity_policy": stat_context.get("monotonicity_policy") if isinstance(stat_context, dict) else None,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(out_path)


def _write_compare_stat_summary_json(
    *,
    out_path: Path,
    stat_context: Dict[str, Any],
) -> str:
    out_path.write_text(json.dumps(stat_context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(out_path)


def _to_num(v: str) -> Optional[float]:
    v = (v or "").strip()
    if not v:
        return None
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _read_stage2_distributions_csv(path: Path) -> DistTable:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        if "bucket" not in fieldnames:
            raise ValueError(f"Missing 'bucket' column in {path}")
        rows = list(r)

    buckets: List[int] = []
    cols: Dict[str, List[Optional[float]]] = {k: [] for k in fieldnames if k != "bucket"}
    for row in rows:
        b = _to_num(row.get("bucket", ""))
        if b is None:
            continue
        buckets.append(int(b))
        for k in cols:
            cols[k].append(_to_num(row.get(k, "")))
    return DistTable(buckets=buckets, cols=cols)


def _iter_stage2_dist_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("stage2_distributions.csv"), key=lambda p: str(p))


def _aggregate_source(dist_paths: Sequence[Path]) -> Optional[DistTable]:
    """
    Aggregate many stage2_distributions.csv files into one mean curve per column.
    This function is kept for backward compatibility; prefer `_aggregate_source_by_n`.
    """
    by_n = _aggregate_source_by_n(dist_paths)
    if not by_n:
        return None
    # Return the most common bucket-count group (legacy behavior).
    n = max(by_n.keys(), key=lambda k: len(by_n[k].buckets))
    return by_n[n]


def _aggregate_source_by_n(dist_paths: Sequence[Path]) -> Dict[int, DistTable]:
    """
    Aggregate many stage2_distributions.csv files into mean curves, grouped by bucket count.

    Returns:
      {n_buckets: DistTable(mean-curves)}
    """
    tables: List[DistTable] = []
    for p in dist_paths:
        try:
            t = _read_stage2_distributions_csv(p)
            if t.buckets:
                tables.append(t)
        except Exception:
            continue
    if not tables:
        return {}

    by_n: Dict[int, List[DistTable]] = {}
    for t in tables:
        by_n.setdefault(len(t.buckets), []).append(t)

    out: Dict[int, DistTable] = {}
    for n, group in by_n.items():
        if not group:
            continue
        buckets = group[0].buckets

        common_cols = set(group[0].cols.keys())
        for t in group[1:]:
            common_cols &= set(t.cols.keys())

        cols_out: Dict[str, List[Optional[float]]] = {}
        for col in sorted(common_cols):
            sums = [0.0] * n
            cnts = [0] * n
            for t in group:
                arr = t.cols.get(col, [])
                if len(arr) != n:
                    continue
                for i, v in enumerate(arr):
                    if v is None:
                        continue
                    sums[i] += float(v)
                    cnts[i] += 1
            cols_out[col] = [(sums[i] / cnts[i]) if cnts[i] > 0 else None for i in range(n)]

        out[n] = DistTable(buckets=buckets, cols=cols_out)
    return out


def _polyline(points: List[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _write_compare_stat_png(
    *,
    out_path: Path,
    title: str,
    buckets: List[int],
    by_feature_series: Dict[str, List[Tuple[str, str, float, List[Optional[float]]]]],
    by_feature_medians: Dict[str, List[Tuple[str, str, str, List[Optional[float]]]]],
) -> Optional[str]:
    """
    Stage2-style compare plot: one PNG per stat with a 2x2 grid (MAG/DIR/VOL/POS).
    Each panel overlays sources (lines only; no point markers) and shows median markers (diamonds).
    """
    def _parse_color(c: str) -> Tuple[int, int, int]:
        c = (c or "").strip()
        if c.startswith("#") and len(c) == 7:
            return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
        return (0, 0, 0)

    def norm(v: Optional[float], mn: Optional[float], mx: Optional[float]) -> Optional[float]:
        if v is None or not math.isfinite(v):
            return None
        if mn is None or mx is None or mx <= mn:
            return 0.5
        return (float(v) - mn) / (mx - mn)

    feature_titles = {
        "MAG": "MAG (high-low)",
        "DIR": "DIR (close-open)",
        "VOL": "VOL (volume)",
        "POS": "POS (close-in-range)",
    }

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")  # type: ignore[attr-defined]
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore

        fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
        ax_list = axes.ravel().tolist()

        # Collect legend entries from first panel only (deduped).
        legend_handles: List[Any] = []
        legend_labels: List[str] = []

        for ax, feat in zip(ax_list, FEATURES):
            series = by_feature_series.get(feat, [])
            if not series:
                ax.set_axis_off()
                continue

            ax.set_title(f"{feature_titles.get(feat, feat)} | scaled 0-1 (per-series)", fontsize=11)
            ax.set_xlabel("Statistics")
            ax.set_ylabel("bucket (Q1..Qn)")
            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_ylim(0.5, len(buckets) + 0.5)
            ax.set_yticks(buckets)
            ax.set_yticklabels([f"Q{b}" for b in buckets])
            ax.grid(True, alpha=0.25)

            base_norm: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
            for label, color, stroke_w, arr in series:
                mn, mx = _series_minmax(arr)
                base_norm[label] = (mn, mx)
                x = [float(norm(v, mn, mx)) if norm(v, mn, mx) is not None else float("nan") for v in arr]
                h = ax.plot(x, buckets, linewidth=stroke_w, color=color, label=label)[0]
                if feat == "MAG":  # collect once
                    if label not in legend_labels:
                        legend_labels.append(label)
                        legend_handles.append(h)

            # Median points
            med_points = by_feature_medians.get(feat, [])
            for base_label, kind, color, arr in med_points:
                mn, mx = base_norm.get(base_label, _series_minmax(arr))
                x = [float(norm(v, mn, mx)) if norm(v, mn, mx) is not None else float("nan") for v in arr]
                marker = "D" if kind == "median" else "X"
                ax.plot(x, buckets, linestyle="None", marker=marker, markersize=4.8, color=color, alpha=0.85, label="_nolegend_")

        # Add a legend marker for median
        if legend_handles:
            legend_handles.append(Line2D([0], [0], linestyle="None", marker="D", markersize=6, color="#333333"))
            legend_labels.append("median")

        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(5, len(legend_labels)),
                frameon=False,
            )

        fig.suptitle(title, fontsize=12)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return str(out_path)
    except Exception:
        pass

    # PIL fallback (same geometry as Stage2 plots: 2400x1280).
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

    width, height = 600, 420
    pad = 20
    title_h = 42
    legend_h = 48

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _font(16)
    font_med = _font(12)
    font_small = _font(11)

    draw.text((pad, pad), title, fill=(0, 0, 0), font=font_title)

    grid_top = title_h + 10
    grid_left = pad
    grid_right = pad
    grid_bottom = pad + legend_h
    cols, rows = 2, 2
    cell_w = (width - grid_left - grid_right) / cols
    cell_h = (height - grid_top - grid_bottom) / rows

    def panel_rect(i: int):
        r = i // cols
        c = i % cols
        x0 = grid_left + c * cell_w
        y0 = grid_top + r * cell_h
        return x0, y0, cell_w, cell_h

    def scale_x(x0: float, w: float, v: float) -> float:
        left = x0 + 55
        right = x0 + w - 15
        return left + float(v) * (right - left)

    def scale_y(y0: float, h: float, q: int) -> float:
        top = y0 + 45
        bottom = y0 + h - 35
        if len(buckets) <= 1:
            return (top + bottom) / 2
        i = buckets.index(q)
        return bottom - (i) * (bottom - top) / (len(buckets) - 1)

    # Legend (bottom)
    lx = pad
    ly = height - pad - 22
    legend_items: List[Tuple[str, Tuple[int, int, int], int]] = []
    for feat in FEATURES:
        for label, color, stroke, _arr in by_feature_series.get(feat, []):
            if label not in [x[0] for x in legend_items]:
                legend_items.append((label, _parse_color(color), max(1, int(round(stroke)))))
    # Draw legend items (line + label)
    xcur = lx
    for label, col, stroke in legend_items[:6]:
        draw.line((xcur, ly, xcur + 28, ly), fill=col, width=stroke)
        draw.text((xcur + 34, ly - 8), label, fill=(0, 0, 0), font=font_med)
        xcur += 34 + 10 + draw.textlength(label, font=font_med) + 26  # type: ignore[attr-defined]
    # Median marker hint
    draw.polygon([(xcur + 8, ly - 6), (xcur + 14, ly), (xcur + 8, ly + 6), (xcur + 2, ly)], fill=(60, 60, 60))
    draw.text((xcur + 18, ly - 8), "median", fill=(0, 0, 0), font=font_med)

    for i, feat in enumerate(FEATURES):
        x0, y0, w, h = panel_rect(i)
        series = by_feature_series.get(feat, [])
        if not series:
            continue

        draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(221, 221, 221), width=1)
        draw.text((x0 + 10, y0 + 8), f"{feature_titles.get(feat, feat)} | scaled 0-1 (per-series)", fill=(0, 0, 0), font=font_small)
        draw.text((x0 + 10, y0 + 22), "x=bucket, y=scaled", fill=(85, 85, 85), font=font_small)

        ax_left = x0 + 50
        ax_right = x0 + w - 20
        ax_top = y0 + 35
        ax_bottom = y0 + h - 30
        draw.line((ax_left, ax_top, ax_left, ax_bottom), fill=(153, 153, 153), width=1)
        draw.line((ax_left, ax_bottom, ax_right, ax_bottom), fill=(153, 153, 153), width=1)

        # x ticks
        for tval in (0.0, 0.5, 1.0):
            x = scale_x(x0, w, tval)
            draw.line((x, ax_bottom, x, ax_bottom + 4), fill=(153, 153, 153), width=1)
            draw.text((x - 9, ax_bottom + 7), f"{tval:.3g}", fill=(85, 85, 85), font=font_small)
        # y ticks
        for q in buckets:
            y = scale_y(y0, h, q)
            draw.line((ax_left - 4, y, ax_left, y), fill=(153, 153, 153), width=1)
            draw.text((x0 + 6, y - 7), f"Q{q}", fill=(85, 85, 85), font=font_small)

        base_norm: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for label, color, stroke_w, arr in series:
            col = _parse_color(color)
            mn, mx = _series_minmax(arr)
            base_norm[label] = (mn, mx)
            pts: List[Tuple[float, float]] = []
            for j, q in enumerate(buckets):
                v = arr[j] if j < len(arr) else None
                nv = norm(v, mn, mx)
                if nv is None:
                    continue
                pts.append((scale_x(x0, w, float(nv)), scale_y(y0, h, q)))
            if len(pts) >= 2:
                draw.line(pts, fill=col, width=max(1, int(round(stroke_w))))

        med_points = by_feature_medians.get(feat, [])
        for base_label, kind, color, arr in med_points:
            col = _parse_color(color)
            mn, mx = base_norm.get(base_label, _series_minmax(arr))
            for j, q in enumerate(buckets):
                v = arr[j] if j < len(arr) else None
                nv = norm(v, mn, mx)
                if nv is None:
                    continue
                x = scale_x(x0, w, float(nv))
                y = scale_y(y0, h, q)
                if kind == "median":
                    draw.polygon([(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)], fill=col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return str(out_path)


def _write_compare_png(
    *,
    out_path: Path,
    title: str,
    buckets: List[int],
    series: List[Tuple[str, str, float, List[Optional[float]]]],  # (label, color, width, values)
    feature_title: str,
    paper: bool = False,
    scale_mode: str = DEFAULT_SCALE_MODE,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
) -> str:
    """
    Write a single-panel compare plot with multiple overlaid lines.

    Axes:
      - x: bucket index Q1..Qn (quantiles of the factor score; Q1 lowest score, Qn highest score)
      - y: statistic values scaled to [0,1] (either per-series min-max, or a shared range across sources)
    """
    # Prefer matplotlib; fall back to PIL.
    paper_pil = False
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")  # type: ignore[attr-defined]
        import matplotlib.pyplot as plt  # type: ignore

        if paper:
            plt.rcParams.update(PAPER_RCPARAMS)

        if paper:
            fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)
            ax.set_title(feature_title, pad=6)
        else:
            # Screen-ish compare-plot geometry: 600x420 pixels = 6x4.2 inches @ 100 DPI.
            fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
        ax.set_title(title, fontsize=11)

        ax.set_xlabel("Factor Score")
        if scale_mode == "common" and scale_min is not None and scale_max is not None:
            ylab = "scaled (0–1, common)" if paper else "scaled (0-1, common-range)"
        else:
            ylab = "scaled (0–1)" if paper else "scaled (0-1, per-series)"
        ax.set_ylabel(ylab)
        ax.set_xlim(0.5, len(buckets) + 0.5)
        ax.set_xticks(buckets)
        ax.set_xticklabels([f"Q{b}" for b in buckets], rotation=(45 if len(buckets) > 10 else 0), ha=("right" if len(buckets) > 10 else "center"))
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(True, alpha=(0.18 if paper else 0.25))
        if not paper:
            ax.tick_params(labelsize=8)
            ax.text(0.01, 0.02, feature_title, transform=ax.transAxes, fontsize=8, color="#555")

        for label, color, stroke_w, arr in series:
            if scale_mode == "common" and scale_min is not None and scale_max is not None:
                mn, mx = float(scale_min), float(scale_max)
            else:
                mn, mx = _series_minmax(arr)
            y0, _mn_used, _mx_used = _normalize_0_1(arr, scale_min=mn, scale_max=mx, clip=(scale_mode == "common"))
            y = [float(v) if v is not None else float("nan") for v in y0]
            ax.plot(buckets, y, linewidth=stroke_w, color=color, label=label)

        if paper:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=2.2)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        else:
            ax.legend(loc="lower center", ncol=min(3, len(series)), frameon=False, fontsize=8)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        # If matplotlib isn't available, fall back to PIL. In --paper mode, render a high-res PNG
        # and also a raster PDF (vector PDF requires matplotlib).
        paper_pil = bool(paper)

    from PIL import Image, ImageDraw, ImageFont  # type: ignore  # type: ignore

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

    # Target compare-plot geometry:
    # - screen: 600x420 (matches prior output)
    # - paper (PIL fallback): ~3.35x2.6 inches @ 300 DPI = 1005x780
    if paper_pil:
        width, height = 1005, 780
        pad = 34
        left = 110
        right = 40
        top = 95
        bottom = 80
        dpi = (300, 300)
        font_title = _font(26)
        font_med = _font(18)
        font_small = _font(16)
    else:
        width, height = 600, 420
        pad = 20
        left = 70
        right = 20
        top = 55
        bottom = 45
        dpi = None
        font_title = _font(16)
        font_med = _font(12)
        font_small = _font(11)

    x0 = left
    y_origin = top
    w = width - left - right
    h = height - top - bottom


    def sx(q: int) -> float:
        # q is a bucket id (e.g. 1..n)
        if len(buckets) <= 1:
            return x0 + w / 2
        i = buckets.index(q)
        return x0 + (i) * (w / (len(buckets) - 1))

    def sy(v: float) -> float:
        # v is already normalized to [0, 1]
        return y_origin + h - float(v) * h

    def _parse_color(c: str):
        c = (c or "").strip()
        if c.startswith("#") and len(c) == 7:
            return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
        return (0, 0, 0)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((pad, pad), title, fill=(0, 0, 0), font=font_title)
    draw.text((pad, pad + 18), feature_title, fill=(0, 0, 0), font=font_med)
    ydesc = "y=scaled 0-1 (common-range)" if (scale_mode == "common" and scale_min is not None and scale_max is not None) else "y=scaled 0-1 (per-series)"
    draw.text((pad, pad + 32), f"x=bucket, {ydesc}", fill=(85, 85, 85), font=font_small)

    # Axes
    draw.line((x0, y_origin, x0, y_origin + h), fill=(153, 153, 153), width=1)
    draw.line((x0, y_origin + h, x0 + w, y_origin + h), fill=(153, 153, 153), width=1)

    # X ticks (buckets)
    tick_step = 1 if len(buckets) <= 10 else 2
    for q in buckets[::tick_step]:
        x = sx(q)
        draw.line((x, y_origin + h, x, y_origin + h + 4), fill=(153, 153, 153), width=1)
        draw.text((x - 10, y_origin + h + 8), f"Q{q}", fill=(0, 0, 0), font=font_small)

    # Y ticks (normalized)
    for tval in (0.0, 0.5, 1.0):
        y = sy(float(tval))
        draw.line((x0 - 4, y, x0, y), fill=(153, 153, 153), width=1)
        draw.text((pad, y - 6), f"{tval:.3g}", fill=(0, 0, 0), font=font_small)

    # Lines
    base_norm: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for label, color, stroke_w, arr in series:
        col = _parse_color(color)
        if scale_mode == "common" and scale_min is not None and scale_max is not None:
            mn, mx = float(scale_min), float(scale_max)
        else:
            mn, mx = _series_minmax(arr)
        base_norm[label] = (mn, mx)
        y_norm, _mn_used, _mx_used = _normalize_0_1(arr, scale_min=mn, scale_max=mx, clip=(scale_mode == "common"))
        pts: List[Tuple[float, float]] = []
        for i, q in enumerate(buckets):
            v = y_norm[i] if i < len(y_norm) else None
            if v is None or not math.isfinite(float(v)):
                continue
            pts.append((sx(q), sy(float(v))))
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=max(1, int(round(stroke_w))))

    # Legend
    lx = x0 + w - 260
    ly = y_origin + 18
    for i, (label, color, stroke_w, _) in enumerate(series):
        y = ly + i * 14
        col = _parse_color(color)
        draw.line((lx, y - 4, lx + 18, y - 4), fill=col, width=max(1, int(round(stroke_w))))
        draw.text((lx + 24, y - 10), label, fill=(0, 0, 0), font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is not None:
        img.save(out_path, format="PNG", dpi=dpi)
    else:
        img.save(out_path, format="PNG")
    if paper_pil:
        # Raster PDF for paper mode when matplotlib isn't available.
        try:
            pdf_path = out_path.with_suffix(".pdf")
            img.save(pdf_path, format="PDF", resolution=300.0)
        except Exception:
            pass
    return str(out_path)


def _write_compare_all_png_from_feature_pngs(
    *,
    out_path: Path,
    feature_pngs: Dict[str, Path],
) -> Optional[str]:
    """
    Stitch existing per-feature compare PNGs into a single 2x2 grid image.

    This intentionally uses the already-rendered feature PNGs to:
      - preserve the same style as the per-feature outputs
      - avoid requiring Matplotlib
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    # Require all 4 features to be present.
    ordered = [("MAG", feature_pngs.get("MAG")), ("DIR", feature_pngs.get("DIR")), ("VOL", feature_pngs.get("VOL")), ("POS", feature_pngs.get("POS"))]
    if any(p is None or not Path(p).exists() for _, p in ordered):
        return None

    imgs: Dict[str, Image.Image] = {}
    max_w, max_h = 0, 0
    for feat, p in ordered:
        im = Image.open(str(p)).convert("RGB")
        imgs[feat] = im
        max_w = max(max_w, int(im.size[0]))
        max_h = max(max_h, int(im.size[1]))

    # Layout: [MAG, DIR; VOL, POS]
    out = Image.new("RGB", (max_w * 2, max_h * 2), (255, 255, 255))
    out.paste(imgs["MAG"], (0, 0))
    out.paste(imgs["DIR"], (max_w, 0))
    out.paste(imgs["VOL"], (0, max_h))
    out.paste(imgs["POS"], (max_w, max_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(out_path), format="PNG")
    return str(out_path)


def _write_compare_all_stat_png(
    *,
    out_path: Path,
    title: str,
    buckets: List[int],
    by_feature_series: Dict[str, List[Tuple[str, str, float, List[Optional[float]]]]],
    paper: bool,
    scale_mode: str,
    common_scale_by_feature: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
) -> Optional[str]:
    """
    Render a 2x2 compare plot for a single stat (MAG/DIR/VOL/POS) with a unified legend.
    """
    # Prefer matplotlib for paper mode (and optional PDF), but support a PIL fallback.
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")  # type: ignore[attr-defined]
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore[assignment]

    feature_titles = {
        "MAG": "MAG (high-low)",
        "DIR": "DIR (close-open)",
        "VOL": "VOL (volume)",
        "POS": "POS (close-in-range)",
    }

    if plt is not None:
        if paper:
            plt.rcParams.update(PAPER_RCPARAMS)
            figsize = (6.9, 5.2)  # ~two single-column panels wide, with room for a shared legend
            dpi = 300
            all_lw_mul = 1.35
        else:
            figsize = (12, 8)
            dpi = 120
            all_lw_mul = 1.45

        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=(not paper))
        ax_list = axes.ravel().tolist()

        handles_by_label: Dict[str, Any] = {}
        for ax, feat in zip(ax_list, FEATURES):
            series = by_feature_series.get(feat, [])
            if not series:
                ax.set_axis_off()
                continue

            ax.set_title(feature_titles.get(feat, feat))
            ax.set_xlabel("bucket (Q1..Qn)")
            ax.set_ylabel("scaled stat (0–1)" if paper else "scaled stat (0-1)")
            ax.set_xlim(0.5, len(buckets) + 0.5)
            ax.set_xticks(buckets)
            ax.set_xticklabels([f"Q{b}" for b in buckets], rotation=(45 if len(buckets) > 10 else 0), ha=("right" if len(buckets) > 10 else "center"))
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.grid(True, alpha=(0.18 if paper else 0.25))

            for label, color, stroke_w, arr in series:
                if str(scale_mode) == "common" and isinstance(common_scale_by_feature, dict):
                    rng = common_scale_by_feature.get(feat)
                else:
                    rng = None
                if str(scale_mode) == "common" and isinstance(rng, tuple) and len(rng) == 2 and rng[0] is not None and rng[1] is not None and float(rng[1]) > float(rng[0]):
                    mn, mx = float(rng[0]), float(rng[1])
                else:
                    mn, mx = _series_minmax(arr)
                y_norm, _mn_used, _mx_used = _normalize_0_1(arr, scale_min=mn, scale_max=mx, clip=(str(scale_mode) == "common"))
                y = [float(v) if v is not None else float("nan") for v in y_norm]
                lw = max(1.0, float(stroke_w) * float(all_lw_mul))
                h = ax.plot(buckets, y, linewidth=lw, color=color, label=label)[0]
                if label not in handles_by_label:
                    handles_by_label[label] = h

        if handles_by_label:
            labels = list(handles_by_label.keys())
            handles = [handles_by_label[k] for k in labels]
            if paper:
                fig.subplots_adjust(bottom=0.18)
                fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, handlelength=2.2)
            else:
                fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=9)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if paper:
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        else:
            fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return str(out_path)

    # PIL fallback (paper-ish): high-res PNG + unified legend.
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return None

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

    if paper:
        width, height = 2070, 1560  # ~6.9x5.2 inches @ 300 DPI
        pad = 50
        gap = 40
        title_h = 0
        legend_h = 145
        font_title = _font(30)
        font_panel = _font(22)
        font_small = _font(18)
        dpi = (300, 300)
        all_lw_mul = 1.35
    else:
        width, height = 1200, 840
        pad = 30
        gap = 24
        title_h = 0
        legend_h = 110
        font_title = _font(20)
        font_panel = _font(16)
        font_small = _font(13)
        dpi = None
        all_lw_mul = 1.45

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    grid_top = pad + title_h
    grid_left = pad
    grid_right = pad
    grid_bottom = pad + legend_h
    cell_w = (width - grid_left - grid_right - gap) / 2
    cell_h = (height - grid_top - grid_bottom - gap) / 2

    def panel_rect(i: int) -> Tuple[int, int, int, int]:
        r = i // 2
        c = i % 2
        x0 = int(grid_left + c * (cell_w + gap))
        y0 = int(grid_top + r * (cell_h + gap))
        x1 = int(x0 + cell_w)
        y1 = int(y0 + cell_h)
        return x0, y0, x1, y1

    def draw_panel(*, rect: Tuple[int, int, int, int], feat: str, series: List[Tuple[str, str, float, List[Optional[float]]]]):
        x0, y0, x1, y1 = rect
        w = x1 - x0
        h = y1 - y0
        draw.rectangle((x0, y0, x1, y1), outline=(221, 221, 221), width=2 if paper else 1)
        draw.text((x0 + 12, y0 + 10), feature_titles.get(feat, feat), fill=(0, 0, 0), font=font_panel)

        ax_left = x0 + (80 if paper else 55)
        ax_right = x1 - (26 if paper else 18)
        ax_top = y0 + (55 if paper else 40)
        ax_bottom = y1 - (55 if paper else 40)
        draw.line((ax_left, ax_top, ax_left, ax_bottom), fill=(153, 153, 153), width=2 if paper else 1)
        draw.line((ax_left, ax_bottom, ax_right, ax_bottom), fill=(153, 153, 153), width=2 if paper else 1)

        # x scale: buckets evenly spaced
        def sx(q: int) -> float:
            if len(buckets) <= 1:
                return (ax_left + ax_right) / 2.0
            i = buckets.index(q)
            return ax_left + i * ((ax_right - ax_left) / (len(buckets) - 1))

        def sy(v: float) -> float:
            return ax_bottom - float(v) * (ax_bottom - ax_top)

        tick_step = 1 if len(buckets) <= 10 else 2
        for q in buckets[::tick_step]:
            x = sx(q)
            draw.line((x, ax_bottom, x, ax_bottom + (8 if paper else 6)), fill=(153, 153, 153), width=2 if paper else 1)
            draw.text((x - (18 if paper else 14), ax_bottom + (10 if paper else 8)), f"Q{q}", fill=(60, 60, 60), font=font_small)

        for tval in (0.0, 0.5, 1.0):
            y = sy(float(tval))
            draw.line((ax_left - (8 if paper else 6), y, ax_left, y), fill=(153, 153, 153), width=2 if paper else 1)
            draw.text((x0 + 10, y - (10 if paper else 8)), f"{tval:.3g}", fill=(60, 60, 60), font=font_small)

        for label, color, stroke_w, arr in series:
            if str(scale_mode) == "common" and isinstance(common_scale_by_feature, dict):
                rng = common_scale_by_feature.get(feat)
            else:
                rng = None
            if str(scale_mode) == "common" and isinstance(rng, tuple) and len(rng) == 2 and rng[0] is not None and rng[1] is not None and float(rng[1]) > float(rng[0]):
                mn, mx = float(rng[0]), float(rng[1])
            else:
                mn, mx = _series_minmax(arr)
            y_norm, _mn_used, _mx_used = _normalize_0_1(arr, scale_min=mn, scale_max=mx, clip=(str(scale_mode) == "common"))
            pts: List[Tuple[float, float]] = []
            for i, q in enumerate(buckets):
                v = y_norm[i] if i < len(y_norm) else None
                if v is None or not math.isfinite(float(v)):
                    continue
                pts.append((sx(q), sy(float(v))))
            if len(pts) >= 2:
                lw = max(2 if paper else 1, int(round(float(stroke_w) * float(all_lw_mul))))
                draw.line(pts, fill=_parse_color(color), width=lw)

    # Draw panels in [MAG, DIR; VOL, POS] order.
    for i, feat in enumerate(("MAG", "DIR", "VOL", "POS")):
        draw_panel(rect=panel_rect(i), feat=feat, series=by_feature_series.get(feat, []))

    # Unified legend: preserve first-seen label order across features.
    legend_items: List[Tuple[str, str, float]] = []
    seen = set()
    for feat in ("MAG", "DIR", "VOL", "POS"):
        for label, color, stroke_w, _arr in by_feature_series.get(feat, []):
            if label in seen:
                continue
            seen.add(label)
            legend_items.append((label, color, stroke_w))

    if legend_items:
        lx = pad
        ly = height - legend_h + (20 if paper else 16)
        x = lx
        y = ly
        max_x = width - pad
        for label, color, stroke_w in legend_items:
            # wrap
            est_w = (140 if paper else 110) + len(label) * (10 if paper else 8)
            if x + est_w > max_x:
                x = lx
                y += (34 if paper else 26)
            lw = max(2 if paper else 1, int(round(float(stroke_w) * float(all_lw_mul))))
            draw.line((x, y + 10, x + (42 if paper else 34), y + 10), fill=_parse_color(color), width=lw)
            draw.text((x + (52 if paper else 42), y), label, fill=(0, 0, 0), font=font_small)
            x += est_w

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is not None:
        img.save(out_path, format="PNG", dpi=dpi)
    else:
        img.save(out_path, format="PNG")
    if paper:
        # Raster PDF fallback (vector PDF is handled by matplotlib branch above).
        try:
            img.save(out_path.with_suffix(".pdf"), format="PDF", resolution=300.0)
        except Exception:
            pass
    return str(out_path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="Results root (default: results)")
    ap.add_argument("--ours_root", default="results/formulas", help="OURS formulas output root")
    ap.add_argument("--alphaagent_root", default="results/alphaagent", help="AlphaAgent stage2 output root")
    ap.add_argument("--gpt_root", default="results/gpt", help="GPT stage2 output root")
    ap.add_argument("--alpha101_root", default="results/alpha101", help="Alpha101 stage2 output root")
    ap.add_argument("--baseline_momentum_root", default="results/baselines/momentum_1d", help="Momentum baseline output root")
    ap.add_argument("--baseline_reversal_root", default="results/baselines/reversal_1d", help="Reversal baseline output root")
    ap.add_argument(
        "--include-baselines",
        action="store_true",
        help="If set, include baseline_momentum_1d and baseline_reversal_1d overlay curves (default: off).",
    )
    ap.add_argument(
        "--stats",
        default="mean,std,q90",
        help="Comma-separated stats to overlay (e.g. mean,std,q90,skewness,kurtosis)",
    )
    ap.add_argument(
        "--scale-mode",
        choices=["common", "per_series"],
        default=DEFAULT_SCALE_MODE,
        help="Scaling for compare plots: 'per_series' (each curve min-max) vs 'common' (shared robust range across sources per feature+stat).",
    )
    ap.add_argument(
        "--scale-q-low",
        type=float,
        default=DEFAULT_SCALE_Q_LOW,
        help="Lower quantile for --scale-mode common fallback range (used when q10/q90 columns aren't available; default: 0.05)",
    )
    ap.add_argument(
        "--scale-q-high",
        type=float,
        default=DEFAULT_SCALE_Q_HIGH,
        help="Upper quantile for --scale-mode common fallback range (used when q10/q90 columns aren't available; default: 0.95)",
    )
    ap.add_argument(
        "--paper",
        action="store_true",
        help="Paper mode: prefer Matplotlib rcParams; if Matplotlib missing, use PIL high-res PNG. Writes PDF too (vector if Matplotlib, else raster).",
    )
    args = ap.parse_args(argv)

    stats = [s.strip() for s in str(args.stats).split(",") if s.strip()]
    if not stats:
        raise SystemExit("No --stats provided.")
    if not (0.0 <= float(args.scale_q_low) < float(args.scale_q_high) <= 1.0):
        raise SystemExit("--scale-q-low and --scale-q-high must satisfy 0 <= low < high <= 1")

    ours_root = Path(args.ours_root)
    ours_dists = _iter_stage2_dist_paths(ours_root)
    if not ours_dists:
        raise SystemExit(f"No OURS stage2_distributions.csv found under {ours_root}")

    sources = [
        ("alphaagent", Path(args.alphaagent_root)),
        ("gpt", Path(args.gpt_root)),
        ("alpha101", Path(args.alpha101_root)),
    ]
    if args.include_baselines:
        sources.extend(
            [
                ("baseline_momentum_1d", Path(args.baseline_momentum_root)),
                ("baseline_reversal_1d", Path(args.baseline_reversal_root)),
            ]
        )
    aggregated: Dict[str, Dict[int, DistTable]] = {}
    for name, root in sources:
        aggregated[name] = _aggregate_source_by_n(_iter_stage2_dist_paths(root))
    source_roots: Dict[str, Path] = {name: root for name, root in sources}

    # Per-OURS output directory, write compare plots.
    for ours_csv in ours_dists:
        ours_dir = ours_csv.parent
        try:
            ours_tab = _read_stage2_distributions_csv(ours_csv)
        except Exception:
            continue
        if not ours_tab.buckets:
            continue

        # Pick only sources aggregated on the same bucket count as this OURS run.
        n = len(ours_tab.buckets)
        ok_aggs: Dict[str, DistTable] = {}
        for name, by_n in aggregated.items():
            tab = by_n.get(n)
            if tab is not None:
                ok_aggs[name] = tab

        for st in stats:
            stat_context = _build_stat_context(
                ours_dir=ours_dir,
                buckets=ours_tab.buckets,
                stat=st,
                ours_tab=ours_tab,
                other_tabs=ok_aggs,
                source_roots=source_roots,
                scale_mode=str(args.scale_mode),
                scale_q_low=float(args.scale_q_low),
                scale_q_high=float(args.scale_q_high),
            )

            # One summary per stat, per OURS output directory.
            summary_out = ours_dir / f"stage2_compare_{st}.summary.json"
            _write_compare_stat_summary_json(out_path=summary_out, stat_context=stat_context)

            feature_pngs: Dict[str, Path] = {}
            by_feature_series: Dict[str, List[Tuple[str, str, float, List[Optional[float]]]]] = {}
            for feat in FEATURES:
                col = f"{feat.lower()}_{st}"
                ours_arr = ours_tab.cols.get(col)
                if ours_arr is None:
                    continue
                series: List[Tuple[str, str, float, List[Optional[float]]]] = []
                # Emphasize OURS and "traditional" (alpha101) with thicker strokes and similar dark tones.
                series.append(("ours", "#111111", 3.2, ours_arr))
                palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
                for i, (name, tab) in enumerate(sorted(ok_aggs.items(), key=lambda x: x[0])):
                    arr = tab.cols.get(col)
                    if arr is None:
                        continue
                    if name == "alpha101":
                        series.append((name, "#444444", 2.6, arr))
                    else:
                        series.append((name, palette[i % len(palette)], 1.8, arr))
                if len(series) <= 1:
                    continue

                by_feature_series[feat] = series
                compare_png = ours_dir / f"stage2_compare_{feat.lower()}_{st}.png"
                scale_cfg = stat_context.get("scaling", {}) if isinstance(stat_context, dict) else {}
                per_feat = scale_cfg.get("per_feature", {}) if isinstance(scale_cfg, dict) else {}
                srange = per_feat.get(feat, {}) if isinstance(per_feat, dict) else {}
                smin = srange.get("min") if isinstance(srange, dict) else None
                smax = srange.get("max") if isinstance(srange, dict) else None
                _write_compare_png(
                    out_path=compare_png,
                    title=f"Stage2 compare | stat={st} | buckets={len(ours_tab.buckets)}",
                    buckets=ours_tab.buckets,
                    series=series,
                    feature_title=f"{feat}",
                    paper=bool(args.paper),
                    scale_mode=str(scale_cfg.get("mode") or DEFAULT_SCALE_MODE),
                    scale_min=(float(smin) if isinstance(smin, (int, float)) else None),
                    scale_max=(float(smax) if isinstance(smax, (int, float)) else None),
                )
                feature_pngs[feat] = compare_png

                analysis_out = ours_dir / f"stage2_compare_{feat.lower()}_{st}.analysis.json"
                _write_compare_analysis_json(
                    out_path=analysis_out,
                    buckets=ours_tab.buckets,
                    feature=feat,
                    stat=st,
                    series=series,
                    stat_context=stat_context,
                )

            # Extra convenience output: a single 2x2 stitched view for this stat.
            all_out = ours_dir / f"stage2_compare_all_{st}.png"
            if all(feat in by_feature_series for feat in FEATURES):
                scale_cfg = stat_context.get("scaling", {}) if isinstance(stat_context, dict) else {}
                per_feat = scale_cfg.get("per_feature", {}) if isinstance(scale_cfg, dict) else {}
                common_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
                if isinstance(per_feat, dict):
                    for feat in FEATURES:
                        srange = per_feat.get(feat, {}) if isinstance(per_feat.get(feat, {}), dict) else {}
                        smin = srange.get("min") if isinstance(srange, dict) else None
                        smax = srange.get("max") if isinstance(srange, dict) else None
                        common_ranges[feat] = (
                            float(smin) if isinstance(smin, (int, float)) else None,
                            float(smax) if isinstance(smax, (int, float)) else None,
                        )
                ok = _write_compare_all_stat_png(
                    out_path=all_out,
                    title=f"Stage2 compare (all) | stat={st} | buckets={len(ours_tab.buckets)}",
                    buckets=ours_tab.buckets,
                    by_feature_series=by_feature_series,
                    paper=bool(args.paper),
                    scale_mode=str(scale_cfg.get("mode") or DEFAULT_SCALE_MODE),
                    common_scale_by_feature=(common_ranges if common_ranges else None),
                )
                if ok is None:
                    _write_compare_all_png_from_feature_pngs(out_path=all_out, feature_pngs=feature_pngs)
            else:
                _write_compare_all_png_from_feature_pngs(out_path=all_out, feature_pngs=feature_pngs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
