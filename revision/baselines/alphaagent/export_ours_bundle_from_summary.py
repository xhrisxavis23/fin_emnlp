#!/usr/bin/env python3
"""
Export bundle-style (react_validation_agent-like) JSON files from summary/ours_stage2_summary.json.

This is useful when you have aggregated Stage2 judgment/evidence in the summary JSON and want to
materialize one-per-formula JSONs under examples_ous_bundle/ (or another directory).

Notes
- The summary JSON usually does NOT contain the formula "definition"/"expr".
  This script tries to resolve definitions from existing packaged workspaces:
    results/workspaces/formulas/**/formula_spec.json
  If a definition can't be found, it will be left empty unless --require-definition is set.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RAW_FEATURES = ("MAG", "DIR", "VOL", "POS")

FEATURE_TITLES = {
    "MAG": "H - L (변동폭)",
    "DIR": "C - O (방향성: 양수=상승, 음수=하락)",
    "VOL": "V (거래량)",
    "POS": "(C - L) / (H - L) (상대적 위치: 0=저가마감, 1=고가마감)",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s or "").strip("_") or "unnamed"


def _fmt4(x: Any) -> str:
    try:
        if x is None:
            return "null"
        v = float(x)
        return f"{v:.4f}"
    except Exception:
        return str(x)


def _pct_change(a: Any, b: Any) -> Optional[float]:
    try:
        x0 = float(a)
        x1 = float(b)
    except Exception:
        return None
    denom = abs(x0)
    if denom <= 1e-12:
        return None
    return (x1 - x0) / denom * 100.0


def _format_trend_line(label: str, bins: List[str], values: List[Any]) -> str:
    parts = []
    for b, v in zip(bins, values):
        parts.append(f"{b}={_fmt4(v)}")
    return f"{label} 추이: " + " \u2192 ".join(parts)


def build_distribution_summary(evidence_packet: Dict[str, Any]) -> str:
    meta = evidence_packet.get("meta") if isinstance(evidence_packet, dict) else None
    if not isinstance(meta, dict):
        meta = {}
    polarity = str(meta.get("polarity") or "").strip() or "higher_is_more_true"
    bin_order = str(meta.get("bin_order") or "").strip()
    bins = evidence_packet.get("bins") if isinstance(evidence_packet, dict) else None
    if not isinstance(bins, list) or not bins:
        bins = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    bins = [str(b) for b in bins]
    n_bins = int(meta.get("actual_n_bins") or len(bins) or 0)

    counts = evidence_packet.get("counts") if isinstance(evidence_packet, dict) else None
    if not isinstance(counts, dict):
        counts = {}
    count_str = ", ".join(f"{b}:{int(counts.get(b, 0) or 0)}" for b in bins)

    feats = evidence_packet.get("features") if isinstance(evidence_packet, dict) else None
    if not isinstance(feats, dict):
        feats = {}

    lines: List[str] = []
    # Reflect actual ordering from evidence meta (preferred), otherwise infer from polarity.
    order_kr = ""
    if "obs_strong" in bin_order and "obs_weak" in bin_order:
        if bin_order.startswith("Q1=obs_strong"):
            order_kr = "강→약"
        elif bin_order.startswith("Q1=obs_weak"):
            order_kr = "약→강"
    if not order_kr:
        order_kr = "약→강" if polarity == "higher_is_more_true" else "강→약"

    lines.append(f"[정렬 기준] Q1→Qk = obs truth({order_kr}). polarity={polarity}")
    lines.append(f"[bin 개수] {n_bins}")
    lines.append(f"[샘플 수] {count_str}")
    lines.append("")

    for key in RAW_FEATURES:
        fobj = feats.get(key)
        if not isinstance(fobj, dict):
            continue
        mean = fobj.get("mean") if isinstance(fobj.get("mean"), list) else []
        std = fobj.get("std") if isinstance(fobj.get("std"), list) else []
        skew = fobj.get("skewness") if isinstance(fobj.get("skewness"), list) else []
        kurt = fobj.get("kurtosis") if isinstance(fobj.get("kurtosis"), list) else []
        q90 = fobj.get("q90") if isinstance(fobj.get("q90"), list) else []

        if not mean or len(mean) < 2:
            continue

        mean0, mean1 = mean[0], mean[-1]
        std0, std1 = (std[0], std[-1]) if std and len(std) >= 2 else (None, None)
        skew0, skew1 = (skew[0], skew[-1]) if skew and len(skew) >= 2 else (None, None)
        kurt0, kurt1 = (kurt[0], kurt[-1]) if kurt and len(kurt) >= 2 else (None, None)
        q900, q901 = (q90[0], q90[-1]) if q90 and len(q90) >= 2 else (None, None)

        lines.append(f"### [{key}] {FEATURE_TITLES.get(key, key)}")
        mp = _pct_change(mean0, mean1)
        if mp is not None:
            lines.append(f"mean: {_fmt4(mean0)} → {_fmt4(mean1)} ({mp:+.1f}%)")
        else:
            lines.append(f"mean: {_fmt4(mean0)} → {_fmt4(mean1)}")

        if std0 is not None and std1 is not None:
            sp = _pct_change(std0, std1)
            if sp is not None:
                lines.append(f"std: {_fmt4(std0)} → {_fmt4(std1)} ({sp:+.1f}%)")
            else:
                lines.append(f"std: {_fmt4(std0)} → {_fmt4(std1)}")

        if skew0 is not None and skew1 is not None:
            sp = _pct_change(skew0, skew1)
            if sp is not None:
                lines.append(f"skewness: {_fmt4(skew0)} → {_fmt4(skew1)} ({sp:+.1f}%)")
            else:
                lines.append(f"skewness: {_fmt4(skew0)} → {_fmt4(skew1)}")

        if kurt0 is not None and kurt1 is not None:
            kp = _pct_change(kurt0, kurt1)
            if kp is not None:
                lines.append(f"kurtosis: {_fmt4(kurt0)} → {_fmt4(kurt1)} ({kp:+.1f}%)")
            else:
                lines.append(f"kurtosis: {_fmt4(kurt0)} → {_fmt4(kurt1)}")

        if q900 is not None and q901 is not None:
            lines.append(f"q90: {_fmt4(q900)} → {_fmt4(q901)}")

        lines.append(_format_trend_line("mean", bins, mean))
        if skew:
            lines.append(_format_trend_line("skewness", bins, skew))
        if kurt:
            lines.append(_format_trend_line("kurtosis", bins, kurt))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class FormulaMeta:
    formula_id: str
    formula_name: str
    definition: str
    polarity: str
    obs_id: str
    obs_description: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_formula_spec_index(spec_root: Path) -> Dict[str, FormulaMeta]:
    """
    Index latest formula_spec.json per formula_id under spec_root.
    """
    best: Dict[str, Tuple[float, FormulaMeta]] = {}
    if not spec_root.exists():
        return {}
    for p in spec_root.rglob("formula_spec.json"):
        try:
            obj = _load_json(p)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        formula = obj.get("formula")
        if not isinstance(formula, dict):
            continue
        fid = str(formula.get("formula_id") or "").strip()
        if not fid:
            continue
        fm = FormulaMeta(
            formula_id=fid,
            formula_name=str(formula.get("formula_name") or fid).strip() or fid,
            definition=str(formula.get("definition") or "").strip(),
            polarity=str(formula.get("polarity") or "higher_is_more_true").strip() or "higher_is_more_true",
            obs_id=str(formula.get("obs_id") or "").strip(),
            obs_description=str(formula.get("obs_description") or "").strip(),
        )
        try:
            mt = p.stat().st_mtime
        except Exception:
            mt = 0.0
        cur = best.get(fid)
        if cur is None or mt >= cur[0]:
            best[fid] = (mt, fm)
    return {k: v for k, (_, v) in best.items()}


def build_existing_bundle_index(bundle_dir: Path) -> Dict[str, FormulaMeta]:
    """
    Best-effort parse existing bundle JSONs in bundle_dir and index formula metadata by formula_id.
    """
    out: Dict[str, FormulaMeta] = {}
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        return out
    for p in bundle_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        try:
            obj = _load_json(p)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        f = obj.get("formula")
        if not isinstance(f, dict):
            continue
        fid = str(f.get("formula_id") or "").strip()
        if not fid:
            continue
        out[fid] = FormulaMeta(
            formula_id=fid,
            formula_name=str(f.get("formula_name") or fid).strip() or fid,
            definition=str(f.get("definition") or "").strip(),
            polarity=str(f.get("polarity") or "higher_is_more_true").strip() or "higher_is_more_true",
            obs_id=str(f.get("obs_id") or "").strip(),
            obs_description=str(f.get("obs_description") or "").strip(),
        )
    return out


def iter_summary_results(summary_obj: Dict[str, Any], outer_keys: List[str]) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Yields (outer_key, outer_obj, result_obj).
    """
    for ok in outer_keys:
        outer = summary_obj.get(ok)
        if not isinstance(outer, dict):
            continue
        results = outer.get("results")
        if not isinstance(results, list):
            continue
        for r in results:
            if isinstance(r, dict):
                yield ok, outer, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="summary/ours_stage2_summary.json", help="Path to ours_stage2_summary.json")
    ap.add_argument("--out-dir", default="examples_ous_bundle", help="Directory to write bundle JSON files into")
    ap.add_argument("--outer", action="append", default=[], help="Which outer key(s) to export (repeatable)")
    ap.add_argument("--all-outers", action="store_true", help="Export all outer_iter_* keys found in the summary")
    ap.add_argument("--only", choices=["all", "passed", "failed"], default="all", help="Filter by verdict")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files if present")
    ap.add_argument(
        "--spec-root",
        default="results/workspaces/formulas",
        help="Directory containing packaged formula workspaces (to resolve definition/polarity/obs_description)",
    )
    ap.add_argument(
        "--bundle-fallback",
        default="examples_ous_bundle",
        help="Directory of existing bundle JSONs to use as fallback for formula metadata",
    )
    ap.add_argument("--require-definition", action="store_true", help="Fail if any exported formula lacks definition")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    spec_root = Path(args.spec_root)
    if not spec_root.is_absolute():
        spec_root = repo_root / spec_root
    fallback_dir = Path(args.bundle_fallback)
    if not fallback_dir.is_absolute():
        fallback_dir = repo_root / fallback_dir

    if not summary_path.exists():
        raise SystemExit(f"Summary not found: {summary_path}")

    summary_obj = _load_json(summary_path)
    if not isinstance(summary_obj, dict):
        raise SystemExit("Unsupported summary JSON shape: expected dict at root")

    if args.all_outers:
        outer_keys = sorted([k for k, v in summary_obj.items() if isinstance(v, dict)])
    elif args.outer:
        outer_keys = args.outer
    else:
        outer_keys = ["outer_iter_1"]

    spec_index = build_formula_spec_index(spec_root)
    bundle_index = build_existing_bundle_index(fallback_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0
    missing_def: List[str] = []

    for outer_key, outer, r in iter_summary_results(summary_obj, outer_keys):
        verdict = str(r.get("verdict") or "").strip().upper()
        if args.only == "passed" and verdict != "PASS":
            continue
        if args.only == "failed" and verdict != "FAIL":
            continue

        fid = str(r.get("formula_id") or "").strip()
        fname = str(r.get("formula_name") or fid).strip() or fid
        if not fid:
            continue

        total += 1

        meta = spec_index.get(fid) or bundle_index.get(fid)
        definition = (meta.definition if meta else "").strip()
        polarity = ""
        obs_id = (meta.obs_id if meta else "").strip() or str(r.get("obs_id") or "").strip()
        obs_desc = (meta.obs_description if meta else "").strip()

        evidence = r.get("evidence_packet") if isinstance(r.get("evidence_packet"), dict) else {}
        if not isinstance(evidence, dict):
            evidence = {}
        pol2 = evidence.get("meta", {}).get("polarity") if isinstance(evidence.get("meta"), dict) else None
        polarity = (
            str(pol2 or "").strip()
            or (meta.polarity if meta else "").strip()
            or str(r.get("polarity") or "").strip()
            or "higher_is_more_true"
        )

        if not definition:
            missing_def.append(fid)

        bundle_obj: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "stage": "stage2",
            "kind": "formula_validation",
            "formula": {
                "formula_id": fid,
                "formula_name": fname,
                "definition": definition,
                "polarity": polarity,
                "obs_id": obs_id,
                "obs_description": obs_desc,
            },
            "input": {
                "distribution_summary": build_distribution_summary(evidence) if evidence else "",
                "evidence_packet": evidence,
            },
            "output": {
                "verdict": verdict,
                "reasoning": str(r.get("reasoning") or "").strip(),
            },
        }

        out_name = _safe_filename(f"react_validation_agent_from_summary_{outer_key}_{fid}.json")
        out_path = out_dir / out_name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        out_path.write_text(json.dumps(bundle_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1

    if missing_def:
        uniq = sorted(set(missing_def))
        msg = f"[export_ours_bundle_from_summary] missing definition for {len(uniq)} formula_id(s): {', '.join(uniq)}"
        if args.require_definition:
            raise SystemExit(msg)
        print(msg)

    print(
        f"[export_ours_bundle_from_summary] total_selected={total} written={written} skipped_existing={skipped} out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
