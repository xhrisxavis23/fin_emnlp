#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class VerdictCounts:
    n_total: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_other: int = 0

    def add(self, other: "VerdictCounts") -> "VerdictCounts":
        return VerdictCounts(
            n_total=self.n_total + other.n_total,
            n_pass=self.n_pass + other.n_pass,
            n_fail=self.n_fail + other.n_fail,
            n_other=self.n_other + other.n_other,
        )

    def pass_rate(self) -> Optional[float]:
        denom = self.n_pass + self.n_fail
        if denom == 0:
            return None
        return self.n_pass / denom


def _normalize_verdict(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip().upper()
    return str(v).strip().upper()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _iter_outer_items(obj: Dict[str, Any], outer_keys: List[str], all_outers: bool) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if all_outers:
        keys = sorted([k for k, v in obj.items() if isinstance(v, dict)])
    elif outer_keys:
        keys = outer_keys
    else:
        keys = ["outer_iter_1"]

    for k in keys:
        v = obj.get(k)
        if isinstance(v, dict):
            yield k, v


def _summarize_outer(outer: Dict[str, Any]) -> Tuple[VerdictCounts, Dict[str, Any]]:
    """
    Prefer outer['results'] verdicts. Fallback to passed_formulas/failed_formulas lists.
    Returns (counts, meta) where meta includes mismatch info (if computable).
    """
    meta: Dict[str, Any] = {}

    results = outer.get("results")
    if isinstance(results, list):
        n_pass = 0
        n_fail = 0
        n_other = 0
        n_total = 0
        for r in results:
            if not isinstance(r, dict):
                n_other += 1
                n_total += 1
                continue
            verdict = _normalize_verdict(r.get("verdict"))
            if verdict == "PASS":
                n_pass += 1
            elif verdict == "FAIL":
                n_fail += 1
            else:
                n_other += 1
            n_total += 1
        counts = VerdictCounts(n_total=n_total, n_pass=n_pass, n_fail=n_fail, n_other=n_other)
    else:
        passed_list = outer.get("passed_formulas")
        failed_list = outer.get("failed_formulas")
        n_pass = len(passed_list) if isinstance(passed_list, list) else 0
        n_fail = len(failed_list) if isinstance(failed_list, list) else 0
        counts = VerdictCounts(n_total=n_pass + n_fail, n_pass=n_pass, n_fail=n_fail, n_other=0)

    # Compare with outer's own counters if present.
    outer_passed = outer.get("passed")
    outer_failed = outer.get("failed")
    if isinstance(outer_passed, int) and isinstance(outer_failed, int):
        meta["declared"] = {"passed": outer_passed, "failed": outer_failed}
        meta["mismatch_declared_vs_computed"] = {
            "passed": int(outer_passed) - int(counts.n_pass),
            "failed": int(outer_failed) - int(counts.n_fail),
        }

    return counts, meta


def _as_jsonable(counts: VerdictCounts) -> Dict[str, Any]:
    scored = counts.n_pass + counts.n_fail
    return {
        "total_scored": scored,
        "total_items": counts.n_total,
        "pass": counts.n_pass,
        "fail": counts.n_fail,
        "other": counts.n_other,
        "pass_rate": counts.pass_rate(),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize PASS/FAIL from summary/ours_stage2_summary.json")
    ap.add_argument("--summary", default="summary/ours_stage2_summary.json", help="Path to ours_stage2_summary.json")
    ap.add_argument("--outer", action="append", default=[], help="Which outer key(s) to summarize (repeatable)")
    ap.add_argument("--all-outers", action="store_true", help="Summarize all outer_iter_* keys found in the summary")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    if not summary_path.exists():
        print(f"Summary not found: {summary_path}", file=sys.stderr)
        return 2

    obj = _load_json(summary_path)
    if not isinstance(obj, dict):
        print(f"Unsupported JSON shape: expected dict at root, got {type(obj).__name__}", file=sys.stderr)
        return 2

    per_outer: Dict[str, Dict[str, Any]] = {}
    overall = VerdictCounts()
    mismatches: List[Dict[str, Any]] = []

    for outer_key, outer in _iter_outer_items(obj, outer_keys=args.outer, all_outers=args.all_outers):
        counts, meta = _summarize_outer(outer)
        overall = overall.add(counts)
        per_outer[outer_key] = {**_as_jsonable(counts), **meta}

        mm = meta.get("mismatch_declared_vs_computed")
        if isinstance(mm, dict) and (mm.get("passed") or 0) != 0 or (mm.get("failed") or 0) != 0:
            mismatches.append({"outer": outer_key, **mm})

    out = {"summary_path": str(summary_path), "overall": _as_jsonable(overall), "by_outer": per_outer}

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    pr = overall.pass_rate()
    pr_str = "n/a" if pr is None else f"{pr:.3f}"
    scored = overall.n_pass + overall.n_fail
    print(
        f"scored={scored} PASS={overall.n_pass} FAIL={overall.n_fail} other={overall.n_other} pass_rate={pr_str}"
    )
    print("\nBy outer:")
    for k in sorted(per_outer.keys()):
        s = per_outer[k]
        pr = s.get("pass_rate")
        pr_str = "n/a" if pr is None else f"{float(pr):.3f}"
        print(
            f"- {k}: scored={s['total_scored']} PASS={s['pass']} FAIL={s['fail']} other={s['other']} pass_rate={pr_str}"
        )

    if mismatches:
        print("\nMismatches (declared vs computed):", file=sys.stderr)
        for m in mismatches:
            print(f"- {m['outer']}: passed_delta={m.get('passed')} failed_delta={m.get('failed')}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

