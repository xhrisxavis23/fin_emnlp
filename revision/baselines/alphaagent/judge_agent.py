#!/usr/bin/env python3
"""
Judge hypothesis <-> factor expression consistency from `export_artifacts.py` CSV.

This is intentionally a standalone script to avoid touching the AlphaAgent pipeline logic.

Input CSV columns expected (from export_artifacts.py):
  - hypothesis
  - concise_knowledge (optional)
  - concise_observation (optional)
  - concise_justification (optional)
  - concise_specification (optional)
  - factor_expression
  - factor_name (optional; ignored by the judge)
  - (optional) factor_description, factor_formulation, workspace_path, factor_py_path, source_llm_log

Output:
  - Same rows with added columns:
      hyp_factor_verdict: PASS|FAIL
      hyp_factor_reasoning: string
      hyp_factor_checks: json string
      judge_mode: llm|heuristic
      judge_error: optional error string (when LLM fails and falls back)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Verdict = str  # "PASS" | "FAIL"


def _md5(s: str) -> str:
    h = hashlib.md5(usedforsecurity=False)
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = [dict(row) for row in r]
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _default_out_path(in_csv: Path) -> Path:
    stem = in_csv.name
    if stem.lower().endswith(".csv"):
        stem = stem[: -len(".csv")]
    return in_csv.with_name(f"{stem}.judged.csv")


def _extract_ints(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\b(\d{1,4})\b", text or "")]


def _extract_horizons_from_hypothesis(hyp: str) -> List[int]:
    # e.g. "5-day", "within a 5-day period", "10D"
    pat = re.compile(r"\b(\d{1,4})\s*(?:-?\s*(?:day|days|d))\b", flags=re.IGNORECASE)
    return [int(m.group(1)) for m in pat.finditer(hyp or "")]


def _looks_like_mean_reversion(hyp: str) -> bool:
    h = (hyp or "").lower()
    return any(k in h for k in ("mean reversion", "mean-reversion", "revert to the mean", "revert", "reversion"))


def _expr_mentions_ma_or_band(expr: str) -> bool:
    e = (expr or "").upper()
    return any(k in e for k in ("TS_MEAN(", "SMA(", "EMA(", "WMA(", "BB_MIDDLE(", "BB_UPPER(", "BB_LOWER("))


def _heuristic_judge(hypothesis: str, factor_name: str, factor_expression: str) -> Tuple[Verdict, str, Dict[str, Any]]:
    hyp = (hypothesis or "").strip()
    expr = (factor_expression or "").strip()
    if not hyp:
        return "FAIL", "Missing hypothesis (empty hypothesis column).", {"has_hypothesis": False}
    if not expr:
        return "FAIL", "Missing factor_expression (empty factor_expression column).", {"has_expression": False}

    horizons = _extract_horizons_from_hypothesis(hyp)
    expr_ints = set(_extract_ints(expr))

    horizon_ok = True
    if horizons:
        horizon_ok = any((h in expr_ints) for h in horizons)

    mean_rev_req = _looks_like_mean_reversion(hyp)
    mean_rev_ok = True
    if mean_rev_req:
        mean_rev_ok = _expr_mentions_ma_or_band(expr) or ("-" in expr and "$close" in expr)

    checks = {
        "horizons_in_hypothesis": horizons,
        "ints_in_expression": sorted(expr_ints),
        "horizon_consistent": bool(horizon_ok),
        "mean_reversion_hypothesis": bool(mean_rev_req),
        "mean_reversion_signal_present": bool(mean_rev_ok),
    }

    if horizon_ok and mean_rev_ok:
        return (
            "PASS",
            "PASS (heuristic): horizon appears consistent and expression shape plausibly matches the hypothesis. "
            f"Hypothesis horizons={horizons or 'none'}, expr_ints={sorted(expr_ints)}; "
            f"mean_reversion_hypothesis={mean_rev_req}, mean_reversion_signal_present={mean_rev_ok}.",
            checks,
        )

    reasons = []
    if not horizon_ok:
        reasons.append(
            f"Horizon mismatch: hypothesis horizons={horizons} not found in factor_name/expression "
            f"(expr_ints={sorted(expr_ints)})."
        )
    if not mean_rev_ok:
        reasons.append(
            "Signal mismatch: hypothesis implies mean-reversion but expression doesn't clearly look like "
            "a deviation-from-mean / band / MA-based construction."
        )
    return "FAIL", "FAIL (heuristic): " + " ".join(reasons), checks


@dataclass(frozen=True)
class LLMJudgeResult:
    verdict: Verdict
    reasoning: str
    checks: Dict[str, Any]


def _llm_judge(hypothesis: str, factor_name: str, factor_expression: str) -> LLMJudgeResult:
    # Local import to keep script usable without LLM deps.
    from alphaagent.oai.llm_utils import APIBackend

    system_prompt = (
        "You are an impartial judge for quant research artifacts.\n"
        "Your task: evaluate whether the provided factor expression is a reasonable implementation of the provided hypothesis.\n"
        "First, think step-by-step privately. Then output ONLY the final judgment as JSON (no extra text).\n"
        "Be strict and conservative: if the mapping is unclear or relies on unstated assumptions, mark FAIL.\n"
        "Evaluation criteria (must consider all):\n"
        "1) Horizon/window consistency (e.g., 5D/10D parameters match the hypothesis timeframe).\n"
        "2) Signal construction consistency (e.g., deviation-from-mean for mean reversion, volatility/volume use if stated).\n"
        "3) Direction/polarity consistency when implied (e.g., oversold -> revert; sign conventions).\n"
        "4) Variable/feature scope consistency (expression should plausibly measure the stated condition).\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "verdict": "PASS" | "FAIL",\n'
        '  "reasoning": "detailed explanation (4-8 sentences). Must mention (a) which horizon/window(s) you inferred from the hypothesis and where they appear (or not) in the expression, and (b) the main expression components that support or contradict the hypothesis.",\n'
        '  "checks": {\n'
        '    "horizon_consistent": true|false,\n'
        '    "signal_consistent": true|false,\n'
        '    "major_mismatch": true|false\n'
        "  },\n"
        '  "primary_evidence": ["short bullet evidence", "..."]\n'
        "}\n"
    )

    user_prompt = (
        f"Hypothesis:\n{hypothesis.strip()}\n\n"
        f"Factor expression:\n{factor_expression.strip()}\n"
    )

    resp = APIBackend().build_messages_and_create_chat_completion(
        user_prompt,
        system_prompt,
        json_mode=True,
        reasoning_flag=False,
        temperature=0.0,
        max_tokens=900,
        shrink_multiple_break=True,
    )
    obj = json.loads(resp)
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in ("PASS", "FAIL"):
        verdict = "FAIL"
    reasoning = str(obj.get("reasoning", "")).strip()
    checks = obj.get("checks", {})
    if not isinstance(checks, dict):
        checks = {"raw_checks": checks}
    # Include primary evidence if present (helpful for debugging).
    pe = obj.get("primary_evidence")
    if pe is not None:
        checks["primary_evidence"] = pe
    return LLMJudgeResult(verdict=verdict, reasoning=reasoning, checks=checks)


def judge_csv(
    *,
    in_csv: Path,
    out_csv: Path,
    mode: str = "auto",
    max_rows: Optional[int] = None,
) -> Path:
    in_fields, rows = _read_csv(in_csv)

    needed = {"hypothesis", "factor_expression"}
    missing = [c for c in sorted(needed) if c not in set(in_fields)]
    if missing:
        raise SystemExit(f"Input CSV missing columns: {missing}. Got columns: {in_fields}")

    extra_fields = [
        "hyp_factor_verdict",
        "hyp_factor_reasoning",
        "hyp_factor_checks",
        "judge_mode",
        "judge_error",
    ]
    out_fields = list(in_fields)
    for f in extra_fields:
        if f not in out_fields:
            out_fields.append(f)

    cache: Dict[str, Dict[str, str]] = {}

    def apply_result(row: Dict[str, str], verdict: Verdict, reasoning: str, checks: Dict[str, Any], judge_mode: str, err: str = ""):
        row["hyp_factor_verdict"] = verdict
        row["hyp_factor_reasoning"] = reasoning
        row["hyp_factor_checks"] = json.dumps(checks, ensure_ascii=False)
        row["judge_mode"] = judge_mode
        row["judge_error"] = err

    processed = 0
    for row in rows:
        if max_rows is not None and processed >= max_rows:
            break

        hyp = row.get("hypothesis", "") or ""
        fname = row.get("factor_name", "") or ""
        expr = row.get("factor_expression", "") or ""
        key = _md5(f"{hyp}\n---\n{expr}")

        if key in cache:
            row.update(cache[key])
            processed += 1
            continue

        # Decide judging mode
        want_llm = mode in ("auto", "llm")
        if want_llm:
            try:
                r = _llm_judge(hyp, fname, expr)
                out = {}
                apply_result(out, r.verdict, r.reasoning, r.checks, "llm")
                cache[key] = out
                row.update(out)
                processed += 1
                continue
            except Exception as e:  # noqa: BLE001
                if mode == "llm":
                    raise
                # auto: fall back to heuristic
                err = f"llm_failed: {type(e).__name__}: {e}"

                v, reason, checks = _heuristic_judge(hyp, fname, expr)
                out = {}
                apply_result(out, v, reason, {**checks, "llm_error": err}, "heuristic", err=err)
                cache[key] = out
                row.update(out)
                processed += 1
                continue

        # heuristic-only
        v, reason, checks = _heuristic_judge(hyp, fname, expr)
        out = {}
        apply_result(out, v, reason, checks, "heuristic")
        cache[key] = out
        row.update(out)
        processed += 1

    _write_csv(out_csv, out_fields, rows)
    return out_csv


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="CSV produced by export_artifacts.py")
    ap.add_argument("--out_csv", default=None, help="Output CSV path (default: <in>.judged.csv)")
    ap.add_argument("--mode", choices=["auto", "llm", "heuristic"], default="auto")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional limit for quick runs.")
    args = ap.parse_args(argv)

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv) if args.out_csv else _default_out_path(in_csv)

    out = judge_csv(in_csv=in_csv, out_csv=out_csv, mode=str(args.mode), max_rows=args.max_rows)
    print(f"Wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
