#!/usr/bin/env python3
"""
Compare factor-generation *process quality* between:
  - ours/left : expression(artifact) | logic(process) | hypothesis
  - alpha/right: expression(artifact) | logic(process) | hypothesis (e.g., expression description)

Unlike `judge_agent.py` (hypothesis <-> expression consistency), this script asks an LLM
to judge which side's *generation process* is better grounded, more testable, and more
faithful to the provided expression/hypothesis text.

Input CSV columns (required):
  - For left/ours side (any one of each group):
      - expression/artifact: ours_expression | ours_artifact | left_expression | left_artifact
      - logic/process      : ours_logic | ours_process | left_logic | left_process | ours_reasoning | left_reasoning
      - hypothesis         : ours_hypothesis | left_hypothesis
  - For right/alpha side (any one of each group):
      - expression/artifact: alpha_expression | alpha_artifact | right_expression | right_artifact
      - logic/process      : alpha_logic | alpha_process | right_logic | right_process | alpha_reasoning | right_reasoning
      - hypothesis         : alpha_hypothesis | right_hypothesis

Output:
  - Same rows with added columns:
      process_winner: OURS|ALPHAAGENT|TIE|INCONCLUSIVE
      process_reasoning: string
      process_scores: json string
      process_checks: json string
      judge_mode: llm|heuristic
      judge_error: optional error string (when LLM fails and falls back)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Winner = str  # "OURS" | "ALPHAAGENT" | "TIE" | "INCONCLUSIVE"


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


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response.")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Could not locate JSON object in LLM response.")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("LLM response JSON is not an object.")
    return obj


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _first_nonempty(row: Dict[str, str], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        v = str(v)
        if v.strip():
            return v
    return ""


def _has_any_field(fieldnames: List[str], keys: List[str]) -> bool:
    fs = set(fieldnames)
    return any(k in fs for k in keys)


def _heuristic_compare(
    *,
    ours_expression: str,
    ours_logic: str,
    ours_hypothesis: str,
    alpha_expression: str,
    alpha_logic: str,
    alpha_hypothesis: str,
    left_name: str = "OURS",
    right_name: str = "ALPHAAGENT",
) -> Tuple[Winner, str, Dict[str, Any], Dict[str, Any]]:
    """
    Lightweight fallback when LLM deps aren't available.
    Intentionally simple: it rewards specificity/testability signals.

    NOTE: This is a heuristic ranking function, not a scientific metric.
    """

    def score_side(expr: str, logic: str, hyp: str) -> Dict[str, Any]:
        expr_n = len(_normalize_text(expr))
        logic_n = len(_normalize_text(logic))
        hyp_n = len(_normalize_text(hyp))

        logic_l = (logic or "").lower()
        hyp_l = (hyp or "").lower()

        nums = re.findall(r"\b\d+(\.\d+)?\b", logic)
        num_count = len(nums)
        has_numbers = num_count > 0

        check_terms = (
            "check",
            "test",
            "validate",
            "verification",
            "falsif",
            "edge case",
            "failure mode",
            "ablation",
            "robust",
            "sensitivity",
            "monotonic",
            "spearman",
            "p-value",
            "quantile",
            "out-of-sample",
            "oos",
            "검증",
            "테스트",
            "근거",
            "반증",
            "엣지",
            "실패",
            "단계",
        )
        check_hits = sum(1 for t in check_terms if t in logic_l)

        # Expression token grounding: does the process mention key expression components?
        tokens = re.findall(r"[A-Za-z_\$][A-Za-z0-9_\$]*", expr or "")
        tokens_u = {t.upper() for t in tokens if t.strip()}
        stop = {
            "CLOSE",
            "OPEN",
            "HIGH",
            "LOW",
            "VOLUME",
            "VOL",
            "RETURN",
            "RET",
            "ABS",
            "LOG",
            "MAX",
            "MIN",
            "MEAN",
            "STD",
            "SUM",
        }
        tokens_u = {t for t in tokens_u if len(t) >= 3 and t not in stop}

        grounding_hits = sum(1 for t in tokens_u if t.lower() in logic_l or t.lower() in hyp_l)
        grounding_ratio = grounding_hits / max(1, len(tokens_u))

        # Simple hypothesis<->logic lexical overlap as a proxy for consistency.
        words_logic = {w for w in re.findall(r"[a-z]{4,}", logic_l)}
        words_hyp = {w for w in re.findall(r"[a-z]{4,}", hyp_l)}
        overlap = len(words_logic & words_hyp)
        union = len(words_logic | words_hyp) or 1
        overlap_ratio = overlap / union

        # Penalize likely "results claims" in the process text (hallucination risk).
        # Use boundaries to avoid false positives (e.g., "monotonicity" contains "ic").
        results_claim_patterns = (
            r"\bsharpe\b",
            r"\bic\b",
            r"\binformation coefficient\b",
            r"\bcagr\b",
            r"\bannualized\b",
            r"\balpha\s*=",
            r"\boutperform(ed|s)?\b",
        )
        claims_results = any(re.search(p, logic_l) for p in results_claim_patterns) and has_numbers
        hallucination_penalty = 2.0 if claims_results else 0.0

        faithfulness = min(5.0, 5.0 * grounding_ratio)
        verification = min(5.0, 0.8 * float(has_numbers) + 0.25 * float(check_hits))
        specificity = min(5.0, 0.6 * float(num_count >= 2) + 0.6 * float(has_numbers) + 0.6 * min(5.0, logic_n / 500.0))
        consistency = min(5.0, 1.5 + 3.5 * overlap_ratio)
        non_hallucination = max(0.0, 5.0 - hallucination_penalty)

        total = float(faithfulness + verification + specificity + consistency + non_hallucination)
        return {
            "expr_len": expr_n,
            "logic_len": logic_n,
            "hyp_len": hyp_n,
            "has_numbers": bool(has_numbers),
            "numbers_count": int(num_count),
            "check_hits": int(check_hits),
            "grounding_hits": int(grounding_hits),
            "grounding_ratio": grounding_ratio,
            "hyp_logic_overlap_ratio": overlap_ratio,
            "claims_results": bool(claims_results),
            "faithfulness": faithfulness,
            "verification": verification,
            "specificity": specificity,
            "consistency": consistency,
            "non_hallucination": non_hallucination,
            "total": total,
        }

    ours = score_side(ours_expression, ours_logic, ours_hypothesis)
    alpha = score_side(alpha_expression, alpha_logic, alpha_hypothesis)

    delta = float(ours["total"]) - float(alpha["total"])
    too_weak = float(ours["total"]) < 6.0 and float(alpha["total"]) < 6.0
    if too_weak:
        winner = "INCONCLUSIVE"
    elif abs(delta) < 2.0:
        winner = "TIE"
    else:
        winner = "OURS" if delta > 0 else "ALPHAAGENT"

    reasoning = (
        "Heuristic fallback: approximated 5 subscores (faithfulness/verification/specificity/consistency/non_hallucination) "
        "from token-grounding, explicit checks, and specificity signals. "
        f"{left_name}_total={ours['total']:.2f}, {right_name}_total={alpha['total']:.2f} (delta={delta:.2f})."
    )
    checks = {
        "left_name": left_name,
        "right_name": right_name,
        "decision_rule": "inconclusive if both_total<6; tie if |delta|<2; else higher total wins",
        "delta_total": delta,
        "tie": bool(winner == "TIE"),
    }
    return winner, reasoning, {"ours": ours, "alphaagent": alpha}, checks


@dataclass(frozen=True)
class LLMCompareResult:
    winner: Winner
    reasoning: str
    scores: Dict[str, Any]
    checks: Dict[str, Any]


def _llm_compare(
    *,
    ours_expression: str,
    ours_logic: str,
    ours_hypothesis: str,
    alpha_expression: str,
    alpha_logic: str,
    alpha_hypothesis: str,
    model: str,
    left_name: str = "OURS",
    right_name: str = "ALPHAAGENT",
) -> LLMCompareResult:
    from alphaagent.oai.llm_utils import APIBackend

    system_prompt = (
        "You are an impartial judge for quantitative-research *process quality*.\n"
        "Do NOT judge backtest performance.\n"
        "Compare two factor-generation traces and decide which process is better.\n"
        "\n"
        "Per side you receive:\n"
        "- expression: the final factor expression\n"
        "- logic: the reasoning/validation logic used while creating or validating the factor\n"
        "- hypothesis: the stated idea (OURS) OR the expression description (ALPHAAGENT)\n"
        "\n"
        "Judge which side is better on ALL criteria:\n"
        "1) Faithfulness: logic/hypothesis actually matches the given expression components.\n"
        "2) Verification quality: logic is testable/falsifiable; cites concrete checks, edge cases, failure modes.\n"
        "3) Specificity: avoids vague claims; uses precise definitions, variables, horizons.\n"
        "4) Internal consistency: no contradictions between hypothesis, logic, expression.\n"
        "5) Minimal hallucination: does not invent data/results not present in the inputs.\n"
        "\n"
        "Scoring rubric (0-5 each): faithfulness, verification, specificity, consistency, non_hallucination.\n"
        "Total score = sum of 5 subscores.\n"
        "\n"
        "Decision rules:\n"
        "- If one side is clearly better (>= 2.0 total points higher), pick it.\n"
        "- If difference < 2.0, return TIE.\n"
        "- If both are too weak/missing to judge, return INCONCLUSIVE.\n"
        "\n"
        "Output ONLY valid JSON with this schema:\n"
        "{\n"
        '  "winner": "OURS" | "ALPHAAGENT" | "TIE" | "INCONCLUSIVE",\n'
        '  "scores": {\n'
        '    "ours": {"faithfulness":n,"verification":n,"specificity":n,"consistency":n,"non_hallucination":n,"total":n},\n'
        '    "alphaagent": {"faithfulness":n,"verification":n,"specificity":n,"consistency":n,"non_hallucination":n,"total":n}\n'
        "  },\n"
        '  "checks": {\n'
        '    "ours_missing_fields": true|false,\n'
        '    "alpha_missing_fields": true|false,\n'
        '    "major_hallucination_detected": true|false\n'
        "  },\n"
        '  "primary_evidence": [\n'
        '    {"side":"OURS|ALPHAAGENT","quote":"<=20 words","reason":"why it matters"},\n'
        "    ...\n"
        "  ],\n"
        '  "reasoning": "4-7 sentences. Mention 1-2 concrete strengths and weaknesses for each side."\n'
        "}\n"
    )

    user_prompt = (
        f"=== OURS ({_normalize_text(left_name)}) ===\n"
        f"[expression]\n{_normalize_text(ours_expression)}\n\n"
        f"[logic]\n{ours_logic.strip()}\n\n"
        f"[hypothesis]\n{ours_hypothesis.strip()}\n\n"
        f"=== ALPHAAGENT ({_normalize_text(right_name)}) ===\n"
        f"[expression]\n{_normalize_text(alpha_expression)}\n\n"
        f"[logic]\n{alpha_logic.strip()}\n\n"
        f"[hypothesis]\n{alpha_hypothesis.strip()}\n"
    )

    backend = APIBackend(chat_model=model) if model else APIBackend()
    resp = backend.build_messages_and_create_chat_completion(
        user_prompt,
        system_prompt,
        json_mode=True,
        reasoning_flag=False,
        temperature=0.0,
        max_tokens=1200,
        shrink_multiple_break=True,
    )
    obj = _extract_first_json_object(resp)

    winner = str(obj.get("winner", "")).strip().upper()
    if winner not in ("OURS", "ALPHAAGENT", "TIE", "INCONCLUSIVE"):
        raise ValueError(f"Invalid winner from LLM: {winner!r}")

    reasoning = str(obj.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("LLM returned empty reasoning.")

    scores = obj.get("scores", {})
    if not isinstance(scores, dict):
        scores = {"raw_scores": scores}
    checks = obj.get("checks", {})
    if not isinstance(checks, dict):
        checks = {"raw_checks": checks}
    pe = obj.get("primary_evidence")
    if pe is not None:
        checks["primary_evidence"] = pe

    return LLMCompareResult(winner=winner, reasoning=reasoning, scores=scores, checks=checks)


def judge_csv(
    *,
    in_csv: Path,
    out_csv: Path,
    mode: str = "auto",
    llm_model: str = "",
    max_rows: Optional[int] = None,
    left_name: str = "OURS",
    right_name: str = "ALPHAAGENT",
) -> Path:
    in_fields, rows = _read_csv(in_csv)

    # Accept multiple schemas: ours/alpha (legacy) or left/right (generic).
    ours_expr_keys = ["ours_expression", "ours_artifact", "left_expression", "left_artifact"]
    ours_logic_keys = ["ours_logic", "ours_process", "left_logic", "left_process", "ours_reasoning", "left_reasoning"]
    ours_hyp_keys = ["ours_hypothesis", "left_hypothesis"]

    alpha_expr_keys = ["alpha_expression", "alpha_artifact", "right_expression", "right_artifact"]
    alpha_logic_keys = [
        "alpha_logic",
        "alpha_process",
        "right_logic",
        "right_process",
        "alpha_reasoning",
        "right_reasoning",
    ]
    alpha_hyp_keys = ["alpha_hypothesis", "right_hypothesis"]

    missing_groups: List[str] = []
    if not _has_any_field(in_fields, ours_expr_keys):
        missing_groups.append(f"left_expression(one of {ours_expr_keys})")
    if not _has_any_field(in_fields, ours_logic_keys):
        missing_groups.append(f"left_logic(one of {ours_logic_keys})")
    if not _has_any_field(in_fields, ours_hyp_keys):
        missing_groups.append(f"left_hypothesis(one of {ours_hyp_keys})")
    if not _has_any_field(in_fields, alpha_expr_keys):
        missing_groups.append(f"right_expression(one of {alpha_expr_keys})")
    if not _has_any_field(in_fields, alpha_logic_keys):
        missing_groups.append(f"right_logic(one of {alpha_logic_keys})")
    if not _has_any_field(in_fields, alpha_hyp_keys):
        missing_groups.append(f"right_hypothesis(one of {alpha_hyp_keys})")
    if missing_groups:
        raise SystemExit(f"Input CSV missing required column groups: {missing_groups}. Got columns: {in_fields}")

    extra_fields = [
        "process_winner",
        "process_reasoning",
        "process_scores",
        "process_checks",
        "judge_mode",
        "judge_error",
    ]
    out_fields = list(in_fields)
    for f in extra_fields:
        if f not in out_fields:
            out_fields.append(f)

    cache: Dict[str, Dict[str, str]] = {}

    def apply_result(
        row: Dict[str, str],
        *,
        winner: Winner,
        reasoning: str,
        scores: Dict[str, Any],
        checks: Dict[str, Any],
        judge_mode: str,
        err: str = "",
    ) -> None:
        row["process_winner"] = winner
        row["process_reasoning"] = reasoning
        row["process_scores"] = json.dumps(scores, ensure_ascii=False)
        row["process_checks"] = json.dumps(checks, ensure_ascii=False)
        row["judge_mode"] = judge_mode
        row["judge_error"] = err

    processed = 0
    for row in rows:
        if max_rows is not None and processed >= max_rows:
            break

        ours_expression = _first_nonempty(row, ours_expr_keys)
        ours_logic = _first_nonempty(row, ours_logic_keys)
        ours_hypothesis = _first_nonempty(row, ours_hyp_keys)
        alpha_expression = _first_nonempty(row, alpha_expr_keys)
        alpha_logic = _first_nonempty(row, alpha_logic_keys)
        alpha_hypothesis = _first_nonempty(row, alpha_hyp_keys)

        key = _md5(
            "\n---\n".join(
                [
                    ours_expression,
                    ours_logic,
                    ours_hypothesis,
                    alpha_expression,
                    alpha_logic,
                    alpha_hypothesis,
                    llm_model,
                    mode,
                    left_name,
                    right_name,
                ]
            )
        )
        if key in cache:
            row.update(cache[key])
            processed += 1
            continue

        want_llm = mode in ("auto", "llm")
        if want_llm:
            try:
                r = _llm_compare(
                    ours_expression=ours_expression,
                    ours_logic=ours_logic,
                    ours_hypothesis=ours_hypothesis,
                    alpha_expression=alpha_expression,
                    alpha_logic=alpha_logic,
                    alpha_hypothesis=alpha_hypothesis,
                    model=llm_model,
                    left_name=left_name,
                    right_name=right_name,
                )
                out: Dict[str, str] = {}
                apply_result(
                    out,
                    winner=r.winner,
                    reasoning=r.reasoning,
                    scores=r.scores,
                    checks=r.checks,
                    judge_mode="llm",
                )
                cache[key] = out
                row.update(out)
                processed += 1
                continue
            except Exception as e:  # noqa: BLE001
                if mode == "llm":
                    raise
                err = f"llm_failed: {type(e).__name__}: {e}"

                winner, reasoning, scores, checks = _heuristic_compare(
                    ours_expression=ours_expression,
                    ours_logic=ours_logic,
                    ours_hypothesis=ours_hypothesis,
                    alpha_expression=alpha_expression,
                    alpha_logic=alpha_logic,
                    alpha_hypothesis=alpha_hypothesis,
                    left_name=left_name,
                    right_name=right_name,
                )
                out = {}
                apply_result(
                    out,
                    winner=winner,
                    reasoning=reasoning,
                    scores={**scores, "llm_error": err},
                    checks={**checks, "llm_error": err},
                    judge_mode="heuristic",
                    err=err,
                )
                cache[key] = out
                row.update(out)
                processed += 1
                continue

        # heuristic-only
        winner, reasoning, scores, checks = _heuristic_compare(
            ours_expression=ours_expression,
            ours_logic=ours_logic,
            ours_hypothesis=ours_hypothesis,
            alpha_expression=alpha_expression,
            alpha_logic=alpha_logic,
            alpha_hypothesis=alpha_hypothesis,
            left_name=left_name,
            right_name=right_name,
        )
        out = {}
        apply_result(out, winner=winner, reasoning=reasoning, scores=scores, checks=checks, judge_mode="heuristic")
        cache[key] = out
        row.update(out)
        processed += 1

    _write_csv(out_csv, out_fields, rows)
    return out_csv


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="CSV containing ours_* and alpha_* fields")
    ap.add_argument("--out_csv", default=None, help="Output CSV path (default: <in>.judged.csv)")
    ap.add_argument("--mode", choices=["auto", "llm", "heuristic"], default="auto")
    ap.add_argument("--llm_model", default="", help="Override chat model (default: LLM_SETTINGS.chat_model)")
    ap.add_argument("--left_name", default="OURS", help="Display name for left/ours side in the judge prompt.")
    ap.add_argument("--right_name", default="ALPHAAGENT", help="Display name for right/alpha side in the judge prompt.")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional limit for quick runs.")
    args = ap.parse_args(argv)

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv) if args.out_csv else _default_out_path(in_csv)

    out = judge_csv(
        in_csv=in_csv,
        out_csv=out_csv,
        mode=str(args.mode),
        llm_model=str(args.llm_model),
        max_rows=args.max_rows,
        left_name=str(args.left_name),
        right_name=str(args.right_name),
    )
    print(f"Wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
