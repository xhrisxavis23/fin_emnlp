#!/usr/bin/env python3
"""
Judge a single pair of "hypothesis - reasoning/process - artifact" without building a CSV.

This is a thin wrapper around `judge_agent_2.py` so you can quickly compare:
  (left/ours)  hypothesis + process + artifact
  (right)      hypothesis + process + artifact

Example:
  python judge_process_pair.py \\
    --left_name "OURS" \\
    --left_hypothesis "..." \\
    --left_process "..." \\
    --left_artifact "..." \\
    --right_name "ALPHAAGENT" \\
    --right_hypothesis "..." \\
    --right_process "..." \\
    --right_artifact "..." \\
    --mode heuristic
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from judge_agent_2 import _heuristic_compare, _llm_compare


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "llm", "heuristic"], default="auto")
    ap.add_argument("--llm_model", default="", help="Override chat model (default: LLM_SETTINGS.chat_model)")

    ap.add_argument("--left_name", default="OURS")
    ap.add_argument("--left_hypothesis", required=True)
    ap.add_argument("--left_process", required=True)
    ap.add_argument("--left_artifact", required=True)

    ap.add_argument("--right_name", default="ALPHAAGENT")
    ap.add_argument("--right_hypothesis", required=True)
    ap.add_argument("--right_process", required=True)
    ap.add_argument("--right_artifact", required=True)

    ap.add_argument("--out", default="", help="Optional JSON output path.")
    args = ap.parse_args(argv)

    result: Dict[str, Any] = {
        "left_name": str(args.left_name),
        "right_name": str(args.right_name),
    }

    if str(args.mode) in ("auto", "llm"):
        try:
            r = _llm_compare(
                ours_expression=str(args.left_artifact),
                ours_logic=str(args.left_process),
                ours_hypothesis=str(args.left_hypothesis),
                alpha_expression=str(args.right_artifact),
                alpha_logic=str(args.right_process),
                alpha_hypothesis=str(args.right_hypothesis),
                model=str(args.llm_model),
                left_name=str(args.left_name),
                right_name=str(args.right_name),
            )
            result.update(
                {
                    "winner": r.winner,
                    "reasoning": r.reasoning,
                    "scores": r.scores,
                    "checks": r.checks,
                    "judge_mode": "llm",
                    "judge_error": "",
                }
            )
        except Exception as e:  # noqa: BLE001
            if str(args.mode) == "llm":
                raise
            err = f"llm_failed: {type(e).__name__}: {e}"
            winner, reasoning, scores, checks = _heuristic_compare(
                ours_expression=str(args.left_artifact),
                ours_logic=str(args.left_process),
                ours_hypothesis=str(args.left_hypothesis),
                alpha_expression=str(args.right_artifact),
                alpha_logic=str(args.right_process),
                alpha_hypothesis=str(args.right_hypothesis),
                left_name=str(args.left_name),
                right_name=str(args.right_name),
            )
            result.update(
                {
                    "winner": winner,
                    "reasoning": reasoning,
                    "scores": {**scores, "llm_error": err},
                    "checks": {**checks, "llm_error": err},
                    "judge_mode": "heuristic",
                    "judge_error": err,
                }
            )
    else:
        winner, reasoning, scores, checks = _heuristic_compare(
            ours_expression=str(args.left_artifact),
            ours_logic=str(args.left_process),
            ours_hypothesis=str(args.left_hypothesis),
            alpha_expression=str(args.right_artifact),
            alpha_logic=str(args.right_process),
            alpha_hypothesis=str(args.right_hypothesis),
            left_name=str(args.left_name),
            right_name=str(args.right_name),
        )
        result.update(
            {
                "winner": winner,
                "reasoning": reasoning,
                "scores": scores,
                "checks": checks,
                "judge_mode": "heuristic",
                "judge_error": "",
            }
        )

    if args.out:
        with open(str(args.out), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

