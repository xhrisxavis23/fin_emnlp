#!/usr/bin/env python3
"""
Example GPT Agent: Hypothesis -> Single Factor (expr + name)

This is a minimal, standalone-ish script that uses an LLM to generate ONE factor formula
compatible with AlphaAgent factor workspaces.

Outputs
  - Creates a new workspace under git_ignore_folder/RD-Agent_workspace/<uuid> containing:
      - factor.py (expr/name filled)
      - daily_pv.h5 symlink (default to git_ignore_folder/factor_implementation_source_data/daily_pv.h5)
      - gpt_factor.json (LLM response + metadata)
      - hypothesis.txt
  - Optionally runs factor.py to produce result.h5.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FactorSpec:
    name: str
    expr: str
    rationale: str
    hypothesis: str
    constraints: list[str]
    model: str
    created_at: str


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


def _is_safe_expr(expr: str) -> tuple[bool, str]:
    """
    factor.py will ultimately eval() the parsed expression. Treat expr as untrusted.
    We enforce a conservative allowlist to block code injection.
    """
    if not expr or not isinstance(expr, str):
        return False, "expr is empty"
    if any(ch in expr for ch in ("\n", "\r", "`", ";")):
        return False, "expr contains newline/backtick/semicolon"
    if "'" in expr or '"' in expr:
        return False, "expr contains quotes"
    lowered = expr.lower()
    banned = [
        "__",
        "import",
        "exec",
        "eval",
        "open(",
        "os.",
        "sys.",
        "subprocess",
        "pathlib",
    ]
    if any(b in lowered for b in banned):
        return False, f"expr contains banned token ({next(b for b in banned if b in lowered)!r})"
    # Very conservative charset.
    if not re.fullmatch(r"[A-Za-z0-9_\$\(\)\[\]\+\-\*\/\.,:<>=!%\s]+", expr):
        return False, "expr contains unexpected characters"
    return True, ""


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "").strip("_")
    return s or "unnamed_factor"


def _call_llm_json(*, model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Prefer AlphaAgent's APIBackend (respects LLM_SETTINGS and supports caching).
    Fall back to raw OpenAI client if needed.
    """
    try:
        from alphaagent.oai.llm_utils import APIBackend  # type: ignore

        backend = APIBackend(chat_model=model)
        return backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            reasoning_flag=False,
            temperature=0.7,
        )
    except Exception:
        pass

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(
            "No available LLM backend. Configure AlphaAgent/OpenAI dependencies or set OPENAI_API_KEY."
        ) from e


def generate_factor_spec(*, hypothesis: str, model: str, seed: int, max_depth: int) -> FactorSpec:
    constraints = [
        'Use ONLY variables like "$open,$high,$low,$close,$volume,$return" (if available).',
        "Use ONLY math ops and function_lib calls (e.g., DELTA, DELAY, EMA, SMA, WMA, TS_MEAN, TS_STD, TS_ZSCORE, TS_CORR, RANK, ABS, SIGN).",
        "Expression must be a SINGLE line and must not contain quotes, semicolons, backticks, or any Python imports.",
        "Keep it simple and interpretable; avoid excessive nesting.",
        "Avoid forward-looking leakage; do not use negative delays.",
        "Make the factor name concise and descriptive in English (PascalCase or with underscores).",
    ]

    system_prompt = (
        "You are a quantitative researcher writing a single Qlib-style factor expression.\n"
        "You must output STRICT JSON with keys: name, expr, rationale.\n"
        "The expr will be evaluated in a restricted environment that provides OHLCV columns like $close and "
        "a function library (DELTA/EMA/SMA/TS_MEAN/...); do not output python code.\n"
        "If unsure about available variables, prefer $close/$open/$high/$low/$volume.\n"
    )
    user_prompt = (
        "Task: Propose ONE alpha factor formula from the hypothesis.\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        f"Randomness seed (for diversity): {seed}\n"
        f"Max nesting depth guideline: {max_depth}\n\n"
        "Constraints:\n- " + "\n- ".join(constraints) + "\n\n"
        "Return JSON only.\n"
        "Example JSON:\n"
        '{\n  "name": "ExampleFactor",\n  "expr": "RANK(DELTA($close, 1) / (TS_STD($close, 20) + 1e-8))",\n'
        '  "rationale": "One sentence explaining why this matches the hypothesis."\n}\n'
    )

    raw = _call_llm_json(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    obj = _extract_first_json_object(raw)
    name = str(obj.get("name", "")).strip()
    expr = str(obj.get("expr", "")).strip()
    rationale = str(obj.get("rationale", "")).strip()

    if not name:
        raise ValueError("LLM returned empty 'name'.")
    if not expr:
        raise ValueError("LLM returned empty 'expr'.")
    ok, why = _is_safe_expr(expr)
    if not ok:
        raise ValueError(f"Unsafe expr rejected: {why}. expr={expr!r}")
    if not rationale:
        raise ValueError("LLM returned empty 'rationale'.")

    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return FactorSpec(
        name=_safe_name(name),
        expr=expr,
        rationale=rationale,
        hypothesis=hypothesis,
        constraints=constraints,
        model=model,
        created_at=created_at,
    )


def _render_factor_py(*, name: str, expr: str) -> str:
    return (
        "\n"
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "from alphaagent.components.coder.factor_coder.expr_parser import parse_expression, parse_symbol\n"
        "from alphaagent.components.coder.factor_coder.function_lib import *\n"
        "\n"
        "\n"
        "def calculate_factor(expr: str, name: str) -> None:\n"
        "    df = pd.read_hdf('./daily_pv.h5', key='data')\n"
        "    expr2 = parse_symbol(expr, df.columns)\n"
        "    expr2 = parse_expression(expr2)\n"
        "\n"
        "    for col in df.columns:\n"
        "        expr2 = expr2.replace(col[1:], f\"df['{col}']\")\n"
        "\n"
        "    df[name] = eval(expr2)\n"
        "    result = df[name].astype(np.float64)\n"
        "\n"
        "    if os.path.exists('result.h5'):\n"
        "        os.remove('result.h5')\n"
        "    result.to_hdf('result.h5', key='data')\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        f"    expr = {json.dumps(expr)}\n"
        f"    name = {json.dumps(name)}\n"
        "    calculate_factor(expr, name)\n"
    )


def _ensure_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
    except FileNotFoundError:
        pass
    link_path.symlink_to(target)


def write_workspace(
    *,
    spec: FactorSpec,
    workspace_root: Path,
    daily_pv_path: Path,
    run_factor: bool,
) -> Path:
    ws_id = uuid.uuid4().hex
    ws_dir = workspace_root / ws_id
    ws_dir.mkdir(parents=True, exist_ok=False)

    (ws_dir / "hypothesis.txt").write_text(spec.hypothesis.strip() + "\n", encoding="utf-8")
    (ws_dir / "gpt_factor.json").write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ws_dir / "factor.py").write_text(_render_factor_py(name=spec.name, expr=spec.expr), encoding="utf-8")
    _ensure_symlink(ws_dir / "daily_pv.h5", daily_pv_path.resolve())

    if run_factor:
        # This requires the same runtime deps as AlphaAgent factor evaluation (pandas/numpy/h5py).
        import subprocess

        subprocess.check_call([sys.executable, "factor.py"], cwd=str(ws_dir))

    return ws_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", default="", help="Natural-language hypothesis to translate into a factor formula")
    ap.add_argument("--hypothesis-file", default="", help="Read hypothesis text from a file")
    ap.add_argument("--model", default="", help="LLM model override (default: alphaagent LLM_SETTINGS.chat_model)")
    ap.add_argument("--seed", type=int, default=0, help="Diversity seed (fed to prompt)")
    ap.add_argument("--max-depth", type=int, default=4, help="Max nesting depth guideline for the LLM")
    ap.add_argument(
        "--workspace-root",
        default="git_ignore_folder/RD-Agent_workspace",
        help="Where to create factor workspace dirs",
    )
    ap.add_argument(
        "--daily-pv",
        default="git_ignore_folder/factor_implementation_source_data/daily_pv.h5",
        help="Path to daily_pv.h5 to symlink into the new workspace",
    )
    ap.add_argument("--print-only", action="store_true", help="Only print the generated spec JSON; do not write workspace")
    ap.add_argument("--run", action="store_true", help="Run factor.py after writing workspace (produces result.h5)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent

    hypothesis = (args.hypothesis or "").strip()
    if args.hypothesis_file:
        hyp_path = Path(args.hypothesis_file)
        if not hyp_path.is_absolute():
            hyp_path = repo_root / hyp_path
        hypothesis = hyp_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not hypothesis:
        raise SystemExit("Provide --hypothesis or --hypothesis-file.")

    if not args.model:
        try:
            from alphaagent.oai.llm_conf import LLM_SETTINGS  # type: ignore

            args.model = LLM_SETTINGS.chat_model
        except Exception:
            args.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    spec = generate_factor_spec(hypothesis=hypothesis, model=args.model, seed=args.seed, max_depth=args.max_depth)

    if args.print_only:
        print(json.dumps(asdict(spec), ensure_ascii=False, indent=2))
        return 0

    ws_root = Path(args.workspace_root)
    if not ws_root.is_absolute():
        ws_root = repo_root / ws_root
    daily_pv = Path(args.daily_pv)
    if not daily_pv.is_absolute():
        daily_pv = repo_root / daily_pv
    if not daily_pv.exists():
        raise SystemExit(f"daily_pv.h5 not found: {daily_pv}")

    ws_dir = write_workspace(spec=spec, workspace_root=ws_root, daily_pv_path=daily_pv, run_factor=args.run)
    print(f"[example_gpt_agent] wrote workspace: {ws_dir}")
    print(f"[example_gpt_agent] factor_name={spec.name}")
    print(f"[example_gpt_agent] expr={spec.expr}")
    if args.run:
        print(f"[example_gpt_agent] wrote: {ws_dir / 'result.h5'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
