from __future__ import annotations

import ast
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas.code import CODE_TOOL

from .knowledge_base import (
    KBFailureEntry,
    KBSuccessEntry,
    PolarsCoSTEERKnowledgeBase,
    signature_for_factor,
)
from .prompts import (
    POLARS_COSTEER_CODE_SYSTEM_PROMPT,
    POLARS_COSTEER_CODE_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


class PolarsCoSTEERCodeAgent:
    """
    CoSTEER-like retry + memory wrapper for polars factor code generation.

    Output contract matches `schemas/code.py`:
      {"codes":[{"filename","implementation","entry_point","description","notes"}]}

    This class is designed to be drop-in usable anywhere `CodeAgent.generate_code/loop`
    output is expected.
    """

    def __init__(
        self,
        model: str,
        kb_path: Path | None = None,
        max_loop: int = 30,
        top_k_similar_success: int = 2,
    ) -> None:
        self.model = model
        self.max_loop = max_loop
        self.top_k_similar_success = top_k_similar_success
        if kb_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            kb_path = repo_root / "log" / "coder_polars_kb.json"
        self.kb = PolarsCoSTEERKnowledgeBase(kb_path)

    def _check_code_safety(self, implementation: str) -> Optional[str]:
        """
        Minimal safety checks:
        - disallow shift(-k) (future)
        - disallow dynamic shift args (must be constant >= 0)
        - disallow referencing label columns like fwd_return_* (data leakage)
        """
        try:
            tree = ast.parse(implementation)
        except SyntaxError as e:
            return f"Syntax Error during safety check: {e}"

        class _Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.error: Optional[str] = None

            def visit_Call(self, node: ast.Call) -> Any:
                if self.error:
                    return
                is_shift = False
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.attr, str):
                    # Disallow dataframe joins to prevent constructing explicit future lookups.
                    if "join" in node.func.attr:
                        self.error = "Safety Violation: join-like operations are not allowed."
                        return
                if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
                    is_shift = True
                if isinstance(node.func, ast.Name) and node.func.id == "shift":
                    is_shift = True
                if is_shift:
                    args_to_check = []
                    if node.args:
                        args_to_check.append(node.args[0])
                    for kw in node.keywords or []:
                        if kw.arg in (None, "n", "periods"):
                            args_to_check.append(kw.value)

                    for arg in args_to_check:
                        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                            self.error = "Safety Violation: shift() with a negative argument detected."
                            return
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
                            self.error = f"Safety Violation: 'shift({arg.value})' detected."
                            return
                        if not isinstance(arg, ast.Constant):
                            self.error = "Safety Violation: shift() argument must be a non-negative constant."
                            return
                self.generic_visit(node)

            def visit_Constant(self, node: ast.Constant) -> Any:
                if self.error:
                    return
                if isinstance(node.value, str) and "fwd_return_" in node.value:
                    self.error = "Safety Violation: referencing label columns like 'fwd_return_*' is forbidden."
                    return
                self.generic_visit(node)

        v = _Visitor()
        v.visit(tree)
        return v.error

    def _format_similar_successes(self, factor_definition: str) -> str:
        entries = self.kb.list_success_entries()
        scored = []
        for e in entries:
            s = SequenceMatcher(None, (e.factor_definition or ""), factor_definition or "").ratio()
            scored.append((s, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for s, e in scored[: self.top_k_similar_success] if s > 0.0]
        if not top:
            return "None"
        parts = []
        for e in top:
            code_resp = e.code_response or {}
            codes = code_resp.get("codes", []) if isinstance(code_resp, dict) else []
            impl = ""
            if codes and isinstance(codes[0], dict):
                impl = codes[0].get("implementation", "") or ""
            parts.append(
                f"- factor_name: {e.factor_name}\n  definition: {e.factor_definition}\n  implementation:\n{impl}\n"
            )
        return "\n".join(parts)

    def _format_previous_failures(self, signature: str) -> str:
        fails = self.kb.get_failures(signature)
        if not fails:
            return "None"
        parts = []
        for f in fails[-5:]:
            parts.append(f"- error: {f.error}\n  implementation:\n{f.implementation}\n")
        return "\n".join(parts)

    def generate_code(
        self,
        factor: Dict[str, Any],
        metadata: list | None = None,
        coding_context: str = "",
        error_msg: str | None = None,
    ) -> Dict[str, Any]:
        # Lazy import to avoid hard dependency at module import time (useful for offline tooling/tests).
        from util.llm_client import call_llm

        factor_name = factor.get("name", "")
        factor_definition = factor.get("definition", "")
        factor_description = factor.get("description", "")

        signature = signature_for_factor(
            factor_name=factor_name,
            factor_definition=factor_definition,
            columns=list(metadata or []),
            coding_context=coding_context or "",
        )

        similar_successes = self._format_similar_successes(factor_definition)
        previous_failures = self._format_previous_failures(signature)
        if error_msg:
            previous_failures = f"{previous_failures}\n\nLATEST_ERROR:\n{error_msg}"

        user_prompt = POLARS_COSTEER_CODE_USER_PROMPT_TEMPLATE.format(
            factor_name=factor_name,
            factor_definition=factor_definition,
            factor_description=factor_description,
            columns=metadata,
            coding_context=coding_context or "None",
            similar_successes=similar_successes,
            previous_failures=previous_failures,
        )

        response_dict = call_llm(
            model=self.model,
            system_prompt=POLARS_COSTEER_CODE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=[CODE_TOOL],
            target_tool_name="code_tool",
            temperature=0.2,
        )
        if isinstance(response_dict, dict) and "codes" in response_dict:
            return response_dict
        return {"codes": []}

    def execute_code(self, code: Dict[str, Any], series_df=None) -> str:
        import polars as pl

        try:
            exec_globals: Dict[str, Any] = {}
            implementation = code.get("implementation", "")
            exec(implementation, exec_globals)

            if series_df is not None:
                # Prevent leakage during validation runs: never expose label columns to generated code.
                if hasattr(series_df, "columns"):
                    forbidden = [c for c in series_df.columns if isinstance(c, str) and c.startswith("fwd_return_")]
                    series_df = series_df.drop(forbidden, strict=False) if forbidden else series_df

                entry_point = code.get("entry_point", "") or "compute_factor"
                entry_name = entry_point.split("(")[0].strip()
                if entry_name.startswith("def "):
                    entry_name = entry_name[4:].strip()
                calc_func = exec_globals.get(entry_name)
                if not callable(calc_func):
                    return f"Entry point '{entry_name}' not callable."
                expr = calc_func(series_df)
                if not isinstance(expr, pl.Expr):
                    return f"Function returned {type(expr)}, expected pl.Expr"
                series_df.select(expr.alias("test_factor"))
            return "Success"
        except Exception as e:
            return str(e)

    def loop(
        self,
        factor: Dict[str, Any],
        series_df=None,
        metadata: list | None = None,
        coding_context: str = "",
    ) -> Dict[str, Any]:
        name = factor.get("name", "")
        definition = factor.get("definition", "")
        signature = signature_for_factor(
            factor_name=name,
            factor_definition=definition,
            columns=list(metadata or []),
            coding_context=coding_context or "",
        )

        cached = self.kb.get_exact_success(signature)
        if cached is not None:
            code_resp = cached.code_response or {}
            codes = code_resp.get("codes", []) if isinstance(code_resp, dict) else []
            if codes and isinstance(codes[0], dict):
                # validate quickly
                safety_error = self._check_code_safety(codes[0].get("implementation", "") or "")
                if not safety_error:
                    if self.execute_code(codes[0], series_df) == "Success":
                        return code_resp

        error_msg = None
        last_code_obj = None

        for attempt in range(self.max_loop):
            code_response = self.generate_code(
                factor=factor,
                metadata=metadata,
                coding_context=coding_context,
                error_msg=error_msg,
            )
            codes = code_response.get("codes", []) or []
            if not codes:
                error_msg = "No code generated."
                continue

            code_obj = codes[0]
            last_code_obj = code_obj

            implementation = code_obj.get("implementation", "") or ""
            safety_error = self._check_code_safety(implementation)
            if safety_error:
                error_msg = safety_error
                self.kb.add_failure(
                    KBFailureEntry(
                        signature=signature,
                        factor_name=name,
                        factor_definition=definition,
                        implementation=implementation,
                        error=error_msg,
                    )
                )
                continue

            result = self.execute_code(code_obj, series_df)
            if result == "Success":
                self.kb.add_success(
                    KBSuccessEntry(
                        signature=signature,
                        factor_name=name,
                        factor_definition=definition,
                        columns=list(metadata or []),
                        coding_context=coding_context or "",
                        code_response=code_response,
                    )
                )
                self.kb.save()
                return code_response

            error_msg = result
            self.kb.add_failure(
                KBFailureEntry(
                    signature=signature,
                    factor_name=name,
                    factor_definition=definition,
                    implementation=implementation,
                    error=error_msg,
                )
            )

            if (attempt + 1) % 3 == 0:
                # persist periodically even for failures
                try:
                    self.kb.save()
                except Exception:
                    pass

        try:
            self.kb.save()
        except Exception:
            pass

        raise RuntimeError(
            f"PolarsCoSTEERCodeAgent failed after {self.max_loop} attempts for factor '{name}'. Last error: {error_msg}"
        )
