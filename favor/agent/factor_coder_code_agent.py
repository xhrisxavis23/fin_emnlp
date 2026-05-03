from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

from agent.base_agent import BaseAgent

if TYPE_CHECKING:
    from util.run_context import RunContext

logger = logging.getLogger(__name__)


def _definition_to_alpha_expression(defn: str) -> str:
    """
    Best-effort converter from the loop's legacy DSL (e.g., close[-1], ts_max(x, n)[-1])
    into the AlphaAgent-style expression DSL used by `coder/factor_coder`.

    This is intentionally heuristic; on failure the refinement loop will ask the LLM to fix it.
    """
    import re

    if not defn or not isinstance(defn, str):
        return ""

    expr = defn.strip()

    # Normalize function names (legacy -> alphaagent-like)
    repl_funcs = {
        r"\bts_max\(": "TS_MAX(",
        r"\bts_min\(": "TS_MIN(",
        r"\bts_mean\(": "TS_MEAN(",
        r"\bts_sum\(": "TS_SUM(",
        r"\bts_rank\(": "TS_RANK(",
        r"\bstddev\(": "TS_STD(",
        r"\bts_std\(": "TS_STD(",
        r"\bsma\(": "SMA(",
        r"\brank\(": "RANK(",
        r"\blog\(": "LOG(",
        r"\babs\(": "ABS(",
        r"\bsqrt\(": "SQRT(",
        r"\bexp\(": "EXP(",
        r"\bsign\(": "SIGN(",
    }
    for pat, rep in repl_funcs.items():
        expr = re.sub(pat, rep, expr, flags=re.IGNORECASE)

    # Variables -> $var (only for common OHLCV-like names)
    # We avoid touching already-$ prefixed tokens.
    base_vars = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tradingvalue",
        "marketcap",
        "sharesoutstanding",
    ]
    for v in sorted(base_vars, key=len, reverse=True):
        expr = re.sub(rf"(?<!\$)\b{v}\b", f"${v}", expr, flags=re.IGNORECASE)

    # Convert lag syntax: $var[-k] / var[-k] -> DELAY($var, k)
    # Apply after $-prefixing so we don't miss variables.
    for _ in range(10):
        new = re.sub(r"(\$[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*-\s*(\d+)\s*\]", r"DELAY(\1, \2)", expr)
        if new == expr:
            break
        expr = new

    # Also convert lag syntax for function outputs: FUNC(...)[-k] -> DELAY(FUNC(...), k)
    for _ in range(10):
        new = re.sub(r"(\b[A-Z_]+\([^\)]*\))\s*\[\s*-\s*(\d+)\s*\]", r"DELAY(\1, \2)", expr)
        if new == expr:
            break
        expr = new

    # Handle SMA(A, n) -> SMA(A, n, 1) (alphaagent signature)
    expr = re.sub(r"\bSMA\(([^,]+),\s*(\d+)\s*\)", r"SMA(\1, \2, 1)", expr)

    return expr


def _polars_to_factor_pandas(series_df) -> "Any":
    """
    Convert loop polars df -> pandas df with MultiIndex (datetime, instrument)
    and columns like '$open', '$close', ...
    """
    import pandas as pd

    required = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in getattr(series_df, "columns", [])]
    if missing:
        raise ValueError(f"series_df is missing required columns for factor_coder backend: {missing}")

    extra_cols = [
        c
        for c in ["tradingvalue", "marketcap", "sharesoutstanding"]
        if c in getattr(series_df, "columns", [])
    ]
    select_cols = [*required, *extra_cols]

    pdf = series_df.select(select_cols).to_pandas()
    pdf = pdf.rename(columns={"timestamp": "datetime", "ticker": "instrument"})
    value_cols = [c for c in pdf.columns if c not in ("datetime", "instrument")]
    pdf = pdf.rename(columns={c: f"${c}" for c in value_cols})
    pdf = pdf.set_index(["datetime", "instrument"]).sort_index()
    return pdf


def _eval_factor_expression(expr: str, df) -> Tuple["Any", str | None]:
    """
    Evaluate AlphaAgent factor expression on pandas dataframe `df` (MultiIndex datetime/instrument).
    Returns (series_or_df, error_msg).
    """
    import numpy as np
    import pandas as pd

    if not expr:
        return None, "Empty expression."

    from coder.factor_coder.expr_parser import parse_expression, parse_symbol
    from coder.factor_coder import function_lib as fl

    try:
        rendered = parse_symbol(expr, df.columns)
        rendered = parse_expression(rendered)

        # Replace bare symbols (open/close/...) with df['$open'] etc, matching the template logic.
        for col in sorted(list(df.columns), key=len, reverse=True):
            if not isinstance(col, str) or not col.startswith("$"):
                continue
            rendered = rendered.replace(col[1:], f"df[{col!r}]")

        env: Dict[str, Any] = {"df": df, "np": np, "pd": pd}
        env.update({k: v for k, v in fl.__dict__.items() if k and k[0].isupper()})
        out = eval(rendered, env)
        return out, None
    except Exception as e:
        return None, str(e)


def _factor_output_to_polars(out, factor_name: str):
    import pandas as pd
    import polars as pl

    if out is None:
        return pl.DataFrame({"timestamp": [], "ticker": [], factor_name: []})

    if isinstance(out, pd.DataFrame):
        if out.shape[1] == 1:
            s = out.iloc[:, 0]
        else:
            s = out.mean(axis=1)
    elif isinstance(out, pd.Series):
        s = out
    else:
        # numpy array or scalar -> coerce to series aligned to index if possible
        try:
            s = pd.Series(out)
        except Exception:
            s = pd.Series([out])

    s = s.astype("float64", errors="ignore")
    df = s.to_frame(name=factor_name).reset_index()
    df = df.rename(columns={"datetime": "timestamp", "instrument": "ticker"})
    return pl.from_pandas(df)


class FactorCoderCodeAgent(BaseAgent):
    """
    Code backend that leverages `coder/factor_coder`'s expression-oriented prompts to:
    - (optionally) convert legacy definition -> AlphaAgent expression
    - execute expression on the current dataset
    - on failure, ask LLM to propose a corrected expression

    Output contract matches `CodeAgent.execute_code_to_factors`:
      returns (polars_df_with_factors, code_results_list)

    NOTE:
    - This backend produces *expressions* (not polars python code). Downstream re-apply needs support.
    """

    def __init__(
        self,
        model: str,
        run_ctx: Optional["RunContext"] = None,
        max_loop: int = 8,
        alignment_check: bool = True,
    ) -> None:
        super().__init__(model=model, run_ctx=run_ctx)
        self.max_loop = max_loop
        self.alignment_check = alignment_check
        self._prompts = None

    def _get_prompts(self):
        if self._prompts is not None:
            return self._prompts
        from core.prompts import Prompts

        p = Prompts(file_path=Path(__file__).resolve().parents[1] / "coder" / "factor_coder" / "prompts_alphaagent.yaml")
        self._prompts = p
        return p

    def _propose_fixed_expr(
        self,
        *,
        factor_name: str,
        factor_description: str,
        former_expression: str,
        former_feedback: str,
        available_columns: List[str],
    ) -> str | None:
        from jinja2 import Environment, StrictUndefined

        from schemas.factor_expression import FACTOR_EXPRESSION_TOOL
        from util.llm_client import call_llm

        prompts = self._get_prompts()

        scenario = (
            "Data is a daily panel with MultiIndex (datetime, instrument).\n"
            f"Available variables: {', '.join('$'+c for c in available_columns)}.\n"
            "Your expression must be executable using only the allowed ops listed in the system prompt."
        )

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompts["evolving_strategy_factor_implementation_v1_system"])
            .render(scenario=scenario)
            .strip("\n")
        )

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompts["evolving_strategy_factor_implementation_v2_user"])
            .render(
                factor_information_str=f"factor_name: {factor_name}\nfactor_description: {factor_description}",
                former_expression=former_expression,
                former_feedback=former_feedback,
                queried_similar_error_knowledge=[],
                error_summary_critics=None,
                similar_successful_factor_description=None,
                similar_successful_expression=None,
                latest_attempt_to_latest_successful_execution=None,
            )
            .strip("\n")
        )

        resp = call_llm(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=[FACTOR_EXPRESSION_TOOL],
            target_tool_name="factor_expression_tool",
            temperature=0.1,
            react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
            react_agent_name="factor_coder_generate",
            context="Stage1: Factor Expression Generation",
        )
        if isinstance(resp, dict):
            expr = resp.get("expr")
            if isinstance(expr, str) and expr.strip():
                return expr.strip()
        return None

    def _comment_on_expression(
        self,
        *,
        factor_name: str,
        factor_description: str,
        expression: str,
        execution_feedback: str,
        available_columns: List[str],
    ) -> str | None:
        from jinja2 import Environment, StrictUndefined
        from util.llm_client import call_llm

        prompts = self._get_prompts()
        scenario = (
            "Data is a daily panel with MultiIndex (datetime, instrument).\n"
            f"Available variables: {', '.join('$'+c for c in available_columns)}.\n"
            "Only allowed operations in the expression are those listed in the system prompt."
        )

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompts["evaluator_code_feedback_v1_system"])
            .render(scenario=scenario)
            .strip("\n")
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompts["evaluator_code_feedback_v1_user"])
            .render(
                factor_information=f"factor_name: {factor_name}\nfactor_description: {factor_description}",
                code=expression,
                execution_feedback=execution_feedback,
                value_feedback=None,
                gt_code=None,
            )
            .strip("\n")
        )

        resp = call_llm(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=None,
            temperature=0.1,
            react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
            react_agent_name="factor_coder_comment",
            context="Stage1: Factor Expression Feedback",
        )
        if isinstance(resp, str) and resp.strip():
            return resp.strip()
        return None

    def execute_code_to_factors(
        self,
        factors_response: Dict[str, Any],
        series_df,
        metadata: list | None = None,
        operators: list | None = None,  # noqa: ARG002
        coding_context: str = "",  # noqa: ARG002
    ):
        import polars as pl

        if metadata is None:
            metadata = list(getattr(series_df, "columns", []))

        # Build a pandas panel once per call.
        base_pdf = _polars_to_factor_pandas(series_df)

        result_df = series_df.clone()
        code_results: List[Dict[str, Any]] = []

        factors = factors_response.get("factors", []) if isinstance(factors_response, dict) else []
        for factor in factors:
            if not isinstance(factor, dict):
                continue

            name = factor.get("name") or "Unknown"
            description = factor.get("description") or ""
            definition = factor.get("definition") or ""

            expr = factor.get("factor_expression") or factor.get("expression")
            if not isinstance(expr, str) or not expr.strip():
                expr = _definition_to_alpha_expression(definition)

            last_error = None
            final_expr = None
            alignment_comment = None
            for attempt in range(max(1, self.max_loop)):
                out, err = _eval_factor_expression(expr, base_pdf)
                if err is None:
                    if self.alignment_check and alignment_comment is None:
                        try:
                            comment = self._comment_on_expression(
                                factor_name=name,
                                factor_description=description,
                                expression=expr,
                                execution_feedback="Execution succeeded.",
                                available_columns=[
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                    "tradingvalue",
                                    "marketcap",
                                    "sharesoutstanding",
                                ],
                            )
                            if comment and "no comment found" not in comment.lower():
                                alignment_comment = comment
                                fixed = self._propose_fixed_expr(
                                    factor_name=name,
                                    factor_description=description,
                                    former_expression=expr,
                                    former_feedback=f"ALIGNMENT_COMMENT:\n{comment}",
                                    available_columns=[
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                        "tradingvalue",
                                        "marketcap",
                                        "sharesoutstanding",
                                    ],
                                )
                                if fixed:
                                    expr = fixed
                                    continue
                        except Exception:
                            pass
                    final_expr = expr
                    factor_pl = _factor_output_to_polars(out, name)
                    # Join into result_df
                    if factor_pl.height > 0:
                        # Overwrite semantics: avoid Polars auto-suffixing ("_right") on name collisions.
                        # We may retry the same factor multiple times; keep a single canonical column name.
                        cols = set(result_df.columns)
                        if name in cols:
                            result_df = result_df.drop(name)
                            cols.discard(name)
                        suffix = "_right"
                        while f"{name}{suffix}" in cols:
                            col_to_drop = f"{name}{suffix}"
                            result_df = result_df.drop(col_to_drop)
                            cols.discard(col_to_drop)
                            suffix = f"{suffix}_right"
                        result_df = result_df.join(factor_pl, on=["timestamp", "ticker"], how="left")
                    else:
                        result_df = result_df.with_columns(pl.lit(None).alias(name))
                    break

                last_error = err
                fixed = self._propose_fixed_expr(
                    factor_name=name,
                    factor_description=description,
                    former_expression=expr,
                    former_feedback=err,
                    available_columns=[
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "tradingvalue",
                        "marketcap",
                        "sharesoutstanding",
                    ],
                )
                if not fixed:
                    break
                expr = fixed

            code_results.append(
                {
                    "factor_name": name,
                    "code_backend": "factor_coder",
                    "codes": [
                        {
                            "filename": "factor_expression",
                            "entry_point": "expression",
                            "implementation": final_expr or expr,
                            "description": description,
                            "notes": (
                                f"last_error={last_error}; alignment_comment={alignment_comment}"
                                if (last_error or alignment_comment)
                                else ""
                            ),
                            "factor_expression": final_expr or expr,
                        }
                    ],
                }
            )

        return result_df, code_results
