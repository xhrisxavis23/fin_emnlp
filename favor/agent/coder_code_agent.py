from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agent.base_agent import BaseAgent

if TYPE_CHECKING:
    from util.run_context import RunContext

logger = logging.getLogger(__name__)


class CoderCodeAgent(BaseAgent):
    """
    Drop-in alternative to `agent/code_agent.py` that uses the CoSTEER-like
    retry + memory backend implemented in `coder/polars_factor_coder/`.

    It preserves the same public method used by finance loops:
      - execute_code_to_factors(factors_response, series_df, ...)
    """

    def __init__(self, model: str, run_ctx: Optional["RunContext"] = None) -> None:
        super().__init__(model=model, run_ctx=run_ctx)
        # Lazy import so importing this module doesn't require the LLM client deps.
        from coder.polars_factor_coder import PolarsCoSTEERCodeAgent

        self._backend = PolarsCoSTEERCodeAgent(model=model)

    def execute_code_to_factors(
        self,
        factors_response: Dict[str, Any],
        series_df,
        metadata: list | None = None,
        operators: list | None = None,
        coding_context: str = "",
    ):
        import polars as pl

        required_columns = [
            "timestamp",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tradingvalue",
            "marketcap",
            "sharesoutstanding",
        ]
        for col in required_columns:
            if col not in series_df.columns:
                raise ValueError(f"Required column {col} not found in series_df")

        if metadata is None:
            metadata = list(series_df.columns)

        result_df = series_df.clone()
        code_results: List[Dict[str, Any]] = []

        factors = factors_response.get("factors", []) or []
        for factor in factors:
            if not isinstance(factor, dict):
                continue
            try:
                name = factor.get("name", "Unknown")
                logger.info(f"[CoderCodeAgent] Processing factor: {name}")

                code_resp = self._backend.loop(
                    factor=factor,
                    series_df=result_df,
                    metadata=metadata,
                    coding_context=coding_context,
                )
                codes = code_resp.get("codes", []) or []
                if not codes or not isinstance(codes[0], dict):
                    continue
                code_obj = codes[0]

                exec_globals: Dict[str, Any] = {}
                implementation = code_obj.get("implementation", "")
                exec(implementation, exec_globals)
                entry_point = code_obj.get("entry_point", "") or "compute_factor"
                entry_point_name = entry_point.split("(")[0].strip()
                if entry_point_name.startswith("def "):
                    entry_point_name = entry_point_name[4:].strip()
                calc_func = exec_globals.get(entry_point_name)
                if not callable(calc_func):
                    raise ValueError(f"Entry point '{entry_point_name}' not callable.")

                factor_expr = calc_func(result_df)
                if isinstance(factor_expr, pl.Expr):
                    result_df = result_df.with_columns(factor_expr.alias(name))
                    code_resp_with_name = dict(code_resp)
                    code_resp_with_name["factor_name"] = name
                    code_results.append(code_resp_with_name)
                else:
                    logger.warning(
                        f"[CoderCodeAgent] Factor {name} returned {type(factor_expr)}, expected pl.Expr. Skipping."
                    )
            except Exception as e:
                logger.error(f"[CoderCodeAgent] Failed to calculate {factor.get('name')}: {e}")
                continue

        return result_df, code_results
