from __future__ import annotations

import logging
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

from agent.base_agent import BaseAgent

if TYPE_CHECKING:
    from util.run_context import RunContext

logger = logging.getLogger(__name__)


class _FinanceFactorScenario:
    """
    Minimal Scenario implementation for CoSTEER factor_coder.
    This is only used to provide prompt context; it does not drive execution.
    """

    @property
    def background(self) -> str:
        return (
            "You are implementing daily cross-sectional alpha factors over KOSPI tickers.\n"
            "Data is a panel indexed by (datetime, instrument) with OHLCV-like features.\n"
            "NOTE: The input dataframe columns are '$open', '$high', '$low', '$close', '$volume',\n"
            "'$tradingvalue', '$marketcap', '$sharesoutstanding' (prefixed with '$').\n"
        )

    @property
    def interface(self) -> str:
        return (
            "Implement factor expression inside the provided jinja2 template.\n"
            "The code will read './daily_pv.h5' (key='data') and must write 'result.h5' (key='data')."
            "\nYou may use pandas.read_hdf / Series.to_hdf as in the template (h5py is also available if needed)."
        )

    @property
    def output_format(self) -> str:
        return "A pandas Series/DataFrame indexed by (datetime, instrument), saved to result.h5."

    @property
    def simulator(self) -> str:
        return "No external simulator; correctness is evaluated by successful execution and internal critics."

    @property
    def rich_style_description(self) -> str:
        return self.get_scenario_all_desc()

    def get_scenario_all_desc(
        self,
        task: Any | None = None,  # noqa: ARG002
        filtered_tag: str | None = None,  # noqa: ARG002
        simple_background: bool | None = None,  # noqa: ARG002
    ) -> str:
        return "\n".join(
            [
                self.background,
                "Interface:\n" + self.interface,
                "Output:\n" + self.output_format,
            ]
        )


def _polars_to_costeer_hdf(series_df, *, out_path: Path) -> None:
    """
    Write the current panel into the exact format expected by `coder/factor_coder/template.jinjia2`:
    - pandas DataFrame with MultiIndex (datetime, instrument)
    - columns are '$open', '$high', ... (prefixed with '$')
    - stored at out_path as HDF key='data'
    """
    import pandas as pd

    required = [
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
    missing = [c for c in required if c not in getattr(series_df, "columns", [])]
    if missing:
        raise ValueError(f"series_df missing required columns for CoSTEER factor_coder: {missing}")

    pdf = series_df.select(required).to_pandas()
    pdf = pdf.rename(columns={"timestamp": "datetime", "ticker": "instrument"})
    value_cols = [c for c in pdf.columns if c not in ("datetime", "instrument")]
    # CoSTEER factor_coder template historically expects `$`-prefixed columns.
    # However, some generated code may still access non-prefixed names (e.g. `df["low"]`).
    # To reduce brittle KeyErrors, we provide both:
    # - `$low` (canonical)
    # - `low`  (alias)
    pdf = pdf.rename(columns={c: f"${c}" for c in value_cols})
    for c in value_cols:
        pdf[c] = pdf[f"${c}"]
    pdf = pdf.set_index(["datetime", "instrument"]).sort_index()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_hdf(out_path, key="data")


def _extract_expr_from_factor_py(code_str: str) -> str:
    # Matches: expr = "..." or expr = '...'
    m = re.search(r"expr\\s*=\\s*['\\\"]([^'\\\"]*)['\\\"]", code_str or "")
    return m.group(1) if m else ""


def _series_from_costeer_output(executed) -> Tuple["Any", str]:
    """
    Normalize CoSTEER factor output read from result.h5 into pandas Series with MultiIndex.
    Returns (series, kind_str).
    """
    import pandas as pd

    if executed is None:
        return None, "none"
    if isinstance(executed, pd.Series):
        return executed, "series"
    if isinstance(executed, pd.DataFrame):
        if executed.shape[1] == 1:
            return executed.iloc[:, 0], "df1"
        return executed.mean(axis=1), "dfN"
    try:
        return pd.Series(executed), "coerced"
    except Exception:
        return None, "unknown"


def _pandas_series_to_polars_frame(series, *, factor_name: str):
    import polars as pl

    if series is None:
        return pl.DataFrame({"timestamp": [], "ticker": [], factor_name: []})
    df = series.to_frame(name=factor_name).reset_index()
    df = df.rename(columns={"datetime": "timestamp", "instrument": "ticker"})
    return pl.from_pandas(df)


class CoSTEERFullCodeAgent(BaseAgent):
    """
    End-to-end CoSTEER (Experiment/Workspace) runner for factor coding.

    It runs `coder/factor_coder/FactorCoSTEER` over FactorTask list and returns:
      (polars_df_with_factors, code_results_list)
    in the same shape as `CodeAgent.execute_code_to_factors`.
    """

    def __init__(
        self,
        model: str,
        run_ctx: Optional["RunContext"] = None,
        data_cache_dir: Path | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        super().__init__(model=model, run_ctx=run_ctx)
        self.scen = _FinanceFactorScenario()
        self.repo_root = Path(__file__).resolve().parents[1]
        self.data_cache_dir = data_cache_dir or (self.repo_root / "git_ignore_folder" / "factor_implementation_source_data_debug")

        # Optional: isolate workspaces when used as a library from another project/workspace.
        # This controls where file-based workspaces are created (RD_AGENT_SETTINGS.workspace_path).
        try:
            from core.conf import RD_AGENT_SETTINGS

            if workspace_root is not None:
                RD_AGENT_SETTINGS.workspace_path = Path(workspace_root)

            # Avoid incorrect cache reuse across different datasets.
            RD_AGENT_SETTINGS.cache_with_pickle = False
        except Exception:
            pass

        # Ensure factor_coder uses our dataset folder.
        try:
            from coder.factor_coder.config import FACTOR_COSTEER_SETTINGS

            FACTOR_COSTEER_SETTINGS.data_folder_debug = str(self.data_cache_dir)
            FACTOR_COSTEER_SETTINGS.data_folder = str(self.data_cache_dir)
        except Exception:
            pass

    def execute_code_to_factors(
        self,
        factors_response: Dict[str, Any],
        series_df,
        metadata: list | None = None,  # noqa: ARG002
        operators: list | None = None,  # noqa: ARG002
        coding_context: str = "",  # noqa: ARG002
    ):
        import polars as pl

        def _get_allowed_expr_symbols() -> list[str]:
            # Keep in sync with `_polars_to_costeer_hdf`.
            base = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tradingvalue",
                "marketcap",
                "sharesoutstanding",
            ]
            return [f"${c}" for c in base] + base

        def _get_allowed_expr_functions() -> list[str]:
            # The factor_coder template eval env exposes upper-case callables from function_lib.
            try:
                from coder.factor_coder import function_lib as fl

                funcs: list[str] = []
                for name, obj in fl.__dict__.items():
                    if not name or not name[0].isupper():
                        continue
                    if callable(obj):
                        funcs.append(name)
                # Keep stable order for prompts/caching.
                return sorted(set(funcs))
            except Exception:
                # Conservative fallback (core subset).
                return [
                    "DELAY",
                    "RANK",
                    "TS_MAX",
                    "TS_MEAN",
                    "TS_MIN",
                    "TS_STD",
                    "TS_SUM",
                    "TS_ZSCORE",
                ]

        def _compile_definition_to_expr_llm(
            *,
            factor_name: str,
            factor_description: str,
            factor_formulation: str,
        ) -> str | None:
            """
            One-shot "spec compiler": turn natural-language formulation into an AlphaAgent expression DSL string.
            This reduces drift between factor intent and executable code by fixing the initial expression.
            """
            try:
                from util.llm_client import call_llm
                from util.json_utils import strip_code_fence
            except Exception:
                return None

            allowed_cols = _get_allowed_expr_symbols()
            allowed_funcs = _get_allowed_expr_functions()

            system_prompt = (
                "You compile factor formulations into a single factor expression.\n"
                "Output MUST be valid JSON, with a single key 'expr'. No extra keys.\n"
                "The expression must be compatible with the provided DSL.\n"
                "Hard constraints:\n"
                "- Use ONLY allowed columns and allowed functions.\n"
                "- Do NOT reference any other columns (e.g., ATR_14, Close).\n"
                "- Do NOT use pandas methods like .rolling/.shift; use TS_* and DELAY.\n"
                "- Rank semantics: use RANK(x) for cross-sectional rank, TS_RANK(x,N) only for time-series rank.\n"
                "- If you need t-1 alignment, write it explicitly with DELAY(col, 1) and/or TS_*(DELAY(col,1), N).\n"
                "- Do NOT write python code; output only the expression string.\n"
            )
            user_prompt = (
                f"Factor name: {factor_name}\n"
                f"Factor description: {factor_description}\n"
                f"Factor formulation (natural language): {factor_formulation}\n\n"
                f"Allowed columns: {', '.join(allowed_cols)}\n"
                f"Allowed functions: {', '.join(allowed_funcs)}\n\n"
                "Return JSON like:\n"
                '{\"expr\": \"TS_ZSCORE($close, 20)\"}\n'
            )

            raw = call_llm(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=None,
                temperature=0.1,
                react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
                react_agent_name="costeer_full_code_agent",
                context="Stage1: Factor Code Generation",
            )
            if not raw:
                return None
            raw = strip_code_fence(raw)
            try:
                obj = json.loads(raw)
            except Exception:
                return None
            expr = obj.get("expr")
            if not isinstance(expr, str) or not expr.strip():
                return None
            return expr.strip()

        def _normalize_definition_to_expr(definition: str) -> str:
            """
            Best-effort normalization of FactorAgent 'definition' into executable DSL.
            Goal: make `definition` and executed `expr` match (single source of truth).

            We prefer an explicit DSL style:
              - use UPPERCASE function names
              - use DELAY(col, 1) instead of x[-1]/shift(1)
            """
            s = (definition or "").strip()
            if not s:
                return ""

            # If the definition already looks like a DSL expression, keep it (with minimal normalization).
            # This is intentionally conservative to avoid semantic drift.
            # Normalize common lowercase function names to the canonical UPPERCASE names used by function_lib.
            repl = {
                r"\bdelay\s*\(": "DELAY(",
                r"\brank\s*\(": "RANK(",
                r"\bts_rank\s*\(": "TS_RANK(",
                r"\bts_min\s*\(": "TS_MIN(",
                r"\bts_max\s*\(": "TS_MAX(",
                r"\bts_mean\s*\(": "TS_MEAN(",
                r"\bstddev\s*\(": "TS_STD(",
                r"\bts_std\s*\(": "TS_STD(",
                r"\bsma\s*\(": "SMA(",
                r"\bema\s*\(": "EMA(",
                r"\babs\s*\(": "ABS(",
                r"\bmin\s*\(": "MIN(",
                r"\bmax\s*\(": "MAX(",
            }
            for pat, rep in repl.items():
                s = re.sub(pat, rep, s, flags=re.IGNORECASE)

            # Convert common bracket lag syntax into DELAY(), if present.
            # Examples: close[-1] -> DELAY(close, 1), TS_MIN(low,10)[-2] -> DELAY(TS_MIN(low,10), 2)
            # NOTE: This is a syntactic normalization only; FactorAgent is instructed to avoid [].
            s = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*\b)\s*\[\s*-(\d+)\s*\]", r"DELAY(\1, \2)", s)
            s = re.sub(r"(\))\s*\[\s*-(\d+)\s*\]", r"DELAY(\1, \2)", s)
            return s.strip()

        # 1) Prepare input data file expected by factor_coder
        daily_h5 = Path(self.data_cache_dir) / "daily_pv.h5"
        _polars_to_costeer_hdf(series_df, out_path=daily_h5)

        # 2) Build FactorTasks
        from coder.factor_coder.factor import FactorTask
        from core.experiment import Experiment

        factors = factors_response.get("factors", []) if isinstance(factors_response, dict) else []
        tasks: List[FactorTask] = []
        for f in factors:
            if not isinstance(f, dict):
                continue
            name = f.get("name") or ""
            if not name:
                continue
            desc = f.get("description") or ""
            definition = f.get("definition") or ""
            # Single source of truth preference order:
            # 1) explicit expr fields (already canonicalized by execution backend)
            # 2) definition-as-DSL (FactorAgent now emits DSL; normalize)
            # 3) LLM spec compiler (definition -> expr)
            # 4) heuristic fallback
            expr = (f.get("expression") or f.get("factor_expression") or "").strip()
            if not expr:
                normalized = _normalize_definition_to_expr(definition)
                if normalized:
                    expr = normalized
            if not expr:
                compiled = _compile_definition_to_expr_llm(
                    factor_name=name,
                    factor_description=desc,
                    factor_formulation=definition,
                )
                if compiled:
                    expr = compiled
            if not expr:
                try:
                    from agent.factor_coder_code_agent import _definition_to_alpha_expression

                    expr = _definition_to_alpha_expression(definition)
                except Exception:
                    expr = ""
            tasks.append(
                FactorTask(
                    factor_name=name,
                    factor_description=desc,
                    factor_formulation=definition,
                    factor_expression=expr,
                    factor_implementation=True,
                )
            )

        exp = Experiment(sub_tasks=tasks)

        # 3) Run CoSTEER end-to-end
        # Use the expression-first CoSTEER runner to keep implementation tightly aligned to the
        # factor_coder template (i.e., evolve `expr` rather than arbitrary python).
        from coder.factor_coder import FactorParser

        developer = FactorParser(scen=self.scen)
        developed = developer.develop(exp)

        # 4) Collect outputs and attach factors to df
        result_df = series_df.clone()
        code_results: List[Dict[str, Any]] = []

        for idx, task in enumerate(tasks):
            ws = None
            try:
                ws = developed.sub_workspace_list[idx]
            except Exception:
                ws = None

            factor_name = task.factor_name
            if ws is None:
                code_results.append(
                    {
                        "factor_name": factor_name,
                        "code_backend": "costeer_full",
                        "codes": [
                            {
                                "filename": "factor.py",
                                "entry_point": "calculate_factor",
                                "implementation": "",
                                "description": task.factor_description,
                                "notes": "No workspace produced.",
                                "factor_expression": task.factor_expression,
                            }
                        ],
                    }
                )
                continue

            try:
                exec_feedback, executed = ws.execute(data_type="Debug")
            except Exception as e:
                exec_feedback, executed = f"execute_error: {e}", None

            series, out_kind = _series_from_costeer_output(executed)
            try:
                factor_pl = _pandas_series_to_polars_frame(series, factor_name=factor_name)
                if factor_pl.height > 0:
                    # Polars join keeps both columns when names collide by appending the default
                    # suffix "_right" to the incoming (right) column. In our pipeline we want the
                    # latest computed factor to overwrite any existing factor column with the same
                    # name (e.g., across retries/refine rounds), otherwise downstream logic will
                    # start seeing "<factor>_right" and lose name alignment with the factor spec.
                    cols = set(result_df.columns)
                    if factor_name in cols:
                        result_df = result_df.drop(factor_name)
                        cols.discard(factor_name)
                    # Drop any previously created collision columns like "<factor>_right", "<factor>_right_right", ...
                    suffix = "_right"
                    while f"{factor_name}{suffix}" in cols:
                        col_to_drop = f"{factor_name}{suffix}"
                        result_df = result_df.drop(col_to_drop)
                        cols.discard(col_to_drop)
                        suffix = f"{suffix}_right"
                    result_df = result_df.join(factor_pl, on=["timestamp", "ticker"], how="left")
                else:
                    result_df = result_df.with_columns(pl.lit(None).alias(factor_name))
            except Exception as e:
                exec_feedback = f"{exec_feedback}\nattach_error: {e}"
                result_df = result_df.with_columns(pl.lit(None).alias(factor_name))

            impl = ""
            try:
                impl = (ws.code_dict or {}).get("factor.py", "") or ws.code
            except Exception:
                impl = ""

            expr_in_impl = _extract_expr_from_factor_py(impl)
            expr_final = expr_in_impl or (task.factor_expression or "")
            # Enforce single source of truth in the returned artifact.
            # If the workspace mutated the expression, prefer the executed expr (from factor.py).
            extra_note = ""
            if expr_in_impl and task.factor_expression and expr_in_impl != task.factor_expression:
                extra_note = " expr_mismatch(task_vs_impl)=true;"

            code_results.append(
                {
                    "factor_name": factor_name,
                    "code_backend": "costeer_full",
                    "codes": [
                        {
                            "filename": "factor.py",
                            "entry_point": "calculate_factor",
                            "implementation": impl,
                            "description": task.factor_description,
                            "notes": f"output_kind={out_kind};{extra_note} feedback={exec_feedback[:500]}",
                            "factor_expression": expr_final,
                        }
                    ],
                }
            )

        return result_df, code_results
