from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from agent.factor_coder_code_agent import (
    _definition_to_alpha_expression,
    _eval_factor_expression,
    _factor_output_to_polars,
    _polars_to_factor_pandas,
)


def choose_single_ticker(price_df: pl.DataFrame, ticker: str | None) -> str:
    tickers = price_df.select(pl.col("ticker").unique()).to_series().to_list()
    tickers = [t for t in tickers if t is not None]
    if not tickers:
        raise ValueError("No tickers found in price_df.")
    if ticker is not None:
        if ticker not in tickers:
            raise ValueError(f"ticker={ticker!r} not found; available={sorted(tickers)[:20]}")
        return ticker
    if len(tickers) > 1:
        raise ValueError(
            f"Multiple tickers detected ({len(tickers)}). Pass ticker=... to select one explicitly."
        )
    return str(tickers[0])


def to_single_ticker_frames(
    price_with_formulas: pl.DataFrame,
    *,
    ticker: str,
    formula_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    one = (
        price_with_formulas.filter(pl.col("ticker") == ticker)
        .sort("timestamp")
        .select(["timestamp", "open", "high", "low", "close", "volume", *formula_names])
    )
    pdf = one.to_pandas()
    pdf = pdf.set_index("timestamp")

    ohlcv_df = pdf[["open", "high", "low", "close", "volume"]].copy()
    formula_df = pdf[formula_names].copy()
    return ohlcv_df, formula_df


@dataclass
class FormulaComputeResult:
    """Result of formula computation with success/failure tracking."""
    df: pl.DataFrame
    failed_formulas: list[dict[str, Any]]  # list of {"name": ..., "definition": ..., "error": ...}


def compute_formula_values(
    price_df: pl.DataFrame,
    formulas: list[dict[str, Any]],
    *,
    fail_on_error: bool = False,
) -> FormulaComputeResult:
    """
    Compute formula values using the factor_coder expression evaluator (no LLM codegen).

    Returns a FormulaComputeResult containing:
    - df: Polars DataFrame with formula columns joined to the original panel
    - failed_formulas: List of formulas that failed to evaluate (for feedback in next iteration)

    If fail_on_error=True, raises ValueError on first failure (legacy behavior).
    """
    if not isinstance(price_df, pl.DataFrame):
        raise TypeError("price_df must be a polars.DataFrame")

    required = {"timestamp", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(price_df.columns))
    if missing:
        raise ValueError(f"price_df missing required columns: {missing}")

    base_pdf = _polars_to_factor_pandas(price_df)
    result_df = price_df.clone()
    failed_formulas: list[dict[str, Any]] = []

    for formula in formulas:
        if not isinstance(formula, dict):
            continue
        name = str(formula.get("name") or "").strip()
        definition = str(formula.get("definition") or "").strip()
        if not name or not definition:
            continue

        expr = _definition_to_alpha_expression(definition)
        out, err = _eval_factor_expression(expr, base_pdf)
        if err is not None:
            error_info = {
                "name": name,
                "definition": definition,
                "expr": expr,
                "error": str(err),
            }
            if fail_on_error:
                raise ValueError(f"Failed to evaluate formula {name!r}: {err}. expr={expr!r}")
            # Log warning and continue
            import logging
            logging.warning(f"Formula {name!r} failed: {err}. expr={expr!r}. Skipping.")
            failed_formulas.append(error_info)
            # Add null column for failed formula
            result_df = result_df.with_columns(pl.lit(None).alias(name))
            continue

        factor_pl = _factor_output_to_polars(out, name)
        if factor_pl.height == 0:
            result_df = result_df.with_columns(pl.lit(None).alias(name))
            continue

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

    return FormulaComputeResult(df=result_df, failed_formulas=failed_formulas)
