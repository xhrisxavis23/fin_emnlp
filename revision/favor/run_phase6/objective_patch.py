"""
Phase 6 monkey-patch: Stage 4 Optuna objective with branching.

Goal:
    Replace `_create_objective` in `run.pipeline.stage4`,
    `run.pipeline.stage4_parallel`, and `run.pipeline.stage4_parallel_per_combo`
    with a version that reads `FAVOR_STAGE4_OBJECTIVE` env var and returns one of:

        ir            (paper-aligned default; identical to original)
        calmar        (= ann_return / max(|max_drawdown|, eps))
        ir_minus_mdd  (= information_ratio - lambda * |max_drawdown|, lambda from env)

Why monkey-patch (not file copy):
    The original `_create_objective` is 190 lines defining nested closures
    (`_simulate_positions_fast_from_signal`, `_calculate_portfolio_returns_fast`,
    `objective`). Copying the whole body to a separate file would duplicate it
    and risk drift. Instead, we re-define the function here and rebind the
    name in all importing modules — frozen source files remain untouched on disk.

Env vars consumed:
    FAVOR_STAGE4_OBJECTIVE  ∈ {"ir", "calmar", "ir_minus_mdd"}    default "ir"
    FAVOR_STAGE4_MDD_LAMBDA float, used only when mode=="ir_minus_mdd"  default 2.0
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Helpers from the frozen stage4 module — top-level functions, safe to import.
from run.pipeline.stage4 import (
    _compute_thresholds_cached,
    _apply_signal,
    _calc_excess_metrics,
    _compute_metrics_from_returns,
)


def _create_objective_v2(
    *,
    is_panel: pd.DataFrame,
    passed_formulas: List[Dict[str, Any]],
    cfg,
    benchmark_returns: pd.Series = None,
    verbose: bool = False,
):
    """Phase 6 objective factory with FAVOR_STAGE4_OBJECTIVE branching."""
    formula_names = [f["name"] for f in passed_formulas if "name" in f]

    objective_mode = os.getenv("FAVOR_STAGE4_OBJECTIVE", "ir").strip().lower()
    if objective_mode not in {"ir", "calmar", "ir_minus_mdd"}:
        objective_mode = "ir"

    try:
        mdd_lambda = float(os.getenv("FAVOR_STAGE4_MDD_LAMBDA", "2.0"))
    except ValueError:
        mdd_lambda = 2.0

    quantile_cache: Dict[tuple, Dict[str, float]] = {}

    horizon_days = int(getattr(cfg.stage4, "horizon_days", 5))
    stop_loss_threshold = getattr(cfg.stage4, "stop_loss_threshold", -0.05)
    avg_cost = float((cfg.qlib.open_cost + cfg.qlib.close_cost) / 2)

    date_codes, unique_dates = pd.factorize(is_panel["timestamp"], sort=True)
    n_dates = int(len(unique_dates))

    closes_all = is_panel["close"].astype(float).to_numpy()
    tick_groups: list[np.ndarray] = []
    tick_closes: list[np.ndarray] = []

    groups = is_panel.groupby("ticker", sort=False).groups
    for idx in groups.values():
        loc = np.asarray(idx, dtype=np.int64)
        tick_groups.append(loc)
        tick_closes.append(closes_all[loc])

    ret1 = np.full(int(is_panel.shape[0]), np.nan, dtype=float)
    for loc, c in zip(tick_groups, tick_closes, strict=True):
        if c.shape[0] >= 2:
            ret1[loc[:-1]] = (c[1:] / c[:-1]) - 1.0

    def _simulate_positions_fast_from_signal(signal_arr: np.ndarray) -> np.ndarray:
        pos = np.zeros(signal_arr.shape[0], dtype=bool)
        for loc, c in zip(tick_groups, tick_closes, strict=True):
            sig_t = signal_arr[loc]
            if not sig_t.any():
                continue

            n = int(sig_t.shape[0])
            pos_t = np.zeros(n, dtype=bool)
            entries = np.flatnonzero(sig_t)
            if entries.size == 0:
                continue

            p = 0
            last_exit = -1
            while p < entries.size:
                e = int(entries[p])
                if e <= last_exit:
                    p += 1
                    continue

                exit_idx = min(e + horizon_days, n)

                if stop_loss_threshold is not None:
                    entry_price = float(c[e])
                    end = min(e + horizon_days, n - 1)
                    if entry_price > 0.0 and (e + 1) <= end:
                        future = (c[(e + 1):(end + 1)] / entry_price) - 1.0
                        bad = np.flatnonzero(future <= float(stop_loss_threshold))
                        if bad.size:
                            exit_idx = (e + 1) + int(bad[0])

                if exit_idx > e:
                    pos_t[e:exit_idx] = True
                last_exit = exit_idx

                p = int(np.searchsorted(entries, last_exit + 1, side="left"))

            pos[loc] = pos_t
        return pos

    def _calculate_portfolio_returns_fast(position: np.ndarray) -> Dict[str, Any]:
        valid = position & np.isfinite(ret1)
        if not valid.any():
            empty_idx = pd.Index(unique_dates)
            z = pd.Series(0.0, index=empty_idx)
            return {
                "gross": z,
                "net": z,
                "eq_gross": (1.0 + z).cumprod(),
                "eq_net": (1.0 + z).cumprod(),
                "holdings": pd.Series(0, index=empty_idx, dtype=int),
                "turnover": z,
                "cost": z,
            }

        holdings = np.bincount(date_codes, weights=valid.astype(np.int8), minlength=n_dates).astype(int)
        denom = holdings[date_codes]

        weight = np.zeros_like(ret1)
        mask = valid & (denom > 0)
        weight[mask] = 1.0 / denom[mask]

        ret1_safe = np.nan_to_num(ret1, nan=0.0)
        gross_by_date = np.bincount(date_codes, weights=weight * ret1_safe, minlength=n_dates).astype(float)

        absdiff = np.zeros_like(weight)
        for loc in tick_groups:
            w = weight[loc]
            if w.size:
                prev = np.empty_like(w)
                prev[0] = 0.0
                prev[1:] = w[:-1]
                absdiff[loc] = np.abs(w - prev)

        turnover_by_date = 0.5 * np.bincount(date_codes, weights=absdiff, minlength=n_dates).astype(float)
        if turnover_by_date.size:
            turnover_by_date[0] = 0.0

        cost_by_date = turnover_by_date * avg_cost

        idx = pd.Index(unique_dates)
        gross = pd.Series(gross_by_date, index=idx)
        cost = pd.Series(cost_by_date, index=idx)
        net = gross - cost

        return {
            "gross": gross,
            "net": net,
            "eq_gross": (1.0 + gross).cumprod(),
            "eq_net": (1.0 + net).cumprod(),
            "holdings": pd.Series(holdings, index=idx, dtype=int),
            "turnover": pd.Series(turnover_by_date, index=idx),
            "cost": cost,
        }

    def objective(trial: Any) -> float:
        step = getattr(cfg.stage4, "threshold_step", 0.05)
        threshold_choices = [
            round(cfg.stage4.threshold_min + i * step, 2)
            for i in range(int((cfg.stage4.threshold_max - cfg.stage4.threshold_min) / step) + 1)
        ]

        threshold_dict = {}
        for fname in formula_names:
            threshold_dict[fname] = trial.suggest_categorical(
                f"threshold_{fname}",
                threshold_choices,
            )

        thresholds = _compute_thresholds_cached(
            train_panel=is_panel,
            passed_formulas=passed_formulas,
            threshold_dict=threshold_dict,
            cache=quantile_cache,
        )

        is_signal = _apply_signal(
            panel=is_panel,
            thresholds=thresholds,
            passed_formulas=passed_formulas,
        )

        is_position = _simulate_positions_fast_from_signal(is_signal.to_numpy(dtype=bool))
        is_returns = _calculate_portfolio_returns_fast(is_position)

        # Resolve (ir, ann_return, mdd) from the proper source (excess vs benchmark when available).
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            portfolio_gross = is_returns["gross"]
            common_idx = portfolio_gross.index.intersection(benchmark_returns.index)
            if len(common_idx) > 1:
                excess_return = portfolio_gross.loc[common_idx] - benchmark_returns.loc[common_idx]
                m = _calc_excess_metrics(excess_return)
            else:
                m = _compute_metrics_from_returns(is_returns)
        else:
            m = _compute_metrics_from_returns(is_returns)

        ir = float(m.get("information_ratio", 0.0))
        ann_return = float(m.get("ann_return", 0.0))
        mdd = float(m.get("max_drawdown", 0.0))

        # Phase 6 objective branching.
        if objective_mode == "calmar":
            score = ann_return / max(abs(mdd), 1e-6)
        elif objective_mode == "ir_minus_mdd":
            score = ir - mdd_lambda * abs(mdd)
        else:  # "ir"
            score = ir

        if np.isnan(score) or np.isinf(score):
            score = -999.0

        if verbose and trial.number % 10 == 0:
            print(
                f"  Trial {trial.number}: mode={objective_mode} "
                f"IR={ir:+.3f} AR={ann_return:+.3f} MDD={mdd:+.3f} score={score:+.3f}"
            )

        return score

    return objective


def apply() -> dict:
    """
    Rebind `_create_objective` in stage4 and its parallel wrappers to the v2 version.

    Returns a dict describing what was patched, useful for logging / assertions.
    Safe to call multiple times (idempotent).
    """
    from run.pipeline import stage4 as _s4
    from run.pipeline import stage4_parallel as _s4p
    from run.pipeline import stage4_parallel_per_combo as _s4ppc

    patched = {}
    for mod_name, mod in (
        ("stage4", _s4),
        ("stage4_parallel", _s4p),
        ("stage4_parallel_per_combo", _s4ppc),
    ):
        old = getattr(mod, "_create_objective", None)
        setattr(mod, "_create_objective", _create_objective_v2)
        patched[mod_name] = {
            "old": getattr(old, "__qualname__", str(old)),
            "new": _create_objective_v2.__qualname__,
        }

    return patched
