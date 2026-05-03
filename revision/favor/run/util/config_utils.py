from __future__ import annotations

from typing import Tuple

from run.config import RDConfig, load_rd_config


def resolve_cfg(cfg: RDConfig | None) -> RDConfig:
    return cfg or load_rd_config()


def resolve_model(model: str | None, cfg: RDConfig) -> str:
    return model or cfg.llm.model_name


def resolve_stage1_params(
    *,
    cfg: RDConfig,
    allowed_columns: list[str] | None,
    refine_rounds: int | None,
) -> tuple[list[str], int]:
    columns = allowed_columns or cfg.stage1.allowed_ohlcv_columns
    rounds = cfg.stage1.refine_rounds if refine_rounds is None else refine_rounds
    return columns, rounds


def resolve_stage2_params(
    *,
    cfg: RDConfig,
    n_quantiles: int | None,
    monotonicity_threshold: float | None,
) -> Tuple[int, float]:
    nq = cfg.stage2.n_quantiles if n_quantiles is None else n_quantiles
    mt = cfg.stage2.monotonicity_threshold if monotonicity_threshold is None else monotonicity_threshold
    return nq, mt


def resolve_stage3_params(
    *,
    cfg: RDConfig,
    horizon_days: int | None,
    monotonicity_threshold: float | None,
) -> tuple[int, float]:
    h = cfg.stage3.horizon_days if horizon_days is None else horizon_days
    mt = cfg.stage3.monotonicity_threshold if monotonicity_threshold is None else monotonicity_threshold
    return h, mt
