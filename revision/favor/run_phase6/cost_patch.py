"""
Phase 6 cost_patch — env-driven override for cfg.qlib.min_cost.

Why this exists:
    `run/config.py:load_rd_config()` (frozen) hardcodes cost defaults per market:
        - CN (CSI500): open=0.0005, close=0.0015, min=5.0
        - US (SP500):  open=0,      close=0.0005, min=0.0
    The existing env-var override block (FAVOR_LLM_*, FAVOR_STAGE4_*, etc.)
    does NOT cover cost fields. For the SP500 × A02 Calmar sweep we want to
    match the revision baseline (`baseline/split_2y_rerun/configs/*_sp500.yaml`)
    which uses min_cost=5 — the SP500 paper-run convention. Hardcoded 0.0
    in config.py would otherwise create an apples-to-oranges comparison.

Behaviour:
    If env `FAVOR_MIN_COST_OVERRIDE` is set to a numeric value, the patch
    wraps `load_rd_config` and rewrites cfg.qlib.min_cost on every load.
    No-op when the env var is unset → safe to apply globally.

Usage:
    from run_phase6.cost_patch import apply as _apply_phase6_cost
    _apply_phase6_cost()        # must run AFTER stage4_* modules are imported
                                # so their `load_rd_config` reference can be
                                # rebound too.

Frozen-rule compliance:
    No modifications to run/config.py or any other 2026-04-28 file. This
    module lives in run_phase6/ (post-04-28 directory) and only patches at
    import time.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def _coerce_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    s = raw.strip()
    if s == "" or s.lower() in ("none", "null", "off", "disable", "disabled"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def apply() -> dict:
    """Install the load_rd_config monkey-patch globally."""
    from run import config as _cfg_module
    original = _cfg_module.load_rd_config

    def patched_load_rd_config():
        cfg = original()
        min_cost_override = _coerce_float(os.getenv("FAVOR_MIN_COST_OVERRIDE"))
        if min_cost_override is not None:
            old = cfg.qlib.min_cost
            cfg.qlib.min_cost = float(min_cost_override)
            if old != cfg.qlib.min_cost:
                print(
                    f"[cost_patch] cfg.qlib.min_cost {old} → {cfg.qlib.min_cost} "
                    f"(FAVOR_MIN_COST_OVERRIDE)"
                )
        return cfg

    # 1) replace in the canonical module
    _cfg_module.load_rd_config = patched_load_rd_config

    # 2) rebind every module that already did `from run.config import load_rd_config`
    rebound = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            current = getattr(mod, "load_rd_config", None)
        except Exception:
            continue
        if current is original:
            try:
                setattr(mod, "load_rd_config", patched_load_rd_config)
                rebound.append(mod_name)
            except Exception:
                pass

    return {
        "target": "run.config.load_rd_config",
        "override_env": "FAVOR_MIN_COST_OVERRIDE",
        "rebound_modules": rebound,
    }
