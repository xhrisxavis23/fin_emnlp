'''
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
'''

from __future__ import annotations

import logging
import json
import pprint
from typing import Any


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("finagent")
    if logger.handlers:
        return logger
    # Silence console output by default.
    # If you want to see logs, configure logging in your entrypoint and/or attach handlers explicitly.
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = _build_logger()


def log_object(
    obj: Any,
    *,
    tag: str = "object",
    level: int = logging.INFO,
    max_len: int = 4000,
) -> None:
    """
    Best-effort structured logging helper expected by some CoSTEER/RAG components.
    """
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = pprint.pformat(obj, width=120, compact=True)
    if max_len and len(text) > max_len:
        text = text[:max_len] + "...(truncated)"
    logger.log(level, f"[{tag}] {text}")


# Backward-compatible: some modules call `logger.log_object(...)`.
if not hasattr(logger, "log_object"):
    setattr(logger, "log_object", log_object)
