# agent/base_agent.py
"""
Base Agent Class - shared base class for all agents

Provides `data_query_tool` usage utilities to all agents:
- Common methods: query construction, execution, result formatting
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from util.json_utils import strip_code_fence

if TYPE_CHECKING:
    from util.run_context import RunContext

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Shared base class for all agents.

    Responsibilities:
    - Manage LLM model configuration (`model`)
    - Integrate `RunContext` for logging and artifact persistence
    - Provide common initialization logic

    Agents inheriting from this:
    - `HypothesisAgent`: convert a concept into a structured behavioral hypothesis
    - `ObservationAgent`: break a hypothesis into observable conditions
    - `FormulaAgent`: turn conditions into formulas and validate them
    - `ValidationAgent`: Stage 2 formula validation (distribution-based)
    - `HypothesisValidationAgent`: Stage 3 hypothesis instance validation (monotonicity-based)
    - `DiagnosticsAgent`: diagnostics/analysis (compat alias)
    """

    def __init__(
        self,
        model: str,
        run_ctx: Optional["RunContext"] = None,
    ):
        """
        Initialize BaseAgent with common parameters.

        Args:
            model: LLM model name (e.g., "gpt-4o-mini")
            run_ctx: Optional run context for logging
        """
        self.model = model
        self.run_ctx = run_ctx
