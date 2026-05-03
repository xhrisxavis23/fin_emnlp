from __future__ import annotations

import logging
import json
from typing import Any, Dict, Optional

from agent.base_agent import BaseAgent
from util.run_context import RunContext
from prompts.observation_agent_prompts import (
    OBSERVATION_SYSTEM_PROMPT,
    OBSERVATION_USER_PROMPT_TEMPLATE,
)
from schemas.observation import OBSERVATION_TOOL
from util.llm_client import call_llm

logger = logging.getLogger(__name__)

class ObservationAgent(BaseAgent):
    """
    Observation Agent (Stage 1, step 2)

    Purpose:
    - Decompose a behavioral hypothesis into concrete and independent observation conditions
    - Provide clear descriptions so `FormulaAgent` can translate each condition into formulas

    Key responsibilities:
    - Convert each component of a hypothesis into observable conditions
    - Ensure each observation condition is independently verifiable
    - Define logical relationships between observation conditions (AND/OR)

    Output schema:
    - hypothesis_id: original hypothesis ID
    - observations: List[{observation_id, description}]
      - observation_id: unique identifier for an observation condition
      - description: natural-language description of the condition

    Example:
    Input hypothesis: "Sell-off rebound"
    Output: [
      {"observation_id": "obs_1", "description": "Price dropped by at least 10% over the last 5 days"},
      {"observation_id": "obs_2", "description": "Volume increased to at least 2× the 20-day average"}
    ]
    """

    def __init__(
        self,
        model: str,
        run_ctx: Optional[RunContext] = None,
    ):
        super().__init__(model=model, run_ctx=run_ctx)

    def plan_observations(
        self,
        hypothesis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate an Observation Plan based on a hypothesis.
        """
        hyp_list = hypothesis.get("hypotheses", []) if isinstance(hypothesis, dict) else []
        hyp_obj = hyp_list[0] if hyp_list and isinstance(hyp_list[0], dict) else (hypothesis if isinstance(hypothesis, dict) else {})
        
        hypothesis_id = (
            (hyp_obj or {}).get("hypothesis_id")
            or (hyp_obj or {}).get("id")
            or "Unknown"
        )
        
        hyp_json = json.dumps(hyp_obj, ensure_ascii=False, indent=2, default=str)
        
        user_prompt = OBSERVATION_USER_PROMPT_TEMPLATE.format(
            hypothesis_id=hypothesis_id,
            hypothesis_json=hyp_json,
            columns="open, high, low, close, volume"
        )
        
        resp = call_llm(
            model=self.model,
            system_prompt=OBSERVATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=[OBSERVATION_TOOL],
            target_tool_name="observation_plan_tool",
            temperature=0.7,
            react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
            react_agent_name="observation_agent",
            context="Stage1: Observation Planning",
        )
        
        if (
            isinstance(resp, dict)
            and isinstance(resp.get("observations"), list)
        ):
            resp.setdefault("hypothesis_id", hypothesis_id)
            return resp

        logger.error("Failed to parse observation plan: %r", resp)
        return {
            "hypothesis_id": hypothesis_id,
            "observations": [],
            "error": "Failed to generate plan"
        }
