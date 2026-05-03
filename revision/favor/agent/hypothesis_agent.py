from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.base_agent import BaseAgent
from prompts.hypothesis_agent_prompts import (
    BEHAVIORAL_HYPOTHESIS_SYSTEM_PROMPT,
    BEHAVIORAL_HYPOTHESIS_REGEN_SYSTEM_PROMPT,
    BEHAVIORAL_HYPOTHESIS_USER_PROMPT_TEMPLATE,
)  # (edited)
from util.run_context import RunContext
from schemas.hypothesis import HYPOTHESIS_TOOL
from util.llm_client import call_llm

logger = logging.getLogger(__name__)

# The behavioral hypothesis prompt was moved to `hypothesis_agent_prompts.py`. (edited)

class HypothesisAgent(BaseAgent):
    """
    Hypothesis Agent (Stage 1, step 1)

    Purpose:
    - Convert an unstructured idea (concept) into a structured behavioral hypothesis
    - Provide clear, machine-consumable fields for downstream steps (`ObservationAgent`, `FormulaAgent`)

    Key features:
    - Avoid duplicates via hypothesis memory (`hypothesis_memory`)
    - Support knowledge-base augmentation (`knowledge`)
    - Incorporate prior feedback (`feedback`)

    Output schema:
    - hypothesis_id: unique identifier
    - behavioral_description: description of the behavioral pattern
    - horizon_days: observation horizon (days)
    - additional_context: extra context
    """

    def __init__(
        self,
        model: str,
        run_ctx: Optional[RunContext] = None,
    ):
        """
        Initialize HypothesisAgent.
        
        Args:
            model: LLM model name
            run_ctx: RunContext for logging and artifacts
        """
        super().__init__(model=model, run_ctx=run_ctx)

    def purpose_hypothesis(
        self,
        concept: str,
        metadata: list = None,
        hypothesis_memory: list = None,
        knowledge: str = "",
        feedback: str = "",
    ) -> Dict[str, Any]:
        """
        Generate ONE behavioral hypothesis based on a concept (and optional feedback).

        Args:
            concept: The main concept (e.g., "Mean Reversion").
            metadata: List of available columns.
            hypothesis_memory: List of previously generated hypotheses to avoid redundancy.
            knowledge: Retrieved knowledge context (e.g., Alpha101, Technical Indicators).
            feedback: Iteration feedback summary (can include validation outcomes).

        Returns:
            Tool-call response dict: {"hypotheses": [ {behavioral hypothesis fields...} ] }
        """

        def _extract_existing_hypotheses(mem: Any) -> tuple[str, str]:
            entries: List[str] = []
            ids: List[str] = []
            if not mem:
                return "None", "None"

            for item in mem:
                if isinstance(item, dict):
                    # Outer-loop memory entries (stage4_to_stage1 / stage3_fail_to_stage1)
                    outer_iter = item.get("outer_iter")
                    horizon_hint = item.get("horizon_days")
                    avg_is = item.get("avg_is_ir", item.get("avg_is_sharpe"))
                    avg_oos = item.get("avg_oos_ir", item.get("avg_oos_sharpe"))
                    next_suggestions = item.get("next_hypothesis_suggestions", "")

                    hyps = item.get("hypotheses", [])
                    if isinstance(hyps, list):
                        for h in hyps:
                            if isinstance(h, dict):
                                hid = h.get("hypothesis_id") or h.get("id")
                                if isinstance(hid, str) and hid.strip():
                                    ids.append(hid.strip())
                                desc = (
                                    h.get("behavioral_description")
                                    or h.get("hypothesis")
                                    or h.get("reason")
                                    or ""
                                )
                                horizon = h.get("horizon_days")

                                parts = []
                                if isinstance(outer_iter, int) and outer_iter > 0:
                                    parts.append(f"iter={outer_iter}")
                                if isinstance(hid, str) and hid.strip():
                                    parts.append(hid.strip())
                                if isinstance(horizon, int):
                                    parts.append(f"h={horizon}d")
                                elif isinstance(horizon_hint, int):
                                    parts.append(f"h={horizon_hint}d")
                                if avg_is is not None or avg_oos is not None:
                                    try:
                                        ais = float(avg_is) if avg_is is not None else float("nan")
                                    except Exception:
                                        ais = float("nan")
                                    try:
                                        aos = float(avg_oos) if avg_oos is not None else float("nan")
                                    except Exception:
                                        aos = float("nan")
                                    if ais == ais and aos == aos:  # not NaN
                                        parts.append(f"IS_IR={ais:+.3f},OOS_IR={aos:+.3f}")

                                header = f"[{', '.join(parts)}]" if parts else ""
                                if isinstance(desc, str) and desc.strip():
                                    entries.append(f"{header} {desc.strip()}".strip())
                                elif header:
                                    entries.append(header)
                            elif isinstance(h, str) and h.strip():
                                entries.append(h.strip())
                    else:
                        desc = item.get("behavioral_description") or item.get("hypothesis") or ""
                        if isinstance(desc, str) and desc.strip():
                            entries.append(desc.strip())

                    # Add concise guidance (avoid huge prompts).
                    if isinstance(next_suggestions, str) and next_suggestions.strip():
                        s = " ".join(next_suggestions.strip().split())
                        entries.append(f"[outer_loop_suggestions] {s[:500]}")

                elif isinstance(item, str) and item.strip():
                    entries.append(item.strip())

            # Keep prompt bounded
            if len(entries) > 8:
                entries = entries[-8:]

            descriptions_text = "\n- ".join(entries) if entries else "None"
            ids_text = ", ".join(sorted(set(ids))) if ids else "None"
            return descriptions_text, ids_text

        existing_hypotheses_text, existing_ids_text = _extract_existing_hypotheses(hypothesis_memory)

        if not knowledge:
            knowledge = "None"
        if not feedback:
            feedback = "None"

        # Outer-loop regeneration mode: when we have prior results in hypothesis_memory,
        # use a dedicated system prompt that encourages material changes and horizon exploration.
        system_prompt = (
            BEHAVIORAL_HYPOTHESIS_REGEN_SYSTEM_PROMPT
            if hypothesis_memory
            else BEHAVIORAL_HYPOTHESIS_SYSTEM_PROMPT
        )

        user_prompt = BEHAVIORAL_HYPOTHESIS_USER_PROMPT_TEMPLATE.format(
            concept_text=concept,
            columns=metadata,
            existing_hypotheses=existing_hypotheses_text,
            existing_ids=existing_ids_text,
            knowledge=knowledge,
            feedback=feedback,
        )
        
        response_dict = call_llm(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=[HYPOTHESIS_TOOL],
            target_tool_name="hypothesis_tool",
            temperature=0.9,
            react_log_path=self.run_ctx.root_dir / "logs/agents" if self.run_ctx else None,
            react_agent_name="hypothesis_agent",
            context="Stage1: Hypothesis Generation",
        )

        def _normalize(resp: Any) -> Dict[str, Any]:
            if isinstance(resp, dict) and isinstance(resp.get("hypotheses"), list):
                return dict(resp)

            # If the model returned a single hypothesis object (unexpected), wrap it.
            if isinstance(resp, dict) and ("hypothesis_id" in resp or "behavioral_description" in resp):
                return _normalize({"hypotheses": [resp]})

            return {"hypotheses": []}

        normalized = _normalize(response_dict)
        if normalized.get("hypotheses"):
            return normalized

        logger.error("Failed to parse behavioral hypothesis response: %r", response_dict)
        return {"hypotheses": []}
