"""
Stage 2: Observation Formula Validation Agent

This module is intentionally self-contained so `stage2.py` can rely on it
without pulling in historical/unused agent frameworks.

It uses the prompts in `validation_agent_prompts.py` and the schema in
`validation_dataclasses.py` to ask an LLM for a strict PASS/FAIL verdict
based on a structured evidence JSON packet.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Optional

from validation_agent_prompts import (
    DISTRIBUTION_JUDGMENT_SYSTEM_PROMPT,
    DISTRIBUTION_JUDGMENT_USER_TEMPLATE,
)
from validation_dataclasses import LLMJudgment


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM response.")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Could not locate JSON object in LLM response.")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("LLM response JSON is not an object.")
    return obj


def _call_llm_json(*, system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Prefer AlphaAgent's APIBackend (respects LLM_SETTINGS and supports caching),
    fall back to raw OpenAI client if needed.
    """
    try:
        from alphaagent.oai.llm_utils import APIBackend  # type: ignore

        backend = APIBackend(chat_model=model)
        return backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            reasoning_flag=False,
            temperature=0.0,
        )
    except Exception:
        pass

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(
            "No available LLM backend. Install/configure AlphaAgent(OpenAI) dependencies or set OPENAI_API_KEY."
        ) from e


class ValidationAgent:
    def __init__(self, *, model: str) -> None:
        self.model = model

    def judge_distribution(
        self,
        *,
        formula_name: str,
        definition: str,
        polarity: str,
        obs_id: str,
        obs_description: str,
        evidence_json: Dict[str, Any],
        distribution_summary: str,
    ) -> LLMJudgment:
        user_prompt = DISTRIBUTION_JUDGMENT_USER_TEMPLATE.format(
            formula_name=formula_name,
            definition=definition,
            polarity=polarity,
            obs_id=obs_id,
            obs_description=obs_description,
            evidence_json=json.dumps(evidence_json, ensure_ascii=False, indent=2),
            distribution_summary=distribution_summary,
        )
        raw = _call_llm_json(
            system_prompt=DISTRIBUTION_JUDGMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
        )
        obj = _extract_first_json_object(raw)

        verdict = str(obj.get("verdict", "")).strip().upper()
        if verdict not in ("PASS", "FAIL"):
            raise ValueError(f"Invalid verdict from LLM: {verdict!r}")
        reasoning = str(obj.get("reasoning", "")).strip()
        if not reasoning:
            raise ValueError("LLM returned empty reasoning.")

        primary_evidence = obj.get("primary_evidence", [])
        if not isinstance(primary_evidence, list):
            primary_evidence = []

        feature_analysis = obj.get("feature_analysis", {})
        if not isinstance(feature_analysis, dict):
            feature_analysis = {}
        feature_analysis = {str(k): str(v).strip() for k, v in feature_analysis.items() if str(v).strip()}

        return LLMJudgment(
            verdict=verdict,
            reasoning=reasoning,
            primary_evidence=primary_evidence,
            feature_analysis=feature_analysis,
            raw_response=json.dumps(obj, ensure_ascii=False),
        )


def run_stage2_llm_judgment(
    *,
    model: str,
    formula_name: str,
    definition: str,
    polarity: str,
    obs_id: str,
    obs_description: str,
    evidence_json: Dict[str, Any],
    distribution_summary: str,
) -> Dict[str, Any]:
    """
    Convenience wrapper for `stage2.py`.
    Returns a JSON-serializable dict.
    """
    agent = ValidationAgent(model=model)
    judgment = agent.judge_distribution(
        formula_name=formula_name,
        definition=definition,
        polarity=polarity,
        obs_id=obs_id,
        obs_description=obs_description,
        evidence_json=evidence_json,
        distribution_summary=distribution_summary,
    )
    return asdict(judgment)
