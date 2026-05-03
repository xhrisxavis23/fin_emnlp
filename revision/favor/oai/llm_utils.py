from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from log import logger
from oai.llm_conf import LLM_SETTINGS


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def calculate_embedding_distance_between_str_list(
    a_list: list[str],
    b_list: list[str],
) -> list[list[float]]:
    distances: list[list[float]] = []
    for a in a_list:
        row: list[float] = []
        for b in b_list:
            sim = SequenceMatcher(None, a or "", b or "").ratio()
            row.append(1.0 - sim)
        distances.append(row)
    return distances


@dataclass
class _ChatSession:
    system_prompt: str
    use_chat_cache: bool = True

    def build_chat_completion(self, *, user_prompt: str, json_mode: bool = False) -> str:
        return APIBackend(use_chat_cache=self.use_chat_cache).build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            json_mode=json_mode,
        )


class APIBackend:
    def __init__(self, use_chat_cache: bool = True) -> None:
        self.use_chat_cache = use_chat_cache

    def build_messages_and_calculate_token(self, *, user_prompt: str, system_prompt: str) -> int:
        return _approx_token_count((system_prompt or "") + "\n" + (user_prompt or ""))

    def build_messages_and_create_chat_completion(
        self,
        user_prompt: str,
        system_prompt: str,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            from util.llm_client import call_llm
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "LLM backend is not available (util.llm_client import failed). "
                "Install/configure OpenAI client and set OPENAI_API_KEY."
            ) from e

        model = kwargs.get("model") or LLM_SETTINGS.model
        temperature = kwargs.get("temperature", 0.3)
        resp = call_llm(
            model=model,
            system_prompt=system_prompt or "",
            user_prompt=user_prompt or "",
            tools=None,
            temperature=temperature,
        )
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        try:
            return str(resp)
        except Exception:
            logger.warning("Unexpected LLM response type: %s", type(resp))
            return ""

    def build_chat_session(self, *, session_system_prompt: str) -> _ChatSession:
        return _ChatSession(system_prompt=session_system_prompt, use_chat_cache=self.use_chat_cache)

