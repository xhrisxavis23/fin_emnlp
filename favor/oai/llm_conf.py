from __future__ import annotations

from core.conf import ExtendedBaseSettings


class LLMSettings(ExtendedBaseSettings):
    class Config:
        env_prefix = "LLM_"

    model: str = "gpt-4o-mini"
    chat_token_limit: int = 16_000
    init_chat_cache_seed: int = 0


LLM_SETTINGS = LLMSettings()

