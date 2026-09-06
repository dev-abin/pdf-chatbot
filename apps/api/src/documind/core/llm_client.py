from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .settings import (
    LLM_PROVIDER,
    OLLAMA_API_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PREF_MODEL,
)

# Module-level cache for the "default" LLM instance
_DEFAULT_LLM_CACHE: BaseChatModel | None = None
_DEFAULT_TEMPERATURE: float = 0.2


def _build_chat_llm(*, temperature: float, **kwargs: Any) -> BaseChatModel:
    """
    Construct a new LLM instance based on LLM_PROVIDER.
    """
    if LLM_PROVIDER == "ollama":
        if not OLLAMA_API_URL:
            raise RuntimeError("OLLAMA_API_URL not set")
        if not PREF_MODEL:
            raise RuntimeError("PREF_MODEL not set for ollama")
        return ChatOllama(
            model=PREF_MODEL,
            base_url=OLLAMA_API_URL,
            temperature=temperature,
            **kwargs,
        )

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        if not PREF_MODEL:
            raise RuntimeError("PREF_MODEL not set for openai")
        return ChatOpenAI(
            model=PREF_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL or None,
            temperature=temperature,
            **kwargs,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER={LLM_PROVIDER}")


def get_chat_llm(
    *, temperature: float = _DEFAULT_TEMPERATURE, **kwargs: Any
) -> BaseChatModel:
    """
    Cached factory.

    - If called with the default temperature and no extra kwargs, returns a cached LLM instance.
    - If called with a different temperature or kwargs, builds a fresh instance (no cache).
    """
    global _DEFAULT_LLM_CACHE

    # Simple caching policy: only cache the most common case
    if temperature == _DEFAULT_TEMPERATURE and not kwargs:
        if _DEFAULT_LLM_CACHE is None:
            _DEFAULT_LLM_CACHE = _build_chat_llm(temperature=temperature)
        return _DEFAULT_LLM_CACHE

    # Non-default configuration: build a one-off instance
    return _build_chat_llm(temperature=temperature, **kwargs)
