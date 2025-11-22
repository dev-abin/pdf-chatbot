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


def get_chat_llm(*, temperature: float = 0.2, **kwargs: Any) -> BaseChatModel:
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
