from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from .settings import (
    EMBEDDING_PROVIDER,
    OLLAMA_API_URL,
    OPENAI_API_KEY,
    PREF_EMBEDDING_MODEL,
)

# Module-level cache
_EMBEDDINGS_CACHE: Embeddings | None = None


def _build_embedding_function() -> Embeddings:
    """
    Build a new Embeddings instance based on EMBEDDING_PROVIDER.
    This is called only once and cached by get_embedding_function().
    """
    if EMBEDDING_PROVIDER == "huggingface":
        if not PREF_EMBEDDING_MODEL:
            raise RuntimeError(
                "PREF_EMBEDDING_MODEL must be set for huggingface embeddings"
            )
        return HuggingFaceEmbeddings(model_name=PREF_EMBEDDING_MODEL)

    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY must be set for openai embeddings")
        if not PREF_EMBEDDING_MODEL:
            raise RuntimeError("PREF_EMBEDDING_MODEL must be set for openai embeddings")
        return OpenAIEmbeddings(
            model=PREF_EMBEDDING_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
        )

    if EMBEDDING_PROVIDER == "ollama":
        if not OLLAMA_API_URL:
            raise RuntimeError("OLLAMA_API_URL must be set for ollama embeddings")
        if not PREF_EMBEDDING_MODEL:
            raise RuntimeError("PREF_EMBEDDING_MODEL must be set for ollama embeddings")
        return OllamaEmbeddings(
            model=PREF_EMBEDDING_MODEL,
            base_url=OLLAMA_API_URL,
        )

    raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}")


def get_embedding_function() -> Embeddings:
    """
    Cached factory. The first call constructs the Embeddings instance,
    subsequent calls return the cached one.
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        _EMBEDDINGS_CACHE = _build_embedding_function()
    return _EMBEDDINGS_CACHE
