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


def get_embedding_function() -> Embeddings:
    """
    Factory that returns an Embeddings implementation based on EMBEDDING_PROVIDER.

    Supported:
      - huggingface -> HuggingFaceEmbeddings
      - openai      -> OpenAIEmbeddings
      - ollama      -> OllamaEmbeddings
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
        # model name: e.g. "text-embedding-3-small" or whatever you set in PREF_EMBEDDING_MODEL
        return OpenAIEmbeddings(
            model=PREF_EMBEDDING_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
        )

    if EMBEDDING_PROVIDER == "ollama":
        if not OLLAMA_API_URL:
            raise RuntimeError("OLLAMA_API_URL must be set for ollama embeddings")
        if not PREF_EMBEDDING_MODEL:
            raise RuntimeError("PREF_EMBEDDING_MODEL must be set for ollama embeddings")
        # OllamaEmbeddings uses model name + base_url
        return OllamaEmbeddings(
            model=PREF_EMBEDDING_MODEL,
            base_url=OLLAMA_API_URL,
        )

    raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}")
