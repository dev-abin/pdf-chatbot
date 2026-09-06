import functools

from langchain_chroma import Chroma

from .embedder import get_embedding_function
from ..core.settings import VECTOR_DIR


@functools.lru_cache(maxsize=1)
def get_cached_vectorstore() -> Chroma:
    """
    Returns a cached instance of the Chroma vectorstore.
    This prevents re-initializing the connection to the database
    and recreating the collections on every request.
    """
    embeddings = get_embedding_function()
    return Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )
