from typing import Any

from ..schemas.chat_schema import RetrievalTrace


def build_retrieval_trace(
    question: str, contexts: list[Any], fallback_used: bool
) -> RetrievalTrace:
    """Create one stable, API-safe account of the retrieval decision."""
    retrieval_query = question
    if contexts:
        metadata = getattr(contexts[0], "metadata", {}) or {}
        retrieval_query = str(metadata.get("retrieval_query", question))
    return RetrievalTrace(
        retrieval_query=retrieval_query,
        retrieved_chunks=len(contexts),
        fallback_used=fallback_used,
        fallback_reason="No grounded document answer was available" if fallback_used else None,
    )
