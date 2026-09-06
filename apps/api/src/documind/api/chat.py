from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..agents.wikipedia import wikipedia_agent_answer
from ..auth.deps import get_current_user
from ..core.logging_config import logger
from ..core.settings import NO_ANSWER_FOUND
from ..db.models import User
from ..observability.interaction_log import log_interaction
from ..observability.trace import build_retrieval_trace
from ..rag.retrieval import answer_with_docs, build_history
from ..schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    SourceCitation,
)

router = APIRouter(tags=["chat"])


def _document_sources(contexts: list[Any]) -> list[SourceCitation]:
    """Expose only safe, page-aware provenance rather than raw vector metadata."""
    sources: list[SourceCitation] = []
    for document in contexts:
        metadata = getattr(document, "metadata", {}) or {}
        page = metadata.get("page")
        sources.append(
            SourceCitation(
                filename=Path(
                    str(metadata.get("filename", metadata.get("source", "document")))
                ).name,
                page=int(page) + 1 if isinstance(page, int) else None,
                excerpt=str(getattr(document, "page_content", ""))[:320],
            )
        )
    return sources


@router.post("/chat/", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Handle user chat queries scoped to a particular thread.

    user-aware:
      - Filters vector search by (user_id, thread_id).
    """
    try:
        logger.info(
            "Chat endpoint | user_id=%s thread_id=%s query='%s' history='%s'",
            current_user.id,
            chat_request.thread_id,
            chat_request.question,
            chat_request.chat_history,
        )

        lc_history = build_history(chat_request.chat_history)

        try:
            answer, retrieved_contexts = answer_with_docs(
                chat_request.question,
                lc_history,
                user_id=current_user.id,
                thread_id=chat_request.thread_id,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Vectorstore not found or empty. Please upload a document first.",
            ) from None

        fallback_used = NO_ANSWER_FOUND in answer
        if fallback_used:
            logger.info(
                "RAG could not answer; falling back to Wikipedia agent | query='%s'",
                chat_request.question,
            )

            try:
                agent_answer = wikipedia_agent_answer(chat_request.question)
                final_answer = agent_answer or NO_ANSWER_FOUND
                sources = [
                    SourceCitation(
                        filename="Wikipedia",
                        excerpt="General-knowledge fallback via bounded ReAct agent.",
                        source_type="wikipedia",
                    )
                ]
            except Exception:
                logger.exception("Wikipedia fallback failed")
                final_answer = NO_ANSWER_FOUND
                sources = []
        else:
            final_answer = answer
            sources = _document_sources(retrieved_contexts)

        logger.info(
            "Chat response ready | user_id=%s | thread_id=%s | answer_preview='%s'...",
            current_user.id,
            chat_request.thread_id,
            final_answer[:200],
        )

        try:
            log_interaction(
                chat_request.question,
                retrieved_contexts,
                final_answer,
            )
        except Exception:
            logger.exception("Failed to log interaction")

        return ChatResponse(
            answer=final_answer,
            sources=sources,
            trace=build_retrieval_trace(
                chat_request.question, retrieved_contexts, fallback_used
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing chat query failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later.",
        ) from e

