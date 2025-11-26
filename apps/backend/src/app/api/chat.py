# apps/backend/src/app/api/chat.py

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..agents.wikipedia import wikipedia_agent_answer
from ..auth.deps import get_current_user
from ..core.logging_config import logger
from ..core.settings import NO_ANSWER_FOUND
from ..db.models import User
from ..rag.log_rag import log_interaction
from ..rag.retrieval import answer_with_docs, build_history
from ..schemas.chat_schema import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


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
            "Chat endpoint called | endpoint=/chat/ | user_id=%s | thread_id=%s | query='%s' | chat_history='%s'",
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

        if NO_ANSWER_FOUND in answer:
            logger.info(
                "RAG could not answer; falling back to Wikipedia agent | query='%s'",
                chat_request.question,
            )

            agent_answer = wikipedia_agent_answer(chat_request.question)
            final_answer = agent_answer or NO_ANSWER_FOUND
            final_contexts: list[Any] = [agent_answer]
        else:
            final_answer = answer
            final_contexts = retrieved_contexts

        logger.info(
            "Chat response ready | user_id=%s | thread_id=%s | answer_preview='%s'...",
            current_user.id,
            chat_request.thread_id,
            final_answer[:200],
        )

        try:
            log_interaction(
                chat_request.question,
                final_contexts,
                final_answer,
            )
        except Exception:
            logger.exception("Failed to log interaction")

        return ChatResponse(answer=final_answer)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing chat query failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later.",
        ) from e
