from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..core.embedding_client import get_embedding_function
from ..core.llm_client import get_chat_llm
from ..core.logging_config import logger
from ..core.settings import NO_ANSWER_FOUND, VECTOR_DIR
from ..prompts.prompt_template import (
    HISTORY_AWARE_QUERY_PROMPT,
    RAG_PROMPT,
    SYSTEM_MESSAGE,
)


# -------------------------------------------------------------------
# Chat history utilities
# -------------------------------------------------------------------
def build_history(
    chat_history: Sequence[tuple[str, str]] | None,
) -> list[BaseMessage]:
    """
    Convert frontend chat_history (list of (question, answer)) into
    LangChain message objects (HumanMessage / AIMessage).
    """
    messages: list[BaseMessage] = []
    if not chat_history:
        return messages

    for user_q, assistant_a in chat_history:
        if user_q:
            messages.append(HumanMessage(content=user_q))
        if assistant_a:
            messages.append(AIMessage(content=assistant_a))

    return messages


def _get_vectorstore() -> Chroma:
    embeddings = get_embedding_function()
    return Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )


def _rewrite_query_with_history(
    user_query: str,
    chat_history: list[BaseMessage],
) -> str:
    """
    Use the LLM to produce a history-aware, standalone query for retrieval.

    This is the same logic you had in utils.rewrite_query_with_history,
    just moved here to keep all RAG logic together.
    """
    llm = get_chat_llm(temperature=0.2)
    chain = HISTORY_AWARE_QUERY_PROMPT | llm | StrOutputParser()

    try:
        rewritten = chain.invoke(
            {"input": user_query, "chat_history": chat_history or []}
        )
        rewritten = (rewritten or "").strip()
        if rewritten:
            logger.debug(
                "Query rewrite | original='%s' | rewritten='%s'",
                user_query,
                rewritten,
            )
            return rewritten
    except Exception:
        logger.exception("History-aware query rewriting failed")

    # Fallback to original if rewriting fails
    return user_query


# -------------------------------------------------------------------
# Conversational RAG with per-thread docs
# -------------------------------------------------------------------
def answer_with_docs(
    question: str,
    lc_history: list[BaseMessage],
    user_id: int,
    thread_id: str | None = None,
    k: int = 5,
) -> tuple[str, list[Any]]:
    """
    Conversational RAG:

      1. Ensure vectorstore exists.
      2. Rewrite question using history-aware LLM (HISTORY_AWARE_QUERY_PROMPT).
      3. Retrieve docs from Chroma using rewritten query, filtered by user_id + thread_id.
      4. Answer using original question + retrieved context + history (SYSTEM_MESSAGE + RAG_PROMPT).

    Returns:
        answer: str
        contexts: list[Document] used as sources.
    """

    # 1) Ensure vectorstore exists
    if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
        raise FileNotFoundError(
            "Vectorstore not found or empty. Please upload a PDF first."
        )

    vectorstore = _get_vectorstore()

    # 2) Build metadata filter and rewrite query
    metadata_filter: dict = {"user_id": user_id}
    if thread_id is not None:
        metadata_filter["thread_id"] = thread_id

    rewritten_query = _rewrite_query_with_history(question, lc_history)

    logger.info(
        "Performing vector search | user_id=%s | thread_id=%s | filter=%s | rewritten='%s'",
        user_id,
        thread_id,
        metadata_filter,
        rewritten_query,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "lambda_mult": 0.5,
            "filter": metadata_filter,
        },
    )

    docs: list[Document] = retriever.invoke(rewritten_query)

    if not docs:
        logger.info(
            "No documents retrieved | question='%s' | rewritten='%s' | filter=%s",
            question,
            rewritten_query,
            metadata_filter,
        )
        return NO_ANSWER_FOUND, []

    logger.info(
        "Retrieved %d docs for user_id=%s thread_id=%s",
        len(docs),
        user_id,
        thread_id,
    )

    # 3) Build RAG prompt with placeholder chat_history
    llm = get_chat_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE),
            ("placeholder", "{chat_history}"),
            ("user", RAG_PROMPT),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    # We answer the original question (not the rewritten one) using
    # context retrieved from the rewritten query.
    try:
        answer = chain.invoke(
            {
                "chat_history": lc_history,
                "context": docs,
                "question": question,
            }
        )
        answer = (answer or "").strip()
    except Exception:
        logger.exception("RAG LLM invocation failed")
        return NO_ANSWER_FOUND, docs

    if not answer:
        logger.warning(
            "LLM returned empty answer | user_id=%s | thread_id=%s",
            user_id,
            thread_id,
        )
        return NO_ANSWER_FOUND, docs

    return answer, docs
