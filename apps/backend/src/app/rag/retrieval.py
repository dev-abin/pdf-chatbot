# utils.py
import os

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from ..core.embedding_client import get_embedding_function
from ..core.llm_client import get_chat_llm
from ..core.logging_config import logger
from ..core.settings import (
    NO_ANSWER_FOUND,
    VECTOR_DIR,
)
from ..prompts.prompt_template import (
    HISTORY_AWARE_QUERY_PROMPT,
    RAG_PROMPT,
    SYSTEM_MESSAGE,
)


# ----------------- Chat history builder -----------------
def build_history(pairs: list[tuple[str, str]]) -> list[BaseMessage]:
    """
    Convert (user, assistant) string pairs into LangChain message objects.
    """
    messages: list[BaseMessage] = []
    for user_q, assistant_a in pairs:
        if user_q:
            messages.append(HumanMessage(content=user_q))
        if assistant_a:
            messages.append(AIMessage(content=assistant_a))
    return messages


def rewrite_query_with_history(user_query: str, chat_history: list[BaseMessage]) -> str:
    """
    Use the LLM to produce a history-aware, standalone query for retrieval.
    LLM chain that turns (chat_history, input) into a standalone search query.
    """
    # provider agnostic LLM
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


# ----------------- Conversational RAG (docs path) -----------------
def answer_with_docs(question: str, chat_history: list[BaseMessage], user_id: int):
    """
    Conversational RAG:
      1. Rewrite question using history-aware LLM.
      2. Retrieve docs from Chroma using rewritten query.
      3. Answer using original question + retrieved context (RAG_PROMPT).
    """

    if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
        raise FileNotFoundError(
            "Vectorstore not found or empty. Please upload a PDF first."
        )

    embeddings = get_embedding_function()
    vectorstore = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )

    rewritten_query = rewrite_query_with_history(question, chat_history)

    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Enable MMR
        search_kwargs={
            "k": 5,  # number of documents to retrieve
            "lambda_mult": 0.5,  # balance relevance vs diversity
            "filter": {"user_id": user_id},
        },
    )

    docs = retriever.invoke(rewritten_query)
    logger.info(f"Retrieved docs \n {docs}")

    if not docs:
        logger.info(
            "No documents retrieved | question='%s' | rewritten='%s'",
            question,
            rewritten_query,
        )
        return NO_ANSWER_FOUND, []

    llm = get_chat_llm(temperature=0.2)

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE),
            ("placeholder", "{chat_history}"),
            ("user", RAG_PROMPT),
        ]
    )

    final_prompt = prompt.format(
        chat_history=chat_history, context=docs, question=rewritten_query
    )
    logger.info(f"final prompt to LLM \n {final_prompt}")

    response = llm.invoke(final_prompt)

    parser = StrOutputParser()

    str_response = parser.invoke(response)

    contexts = []
    for doc in docs:
        contexts.append(doc.page_content)

    return str_response or NO_ANSWER_FOUND, contexts
