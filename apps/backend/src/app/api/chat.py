from fastapi import APIRouter, HTTPException
from ..rag.evaluate_rag import log_interaction
from ..rag.retrieval import answer_with_docs,build_history
from ..agents.wikipedia import wikipedia_agent_answer
from ..core.settings import NO_ANSWER_FOUND
from ..core.logging_config import logger
from ..schemas.chat_schema import ChatRequest
router = APIRouter()


@router.post("/chat/")
async def chat(chat_request: ChatRequest):
    """
    Handle user chat queries by retrieving relevant information from a vector store
    or falling back to a Wikipedia ReAct agent.

    Flow:
      1. Conversational RAG over uploaded PDFs:
           - history-aware query rewrite (LLM-based)
           - retrieve from Chroma
           - answer using RAG prompt
      2. If answer is NO_ANSWER_FOUND:
           - fallback to Wikipedia ReAct agent.
    """
    try:
        logger.info(
            "Chat endpoint called | endpoint=/chat/ | query='%s' | chat_history='%s'",
            chat_request.question,
            chat_request.chat_history,
        )

        lc_history = build_history(chat_request.chat_history)

        # 1) Try answering from internal documents (RAG)
        try:
            answer, retrieved_contexts = answer_with_docs(
                chat_request.question,
                lc_history
            )
        except FileNotFoundError:
            # No vectorstore yet
            raise HTTPException(
                status_code=404,
                detail="Vectorstore not found or empty. Please upload a document first.",
            )

        if NO_ANSWER_FOUND in answer:
            logger.info(
                "RAG could not answer; falling back to Wikipedia agent | query='%s'",
                chat_request.question,
            )

            # 2) Fallback to Wikipedia agent
            agent_answer = wikipedia_agent_answer(chat_request.question)
            final_answer = agent_answer or NO_ANSWER_FOUND
            final_contexts = [agent_answer]
        else:
            final_answer = answer
            final_contexts = retrieved_contexts

        logger.info(
            "Chat response ready | answer_preview='%s'",
            final_answer[:200],
        )

        try:
            log_interaction(chat_request.question, final_contexts, final_answer)
        except Exception:
            logger.exception("Failed to log interaction")

        return {"answer": final_answer}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing chat query failed")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )
