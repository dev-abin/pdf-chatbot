# apps/backend/src/app/rag/log_rag.py
import json
import uuid
from datetime import UTC, datetime

from ..core.logging_config import rag_logger


def log_interaction(query, contexts, llm_response, ground_truth=None) -> bool:
    """
    Log user interaction details for a RAG query as a JSON line.
    """
    interaction = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "contexts": contexts,
        "llm_response": llm_response,
        "ground_truth": ground_truth,
    }
    rag_logger.info(json.dumps(interaction, ensure_ascii=False))
    return True
