import json
import uuid
from datetime import UTC, datetime
from ..core.logging_config import logger, rag_logger

# ----------------- RAG interaction logging -----------------
def log_interaction(query, contexts, llm_response, ground_truth=None):
    """
    Log user interaction details for a Retrieval-Augmented Generation query.
    """
    interaction = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "contexts": contexts,
        "llm_response": llm_response,
        "ground_truth": ground_truth,
    }
    rag_logger.info(interaction)
    return True



