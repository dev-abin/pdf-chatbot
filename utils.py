
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from typing import List, Tuple
import json
import uuid
from datetime import datetime, timezone
from logger import rag_logger, logger

import os
# This checks for the environment variable `OLLAMA_API_URL`.
# If it's not set, it defaults to localhost (for local dev).
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

PREF_MODEL = "mistral"


# Define a request model for chat input
class ChatRequest(BaseModel):
    question: str   # User's question
    chat_history: List[Tuple[str, str]] = []    # Chat history as a list of question-answer pairs


# Define custom prompt
prompt_template = """
You are an AI assistant. Use only the information provided in the context to answer the user's question.

Context:
{context}

Question: {question}

Rules:
- If the context does not contain information relevant to the question, respond strictly with: "No answer found"
- Do not make assumptions.
- Do not try to be helpful beyond the context.
- Do not guess or provide partial answers.

Answer:
"""


# === Wikipedia Fallback Agent ===
def get_wikipedia_agent():
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Search Wikipedia for general knowledge."
        )
    ]

    # Custom system message to enforce format
    agent_kwargs = {
    "prefix": """You are a helpful AI assistant.
    You can use tools to look up relevant information.
    Always return answers in this format:
    Question: <user question>
    Thought: <step-by-step reasoning>
    Action: <tool name>(<input>)
    Observation: <tool result>
    ... (repeat Thought/Action/Observation as needed)
    Final Answer: <your final answer to the user>""",
    "suffix": """Begin!\n\n
    Question: {input}\n
    {agent_scratchpad}
    Thought:"""
    }

    return initialize_agent(
        tools=tools,
        llm=OllamaLLM(model=PREF_MODEL, base_url=OLLAMA_API_URL),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs=agent_kwargs,
        verbose=True,
        handle_parsing_errors=True
    )


def log_interaction(query, contexts, llm_response, ground_truth=None):
    interaction = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "contexts": contexts,
        "llm_response": llm_response,
        "ground_truth": ground_truth,
    }
    rag_logger.info(json.dumps(interaction))


# Load logged data
def load_logged_interactions(filepath):
    records = []
    with open(filepath, "r") as f:
        for line in f:
            raw = json.loads(line)
            if "message" in raw:  # logged as a nested JSON string
                data = json.loads(raw["message"])
                record = {
                    "user_input": data["query"],
                    "response": data["llm_response"],
                    "retrieved_contexts": data["contexts"],
                    "reference": data["ground_truth"]
                }
                records.append(record)
    return records


def evaluate_rag_response():
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import evaluate
    from ragas.run_config import RunConfig
    import os

    # Get absolute path to the directory of the current script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct full path to logs directory
    log_file_path = os.path.join(BASE_DIR, "logs", "rag_interactions.jsonl")

    # get the logged records
    records = load_logged_interactions(log_file_path)

    # take latest_record
    records = [records[-1]]

    # create dataset
    dataset = EvaluationDataset.from_list(records)

    # wrap llm
    evaluator_llm = LangchainLLMWrapper(OllamaLLM(model=PREF_MODEL, base_url= OLLAMA_API_URL))

    # spcify the embedding fn
    embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # wrap embedding
    embedding   = LangchainEmbeddingsWrapper(embedding_fn)

    # specify metrics
    metrics = [LLMContextPrecisionWithoutReference(), Faithfulness(), ResponseRelevancy()]

    # only 1 concurrent metric calls at a time
    run_config = RunConfig(timeout=300, max_workers=1)

    # evaluate
    results = evaluate(dataset=dataset,metrics= metrics, llm=evaluator_llm, embeddings=embedding, run_config=run_config)
    
    logger.info(results)

    return True


if __name__=="__main__":
    evaluate_rag_response()
