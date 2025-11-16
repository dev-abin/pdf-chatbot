# utils.py
import os
import uuid
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import create_react_agent, AgentExecutor
from backend.core.settings import (
    VECTOR_DIR,
    OLLAMA_API_URL,
    PREF_MODEL,
    PREF_EMBEDDING_MODEL,
    NO_ANSWER_FOUND,
)
from backend.core.logging_config import logger, rag_logger
from backend.prompts.prompt_template import HISTORY_AWARE_QUERY_PROMPT,WIKIPEDIA_AGENT_PROMPT,RAG_PROMPT, SYSTEM_MESSAGE


# ----------------- Chat history builder -----------------
def build_history(pairs: List[Tuple[str, str]]) -> List[BaseMessage]:
    """
    Convert (user, assistant) string pairs into LangChain message objects.
    """
    messages: List[BaseMessage] = []
    for user_q, assistant_a in pairs:
        if user_q:
            messages.append(HumanMessage(content=user_q))
        if assistant_a:
            messages.append(AIMessage(content=assistant_a))
    return messages


def rewrite_query_with_history(
    user_query: str, chat_history: List[BaseMessage]
) -> str:
    """
    Use the LLM to produce a history-aware, standalone query for retrieval.
    LLM chain that turns (chat_history, input) into a standalone search query.
    """

    llm = ChatOllama(
        model=PREF_MODEL,
        base_url=OLLAMA_API_URL,
        temperature=0.0,  # deterministic for routing
    )
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
def answer_with_docs(
    question: str, chat_history: List[BaseMessage]
    ):
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
    
    embeddings = HuggingFaceEmbeddings(model_name=PREF_EMBEDDING_MODEL)
    vectorstore = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings,
        )
    
    rewritten_query = rewrite_query_with_history(question, chat_history)
    
    retriever = vectorstore.as_retriever(
                        search_type="mmr",      # Enable MMR
                        search_kwargs={
                            "k": 5,             # number of documents to retrieve
                            "lambda_mult": 0.5  # balance relevance vs diversity
                            }
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

    llm = ChatOllama(
        model=PREF_MODEL,
        base_url=OLLAMA_API_URL,
        temperature=0.2,
    )
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages(
        [('system', SYSTEM_MESSAGE),
         ('placeholder', '{chat_history}'),
         ('user', RAG_PROMPT)
        ]
    )
    
    final_prompt = prompt.format(chat_history=chat_history, context = docs, question = rewritten_query)
    logger.info(f'final prompt to LLM \n {final_prompt}')
    
    response = llm.invoke(final_prompt)
    
    parser = StrOutputParser()

    str_response = parser.invoke(response)

    contexts = []
    for doc in docs:
        contexts.append(doc.page_content)

    return str_response or NO_ANSWER_FOUND, contexts


# ----------------- Wikipedia ReAct Agent (fallback) -----------------
_wiki_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=_wiki_api)


@lru_cache(maxsize=1)
def get_wikipedia_agent() -> AgentExecutor:
    """
    ReAct-style Wikipedia agent using create_react_agent.

    Tools:
      - wikipedia_tool (WikipediaQueryRun)
    """
    tools = [wikipedia_tool]

    llm = ChatOllama(
        model=PREF_MODEL,
        base_url=OLLAMA_API_URL,
        temperature=0.2,
    )

    react_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=WIKIPEDIA_AGENT_PROMPT,
    )

    agent_executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,                 # avoid stdout spam; we log ourselves
        return_intermediate_steps=True, # important: so we can see tool calls
        handle_parsing_errors=True,
        max_iterations=3,              # you can lower this if you want
        early_stopping_method="force",
    )

    logger.info(
        "Initialized Wikipedia ReAct agent with tools: %s",
        [t.name for t in tools],
    )
    return agent_executor



def wikipedia_agent_answer(question: str) -> str:
    """
    Use the Wikipedia ReAct agent to answer general-knowledge questions.

    - Logs each tool step (tool name, input, observation preview).
    - If the agent hits its iteration limit/time limit, fall back to a direct
      WikipediaQueryRun.run call.
    """
    agent = get_wikipedia_agent()
    result = agent.invoke({"input": question})

    # Log intermediate steps (tool calls + observations)
    intermediate_steps = result.get("intermediate_steps", [])
    for idx, (action, observation) in enumerate(intermediate_steps):
        tool_name = getattr(action, "tool", "")
        tool_input = getattr(action, "tool_input", None)
        obs_preview = str(observation)[:300].replace("\n", " ")
        logger.info(
            "Wikipedia agent step %d | tool=%s | input=%s | observation=%s",
            idx,
            tool_name,
            tool_input,
            obs_preview,
        )

    output = (result.get("output") or "").strip()

    # Handle iteration-limit failure case
    if "Agent stopped due to iteration limit or time limit." in output:
        logger.warning(
            "Wikipedia agent hit iteration/time limit for query='%s'. ",
            question,
        )
    else:
        output = output + " (Source: Wikipedia)"

    return output



# ----------------- RAG interaction logging -----------------
def log_interaction(query, contexts, llm_response, ground_truth=None):
    """
    Log user interaction details for a Retrieval-Augmented Generation query.
    """
    interaction = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "contexts": contexts,
        "llm_response": llm_response,
        "ground_truth": ground_truth,
    }
    rag_logger.info(interaction)
    return True


# Load logged data
def load_logged_interactions(filepath):
    """Load logged interaction records from a JSON log file.

    This function reads a log file containing JSON-formatted interaction records generated 
    by the Retrieval-Augmented Generation (RAG) system. Each line in the file is
    expected to be a JSON object, nested  with a 'message' field containing the
    interaction data. The function extracts relevant fields (query, response, contexts, and
    ground truth) and returns them as a list of structured dictionaries.

    Args:
        filepath (str): The path to the log file containing JSON-formatted interaction records.

    Returns:
        list: A list of dictionaries, each containing:
            - user_input (str): The user's query or question.
            - response (str): The response generated by the language model.
            - retrieved_contexts (list): A list of retrieved context strings used to generate the response.
            - reference (str or None): The ground truth or expected response, if available.
    """
    # Initialize an empty list to store the parsed records
    records = []

    try:
        # Open the log file in read mode
        with open(filepath, "r") as f:
            for line in f:
                # Parse the line as a JSON object
                raw = json.loads(line)

                # Check if the JSON object has a nested 'message' field
                if "message" in raw:
                    # Parse the nested 'message' field as another JSON object
                    data = json.loads(raw["message"])

                    # Create a structured record with relevant fields
                    record = {
                        "user_input": data["query"],
                        "response": data["llm_response"],
                        "retrieved_contexts": data["contexts"],
                        "reference": data["ground_truth"]
                    }

                    # Append the record to the list
                    records.append(record)

    except Exception:
        # Catch any errors, log them, and return an empty list
        logger.exception("load_logged_interactions failed")
        return []

    # Return the list of parsed records
    return records


def evaluate_rag_response():
    """Evaluate the performance of a RAG system using logged interaction data.

    This function loads the most recent interaction record from a JSONL log file and evaluates
    the Retrieval-Augmented Generation (RAG) system's performance using the `ragas` library.
    It assesses the response using metrics such as Faithfulness, Context Precision, and Response
    Relevancy, leveraging an Ollama LLM and HuggingFace embeddings. The evaluation results are
    logged, and the function returns True to indicate successful execution.

    Args:
        None: This function does not take any parameters

    Returns:
        bool: Returns True if the evaluation was successful, False if an error occurs
    """
    try:
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

        # Select only the most recent record for evaluation
        records = [records[-1]]

        # Create an EvaluationDataset from the selected record
        dataset = EvaluationDataset.from_list(records)

        # Initialize the evaluator LLM using Ollama with predefined model and API URL
        evaluator_llm = LangchainLLMWrapper(OllamaLLM(model=PREF_MODEL, base_url= OLLAMA_API_URL))

        # spcify the embedding fn
        embedding_fn = HuggingFaceEmbeddings(model_name=PREF_EMBEDDING_MODEL)

        # Wrap the embeddings for compatibility with ragas
        embedding   = LangchainEmbeddingsWrapper(embedding_fn)

        # Define the metrics for evaluation
        metrics = [
                LLMContextPrecisionWithoutReference(), # Measures precision of retrieved contexts
                Faithfulness(),                        # Assesses factual accuracy of the response
                ResponseRelevancy()                    # Evaluates relevance of the response to the query
            ]

        # Configure evaluation to run with 1 concurrent metric call and a 300-second timeout
        run_config = RunConfig(timeout=300, max_workers=1)

        # Perform the evaluation using the dataset, metrics, LLM, and embeddings
        results = evaluate(dataset=dataset,metrics= metrics, llm=evaluator_llm, embeddings=embedding, run_config=run_config)
        
        logger.info(results)

        return True
    
    except Exception:
            # Log any unexpected errors during evaluation and return False
            logger.exception("evaluate_rag_response failed")
            return False


if __name__=="__main__":
    evaluate_rag_response()
