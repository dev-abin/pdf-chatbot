from functools import lru_cache
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from ..core.settings import (
    VECTOR_DIR,
    OLLAMA_API_URL,
    PREF_MODEL,
    PREF_EMBEDDING_MODEL,
    NO_ANSWER_FOUND,
)
from ..core.logging_config import logger, rag_logger
from ..prompts.prompt_template import HISTORY_AWARE_QUERY_PROMPT,WIKIPEDIA_AGENT_PROMPT,RAG_PROMPT, SYSTEM_MESSAGE


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
