from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .loader import load_prompt

# -----------------------------
# Conversational history-aware query rewriting
# -----------------------------
HISTORY_AWARE_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            load_prompt("history_rewrite_system"),
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            load_prompt("history_rewrite_user"),
        ),
    ]
)


# -----------------------------
# Wikipedia ReAct Agent Prompt
# -----------------------------
WIKIPEDIA_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            load_prompt("wikipedia_react_system"),
        ),
        (
            "human",
            load_prompt("wikipedia_react_user"),
        ),
    ]
)


SYSTEM_MESSAGE = load_prompt("system")

# -----------------------------
# RAG Answering Prompt
# -----------------------------
RAG_PROMPT = load_prompt("rag")
