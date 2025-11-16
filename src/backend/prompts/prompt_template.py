from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.core.settings import NO_ANSWER_FOUND


# -----------------------------  
# Conversational history-aware query rewriting  
# -----------------------------
HISTORY_AWARE_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You rewrite the user's question into a standalone search query "
                "for retrieving relevant documents.\n"
                "Use the chat history to resolve references.\n\n"
                "Rules:\n"
                "- Output only the rewritten query.\n"
                "- Do NOT add explanations.\n"
            ),
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "User question: {input}\n\nRewrite this as a standalone search query:",
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
            (
                "You are a helpful AI assistant that can use tools to look up information.\n\n"
                "TOOLS:\n"
                "{tools}\n\n"
                "VALID TOOL NAMES:\n"
                "{tool_names}\n\n"
                "Rules:\n"
                "1) You may call a tool AT MOST ONE TIME.\n"
                "2) When calling a tool, output exactly:\n"
                "   Action: wikipedia\n"
                "   Action Input: <query>\n"
                "3) After Observation, DO NOT call tools again.\n"
                "4) Provide the final answer in normal prose.\n"
                "5) Final answer must NOT contain Action lines.\n"
            ),
        ),
        (
            "human",
            (
                "Question: {input}\n\n"
                "Use this format:\n"
                "Thought: <reasoning>\n"
                "Action: wikipedia\n"
                "Action Input: <query>\n"
                "Observation: <tool result>\n"
                "Thought: <reasoning>\n"
                "Final Answer: <your answer>\n\n"
                "{agent_scratchpad}"
            ),
        ),
    ]
)


# -----------------------------  
# RAG Answering Prompt  
# -----------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a highly reliable and factual assistant.\n"
                "Answer strictly based ONLY on the provided documents.\n"
                f"If the documents are insufficient, respond exactly:\n\"{NO_ANSWER_FOUND}\""
            ),
        ),
        (
            "human",
            (
                "You are given the following documents:\n\n"
                "{context}\n\n"
                "Instructions:\n"
                "- Use only these documents.\n"
                "- Cite every fact as: (Source: <source>, Page: <page>).\n"
                "- Answer concisely.\n\n"
                "User question: {question}"
            ),
        ),
    ]
)
