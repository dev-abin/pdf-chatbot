from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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



SYSTEM_MESSAGE="""You are an highly reliable and factual analyst AI.
Your sole purpose is to synthesize information and answer the user's question strictly based ONLY on the 
provided documents Follow the given instructions carefully to produce accurate and well-structured answers"""

# -----------------------------  
# RAG Answering Prompt  
# -----------------------------
RAG_PROMPT ="""
<document>
{context}
</document>

### Instructions:
- NEVER use prior knowledge or assumptions. Every part of your answer must be directly supported by the content within <document>.
- Before writing the final output:
    Identify and extract all relevant verbatim quotes (within double quotes) or evidence from <document>.
- When writing the final answer:
    - Clearly site the source and page number for each piece of information used.
    - Format citations as: (Source: <source>, Page: <page>).
    - Keep your answer clear, concise and factual directly responding to the question.
- if the document is insufficient, incomplete, or unrelated, respond strictly with:
    "The provided documents do not contain enough information to answer this question."
- Maintain a professional tone, cite relevant portions where possible, and avoid repetition.

Now answer the following question below:
Question: {question}
"""


prompt = """
Answer the following questions as best you can, but dont use your existing knowledge.
You have access to the following tools:
{tools}

Follow the ReAct framework:

1. Question: The input question you must answer
2. Thought: Always think step-by-step about what to do next
3. Action: Choose one of the available tools [{tool_names}]
4. Action Input: Provide the input parameters for the selected action
5. Observation: Record the result of the action

Repeat the Thought -> Action -> Action Input -> Observation cycle as many times as needed until you have enough information.
Thought: I now know the Final Answer
Final Answer: The final response to the original question.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""