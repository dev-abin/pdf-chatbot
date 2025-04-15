
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from typing import List, Tuple

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