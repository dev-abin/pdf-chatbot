
from pydantic import BaseModel
from typing import List, Tuple


# Define a request model for chat input
class ChatRequest(BaseModel):
    question: str   # User's question
    chat_history: List[Tuple[str, str]] = []    # Chat history as a list of question-answer pairs