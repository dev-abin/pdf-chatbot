from pydantic import BaseModel


# Define a request model for chat input
class ChatRequest(BaseModel):
    question: str  # User's question
    chat_history: list[
        tuple[str, str]
    ] = []  # Chat history as a list of question-answer pairs
