# apps/backend/src/app/schemas/chat_schema.py


from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    chat_history: list[tuple[str, str]] | None = None
    thread_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] | None = None
