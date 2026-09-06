# apps/backend/src/app/schemas/chat_schema.py


from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4_000)
    chat_history: list[tuple[str, str]] | None = None
    thread_id: str


class SourceCitation(BaseModel):
    filename: str
    page: int | None = None
    excerpt: str
    source_type: str = "document"


class RetrievalTrace(BaseModel):
    retrieval_query: str
    retrieved_chunks: int
    fallback_used: bool
    fallback_reason: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = []
    trace: RetrievalTrace

