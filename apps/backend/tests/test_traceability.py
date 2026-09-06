from app.api.chat import _document_sources
from app.schemas.chat_schema import ChatResponse, RetrievalTrace
from langchain_core.documents import Document


def test_document_sources_expose_filename_page_and_excerpt():
    document = Document(
        page_content="A short supporting passage.",
        metadata={"source": "/data/files/policy.pdf", "page": 2},
    )

    sources = _document_sources([document])

    assert sources[0].filename == "policy.pdf"
    assert sources[0].page == 3
    assert sources[0].excerpt == "A short supporting passage."


def test_chat_response_requires_an_auditable_trace():
    response = ChatResponse(
        answer="Grounded answer",
        trace=RetrievalTrace(
            retrieval_query="retention policy",
            retrieved_chunks=2,
            fallback_used=False,
        ),
    )

    assert response.trace.retrieved_chunks == 2

