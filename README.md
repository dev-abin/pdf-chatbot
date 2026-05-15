# PDFQuery

## Overview

**PDFQuery** is an AI-powered PDF chatbot that lets users upload documents, build a vector index, and ask questions through a conversational interface. The backend is a FastAPI service that handles uploads, embedding creation, and retrieval-augmented generation (RAG), while the frontend is a Streamlit UI that calls the backend APIs.

## Architecture

- **Backend (FastAPI)**
  - Upload endpoint ingests PDF/DOCX/TXT files, extracts text, chunks it, and stores embeddings in Chroma.
  - Chat endpoint performs retrieval against the user/thread-scoped vector store and uses an LLM to answer.
  - If the RAG pipeline can’t answer, the service falls back to a Wikipedia agent.
- **Frontend (Streamlit)**
  - Provides a simple UI for authentication, uploads, and chat interaction.

## End-to-end flow

1. **Upload**
   - The client uploads a PDF/DOCX/TXT file with a `thread_id`.
   - The backend stores the raw file and registers it in the database.
2. **Text extraction + preprocessing**
   - PDFs are parsed with PyMuPDF; if no text is found, OCR is applied.
   - DOCX/TXT files are parsed with the appropriate loader.
   - Extracted documents are cleaned before indexing.
3. **Chunking + embeddings**
   - Cleaned documents are split into chunks.
   - Embeddings are generated and stored in Chroma with metadata (`user_id`, `thread_id`, `document_id`).
4. **Chat**
   - The chat endpoint retrieves relevant chunks by user/thread scope.
   - A final answer is generated; if no answer is found, a Wikipedia agent is used as a fallback.

## Project layout

```
apps/
  backend/   # FastAPI service (RAG, uploads, auth)
  frontend/  # Streamlit UI
```

## Local development

> **Requirements:** Python 3.12+ and a configured LLM + embedding provider.

### Backend

```bash
cd apps/backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

### Frontend

```bash
cd apps/frontend
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run rag_ui/app.py
```

## Configuration

The backend expects the following environment variables (typically via `.env` or your container setup):

- `DATABASE_URL`
- `JWT_SECRET`
- `LLM_PROVIDER` (`ollama` or `openai`)
- `PREF_MODEL`
- `EMBEDDING_PROVIDER` (`huggingface` or `openai`)
- `PREF_EMBEDDING_MODEL`
- `OLLAMA_API_URL` (required when using Ollama)
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` (required for OpenAI)

## API quick reference

- `POST /auth/...` for authentication
- `POST /api/upload-files/` to upload and index documents
- `POST /api/chat/` to query a thread-scoped RAG session

## Notes

This project is under active development. If you’re extending the flow, keep the upload → preprocess → chunk → embed → retrieve pipeline consistent so the vector metadata stays aligned with chat filtering.
