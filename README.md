# DocuMind

Open-source, self-hosted document QA using agentic RAG.

DocuMind answers questions over PDF, DOCX, and text files with FastAPI, Streamlit, LangChain, persistent ChromaDB, and Hugging Face embeddings. It is structured as an AI-engineering project: document evidence is isolated by user and conversation, responses include page-aware citations and traces, scanned PDFs can use OCR, and a bounded Wikipedia ReAct agent handles out-of-corpus questions.

## Capabilities

- Persistent ChromaDB with user/thread metadata filtering and MMR retrieval.
- PDF, DOCX, and TXT ingestion; OCR fallback for scanned PDFs.
- Hugging Face embeddings with Ollama or OpenAI-compatible chat generation.
- History-aware retrieval, grounded answer generation, and citation requirements.
- Wikipedia ReAct fallback with bounded tool use for queries beyond the document corpus.
- API traceability: rewritten retrieval query, chunk count, fallback decision, sources, and excerpts.
- Docker Compose deployment, tests, linting, and a RAGAS evaluation dataset template.

## Quick start

Copy `.env.example` to `.env`, then run `docker compose up --build`.

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for the API. The default configuration persists Chroma and SQLite metadata locally; Postgres is optional for a team deployment.

For local generation, install Ollama, then run `ollama pull llama3.2` and `ollama serve`.

## Local development

From `apps/backend`, create a virtual environment, run `pip install -e ".[dev]"`, then run `uvicorn app.main:app --reload`. In a second terminal, install `apps/frontend` and run `streamlit run rag_ui/app.py`.

## API workflow

1. Register and log in under `/auth`.
2. Upload a document through `POST /api/upload-files/` with a `thread_id`.
3. Query `POST /api/chat/` for an answer, sources, and retrieval trace.

## Repository layout

- `apps/backend/` — FastAPI, RAG pipeline, auth, Chroma persistence.
- `apps/frontend/` — Streamlit document QA console.
- `docs/` — architecture and production scaling notes.
- `evals/` — RAGAS dataset template.
- `docker-compose.yml` — self-hosted deployment.

## Evaluation and production path

Use RAGAS to measure context precision, context recall, faithfulness, and answer relevancy whenever you change chunking, retrieval, embeddings, or prompts. The target is a reproducible before/after score on a fixed evaluation dataset, not an unverified claim.

The next production increments are background ingestion workers, RBAC, document retention/deletion, hybrid retrieval with reranking, OpenTelemetry traces, and CI evaluation gates. See [architecture notes](docs/architecture.md).

