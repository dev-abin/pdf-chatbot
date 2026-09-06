# Architecture

## Runtime

DocuMind has a FastAPI backend and Streamlit UI. ChromaDB is embedded and persisted on the backend volume; SQLite is the default metadata store. This gives a simple self-hosted startup path while retaining clean boundaries for production replacements.

## Agentic RAG flow

1. Parse and chunk an uploaded document while preserving user, thread, document, filename, and page metadata.
2. Rewrite a follow-up question into a standalone query using chat history.
3. Run metadata-filtered MMR retrieval against Chroma.
4. Generate only from retrieved context and require citations.
5. If context is absent or insufficient, invoke a maximum-three-iteration ReAct agent with one Wikipedia tool.
6. Return sources and retrieval/fallback trace in the API response.

The bounded fallback prevents expensive, unstructured tool loops and makes behaviour testable.

## Evaluation plan

Build a versioned RAGAS dataset per domain. Measure context precision, context recall, faithfulness, and answer relevancy before and after changes to chunking, embeddings, prompts, or retrieval. Record score deltas and qualitative failures in pull requests.

## Scaling path

For a team deployment, move ingestion to a worker queue, use Postgres instead of SQLite, add RBAC and retention/deletion APIs, introduce reranking and hybrid search, and emit OpenTelemetry spans.

