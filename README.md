# PDF Chat

A small, local-first PDF question-answering app built to be easy to run, explain, and extend in an interview.

Upload one or more text-based PDFs, ask questions, and inspect the passages used to answer them. Retrieval is entirely local. If [Ollama](https://ollama.com/) is running, the app uses it to turn retrieved context into a conversational answer; otherwise it still works by showing the most relevant cited passages.

## Why this version

The previous prototype split a single-user demo across separate frontend and backend apps, authentication, Postgres, Chroma, LangChain, OCR, and a remote-model configuration. That made first-run setup fragile and hid the actual product idea.

This version keeps the interesting engineering:

- PDF text extraction with page-level citations
- Deterministic chunking and local TF-IDF retrieval
- Optional local LLM generation with a graceful fallback
- A clean Streamlit interface, tests, linting, and CI

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e "[dev]"
streamlit run app.py
```

Open the local URL Streamlit prints, upload PDFs, choose **Index documents**, then ask a question.

### Optional: generated answers with Ollama

The app is fully usable without an LLM. For generated answers, install Ollama and pull a model:

```bash
ollama pull llama3.2
ollama serve
```

Copy `.env.example` to `.env` to choose another installed model. The app automatically falls back to cited passages if Ollama is unavailable.

## Project layout

```text
app.py                     # Streamlit UI and application flow
src/pdf_chatbot/
  documents.py             # PDF extraction and page-aware chunking
  retrieval.py             # Local TF-IDF ranking
  assistant.py             # Optional Ollama generation + fallback
tests/                     # Fast unit tests for core behavior
.github/workflows/ci.yml   # Lint and test checks
```

## Development

```bash
ruff check .
pytest
```

## Limitations and next steps

This project deliberately targets text-based PDFs. Scanned PDFs need OCR, and a future production version would add background indexing, persistent per-user indexes, access control, and evaluated semantic retrieval. Keeping these out of the default path makes the demo reliable while leaving clear extension points.
