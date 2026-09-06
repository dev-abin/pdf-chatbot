"""Answer generation with an optional local Ollama integration."""

from __future__ import annotations

import os
from typing import Sequence

from .retrieval import RankedChunk


def _context(ranked: Sequence[RankedChunk]) -> str:
    return "\\n\\n".join(
        f"[Source: {item.chunk.source_name}, page {item.chunk.page_number}]\\n{item.chunk.text}"
        for item in ranked
    )


def _extractive_answer(ranked: Sequence[RankedChunk]) -> str:
    if not ranked:
        return "I could not find matching text in the indexed PDFs. Try a more specific question."
    excerpts = "\\n\\n".join(
        f"- {item.chunk.text[:420].rstrip()}… ({item.chunk.source_name}, p. {item.chunk.page_number})"
        for item in ranked
    )
    return "Here are the most relevant passages from your documents:\\n\\n" + excerpts


def answer_question(question: str, ranked: Sequence[RankedChunk]) -> tuple[str, bool]:
    """Use local Ollama when available; otherwise return cited retrieved passages."""
    if not ranked:
        return _extractive_answer(ranked), False

    try:
        from ollama import chat

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer only from the supplied context. If it is insufficient, say so. Cite source name and page number.",
                },
                {"role": "user", "content": f"Context:\\n{_context(ranked)}\\n\\nQuestion: {question}"},
            ],
        )
        return response["message"]["content"].strip(), True
    except Exception:
        return _extractive_answer(ranked), False
