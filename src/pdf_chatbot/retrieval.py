"""Lightweight, transparent local retrieval with TF-IDF."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .documents import Chunk


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float


def rank_chunks(chunks: Sequence[Chunk], question: str, limit: int = 3) -> list[RankedChunk]:
    """Return the most relevant chunks for a question, highest score first."""
    if not chunks or not question.strip() or limit < 1:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks] + [question])
    except ValueError:
        return []
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    indices = scores.argsort()[::-1][:limit]
    return [RankedChunk(chunk=chunks[index], score=float(scores[index])) for index in indices]
