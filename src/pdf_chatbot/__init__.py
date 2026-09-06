"""Local-first PDF question answering package."""

from .documents import Chunk, extract_pdf_pages, split_pages
from .retrieval import RankedChunk, rank_chunks

__all__ = ["Chunk", "RankedChunk", "extract_pdf_pages", "rank_chunks", "split_pages"]
