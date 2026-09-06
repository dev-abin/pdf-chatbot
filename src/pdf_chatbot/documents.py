"""PDF text extraction and deterministic chunking."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    page_number: int
    source_name: str


def _clean_text(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip()


def extract_pdf_pages(source: BinaryIO) -> list[Page]:
    """Extract non-empty text pages from a PDF file-like object."""
    reader = PdfReader(source)
    pages: list[Page] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = _clean_text(pdf_page.extract_text() or "")
        if text:
            pages.append(Page(number=number, text=text))
    return pages


def split_pages(
    pages: Iterable[Page],
    source_name: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Split page text into overlapping character chunks while retaining citations."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for page in pages:
        for start in range(0, len(page.text), step):
            text = page.text[start : start + chunk_size]
            if text:
                chunks.append(Chunk(text=text, page_number=page.number, source_name=source_name))
            if start + chunk_size >= len(page.text):
                break
    return chunks
