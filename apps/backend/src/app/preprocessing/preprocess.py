from ..core.logging_config import logger


def preprocess_file_content(raw_docs):
    """
    Clean and normalize file content for downstream embedding/RAG.

    Steps per page:
      - Fix hyphenated words split across lines.
      - Normalize Unicode to NFC.
      - Remove control chars and weird symbols, keep letters/numbers/punctuation/whitespace.
      - Normalize excessive newlines and spaces.
    """
    import re
    import unicodedata

    import regex  # For Unicode property support

    cleaned_docs = []
    logger.info("Preprocessing file content | pages=%d", len(raw_docs))

    try:
        for i, doc in enumerate(raw_docs):
            text = doc.page_content or ""

            # Fix hyphenated words split across lines: "ex-\nample" -> "example"
            text = re.sub(r"-\s*\n\s*", "", text)

            # Normalize Unicode for consistent character representation
            text = unicodedata.normalize("NFC", text)

            # Remove non-text noise but keep letters, numbers, punctuation, whitespace
            # NOTE: This keeps newlines and periods, commas, etc.
            text = regex.sub(r"[^\p{L}\p{N}\p{P}\s]", " ", text)

            # Standardize paragraph breaks: 3+ newlines -> exactly 2
            text = re.sub(r"\n{3,}", "\n\n", text)

            # Replace multiple spaces/tabs with a single space
            text = re.sub(r"[ \t]+", " ", text)

            # Strip leading/trailing whitespace
            text = text.strip()

            original_len = len(doc.page_content or "")
            cleaned_len = len(text)
            logger.debug(
                "Preprocess page %d | original_len=%d | cleaned_len=%d",
                i,
                original_len,
                cleaned_len,
            )

            doc.page_content = text
            cleaned_docs.append(doc)

    except Exception:
        logger.exception("preprocess_file_content failed")
        return []

    logger.info("Preprocessing complete | cleaned_pages=%d", len(cleaned_docs))
    return cleaned_docs
