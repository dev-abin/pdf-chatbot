from langchain_core.documents import Document
from backend.core.logging_config import logger
import fitz  # PyMuPDF
import numpy as np
import easyocr


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
            text = re.sub(r'-\s*\n\s*', '', text)

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



def extract_pdf_content_ocr(pdf_path, lang='en', use_angle_cls=True, zoom=3):
    """
    Extract text content from a PDF file using EasyOCR.
    
    Args:
        pdf_path (str): Path to the PDF file
        lang (str): Language code for OCR (default: 'en')
        use_angle_cls (bool): Kept for API compatibility; EasyOCR automatically handles orientation reasonably well.
        zoom (int): Zoom factor for image resolution (default: 3, higher = better quality)
    
    Returns:
        list: List of Document objects compatible with LangChain (same format as PyPDFLoader)
    """
    try:
        # EasyOCR expects a list of language codes, e.g. ['en']
        reader = easyocr.Reader([lang])

        doc = fitz.open(pdf_path)
        documents = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render page to high-res image
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

            # Convert Pixmap to NumPy array (H, W, C)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8)
            img_np = img_np.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA → RGB
                img_np = img_np[:, :, :3]

            # Perform OCR
            # result: list of [bbox, text, confidence]
            result = reader.readtext(img_np, detail=1, paragraph=False)

            text_lines = [line[1] for line in result] if result else []
            ocr_text = "\n".join(text_lines)

            document = Document(
                page_content=ocr_text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                },
            )
            documents.append(document)

        doc.close()
        return documents

    except Exception:
        logger.exception("extract_pdf_content_ocr failed")
        return []

