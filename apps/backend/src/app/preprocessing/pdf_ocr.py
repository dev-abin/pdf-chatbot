# apps/backend/src/app/preprocessing/pdf_ocr.py

import easyocr
import fitz  # PyMuPDF
import numpy as np
from langchain_core.documents import Document

from ..core.logging_config import logger

# Cache readers by language
_OCR_READERS: dict[str, easyocr.Reader] = {}


def _get_reader(lang: str = "en") -> easyocr.Reader:
    if lang not in _OCR_READERS:
        _OCR_READERS[lang] = easyocr.Reader([lang])
    return _OCR_READERS[lang]


def extract_pdf_content_ocr(
    pdf_path: str, lang: str = "en", zoom: int = 3
) -> list[Document]:
    """
    Extract text content from a PDF file using EasyOCR.
    Returns a list of LangChain Document objects (one per page).
    """
    try:
        reader = _get_reader(lang)
        doc = fitz.open(pdf_path)
        documents: list[Document] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

            img_np = np.frombuffer(pix.samples, dtype=np.uint8)
            img_np = img_np.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_np = img_np[:, :, :3]

            result = reader.readtext(img_np, detail=1, paragraph=False)
            text_lines = [line[1] for line in result] if result else []
            ocr_text = "\n".join(text_lines)

            documents.append(
                Document(
                    page_content=ocr_text,
                    metadata={"source": pdf_path, "page": page_num},
                )
            )

        doc.close()
        return documents

    except Exception:
        logger.exception("extract_pdf_content_ocr failed")
        return []
