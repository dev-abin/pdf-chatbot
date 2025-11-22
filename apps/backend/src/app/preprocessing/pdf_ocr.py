import easyocr
import fitz  # PyMuPDF
import numpy as np
from langchain_core.documents import Document

from ..core.logging_config import logger


def extract_pdf_content_ocr(pdf_path, lang="en", zoom=3):
    """
    Extract text content from a PDF file using EasyOCR.
    Args:
        pdf_path (str): Path to the PDF file
        lang (str): Language code for OCR (default: 'en')
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
