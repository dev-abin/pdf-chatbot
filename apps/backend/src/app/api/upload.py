import hashlib
import os

from fastapi import APIRouter, HTTPException, UploadFile, status
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.embedding_client import get_embedding_function
from ..core.logging_config import logger
from ..core.settings import FILE_DIR, FILE_EXTENSIONS, VECTOR_DIR
from ..preprocessing.pdf_ocr import extract_pdf_content_ocr
from ..preprocessing.preprocess import preprocess_file_content

router = APIRouter()


@router.post("/upload-files/")
async def upload_file(file: UploadFile):
    """
    Upload and process a PDF, DOCX, TXT document if not already processed.
    """
    try:
        logger.info(
            "Upload files endpoint called | endpoint=/upload-files/ | uploaded_file='%s'",
            file.filename,
        )

        if file.filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file has no filename.",
            )

        filename = file.filename

        if not filename.endswith(FILE_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF/DOCX or TXT file",
            )

        _, ext = os.path.splitext(filename)

        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        file_path = os.path.join(FILE_DIR, f"{file_hash}{ext}")

        if os.path.exists(file_path):
            logger.info("%s has already been processed.", filename)
            return {"message": f"{filename} has already been processed."}

        logger.info("Storing the new file contents and updating vectorstore")

        # Persist file
        with open(file_path, "wb") as newfile:
            newfile.write(content)

        # Reuse shared vectorstore
        embeddings = get_embedding_function()
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings,
        )

        logger.info("Initialized Chroma vectorstore at %s", VECTOR_DIR)

        # Load content
        if filename.endswith(".pdf"):
            logger.info("Processing PDF file")
            pdf_loader = PyMuPDFLoader(file_path)
            raw_docs = pdf_loader.load()

            if not raw_docs or all(doc.page_content.strip() == "" for doc in raw_docs):
                logger.info(
                    "No extractable text found. PDF may be scanned or image-based. Proceeding to OCR."
                )
                raw_docs = extract_pdf_content_ocr(file_path)

        elif filename.endswith(".docx"):
            logger.info("Processing DOCX file")
            docx_loader = Docx2txtLoader(file_path)
            raw_docs = docx_loader.load()

        elif filename.endswith(".txt"):
            logger.info("Processing TXT file")
            txt_loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = txt_loader.load()
        else:
            # Should not reach here due to earlier extension check
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type.",
            )

        # Preprocess and chunk
        cleaned_docs = preprocess_file_content(raw_docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )
        docs = splitter.split_documents(cleaned_docs)

        # Add to vectorstore
        vectorstore.add_documents(docs)

        logger.info("Successfully processed %s", filename)
        return {"message": f"Successfully processed {filename}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("%s processing failed", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        ) from e
