import hashlib
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..preprocessing.preprocess import preprocess_file_content
from ..preprocessing.pdf_ocr import extract_pdf_content_ocr
from ..core.settings import FILE_EXTENSIONS, FILE_DIR, VECTOR_DIR, PREF_EMBEDDING_MODEL
from ..core.logging_config import logger

router = APIRouter()


@router.post("/upload-files/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a PDF, DOCX, TXT document if not already processed.
    """
    try:
        logger.info(
            "Upload files endpoint called | endpoint=/upload-files/ | uploaded_file='%s'",
            file.filename,
        )

        if not file.filename.endswith(FILE_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF/DOCX or TXT file",
            )

        _, ext = os.path.splitext(file.filename)

        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        file_path = os.path.join(FILE_DIR, f"{file_hash}{ext}")

        if os.path.exists(file_path):
            logger.info("%s has already been processed.", file.filename)
            return {"message": f"{file.filename} has already been processed."}

        logger.info("Storing the new file contents and updating vectorstore")

        # Persist file
        with open(file_path, "wb") as newfile:
            newfile.write(content)

        # Reuse shared vectorstore
        embeddings = HuggingFaceEmbeddings(model_name=PREF_EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings,
        )
        
        logger.info("Initialized Chroma vectorstore at %s", VECTOR_DIR)

        # Load content
        if file.filename.endswith(".pdf"):
            logger.info("Processing PDF file")
            loader = PyMuPDFLoader(file_path)
            raw_docs = loader.load()

            if not raw_docs or all(
                doc.page_content.strip() == "" for doc in raw_docs
            ):
                logger.info(
                    "No extractable text found. PDF may be scanned or image-based. Proceeding to OCR."
                )
                raw_docs = extract_pdf_content_ocr(file_path)

        elif file.filename.endswith(".docx"):
            logger.info("Processing DOCX file")
            loader = Docx2txtLoader(file_path)
            raw_docs = loader.load()

        elif file.filename.endswith(".txt"):
            logger.info("Processing TXT file")
            loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()
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

        logger.info("Successfully processed %s", file.filename)
        return {"message": f"Successfully processed {file.filename}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("%s processing failed", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )
