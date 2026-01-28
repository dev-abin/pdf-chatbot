# apps/backend/src/app/api/upload.py

import hashlib
import os

import anyio
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from ..auth.deps import get_current_user
from ..core.embedding_client import get_embedding_function
from ..core.logging_config import logger
from ..core.settings import FILE_DIR, FILE_EXTENSIONS, VECTOR_DIR
from ..db.base import get_db
from ..db.models import Document, User
from ..preprocessing.pdf_ocr import extract_pdf_content_ocr
from ..preprocessing.preprocess import preprocess_file_content
from ..schemas.document_schema import UploadResponse

router = APIRouter(prefix="/api", tags=["upload"])
MAX_FILE_BYTES = 30 * 1024 * 1024  # 30 MB


async def _load_documents(file_path: str, filename: str):
    if filename.endswith(".pdf"):
        logger.info("Processing PDF file")
        pdf_loader = PyMuPDFLoader(file_path)
        raw_docs = await anyio.to_thread.run_sync(pdf_loader.load)

        if not raw_docs or all(doc.page_content.strip() == "" for doc in raw_docs):
            logger.info(
                "No extractable text found. PDF may be scanned or image-based. Proceeding to OCR."
            )
            raw_docs = await anyio.to_thread.run_sync(extract_pdf_content_ocr, file_path)
        return raw_docs

    loaders = {
        ".docx": (Docx2txtLoader, "DOCX"),
        ".txt": (TextLoader, "TXT"),
    }
    _, ext = os.path.splitext(filename)
    loader_entry = loaders.get(ext)
    if not loader_entry:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )
    loader_cls, label = loader_entry
    logger.info("Processing %s file", label)
    if ext == ".txt":
        loader = loader_cls(file_path, encoding="utf-8")
    else:
        loader = loader_cls(file_path)
    return await anyio.to_thread.run_sync(loader.load)


@router.post("/upload-files/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile,
    thread_id: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload and process a PDF, DOCX, TXT document if not already processed.
      - Tied to authenticated user via Document table.
      - Scoped to a specific thread via thread_id.
      - Chroma docs tagged with user_id and thread_id.
    """
    try:
        logger.info(
            "Upload files endpoint called | endpoint=/upload-files/ | uploaded_file='%s' | user_id=%s | thread_id=%s",
            file.filename,
            current_user.id,
            thread_id,
        )

        if file.filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file has no filename.",
            )

        filename = file.filename.lower()

        if not filename.endswith(FILE_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF/DOCX or TXT file",
            )

        _, ext = os.path.splitext(filename)

        content = await file.read()
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum allowed size is 20 MB.",
            )

        file_hash = hashlib.sha256(content).hexdigest()
        file_path = os.path.join(FILE_DIR, f"{file_hash}{ext}")

        if os.path.exists(file_path):
            logger.info("%s has already been processed.", filename)
            return {
                "message": f"{filename} has already been processed.",
                "filename": filename,
            }

        logger.info("Storing the new file contents and updating vectorstore")

        async with await anyio.open_file(file_path, "wb") as newfile:
            await newfile.write(content)

        doc = Document(
            owner_id=current_user.id,
            filename=filename,
            storage_path=file_path,
            thread_id=thread_id,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        embeddings = get_embedding_function()
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings,
        )

        logger.info("Initialized Chroma vectorstore at %s", VECTOR_DIR)

        raw_docs = await _load_documents(file_path, filename)

        cleaned_docs = preprocess_file_content(raw_docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )
        docs = splitter.split_documents(cleaned_docs)

        metadatas = [
            {
                **(d.metadata or {}),
                "user_id": current_user.id,
                "document_id": doc.id,
                "filename": filename,
                "thread_id": thread_id,
            }
            for d in docs
        ]

        vectorstore.add_documents(docs, metadatas=metadatas)

        logger.info(
            "Successfully processed %s for user_id=%s thread_id=%s",
            filename,
            current_user.id,
            thread_id,
        )
        return UploadResponse(
            message=f"Successfully processed {filename}",
            document_id=doc.id,
            filename=filename,
            thread_id=thread_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("%s processing failed", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        ) from e
