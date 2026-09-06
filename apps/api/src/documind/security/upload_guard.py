from pathlib import Path


MAX_DOCUMENT_BYTES = 30 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UploadValidationError(ValueError):
    """Raised when an upload cannot safely enter the ingestion pipeline."""


def validate_document(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError("File must be a PDF, DOCX, or TXT file.")
    if not content:
        raise UploadValidationError("Uploaded file is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise UploadValidationError("File too large. Maximum allowed size is 30 MB.")
    return suffix
