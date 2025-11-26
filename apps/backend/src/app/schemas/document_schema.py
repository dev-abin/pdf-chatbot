from datetime import datetime

from pydantic import BaseModel


class DocumentBase(BaseModel):
    id: int
    owner_id: int
    filename: str
    storage_path: str
    created_at: datetime
    thread_id: str | None = None

    class Config:
        orm_mode = True


class UploadResponse(BaseModel):
    message: str
    document_id: int | None = None
    filename: str | None = None
    thread_id: str | None = None
