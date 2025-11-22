import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.upload import router as upload_router
from .api.chat import router as chat_router
from .core.logging_config import setup_logging, logger

# ---------------------------
# Logging
# ---------------------------
setup_logging()  # important: call once at startup
logger.info("FastAPI app module started")

# ---------------------------
# Create FastAPI application
# ---------------------------
app = FastAPI(
    title="Agentic RAG Chat Service",
    description="PDF Chatbot + Wikipedia fallback (Agentic RAG Architecture)",
    version="1.0.0",
)

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Include Routers
# ---------------------------
app.include_router(upload_router)
app.include_router(chat_router)

# ---------------------------
# Local run
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
