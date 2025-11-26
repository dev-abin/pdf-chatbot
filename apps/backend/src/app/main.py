from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.chat import router as chat_router
from .api.upload import router as upload_router
from .auth.routes import router as auth_router
from .core.logging_config import logger, setup_logging
from .db.base import Base, engine

# ---------------------------
# Logging
# ---------------------------
setup_logging()
logger.info("FastAPI app module started")


# ---------------------------
# Lifespan Events
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Creating DB tables (if not present)")
    Base.metadata.create_all(bind=engine)

    yield

    # Shutdown (optional cleanup)
    logger.info("Application shutdown")


# ---------------------------
# Create FastAPI application
# ---------------------------
app = FastAPI(
    title="Agentic RAG Chat Service",
    description="PDF Chatbot + Wikipedia fallback (Agentic RAG Architecture) ",
    version="1.0.0",
    lifespan=lifespan,
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
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(chat_router, prefix="/api", tags=["chat"])

# ---------------------------
# Local run
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
