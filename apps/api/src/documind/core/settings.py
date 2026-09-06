import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from a .env file if present.
# In Docker, env vars come from docker-compose; load_dotenv() is a no-op if the file isn't there.
load_dotenv()

# --------------------------------------------------------------------
# Storage directories
# --------------------------------------------------------------------
# API code lives in apps/api; runtime artifacts live at the repository root locally.
# Docker overrides the data directory to its mounted /app/data volume.
APP_DIR = Path(__file__).resolve().parents[3]
PROJECT_DIR = Path(__file__).resolve().parents[5]

# Project directories
DATA_DIR = Path(os.getenv("DOCUMIND_DATA_DIR", str(PROJECT_DIR / "data")))
FILE_DIR = DATA_DIR / "files"
VECTOR_DIR = DATA_DIR / "vectors"
LOG_DIR = Path(os.getenv("DOCUMIND_LOG_DIR", str(PROJECT_DIR / "logs")))

LOGGING_CONFIG_PATH = APP_DIR / "configs" / "logging.yaml"

os.makedirs(FILE_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Core backend config
# --------------------------------------------------------------------
class Settings(BaseSettings):
    """Validated runtime configuration; environment variables override local defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "ollama"
    pref_model: str = "llama3.2"
    embedding_provider: str = "huggingface"
    pref_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    jwt_secret: str = "local-development-secret-change-me"
    database_url: str | None = None
    ollama_api_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_base_url: str | None = None


settings = Settings()
LLM_PROVIDER = settings.llm_provider
PREF_MODEL = settings.pref_model
EMBEDDING_PROVIDER = settings.embedding_provider
PREF_EMBEDDING_MODEL = settings.pref_embedding_model
SECRET_KEY = settings.jwt_secret
DATABASE_URL = settings.database_url or f"sqlite:///{DATA_DIR / 'documind.db'}"
OLLAMA_API_URL = settings.ollama_api_url
OPENAI_API_KEY = settings.openai_api_key
OPENAI_BASE_URL = settings.openai_base_url


# --------------------------------------------------------------------
# Validation (only for backend-critical vars)
# --------------------------------------------------------------------

if LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")
    if not PREF_MODEL:
        raise ValueError("PREF_MODEL required when LLM_PROVIDER=openai")

if EMBEDDING_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required when EMBEDDING_PROVIDER=openai")
    if not PREF_EMBEDDING_MODEL:
        raise ValueError("PREF_EMBEDDING_MODEL required when EMBEDDING_PROVIDER=openai")

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
FILE_EXTENSIONS = (".pdf", ".docx", ".txt")


# a string literal for answer not found message
NO_ANSWER_FOUND = (
    "The provided documents do not contain enough information to answer this question."
)

