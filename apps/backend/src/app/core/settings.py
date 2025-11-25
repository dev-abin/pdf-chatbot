import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file if present.
# In Docker, env vars come from docker-compose; load_dotenv() is a no-op if the file isn't there.
load_dotenv()

# --------------------------------------------------------------------
# Storage directories
# --------------------------------------------------------------------
# Resolve project root explicitly
BASE_DIR = Path(__file__).resolve().parents[3]

# Project directories
DATA_DIR = BASE_DIR / "data"
FILE_DIR = DATA_DIR / "files"
VECTOR_DIR = DATA_DIR / "vectors"
LOG_DIR = BASE_DIR / "logs"

LOGGING_CONFIG_PATH = BASE_DIR / "configs" / "logging.yaml"

os.makedirs(FILE_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Core backend config (required)
# --------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # e.g. "ollama", "openai"
PREF_MODEL = os.getenv("PREF_MODEL")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
PREF_EMBEDDING_MODEL = os.getenv("PREF_EMBEDDING_MODEL")

SECRET_KEY = os.getenv("JWT_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL")

# Ollama-specific
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

# OpenAI / compatible
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL"
)  # optional (for Azure, self-hosted, etc.)


# --------------------------------------------------------------------
# Frontend-facing URLs (optional for backend; required for frontend)
# --------------------------------------------------------------------
UPLOAD_FILE_URL = os.getenv("UPLOAD_FILE_URL")
CHAT_API_URL = os.getenv("CHAT_API_URL")

# --------------------------------------------------------------------
# Validation (only for backend-critical vars)
# --------------------------------------------------------------------
required_vars = {
    "OLLAMA_API_URL": OLLAMA_API_URL,
    "PREF_MODEL": PREF_MODEL,
    "EMBEDDING_PROVIDER": EMBEDDING_PROVIDER,
    "PREF_EMBEDDING_MODEL": PREF_EMBEDDING_MODEL,
}

missing = [key for key, value in required_vars.items() if not value]
if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

if LLM_PROVIDER == "ollama" and not OLLAMA_API_URL:
    raise ValueError("OLLAMA_API_URL required when LLM_PROVIDER=ollama")

if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")

if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY required when EMBEDDING_PROVIDER=openai")

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
FILE_EXTENSIONS = (".pdf", ".docx", ".txt")


# a string literal for answer not found message
NO_ANSWER_FOUND = (
    "The provided documents do not contain enough information to answer this question."
)
