import os
from typing import Final


def require_env(name: str) -> str:
    """
    Read a required environment variable.

    Raises:
        RuntimeError: if the variable is missing or empty.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# FastAPI backend URLs
CHAT_API_URL: Final[str] = require_env("CHAT_API_URL")
UPLOAD_FILE_URL: Final[str] = require_env("UPLOAD_FILE_URL")
AUTH_LOGIN_URL: Final[str] = require_env("AUTH_LOGIN_URL")
AUTH_REGISTER_URL: Final[str] = require_env("AUTH_REGISTER_URL")
