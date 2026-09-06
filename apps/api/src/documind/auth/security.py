"""
This file contains all the security/cryptography functions for the authentication system.
It handles password hashing and JWT token management.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from ..core.settings import SECRET_KEY

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Uses bcrypt (very secure hashing algorithm)
    One-way encryption (can't reverse it to get original password)
    Each hash is unique even for the same password (due to "salt")
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """
    Takes data (usually user ID)
    Adds expiration time (60 minutes from now)
    Encrypts it with SECRET_KEY
    Returns the token string
    """
    to_encode = data.copy()
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(UTC) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict[str, Any]:
    """
    Decrypts the token using SECRET_KEY
    Checks if signature is valid (not tampered with)
    Checks if token hasn't expired
    Returns the payload data
    Raises JWTError if anything is wrong
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise exc
