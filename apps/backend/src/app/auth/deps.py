"""
This is authentication middleware for FastAPI. It protects API routes by verifying that users are logged in.

How it works:
Extracts the JWT token from the request header (Authorization: Bearer <token>)
Decodes and validates the token using decode_token()
Extracts the user ID from the token payload (stored as "sub")
Looks up the user in the database
Returns the User object if everything is valid
Raises 401 Unauthorized if anything fails

jose (JavaScript Object Signing and Encryption) is a Python library for working with JWT tokens (JSON Web Tokens).
JWTs are encoded strings that contain user information and are used for authentication

The flow in app:
User logs in → Backend creates JWT with their user ID
Frontend stores token (localStorage/cookie)
Frontend sends token with every request (in Authorization header)
get_current_user() validates the token and retrieves the user
Route handler receives authenticated User object
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User
from .security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError as e:
        raise credentials_exception from e

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user
