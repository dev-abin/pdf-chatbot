"""
This creates two API endpoints for user authentication: registration and login.

1. /auth/register - Create a new account
User sends email + password
Check if email already exists in database
If exists → return error "User already exists"
If new → hash the password (never store plain passwords!)
Create new User record in database
Return the new user's info (id, email, role)

2. /auth/login - Sign in to existing account
User sends email + password
Look up user by email in database
Verify the password matches the stored hash
If wrong → return 401 "Incorrect email or password"
If correct → create a JWT token containing the user's ID
Return the token + user info

The Complete Flow
1. User registers    → Account created with hashed password
2. User logs in      → Receives JWT token
3. User stores token → (in frontend: localStorage/cookie)
4. User makes requests → Sends token in Authorization header
5. Backend verifies token → Uses get_current_user() dependency
6. User accesses protected routes → ✓ Authenticated!
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db.base import get_db
from ..db.models import User
from ..schemas.auth_schema import LoginRequest, RegisterRequest
from .security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists",
        )
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        is_active=True,
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
    }


@router.post("/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    access_token = create_access_token({"sub": str(user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role,
        },
    }
