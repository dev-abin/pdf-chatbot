# apps/backend/src/app/schemas/auth_schema.py

from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    id: int
    email: EmailStr
    is_active: bool
    role: str
    created_at: datetime

    class Config:
        orm_mode = True


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    user: UserBase


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserBase
