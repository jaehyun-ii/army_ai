"""
User schemas (aligned with database schema).
DB requires: username (NOT NULL, unique), email (nullable), password_hash
"""
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional
from datetime import datetime
from uuid import UUID

from app.models.user import UserRole


class UserBase(BaseModel):
    """Base user schema (aligned with DB schema)."""

    username: str  # DB: NOT NULL, unique
    email: Optional[EmailStr] = None  # DB: nullable
    role: UserRole = UserRole.USER


class UserCreate(UserBase):
    """Schema for creating a user."""

    password: str


class UserLogin(BaseModel):
    """Schema for user login (using username)."""

    username: str  # Changed from email to username
    password: str


class UserUpdate(BaseModel):
    """Schema for updating a user."""

    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """Schema for user response."""

    id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""

    user_id: Optional[UUID] = None
    email: Optional[str] = None
