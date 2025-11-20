"""
Authentication service for handling login and token generation.

This service centralizes authentication business logic, eliminating
code duplication between /login and /login-json endpoints.

ALIGNED WITH DATABASE SCHEMA: Uses username as primary identifier.

SOLID Principles:
- SRP: Single responsibility - authentication and token generation
- DIP: Depends on abstractions (CRUD, security functions)
"""
from datetime import timedelta
from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import UnauthorizedError, ForbiddenError
from app.core.security import create_access_token
from app.crud.user import authenticate_user
from app.schemas.user import Token
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

from app.database import get_db
from app.crud import user as crud_user
from app.models.user import User


class AuthService:
    """
    Service for user authentication and token management.

    Centralizes authentication logic to avoid duplication across endpoints.
    Uses username as primary identifier (aligned with DB schema).
    """

    async def authenticate_and_create_token(
        self,
        db: AsyncSession,
        username: str,
        password: str,
    ) -> Token:
        """
        Authenticate user and create access token (using username).

        Args:
            db: Database session
            username: Username (primary identifier in DB schema)
            password: User password

        Returns:
            Token with access_token and token_type

        Raises:
            UnauthorizedError: If credentials are invalid
            ForbiddenError: If user is inactive

        Example:
            token = await auth_service.authenticate_and_create_token(
                db, "john_doe", "password123"
            )
        """
        # Step 1: Authenticate user by username
        user = await authenticate_user(db, username, password)
        if not user:
            raise UnauthorizedError(detail="Incorrect username or password")

        # Step 2: Check if user is active
        if not user.is_active:
            raise ForbiddenError(detail="Inactive user")

        # Step 3: Create access token with user information
        access_token_expires = timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        access_token = create_access_token(
            data={
                "sub": str(user.id),  # Use user ID as subject
                "username": user.username,
                "email": user.email,
                "role": user.role,
            },
            expires_delta=access_token_expires
        )

        return Token(access_token=access_token, token_type="bearer")

    def create_token_response(self, token: Token) -> Dict[str, str]:
        """
        Create standardized token response dict.

        Args:
            token: Token object

        Returns:
            Dict with access_token and token_type
        """
        return {
            "access_token": token.access_token,
            "token_type": token.token_type
        }


# Global service instance
auth_service = AuthService()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/login")

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("username")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await crud_user.get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_active_admin(current_user: User = Depends(get_current_active_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user
