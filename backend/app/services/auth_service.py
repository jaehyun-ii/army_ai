"""
Authentication service for handling login and token generation.

This service centralizes authentication business logic, eliminating
code duplication between /login and /login-json endpoints.

ALIGNED WITH DATABASE SCHEMA: Uses username as primary identifier.

SECURITY FEATURES:
- Login attempt tracking (5 failed attempts = 30 min lockout)
- Account lockout management
- Single session enforcement (only one concurrent login per user)

SOLID Principles:
- SRP: Single responsibility - authentication and token generation
- DIP: Depends on abstractions (CRUD, security functions)
"""
from datetime import timedelta
from typing import Dict
import secrets
import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import UnauthorizedError, ForbiddenError
from app.core.security import create_access_token
from app.crud.user import (
    authenticate_user,
    is_account_locked,
    increment_failed_login,
    reset_failed_login,
    set_session_id,
    get_user_by_username
)
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

        Security features:
        - Checks account lockout status
        - Tracks failed login attempts (5 attempts = 30 min lockout)
        - Enforces single session (kicks out other sessions)
        - Resets failed attempts on successful login

        Args:
            db: Database session
            username: Username (primary identifier in DB schema)
            password: User password

        Returns:
            Token with access_token and token_type

        Raises:
            ForbiddenError: If account is locked or inactive
            UnauthorizedError: If credentials are invalid

        Example:
            token = await auth_service.authenticate_and_create_token(
                db, "john_doe", "password123"
            )
        """
        # Step 1: Get user by username first (to check lock status)
        user = await get_user_by_username(db, username)

        # Step 2: Check if account is locked
        if user and await is_account_locked(user):
            # Calculate remaining lock time
            remaining = (user.locked_until - datetime.datetime.utcnow()).total_seconds() / 60
            raise ForbiddenError(
                detail=f"계정이 잠겨있습니다. {int(remaining)}분 후에 다시 시도해주세요. "
                       f"(로그인 실패 5회 초과)"
            )

        # Step 3: Authenticate user by username and password
        authenticated_user = await authenticate_user(db, username, password)
        if not authenticated_user:
            # Increment failed login attempts if user exists
            if user:
                await increment_failed_login(db, user)
                remaining_attempts = max(0, 5 - user.failed_login_attempts)
                if remaining_attempts > 0:
                    raise UnauthorizedError(
                        detail=f"아이디 또는 비밀번호가 올바르지 않습니다. "
                               f"(남은 시도 횟수: {remaining_attempts}회)"
                    )
                else:
                    raise ForbiddenError(
                        detail="로그인 실패 5회 초과로 계정이 30분간 잠겼습니다."
                    )
            raise UnauthorizedError(detail="아이디 또는 비밀번호가 올바르지 않습니다.")

        # Step 4: Check if user is active
        if not authenticated_user.is_active:
            raise ForbiddenError(detail="비활성화된 계정입니다.")

        # Step 5: Generate unique session ID for single login enforcement
        session_id = secrets.token_urlsafe(32)

        # Step 6: Reset failed login attempts and set session
        await reset_failed_login(db, authenticated_user)
        await set_session_id(db, authenticated_user, session_id)

        # Step 7: Create access token with user information and session ID
        access_token_expires = timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        access_token = create_access_token(
            data={
                "sub": str(authenticated_user.id),  # Use user ID as subject
                "username": authenticated_user.username,
                "email": authenticated_user.email,
                "role": authenticated_user.role,
                "session_id": session_id,  # Include session ID in token
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
    """
    Get current user from JWT token.

    Security: Validates session ID to enforce single login per user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    session_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="다른 위치에서 로그인되어 현재 세션이 종료되었습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("username")
        session_id: str = payload.get("session_id")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await crud_user.get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception

    # Check session ID for single login enforcement
    if session_id and user.current_session_id:
        if session_id != user.current_session_id:
            # User logged in from another location
            raise session_exception

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
