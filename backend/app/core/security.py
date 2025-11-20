"""
Security utilities for authentication and authorization.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import jwt
from jwt import PyJWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings


security = HTTPBearer(auto_error=False)


def create_access_token(data: Dict[str, Any], expires_delta: timedelta = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT access token.

    Args:
        token: JWT token to decode

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_token(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    """
    Verify JWT token and extract email (authentication only).

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Email from token payload

    Raises:
        HTTPException: If authentication fails
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)

    email: str = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return email


async def get_user_by_email(db: AsyncSession, email: str):
    """
    Fetch user from database by email (user lookup only).

    Args:
        db: Database session
        email: User email

    Returns:
        User object

    Raises:
        HTTPException: If user not found or inactive
    """
    from app.models.user import User
    result = await db.execute(
        select(User).filter(User.email == email, User.is_active == True)
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = None
):
    """
    Get the current authenticated user from JWT token.

    This is a convenience function that combines verify_token and get_user_by_email.

    RECOMMENDED APPROACH - Use the separated functions directly:
    - Use verify_token() for authentication only (lightweight)
    - Use get_user_by_email() for fetching user details from DB

    This function is maintained for backward compatibility but may be deprecated
    in future versions in favor of explicit separation of authentication and lookup.

    Args:
        credentials: HTTP authorization credentials
        db: Database session (optional). If provided, fetches full user from database.

    Returns:
        User object if db provided, otherwise dict with email

    Raises:
        HTTPException: If authentication fails or user not found

    Example usage:
        # Lightweight auth check (no DB hit)
        email = verify_token(credentials)

        # Full user lookup when needed
        user = await get_user_by_email(db, email)
    """
    # Step 1: Authentication - verify token and extract email
    email = verify_token(credentials)

    # Step 2: Optional user lookup - fetch from database if session provided
    if db is not None:
        return await get_user_by_email(db, email)

    # Return email for lightweight auth checks
    return {"email": email}
