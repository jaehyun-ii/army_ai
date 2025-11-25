"""
User CRUD operations (aligned with database schema).
DB schema uses username (NOT NULL, unique) as primary identifier.
"""
from typing import Optional, List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from passlib.context import CryptContext
import datetime

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.core.password_validator import validate_password

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Get user by username (primary identifier in DB schema)."""
    result = await db.execute(
        select(User).filter(User.username == username, User.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email (optional field in DB schema)."""
    if not email:
        return None
    result = await db.execute(
        select(User).filter(User.email == email, User.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[User]:
    """Get user by id."""
    result = await db.execute(
        select(User).filter(User.id == user_id, User.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, user_create: UserCreate) -> User:
    """
    Create a new user (using username as primary identifier).

    Validates password against security policy before creation.
    """
    # Validate password against security policy
    is_valid, errors = validate_password(user_create.password, user_create.username)
    if not is_valid:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"비밀번호 정책 위반: {', '.join(errors)}"
        )

    hashed_password = get_password_hash(user_create.password)

    db_user = User(
        username=user_create.username,  # DB: NOT NULL, unique
        email=user_create.email,  # DB: nullable
        password_hash=hashed_password,
        role=user_create.role,
        failed_login_attempts=0,
    )

    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    return db_user


async def authenticate_user(
    db: AsyncSession, username: str, password: str
) -> Optional[User]:
    """Authenticate a user by username."""
    user = await get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[User]:
    """Get all non-deleted users."""
    result = await db.execute(
        select(User)
        .filter(User.deleted_at.is_(None))
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def update_user(db: AsyncSession, user: User, user_update: UserUpdate) -> User:
    """Update a user's information."""
    update_data = user_update.dict(exclude_unset=True)
    if "password" in update_data:
        update_data["password_hash"] = get_password_hash(update_data.pop("password"))
    
    for key, value in update_data.items():
        setattr(user, key, value)
        
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def delete_user(db: AsyncSession, user_id: UUID) -> Optional[User]:
    """Soft delete a user by setting the deleted_at timestamp."""
    result = await db.execute(
        select(User).filter(User.id == user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()

    if user:
        user.deleted_at = datetime.datetime.utcnow()
        user.is_active = False
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return user


async def increment_failed_login(db: AsyncSession, user: User) -> None:
    """Increment failed login attempts and lock account if threshold exceeded."""
    user.failed_login_attempts += 1

    # Lock account for 30 minutes after 5 failed attempts
    if user.failed_login_attempts >= 5:
        user.locked_until = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
        user.is_active = False

    db.add(user)
    await db.commit()
    await db.refresh(user)


async def reset_failed_login(db: AsyncSession, user: User) -> None:
    """Reset failed login attempts after successful login."""
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = datetime.datetime.utcnow()

    db.add(user)
    await db.commit()
    await db.refresh(user)


async def is_account_locked(user: User) -> bool:
    """Check if account is currently locked."""
    if user.locked_until is None:
        return False

    # Check if lock period has expired
    if datetime.datetime.utcnow() > user.locked_until:
        return False

    return True


async def set_session_id(db: AsyncSession, user: User, session_id: str) -> None:
    """Set current session ID for single login enforcement."""
    user.current_session_id = session_id
    db.add(user)
    await db.commit()
    await db.refresh(user)


async def clear_session_id(db: AsyncSession, user: User) -> None:
    """Clear current session ID on logout."""
    user.current_session_id = None
    db.add(user)
    await db.commit()
    await db.refresh(user)
