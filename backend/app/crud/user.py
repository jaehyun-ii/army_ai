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
    """Create a new user (using username as primary identifier)."""
    hashed_password = get_password_hash(user_create.password)

    db_user = User(
        username=user_create.username,  # DB: NOT NULL, unique
        email=user_create.email,  # DB: nullable
        password_hash=hashed_password,
        role=user_create.role,
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
