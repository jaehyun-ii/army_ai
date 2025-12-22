"""
Database connection and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import AsyncGenerator

from app.core.config import settings

# Create async engine
# SSOT: All database configuration comes from settings
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    future=True,
    pool_pre_ping=settings.DATABASE_POOL_PRE_PING,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
)

# Create sync engine for non-async operations (e.g., auto-loader)
# Sync engine uses half the pool size of async engine
sync_database_url = settings.DATABASE_URL.replace('+asyncpg', '').replace('postgresql+asyncpg', 'postgresql')
sync_pool_size = max(1, settings.DATABASE_POOL_SIZE // 2)
sync_max_overflow = max(1, settings.DATABASE_MAX_OVERFLOW // 2)

sync_engine = create_engine(
    sync_database_url,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=settings.DATABASE_POOL_PRE_PING,
    pool_size=sync_pool_size,
    max_overflow=sync_max_overflow,
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create sync session factory
SessionLocal = sessionmaker(
    sync_engine,
    class_=Session,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
