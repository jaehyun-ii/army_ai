"""
Base CRUD operations.
"""
from typing import Generic, TypeVar, Type, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.sql import Select
from pydantic import BaseModel
from uuid import UUID

from app.database import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDRead(Generic[ModelType]):
    """Read-only CRUD operations."""

    def __init__(self, model: Type[ModelType]):
        self.model = model

    async def get(self, db: AsyncSession, id: UUID) -> Optional[ModelType]:
        """Get a single record by ID."""
        result = await db.execute(
            select(self.model).where(self.model.id == id, self.model.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_multi(
        self, db: AsyncSession, *, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """Get multiple records."""
        result = await db.execute(
            select(self.model)
            .where(self.model.deleted_at.is_(None))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


class CRUDWrite(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Write operations for CRUD implementations."""

    def __init__(self, model: Type[ModelType]):
        self.model = model

    async def create(self, db: AsyncSession, *, obj_in: CreateSchemaType, owner_id: Optional[UUID] = None) -> ModelType:
        """Create a new record."""
        obj_data = obj_in.model_dump()
        if owner_id and hasattr(self.model, "owner_id"):
            obj_data["owner_id"] = owner_id
        if hasattr(self.model, "created_by") and owner_id:
            obj_data["created_by"] = owner_id
        if hasattr(self.model, "uploaded_by") and owner_id:
            obj_data["uploaded_by"] = owner_id

        db_obj = self.model(**obj_data)
        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj

    async def update(
        self, db: AsyncSession, *, db_obj: ModelType, obj_in: UpdateSchemaType
    ) -> ModelType:
        """Update a record."""
        obj_data = obj_in.model_dump(exclude_unset=True)
        for field, value in obj_data.items():
            setattr(db_obj, field, value)

        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj


class CRUDDelete(Generic[ModelType]):
    """Deletion operations."""

    def __init__(self, model: Type[ModelType]):
        self.model = model

    async def soft_delete(self, db: AsyncSession, *, id: UUID) -> bool:
        """Soft delete a record."""
        from sqlalchemy.sql import func

        result = await db.execute(
            update(self.model)
            .where(self.model.id == id, self.model.deleted_at.is_(None))
            .values(deleted_at=func.now())
        )
        await db.flush()
        return result.rowcount > 0

    async def hard_delete(self, db: AsyncSession, *, id: UUID) -> bool:
        """Hard delete a record."""
        result = await db.execute(delete(self.model).where(self.model.id == id))
        await db.flush()
        return result.rowcount > 0


class CRUDBase(
    CRUDRead[ModelType],
    CRUDWrite[ModelType, CreateSchemaType, UpdateSchemaType],
    CRUDDelete[ModelType],
    Generic[ModelType, CreateSchemaType, UpdateSchemaType],
):
    """Full CRUD interface combining read/write/delete operations."""

    def __init__(self, model: Type[ModelType]):
        CRUDRead.__init__(self, model)
        CRUDWrite.__init__(self, model)
        CRUDDelete.__init__(self, model)


class ReadOnlyCRUD(CRUDRead[ModelType]):
    """Convenience class for read-only repositories."""


class WriteCRUD(CRUDWrite[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Convenience class for repositories that only write/update data."""


class DeleteCRUD(CRUDDelete[ModelType]):
    """Convenience class for repositories that only delete data."""
