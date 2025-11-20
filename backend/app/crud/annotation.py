"""
CRUD operations for Annotation model.
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.annotation import Annotation
from app.schemas.annotation import AnnotationCreate, AnnotationUpdate


class CRUDAnnotation(CRUDBase[Annotation, AnnotationCreate, AnnotationUpdate]):
    """CRUD operations for Annotation model."""

    async def get_by_image(
        self,
        db: AsyncSession,
        *,
        image_2d_id: UUID,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Annotation]:
        """Get all annotations for a specific 2D image."""
        result = await db.execute(
            select(Annotation)
            .where(
                and_(
                    Annotation.image_2d_id == image_2d_id,
                    Annotation.deleted_at.is_(None)
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_frame(
        self,
        db: AsyncSession,
        *,
        rt_frame_id: UUID,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Annotation]:
        """Get all annotations for a specific real-time frame."""
        result = await db.execute(
            select(Annotation)
            .where(
                and_(
                    Annotation.rt_frame_id == rt_frame_id,
                    Annotation.deleted_at.is_(None)
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_class(
        self,
        db: AsyncSession,
        *,
        class_name: str,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Annotation]:
        """Get all annotations for a specific class."""
        result = await db.execute(
            select(Annotation)
            .where(
                and_(
                    Annotation.class_name == class_name,
                    Annotation.deleted_at.is_(None)
                )
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


# Instantiate CRUD object
annotation = CRUDAnnotation(Annotation)
