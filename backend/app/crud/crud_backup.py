"""
CRUD operations for backups.
"""
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.crud.base import CRUDBase
from app.models.backup import Backup
from app.schemas.backup import BackupCreate, BackupUpdate


class CRUDBackup(CRUDBase[Backup, BackupCreate, BackupUpdate]):
    """CRUD operations for Backup model."""

    async def get_all_active(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100
    ) -> List[Backup]:
        """Get all active (non-deleted) backups."""
        query = (
            select(self.model)
            .where(self.model.deleted_at.is_(None))
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_total_size(self, db: AsyncSession) -> int:
        """Get total size of all active backups."""
        query = select(func.sum(self.model.total_size_bytes)).where(
            self.model.deleted_at.is_(None)
        )
        result = await db.execute(query)
        total = result.scalar()
        return total or 0

    async def soft_delete(self, db: AsyncSession, *, id: str) -> Optional[Backup]:
        """Soft delete a backup."""
        backup = await self.get(db, id=id)
        if backup:
            from datetime import datetime, timezone
            backup.deleted_at = datetime.now(timezone.utc)
            db.add(backup)
            await db.commit()
            await db.refresh(backup)
        return backup


backup = CRUDBackup(Backup)
