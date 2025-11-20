"""
CRUD operations for experiment tags.

PHASE 2 NEW: Tag management CRUD.
"""
from typing import Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from app.models.experiment import ExperimentTag
from app.schemas.experiment import ExperimentTagCreate, ExperimentTagUpdate
from app.crud.base import CRUDBase


class CRUDExperimentTag(CRUDBase[ExperimentTag, ExperimentTagCreate, ExperimentTagUpdate]):
    """CRUD operations for ExperimentTag model."""

    def get_by_name(self, db: Session, *, name: str) -> Optional[ExperimentTag]:
        """Get tag by name (case-insensitive)."""
        return (
            db.query(self.model)
            .filter(
                and_(
                    func.lower(self.model.name) == name.lower(),
                    self.model.deleted_at.is_(None)
                )
            )
            .first()
        )

    def get_or_create(
        self, db: Session, *, name: str, color: Optional[str] = None
    ) -> ExperimentTag:
        """Get existing tag or create new one."""
        tag = self.get_by_name(db, name=name)
        if tag:
            return tag

        obj_in = ExperimentTagCreate(name=name, color=color)
        return self.create(db, obj_in=obj_in)

    def search_by_name(
        self, db: Session, *, query: str, skip: int = 0, limit: int = 100
    ) -> List[ExperimentTag]:
        """Search tags by name (for autocomplete)."""
        return (
            db.query(self.model)
            .filter(
                and_(
                    func.lower(self.model.name).contains(query.lower()),
                    self.model.deleted_at.is_(None)
                )
            )
            .order_by(self.model.name)
            .offset(skip)
            .limit(limit)
            .all()
        )


experiment_tag = CRUDExperimentTag(ExperimentTag)
