"""
CRUD operations for experiments.

PHASE 2 UPDATE: Added tag management methods.
"""
from typing import Optional, List
from uuid import UUID
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_
from datetime import datetime

from app.models.experiment import Experiment, ExperimentStatus, ExperimentTag, ExperimentTagAssignment
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate
from app.crud.base import CRUDBase


class CRUDExperiment(CRUDBase[Experiment, ExperimentCreate, ExperimentUpdate]):
    """
    CRUD operations for Experiment model.

    PHASE 2 UPDATE: Includes tag management.
    """

    def get(self, db: Session, id: UUID) -> Optional[Experiment]:
        """Get experiment by ID with tags preloaded."""
        return (
            db.query(self.model)
            .options(joinedload(self.model.tag_assignments).joinedload(ExperimentTagAssignment.tag))
            .filter(and_(self.model.id == id, self.model.deleted_at.is_(None)))
            .first()
        )

    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[Experiment]:
        """Get multiple experiments with tags preloaded."""
        return (
            db.query(self.model)
            .options(joinedload(self.model.tag_assignments).joinedload(ExperimentTagAssignment.tag))
            .filter(self.model.deleted_at.is_(None))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_status(
        self, db: Session, *, status: ExperimentStatus, skip: int = 0, limit: int = 100
    ) -> List[Experiment]:
        """Get experiments by status with tags preloaded."""
        return (
            db.query(self.model)
            .options(joinedload(self.model.tag_assignments).joinedload(ExperimentTagAssignment.tag))
            .filter(
                and_(
                    self.model.status == status,
                    self.model.deleted_at.is_(None)
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

    def create(self, db: Session, *, obj_in: ExperimentCreate) -> Experiment:
        """
        Create experiment with tags.

        PHASE 2 UPDATE: Handles tag_names field.
        """
        obj_data = obj_in.model_dump(exclude={"tag_names"})
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        db.flush()  # Get ID before adding tags

        # Handle tags
        if obj_in.tag_names:
            self._assign_tags(db, experiment=db_obj, tag_names=obj_in.tag_names)

        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self, db: Session, *, db_obj: Experiment, obj_in: ExperimentUpdate
    ) -> Experiment:
        """
        Update experiment with tags.

        PHASE 2 UPDATE: Handles tag_names field.
        """
        obj_data = obj_in.model_dump(exclude_unset=True, exclude={"tag_names"})

        for field, value in obj_data.items():
            setattr(db_obj, field, value)

        # Handle tags (replace all if provided)
        if obj_in.tag_names is not None:
            # Remove existing assignments
            db.query(ExperimentTagAssignment).filter(
                ExperimentTagAssignment.experiment_id == db_obj.id
            ).delete()
            db.flush()

            # Add new assignments
            if obj_in.tag_names:
                self._assign_tags(db, experiment=db_obj, tag_names=obj_in.tag_names)

        db.commit()
        db.refresh(db_obj)
        return db_obj

    def _assign_tags(self, db: Session, *, experiment: Experiment, tag_names: List[str]) -> None:
        """
        Assign tags to experiment by tag names.

        Creates tags if they don't exist (auto-create).
        """
        for tag_name in tag_names:
            tag_name = tag_name.strip()
            if not tag_name:
                continue

            # Get or create tag
            tag = db.query(ExperimentTag).filter(
                and_(
                    ExperimentTag.name == tag_name,
                    ExperimentTag.deleted_at.is_(None)
                )
            ).first()

            if not tag:
                tag = ExperimentTag(name=tag_name)
                db.add(tag)
                db.flush()

            # Create assignment
            assignment = ExperimentTagAssignment(
                experiment_id=experiment.id,
                tag_id=tag.id
            )
            db.add(assignment)

    def get_by_tag(
        self, db: Session, *, tag_name: str, skip: int = 0, limit: int = 100
    ) -> List[Experiment]:
        """Get experiments by tag name."""
        return (
            db.query(self.model)
            .join(ExperimentTagAssignment)
            .join(ExperimentTag)
            .options(joinedload(self.model.tag_assignments).joinedload(ExperimentTagAssignment.tag))
            .filter(
                and_(
                    ExperimentTag.name == tag_name,
                    ExperimentTag.deleted_at.is_(None),
                    self.model.deleted_at.is_(None)
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

    def start_experiment(self, db: Session, *, experiment_id: UUID) -> Optional[Experiment]:
        """Start an experiment (update status and started_at)."""
        experiment = self.get(db, id=experiment_id)
        if experiment and experiment.status == ExperimentStatus.DRAFT:
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.utcnow()
            db.commit()
            db.refresh(experiment)
        return experiment

    def complete_experiment(
        self, db: Session, *, experiment_id: UUID, results_summary: dict
    ) -> Optional[Experiment]:
        """Complete an experiment (update status, ended_at, and results_summary)."""
        experiment = self.get(db, id=experiment_id)
        if experiment and experiment.status == ExperimentStatus.RUNNING:
            experiment.status = ExperimentStatus.COMPLETED
            experiment.ended_at = datetime.utcnow()
            experiment.results_summary = results_summary
            db.commit()
            db.refresh(experiment)
        return experiment

    def fail_experiment(self, db: Session, *, experiment_id: UUID) -> Optional[Experiment]:
        """Fail an experiment (update status and ended_at)."""
        experiment = self.get(db, id=experiment_id)
        if experiment and experiment.status == ExperimentStatus.RUNNING:
            experiment.status = ExperimentStatus.FAILED
            experiment.ended_at = datetime.utcnow()
            db.commit()
            db.refresh(experiment)
        return experiment


experiment = CRUDExperiment(Experiment)
