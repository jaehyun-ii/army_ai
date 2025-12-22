"""
Experiment models.
"""
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum as SQLEnum, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.database import Base


class ExperimentStatus(str, enum.Enum):
    """Experiment status enum."""

    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class Experiment(Base):
    """
    Experiment model for research organization.

    PHASE 2 UPDATE: Removed tags JSONB column. Use tag_assignments relationship
    to access experiment tags.
    """

    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    objective = Column(Text)
    hypothesis = Column(Text)
    status = Column(SQLEnum(ExperimentStatus, name="experiment_status_enum", values_callable=lambda x: [e.value for e in x]), nullable=False, default=ExperimentStatus.DRAFT.value)
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    config = Column(JSONB)
    results_summary = Column(JSONB)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    attack_datasets_2d = relationship("AttackDataset2D", back_populates="experiment")
    eval_runs = relationship("EvalRun", back_populates="experiment")
    tag_assignments = relationship("ExperimentTagAssignment", back_populates="experiment", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_experiment_name"),
        CheckConstraint(
            "(status = 'draft' AND started_at IS NULL AND ended_at IS NULL) OR "
            "(status = 'running' AND started_at IS NOT NULL AND ended_at IS NULL) OR "
            "(status IN ('completed', 'failed', 'archived') AND started_at IS NOT NULL AND ended_at IS NOT NULL AND ended_at >= started_at)",
            name="chk_experiment_status_time",
        ),
        CheckConstraint("config IS NULL OR jsonb_typeof(config) = 'object'", name="chk_experiment_config"),
        CheckConstraint("results_summary IS NULL OR jsonb_typeof(results_summary) = 'object'", name="chk_experiment_results"),
    )

    @property
    def tags(self):
        """Get list of tag objects for this experiment."""
        return [assignment.tag for assignment in self.tag_assignments]


class ExperimentTag(Base):
    """
    Experiment tag definition.

    PHASE 2 NEW: Replaces experiments.tags JSONB column with normalized tag table.
    """

    __tablename__ = "experiment_tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    color = Column(String(20))
    description = Column(Text)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    assignments = relationship("ExperimentTagAssignment", back_populates="tag", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_tag_name"),
    )

    def __repr__(self):
        return f"<ExperimentTag(id={self.id}, name={self.name})>"


class ExperimentTagAssignment(Base):
    """
    Many-to-many relationship between experiments and tags.

    PHASE 2 NEW: Junction table for experiment-tag assignments.
    """

    __tablename__ = "experiment_tag_assignments"

    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(UUID(as_uuid=True), ForeignKey("experiment_tags.id", ondelete="CASCADE"), primary_key=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    experiment = relationship("Experiment", back_populates="tag_assignments")
    tag = relationship("ExperimentTag", back_populates="assignments")

    def __repr__(self):
        return f"<ExperimentTagAssignment(experiment_id={self.experiment_id}, tag_id={self.tag_id})>"
