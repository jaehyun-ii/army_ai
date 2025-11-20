"""
Evaluation models for pre/post-attack dataset evaluation.
"""
from sqlalchemy import Column, String, Text, DateTime, Enum as SQLEnum, Integer, CheckConstraint, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
import enum

from app.database import Base


class EvalStatus(str, enum.Enum):
    """Evaluation run status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class EvalPhase(str, enum.Enum):
    """Evaluation phase (pre-attack baseline or post-attack adversarial)."""
    PRE_ATTACK = "pre_attack"
    POST_ATTACK = "post_attack"


class DatasetDimension(str, enum.Enum):
    """Dataset dimension (2D or 3D)."""
    TWO_D = "2d"
    THREE_D = "3d"


class EvalDatasetType(str, enum.Enum):
    """Evaluation dataset type (base or attack)."""
    BASE = "base"
    ATTACK = "attack"


class EvalRun(Base):
    """
    Evaluation run header.

    - pre_attack: Baseline evaluation on clean dataset (requires base_dataset_id)
    - post_attack: Adversarial evaluation on attacked dataset (requires attack_dataset_id)
    """
    __tablename__ = "eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    phase = Column(SQLEnum(EvalPhase, name="eval_phase_enum", values_callable=lambda x: [e.value for e in x]), nullable=False)

    # Foreign keys
    model_id = Column(UUID(as_uuid=True), ForeignKey("od_models.id", ondelete="RESTRICT"), nullable=False)
    # 2D datasets only (3D datasets removed)
    base_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_2d.id", ondelete="RESTRICT"), nullable=True)
    attack_dataset_id = Column(UUID(as_uuid=True), ForeignKey("attack_datasets_2d.id", ondelete="RESTRICT"), nullable=True)
    # Experiment linkage
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Evaluation configuration and results
    params = Column(JSONB)  # score threshold, NMS, IoU, etc.
    metrics_summary = Column(JSONB)  # mAP, mAR, F1, etc.

    # Execution tracking
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(SQLEnum(EvalStatus, name="eval_status_enum", values_callable=lambda x: [e.value for e in x]), nullable=False, default=EvalStatus.QUEUED.value)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    items = relationship("EvalItem", back_populates="run", cascade="all, delete-orphan")
    experiment = relationship("Experiment", back_populates="eval_runs")

    # Check constraints (defined in SQL schema, documented here)
    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_eval_name"),
        CheckConstraint(
            "params IS NULL OR jsonb_typeof(params)='object'",
            name="chk_eval_params"
        ),
        CheckConstraint(
            "metrics_summary IS NULL OR jsonb_typeof(metrics_summary)='object'",
            name="chk_eval_metrics_summary"
        ),
        CheckConstraint(
            "(phase = 'pre_attack' AND base_dataset_id IS NOT NULL AND attack_dataset_id IS NULL) OR "
            "(phase = 'post_attack' AND base_dataset_id IS NOT NULL AND attack_dataset_id IS NOT NULL)",
            name="chk_eval_phase_requirements"
        ),
        CheckConstraint(
            "ended_at IS NULL OR started_at IS NULL OR ended_at >= started_at",
            name="chk_eval_time_range"
        ),
    )

    def __repr__(self):
        return f"<EvalRun(id={self.id}, name={self.name}, phase={self.phase}, status={self.status})>"


class EvalItem(Base):
    """
    Per-image evaluation results.

    Stores predictions, ground truth, and per-item metrics for each image
    evaluated in an evaluation run.

    PHASE 2 UPDATE: Removed file_name and storage_key columns (duplicates of images_2d).
    Use image_2d relationship to access file metadata.
    """
    __tablename__ = "eval_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False)

    # Image reference (2D only)
    image_2d_id = Column(UUID(as_uuid=True), ForeignKey("images_2d.id", ondelete="SET NULL"), nullable=True)
    dataset_type = Column(SQLEnum(EvalDatasetType, name="eval_dataset_type_enum", values_callable=lambda x: [e.value for e in x]), nullable=True)

    # Evaluation data
    ground_truth = Column(JSONB)  # GT bounding boxes/classes
    prediction = Column(JSONB)    # Model output (boxes/scores/classes)
    metrics = Column(JSONB)       # Per-item metrics (IoU, AP@IoU, TP/FP/FN, etc.)
    notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    run = relationship("EvalRun", back_populates="items")
    image_2d = relationship("Image2D", foreign_keys=[image_2d_id])

    # Check constraints
    __table_args__ = (
        CheckConstraint(
            "ground_truth IS NULL OR jsonb_typeof(ground_truth)='object' OR jsonb_typeof(ground_truth)='array'",
            name="chk_eval_item_gt"
        ),
        CheckConstraint(
            "prediction IS NULL OR jsonb_typeof(prediction)='object' OR jsonb_typeof(prediction)='array'",
            name="chk_eval_item_pred"
        ),
        CheckConstraint(
            "metrics IS NULL OR jsonb_typeof(metrics)='object'",
            name="chk_eval_item_metrics"
        ),
    )

    def __repr__(self):
        return f"<EvalItem(id={self.id}, run_id={self.run_id}, image_2d_id={self.image_2d_id})>"


class EvalList(Base):
    """
    Evaluation list/playlist for organizing and comparing multiple evaluation runs.
    """
    __tablename__ = "eval_lists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    items = relationship("EvalListItem", back_populates="list", cascade="all, delete-orphan")

    # Check constraints
    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_eval_lists_name"),
    )

    def __repr__(self):
        return f"<EvalList(id={self.id}, name={self.name})>"


class EvalListItem(Base):
    """
    Junction table linking evaluation lists to evaluation runs.
    """
    __tablename__ = "eval_list_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    list_id = Column(UUID(as_uuid=True), ForeignKey("eval_lists.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False)
    sort_order = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    list = relationship("EvalList", back_populates="items")

    def __repr__(self):
        return f"<EvalListItem(id={self.id}, list_id={self.list_id}, run_id={self.run_id})>"
