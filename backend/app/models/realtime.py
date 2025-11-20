"""
Real-time performance measurement models (simplified).
"""
from sqlalchemy import Column, String, Text, Integer, Numeric, DateTime, ForeignKey, Boolean, Enum as SQLEnum, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.database import Base


class RTRunStatus(str, enum.Enum):
    """Real-time capture run status enum."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class RTCaptureRun(Base):
    """Real-time capture run model (matches database schema)."""

    __tablename__ = "rt_capture_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(
        UUID(as_uuid=True), ForeignKey("od_models.id", ondelete="RESTRICT"), nullable=True
    )
    window_seconds = Column(Integer, nullable=False, default=5)
    frames_expected = Column(Integer, nullable=False, default=10)
    fps_target = Column(Numeric(6, 3))
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    ended_at = Column(DateTime(timezone=True))
    status = Column(SQLEnum(RTRunStatus, name="rt_run_status_enum", values_callable=lambda x: [e.value for e in x]), nullable=False, default=RTRunStatus.RUNNING.value)
    notes = Column(Text)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("window_seconds > 0", name="chk_rt_window"),
        CheckConstraint("frames_expected > 0", name="chk_rt_frames_expected"),
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="chk_rt_run_time_range"),
        CheckConstraint("fps_target IS NULL OR fps_target > 0", name="chk_rt_fps_positive"),
    )


class RTFrame(Base):
    """Real-time frame model (matches database schema)."""

    __tablename__ = "rt_frames"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("rt_capture_runs.id", ondelete="CASCADE"), nullable=False)
    seq_no = Column(Integer, nullable=False)
    captured_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    storage_key = Column(Text)
    width = Column(Integer)
    height = Column(Integer)
    mime_type = Column(String(100))
    metadata_ = Column("metadata", JSONB)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    annotations = relationship("Annotation", back_populates="rt_frame")

    __table_args__ = (
        CheckConstraint("seq_no > 0", name="chk_rt_frames_seq"),
        CheckConstraint(
            "(width IS NULL AND height IS NULL) OR (width > 0 AND height > 0)", name="chk_rt_frames_wh"
        ),
    )
