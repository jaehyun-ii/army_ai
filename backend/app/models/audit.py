"""
Audit log model.
"""
from sqlalchemy import Column, String, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from app.database import Base


class AuditLog(Base):
    """Audit log model."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(200), nullable=False)
    target_type = Column(String(100))
    target_id = Column(UUID(as_uuid=True))
    details = Column(JSONB)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint("char_length(action) > 0", name="chk_audit_action"),
        CheckConstraint("target_type IS NULL OR char_length(target_type) > 0", name="chk_audit_target_type"),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, actor_id={self.actor_id})>"
