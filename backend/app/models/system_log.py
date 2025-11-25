"""
System log model.
"""
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Text, CheckConstraint, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
import enum

from app.database import Base


class LogLevel(str, enum.Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SystemLog(Base):
    """System log model for tracking system events, user actions, and errors."""

    __tablename__ = "system_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Log metadata
    log_level = Column(SQLEnum(LogLevel, name="log_level_enum"), nullable=False, default=LogLevel.INFO)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    # Source information
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    username = Column(String(100))
    ip_address = Column(INET, index=True)
    user_agent = Column(Text)

    # Log content
    action = Column(String(200), nullable=False, index=True)
    module = Column(String(100), index=True)
    message = Column(Text, nullable=False)
    details = Column(JSONB)

    # Request context
    request_id = Column(String(100), index=True)
    endpoint = Column(String(500))
    method = Column(String(10))
    status_code = Column(Integer)
    response_time_ms = Column(Integer)

    # Error information
    error_type = Column(String(200))
    error_message = Column(Text)
    stack_trace = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="system_logs")

    __table_args__ = (
        CheckConstraint("char_length(action) > 0", name="chk_system_logs_action"),
        CheckConstraint("char_length(message) > 0", name="chk_system_logs_message"),
        CheckConstraint("details IS NULL OR jsonb_typeof(details) = 'object'", name="chk_system_logs_details"),
        CheckConstraint("response_time_ms IS NULL OR response_time_ms >= 0", name="chk_system_logs_response_time"),
    )

    def __repr__(self):
        return f"<SystemLog(id={self.id}, level={self.log_level}, action={self.action}, user_id={self.user_id})>"

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            "log_level": self.log_level.value if self.log_level else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "username": self.username,
            "ip_address": str(self.ip_address) if self.ip_address else None,
            "user_agent": self.user_agent,
            "action": self.action,
            "module": self.module,
            "message": self.message,
            "details": self.details,
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
