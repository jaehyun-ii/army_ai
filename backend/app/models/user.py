"""
User model.
"""
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
import enum

from app.database import Base


class UserRole(str, enum.Enum):
    """User role enum."""

    USER = "user"
    ADMIN = "admin"


class User(Base):
    """User model (aligned with database schema)."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), nullable=False, index=True)  # DB: NOT NULL, unique via index
    email = Column(String(255), nullable=True)  # DB: nullable (optional)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole, name="user_role_enum", values_callable=lambda x: [e.value for e in x]), nullable=False, default=UserRole.USER.value)
    is_active = Column(Boolean, nullable=False, default=True)

    # Security fields
    failed_login_attempts = Column(Integer, nullable=False, default=0)  # Track failed login attempts
    locked_until = Column(DateTime(timezone=True), nullable=True)  # Account lock timestamp
    current_session_id = Column(String(255), nullable=True)  # Current active session (for single login)
    last_login_at = Column(DateTime(timezone=True), nullable=True)  # Last successful login

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"
