"""
Backup model for tracking database and storage backups.
"""
from sqlalchemy import Column, String, Integer, DateTime, JSON, Boolean, BigInteger
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.sql import func
from app.database import Base
import uuid


class Backup(Base):
    """Model for backup records."""
    __tablename__ = "backups"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    type = Column(
        ENUM('manual', 'scheduled', name='backup_type_enum', create_type=False),
        nullable=False
    )
    status = Column(
        ENUM('in-progress', 'success', 'failed', name='backup_status_enum', create_type=False),
        nullable=False,
        default='in-progress'
    )

    # File information
    db_file_path = Column(String, nullable=True)  # Path to database backup file
    storage_file_path = Column(String, nullable=True)  # Path to storage backup file
    # Use BigInteger to support backup sizes > 2GB (int32 limit)
    total_size_bytes = Column(BigInteger, nullable=True)  # Total size in bytes

    # Metadata
    metadata_ = Column("metadata", JSON, nullable=True)
    error_message = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    restored_at = Column(DateTime(timezone=True), nullable=True)  # When this backup was last restored
    deleted_at = Column(DateTime(timezone=True), nullable=True)
