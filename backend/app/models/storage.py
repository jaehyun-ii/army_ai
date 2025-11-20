"""
Storage objects model.

PHASE 3 NEW: Centralized storage metadata management.
"""
from sqlalchemy import Column, String, BigInteger, Integer, Text, DateTime, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from app.database import Base


class StorageObject(Base):
    """
    Centralized storage object metadata.

    Replaces duplicate storage fields across:
    - model_artifacts (storage_key, file_name, size_bytes, sha256, content_type)
    - images_2d (storage_key, file_name, width, height, mime_type)
    - patches_2d (storage_key, file_name, size_bytes, sha256)
    - rt_frames (storage_key, width, height, mime_type)
    """
    __tablename__ = "storage_objects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    storage_key = Column(Text, nullable=False)
    file_name = Column(String(1024), nullable=False)
    size_bytes = Column(BigInteger)
    sha256 = Column(String(64))
    content_type = Column(String(200))

    # Image-specific (nullable for non-images)
    width = Column(Integer)
    height = Column(Integer)

    # Generic metadata
    metadata = Column(JSONB)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "(width IS NULL AND height IS NULL) OR (width > 0 AND height > 0)",
            name="chk_storage_dimensions"
        ),
        CheckConstraint(
            "size_bytes IS NULL OR size_bytes >= 0",
            name="chk_storage_size"
        ),
    )

    def __repr__(self):
        return f"<StorageObject(id={self.id}, storage_key={self.storage_key}, file_name={self.file_name})>"
