"""
3D Dataset models (placeholder for annotations compatibility).
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class Dataset3D(Base):
    """3D dataset model."""

    __tablename__ = "datasets_3d"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    storage_path = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    images = relationship("Image3D", back_populates="dataset", lazy="selectin")

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_datasets_3d_name"),
    )


class Image3D(Base):
    """3D Image model."""

    __tablename__ = "images_3d"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_3d.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(1024), nullable=False)
    storage_key = Column(Text, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    depth = Column(Integer)
    mime_type = Column(String(100))
    metadata_ = Column("metadata", JSONB)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    dataset = relationship("Dataset3D", back_populates="images")
    # annotations relationship removed - image_3d_id column removed from annotations table

    __table_args__ = (
        CheckConstraint(
            "(width IS NULL AND height IS NULL AND depth IS NULL) OR (width > 0 AND height > 0)",
            name="chk_images_3d_dimensions",
        ),
        CheckConstraint("char_length(file_name) > 0", name="chk_images_3d_file_name"),
    )
