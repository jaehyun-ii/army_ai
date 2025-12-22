"""
Class definitions models.

PHASE 3 NEW: Normalized class/label management.
Replaces od_models.labelmap JSONB and datasets_2d.metadata.classes
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class ClassDefinition(Base):
    """
    Object detection class/label definition.

    Centralizes class definitions used across models and datasets.
    """
    __tablename__ = "class_definitions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    category = Column(String(100))
    description = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    model_assignments = relationship("ModelClass", back_populates="class_def", cascade="all, delete-orphan")
    dataset_assignments = relationship("DatasetClass", back_populates="class_def", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_class_name"),
    )

    def __repr__(self):
        return f"<ClassDefinition(id={self.id}, name={self.name})>"


class ModelClass(Base):
    """
    Model-Class mapping table.

    Replaces od_models.labelmap JSONB field.
    Maps model internal class indices to class definitions.
    """
    __tablename__ = "model_classes"

    model_id = Column(UUID(as_uuid=True), ForeignKey("od_models.id", ondelete="CASCADE"), primary_key=True)
    class_index = Column(Integer, nullable=False, primary_key=True)
    class_id = Column(UUID(as_uuid=True), ForeignKey("class_definitions.id", ondelete="CASCADE"), nullable=False)

    # Relationships
    class_def = relationship("ClassDefinition", back_populates="model_assignments")

    def __repr__(self):
        return f"<ModelClass(model_id={self.model_id}, class_index={self.class_index}, class_id={self.class_id})>"


class DatasetClass(Base):
    """
    Dataset-Class mapping table.

    Replaces datasets_2d.metadata.classes field.
    """
    __tablename__ = "dataset_classes"

    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_2d.id", ondelete="CASCADE"), primary_key=True)
    class_id = Column(UUID(as_uuid=True), ForeignKey("class_definitions.id", ondelete="CASCADE"), primary_key=True)
    sort_order = Column(Integer, nullable=False, default=0)

    # Relationships
    class_def = relationship("ClassDefinition", back_populates="dataset_assignments")

    def __repr__(self):
        return f"<DatasetClass(dataset_id={self.dataset_id}, class_id={self.class_id})>"
