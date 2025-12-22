"""
2D Dataset models.
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Enum as SQLEnum, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.database import Base


class AttackType(str, enum.Enum):
    """2D attack type enum."""

    PATCH = "patch"
    NOISE = "noise"


class Dataset2D(Base):
    """2D dataset model."""

    __tablename__ = "datasets_2d"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    storage_path = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships (aligned with existing models only)
    images = relationship("Image2D", back_populates="dataset", lazy="selectin")
    attack_datasets = relationship(
        "AttackDataset2D",
        back_populates="base_dataset",
        foreign_keys="[AttackDataset2D.base_dataset_id]"
    )
    # Removed: inference_metadata, class_statistics (tables don't exist in DB schema)

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_datasets_2d_name"),
    )


class Image2D(Base):
    """2D Image model."""

    __tablename__ = "images_2d"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_2d.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(1024), nullable=False)
    storage_key = Column(Text, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    mime_type = Column(String(100))
    metadata_ = Column("metadata", JSONB)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships (aligned with existing models only)
    dataset = relationship("Dataset2D", back_populates="images")
    annotations = relationship("Annotation", back_populates="image_2d")

    __table_args__ = (
        CheckConstraint(
            "(width IS NULL AND height IS NULL) OR (width > 0 AND height > 0)",
            name="chk_images_2d_dimensions",
        ),
        CheckConstraint("char_length(file_name) > 0", name="chk_images_2d_file_name"),
    )


class Patch2D(Base):
    """2D adversarial patch model."""

    __tablename__ = "patches_2d"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    target_model_id = Column(
        UUID(as_uuid=True), ForeignKey("od_models.id", ondelete="RESTRICT"), nullable=True  # DB: NULL allowed
    )
    source_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_2d.id", ondelete="SET NULL"))
    target_class = Column(String(200))
    method = Column(String(200))
    hyperparameters = Column(JSONB)
    patch_metadata = Column(JSONB)
    # NEW: Storage information
    storage_key = Column(Text)
    file_name = Column(String(1024))
    size_bytes = Column(Integer)
    sha256 = Column(String(64))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_patches_name"),
        CheckConstraint(
            "jsonb_typeof(hyperparameters) = 'object' OR hyperparameters IS NULL",
            name="chk_patches_hyperparameters",
        ),
    )


class AttackDataset2D(Base):
    """2D attack dataset model."""

    __tablename__ = "attack_datasets_2d"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    attack_type = Column(SQLEnum(AttackType, name="attack_type_enum", values_callable=lambda x: [e.value for e in x]), nullable=False)
    target_model_id = Column(
        UUID(as_uuid=True), ForeignKey("od_models.id", ondelete="RESTRICT"), nullable=True  # DB: NULL allowed
    )
    base_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_2d.id", ondelete="RESTRICT"), nullable=False)  # DB: NOT NULL
    output_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_2d.id", ondelete="SET NULL"), nullable=True)  # NEW: Output dataset reference
    target_class = Column(String(200))
    patch_id = Column(UUID(as_uuid=True), ForeignKey("patches_2d.id", ondelete="RESTRICT"))
    parameters = Column(JSONB)
    # NEW: Experiment linkage
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="SET NULL"))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    base_dataset = relationship("Dataset2D", back_populates="attack_datasets", foreign_keys=[base_dataset_id])
    output_dataset = relationship("Dataset2D", foreign_keys=[output_dataset_id])
    experiment = relationship("Experiment", back_populates="attack_datasets_2d")

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_attack_name"),
        CheckConstraint(
            "parameters IS NULL OR jsonb_typeof(parameters) = 'object'",
            name="chk_attack_parameters_json",
        ),
        CheckConstraint(
            "(attack_type = 'patch' AND patch_id IS NOT NULL) OR (attack_type <> 'patch' AND patch_id IS NULL)",
            name="chk_attack_patch_id_required",
        ),
    )
