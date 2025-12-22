"""
3D Dataset / Patch / Attack models (aligned with SQL schema).
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Enum as SQLEnum, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base
from app.models.dataset_2d import AttackType  # reuse same enum values ('patch', 'noise')


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
    attack_datasets = relationship(
        "AttackDataset3D",
        back_populates="output_dataset",
        foreign_keys="[AttackDataset3D.output_dataset_id]",
    )

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
    annotations = relationship("Annotation", back_populates="image_3d")

    __table_args__ = (
        CheckConstraint(
            "(width IS NULL AND height IS NULL AND depth IS NULL) OR (width > 0 AND height > 0)",
            name="chk_images_3d_dimensions",
        ),
        CheckConstraint("char_length(file_name) > 0", name="chk_images_3d_file_name"),
    )


class Patch3D(Base):
    """3D adversarial patch model."""

    __tablename__ = "patches_3d"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    target_model_id = Column(UUID(as_uuid=True), ForeignKey("od_models.id", ondelete="RESTRICT"))
    source_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_3d.id", ondelete="SET NULL"))
    target_class = Column(String(200))
    method = Column(String(200))
    hyperparameters = Column(JSONB)
    patch_metadata = Column(JSONB)
    storage_key = Column(Text)
    file_name = Column(String(1024))
    size_bytes = Column(Integer)
    sha256 = Column(String(64))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_patches_3d_name"),
        CheckConstraint("hyperparameters IS NULL OR jsonb_typeof(hyperparameters)='object'", name="chk_patches_3d_hparams"),
    )


class AttackDataset3D(Base):
    """3D adversarial (attacked) dataset model."""

    __tablename__ = "attack_datasets_3d"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    attack_type = Column(
        SQLEnum(AttackType, name="attack_type_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    target_model_id = Column(UUID(as_uuid=True), ForeignKey("od_models.id", ondelete="RESTRICT"))
    output_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets_3d.id", ondelete="SET NULL"))
    target_class = Column(String(200))
    patch_id = Column(UUID(as_uuid=True), ForeignKey("patches_3d.id", ondelete="RESTRICT"))
    parameters = Column(JSONB)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="SET NULL"))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    output_dataset = relationship("Dataset3D", foreign_keys=[output_dataset_id], back_populates="attack_datasets")

    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="chk_attack_datasets_3d_name"),
        CheckConstraint("parameters IS NULL OR jsonb_typeof(parameters) = 'object'", name="chk_attack_datasets_3d_parameters"),
    )
