"""
Annotation models for storing detection/inference results.
Aligned with database schema (complete_schema.sql lines 563-625).
"""
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, Numeric, Boolean, Enum as SQLEnum, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.database import Base


class AnnotationType(str, enum.Enum):
    """Annotation type enum."""
    BBOX = "bbox"
    POLYGON = "polygon"
    KEYPOINT = "keypoint"
    SEGMENTATION = "segmentation"


class Annotation(Base):
    """Annotation model for storing detection/inference results.

    Supports both 2D and 3D images, as well as real-time frames.
    """

    __tablename__ = "annotations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Image references (one of these must be set)
    image_2d_id = Column(UUID(as_uuid=True), ForeignKey("images_2d.id", ondelete="CASCADE"))
    rt_frame_id = Column(UUID(as_uuid=True), ForeignKey("rt_frames.id", ondelete="CASCADE"))

    annotation_type = Column(
        SQLEnum(AnnotationType, name="annotation_type_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=AnnotationType.BBOX
    )

    # Class information
    class_name = Column(String(200), nullable=False)
    class_index = Column(Integer)

    # Bounding box (for bbox type) - using Numeric to match SQL schema
    bbox_x = Column(Numeric(10, 2))
    bbox_y = Column(Numeric(10, 2))
    bbox_width = Column(Numeric(10, 2))
    bbox_height = Column(Numeric(10, 2))

    # Polygon/segmentation data (for polygon/segmentation types)
    polygon_data = Column(JSONB)

    # Keypoints (for keypoint type)
    keypoints = Column(JSONB)

    # Additional metadata
    confidence = Column(Numeric(5, 4), default=1.0)
    is_crowd = Column(Boolean, default=False)
    area = Column(Numeric(12, 2))
    metadata_ = Column("metadata", JSONB)  # Use metadata_ to avoid SQLAlchemy reserved word

    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    image_2d = relationship("Image2D", back_populates="annotations")
    rt_frame = relationship("RTFrame", back_populates="annotations")

    __table_args__ = (
        CheckConstraint("char_length(class_name) > 0", name="chk_annotation_class_name"),
        CheckConstraint("class_index IS NULL OR class_index >= 0", name="chk_annotation_class_index"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="chk_annotation_confidence"),
        CheckConstraint(
            "annotation_type <> 'bbox' OR (bbox_x IS NOT NULL AND bbox_y IS NOT NULL AND bbox_width > 0 AND bbox_height > 0)",
            name="chk_annotation_bbox"
        ),
        CheckConstraint(
            "annotation_type <> 'polygon' OR (polygon_data IS NOT NULL AND jsonb_typeof(polygon_data) = 'array')",
            name="chk_annotation_polygon"
        ),
        CheckConstraint(
            "annotation_type <> 'keypoint' OR (keypoints IS NOT NULL AND jsonb_typeof(keypoints) = 'array')",
            name="chk_annotation_keypoints"
        ),
        CheckConstraint("area IS NULL OR area >= 0", name="chk_annotation_area_nonneg"),
        CheckConstraint(
            "(image_2d_id IS NOT NULL AND rt_frame_id IS NULL) OR "
            "(image_2d_id IS NULL AND rt_frame_id IS NOT NULL)",
            name="chk_annotation_image_xor"
        ),
    )
