"""
Prediction model.

PHASE 3 NEW: Normalized prediction storage.
Replaces eval_items.prediction JSONB field.
"""
from sqlalchemy import Column, String, Integer, Numeric, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class Prediction(Base):
    """
    Individual object detection prediction.

    PHASE 3 NEW: Replaces eval_items.prediction JSONB with normalized table.
    Stores individual bbox predictions for query and analysis.
    """
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    eval_item_id = Column(UUID(as_uuid=True), ForeignKey("eval_items.id", ondelete="CASCADE"), nullable=False)
    
    # Class information
    class_id = Column(UUID(as_uuid=True), ForeignKey("class_definitions.id", ondelete="SET NULL"))
    class_name = Column(String(200), nullable=False)
    class_index = Column(Integer)

    # Normalized bbox coordinates (0.0 - 1.0)
    bbox_x1 = Column(Numeric(10, 6), nullable=False)
    bbox_y1 = Column(Numeric(10, 6), nullable=False)
    bbox_x2 = Column(Numeric(10, 6), nullable=False)
    bbox_y2 = Column(Numeric(10, 6), nullable=False)

    # Confidence score
    confidence = Column(Numeric(5, 4), nullable=False)

    # Additional prediction-specific data
    metadata = Column(JSONB)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    eval_item = relationship("EvalItem", foreign_keys=[eval_item_id])
    class_def = relationship("ClassDefinition", foreign_keys=[class_id])

    __table_args__ = (
        CheckConstraint("char_length(class_name) > 0", name="chk_class_name"),
        CheckConstraint("class_index IS NULL OR class_index >= 0", name="chk_class_index"),
        CheckConstraint("bbox_x1 >= 0 AND bbox_x1 <= 1", name="chk_bbox_x1"),
        CheckConstraint("bbox_y1 >= 0 AND bbox_y1 <= 1", name="chk_bbox_y1"),
        CheckConstraint("bbox_x2 >= 0 AND bbox_x2 <= 1", name="chk_bbox_x2"),
        CheckConstraint("bbox_y2 >= 0 AND bbox_y2 <= 1", name="chk_bbox_y2"),
        CheckConstraint("bbox_x2 > bbox_x1 AND bbox_y2 > bbox_y1", name="chk_bbox_valid"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="chk_confidence"),
    )

    def __repr__(self):
        return f"<Prediction(id={self.id}, class_name={self.class_name}, confidence={self.confidence})>"

    @property
    def bbox_dict(self):
        """Return bbox as dictionary."""
        return {
            "x1": float(self.bbox_x1),
            "y1": float(self.bbox_y1),
            "x2": float(self.bbox_x2),
            "y2": float(self.bbox_y2),
        }

    @property
    def bbox_width(self):
        """Calculate bbox width."""
        return float(self.bbox_x2 - self.bbox_x1)

    @property
    def bbox_height(self):
        """Calculate bbox height."""
        return float(self.bbox_y2 - self.bbox_y1)

    @property
    def bbox_area(self):
        """Calculate bbox area."""
        return self.bbox_width * self.bbox_height
