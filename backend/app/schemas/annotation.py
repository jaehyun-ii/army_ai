"""
Annotation schemas for API requests/responses.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Any, Dict, List
from uuid import UUID
from datetime import datetime
from decimal import Decimal

from app.models.annotation import AnnotationType


class AnnotationBase(BaseModel):
    """Base annotation schema."""
    annotation_type: AnnotationType = AnnotationType.BBOX
    class_name: str
    class_index: Optional[int] = None
    confidence: Optional[Decimal] = Field(default=Decimal("1.0"))
    is_crowd: bool = False
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")


class AnnotationBBoxCreate(AnnotationBase):
    """Schema for creating bounding box annotations."""
    annotation_type: AnnotationType = AnnotationType.BBOX
    bbox_x: Decimal
    bbox_y: Decimal
    bbox_width: Decimal
    bbox_height: Decimal


class AnnotationCreate(BaseModel):
    """Schema for creating annotations (flexible)."""
    annotation_type: AnnotationType
    class_name: str
    class_index: Optional[int] = None

    # Image references (one of these must be set)
    image_2d_id: Optional[UUID] = None
    rt_frame_id: Optional[UUID] = None

    # Bounding box fields
    bbox_x: Optional[Decimal] = None
    bbox_y: Optional[Decimal] = None
    bbox_width: Optional[Decimal] = None
    bbox_height: Optional[Decimal] = None

    # Polygon data
    polygon_data: Optional[List[Any]] = None

    # Keypoints
    keypoints: Optional[List[Any]] = None

    confidence: Optional[Decimal] = Field(default=Decimal("1.0"))
    is_crowd: bool = False
    area: Optional[Decimal] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")


class AnnotationUpdate(BaseModel):
    """Schema for updating annotations."""
    annotation_type: Optional[AnnotationType] = None
    class_name: Optional[str] = None
    class_index: Optional[int] = None

    # Bounding box fields
    bbox_x: Optional[Decimal] = None
    bbox_y: Optional[Decimal] = None
    bbox_width: Optional[Decimal] = None
    bbox_height: Optional[Decimal] = None

    # Polygon data
    polygon_data: Optional[List[Any]] = None

    # Keypoints
    keypoints: Optional[List[Any]] = None

    confidence: Optional[Decimal] = None
    is_crowd: Optional[bool] = None
    area: Optional[Decimal] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")


class AnnotationResponse(BaseModel):
    """Schema for annotation response."""
    id: UUID
    image_2d_id: Optional[UUID] = None
    image_3d_id: Optional[UUID] = None
    rt_frame_id: Optional[UUID] = None
    annotation_type: AnnotationType
    class_name: str
    class_index: Optional[int] = None

    # Bounding box fields
    bbox_x: Optional[Decimal] = None
    bbox_y: Optional[Decimal] = None
    bbox_width: Optional[Decimal] = None
    bbox_height: Optional[Decimal] = None

    # Polygon data
    polygon_data: Optional[Dict[str, Any]] = None

    # Keypoints
    keypoints: Optional[Dict[str, Any]] = None

    confidence: Optional[Decimal] = None
    is_crowd: bool = False
    area: Optional[Decimal] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class AnnotationDetectionInfo(BaseModel):
    """Schema for detection info (compatible with YOLO metadata format)."""
    class_name: str = Field(alias="class")
    class_id: Optional[int] = Field(None, alias="class_id")
    confidence: float
    bbox: Dict[str, float]  # {"x1": ..., "y1": ..., "x2": ..., "y2": ...}

    model_config = ConfigDict(populate_by_name=True)
