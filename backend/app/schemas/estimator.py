"""
Pydantic schemas for estimator endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class EstimatorFramework(str, Enum):
    """Supported ML frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class EstimatorType(str, Enum):
    """Supported estimator types."""
    YOLO = "yolo"
    FASTER_RCNN = "faster_rcnn"
    RT_DETR = "rt_detr"
    EFFICIENTDET = "efficientdet"


class LoadEstimatorRequest(BaseModel):
    """Request to load an estimator."""
    estimator_id: str = Field(..., description="Unique identifier for this estimator instance")
    framework: EstimatorFramework = Field(..., description="ML framework (pytorch/tensorflow)")
    estimator_type: EstimatorType = Field(..., description="Type of estimator")

    # Support both model_id (from DB) and model_path (direct file)
    model_id: Optional[str] = Field(None, description="UUID of model in database (alternative to model_path)")
    model_path: Optional[str] = Field(None, description="Direct path to model weights file (alternative to model_id)")

    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional configuration (input_size, class_names, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "estimator_id": "my_yolo_v8",
                "framework": "pytorch",
                "estimator_type": "yolo",
                "model_id": "550e8400-e29b-41d4-a716-446655440000",
                "config": {
                    "input_size": [640, 640],
                    "class_names": ["person", "car", "bicycle"]
                }
            }
        }


class LoadEstimatorResponse(BaseModel):
    """Response after loading estimator."""
    estimator_id: str = Field(..., description="Unique ID for this estimator instance")
    status: str = Field(..., description="Loading status (loaded/failed)")
    message: str = Field(..., description="Status message")
    framework: EstimatorFramework
    estimator_type: EstimatorType
    model_id: Optional[str] = Field(None, description="Model ID from database (if loaded from DB)")
    model_path: Optional[str] = Field(None, description="Model file path")
    supports_adversarial_attack: bool = Field(..., description="Whether this estimator supports adversarial attacks")


class PredictRequest(BaseModel):
    """Request for prediction."""
    image_path: Optional[str] = Field(None, description="Path to image file")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "image_path": "storage/datasets/test_image.jpg",
                "conf_threshold": 0.25,
                "iou_threshold": 0.45
            }
        }


class BBox(BaseModel):
    """Bounding box in xyxy format."""
    x1: float
    y1: float
    x2: float
    y2: float


class YoloBBox(BaseModel):
    """Bounding box in YOLO format (normalized x_center, y_center, width, height)."""
    x_center: float
    y_center: float
    width: float
    height: float


class Detection(BaseModel):
    """Single detection result."""
    bbox: BBox | YoloBBox
    class_id: int
    class_name: str | None = None
    confidence: float


class PredictResponse(BaseModel):
    """Prediction response."""
    estimator_id: str
    num_detections: int
    detections: List[Detection]


class EstimatorListResponse(BaseModel):
    """List of loaded estimators."""
    estimators: List[Dict[str, Any]]
    count: int


class EstimatorStatusResponse(BaseModel):
    """Status of a specific estimator."""
    estimator_id: str
    status: str
    class_name: Optional[str] = None
    supports_adversarial_attack: Optional[bool] = None
    message: str
