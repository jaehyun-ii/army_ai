"""
Schemas for custom model upload and management.
"""
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID


class ModelFolderParseResponse(BaseModel):
    """Response schema for model folder parsing."""
    model_config = ConfigDict(protected_namespaces=())

    is_valid: bool
    model_name: str
    version: str
    framework: str
    class_names: List[str]
    input_size: List[int]
    num_classes: int
    has_weights: bool
    has_config: bool
    has_adapter: bool
    weights_filename: Optional[str] = None
    description: Optional[str] = None
    framework_version: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    architecture: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = Field(default_factory=list)


class ModelUploadRequest(BaseModel):
    """Request schema for uploading custom model."""
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., min_length=1, max_length=200)
    version: str = Field(..., min_length=1, max_length=64)
    description: Optional[str] = None
    framework: str = Field(..., description="Model framework (pytorch, tensorflow, onnx, etc.)")


class ModelUploadResponse(BaseModel):
    """Response schema for model upload."""
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version_id: str
    status: str
    message: str


class ModelInferenceRequest(BaseModel):
    """Request schema for model inference."""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_image_source(self):
        """Ensure at least one image source is provided."""
        if self.image_url is None and self.image_base64 is None:
            raise ValueError('Either image_url or image_base64 must be provided')
        return self


class BoundingBoxResponse(BaseModel):
    """Bounding box in YOLO normalized format (0-1)."""
    x_center: float
    y_center: float
    width: float
    height: float


class DetectionResponse(BaseModel):
    """Single detection in response."""
    bbox: BoundingBoxResponse
    class_id: int
    class_name: str
    confidence: float


class ModelInferenceResponse(BaseModel):
    """Response schema for model inference."""
    detections: List[DetectionResponse]
    inference_time_ms: Optional[float] = None
    model_info: Dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    """Response schema for listing custom models."""
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_name: str
    version: str
    framework: str
    is_loaded: bool
    num_classes: int
    created_at: Optional[datetime] = None


class ModelInfoResponse(BaseModel):
    """Detailed model information."""
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_name: str
    version: str
    framework: str
    is_loaded: bool
    class_names: List[str]
    num_classes: int
    config: Dict[str, Any]
    created_at: Optional[datetime] = None
