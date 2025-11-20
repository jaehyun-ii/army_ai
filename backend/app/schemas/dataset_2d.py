"""
2D Dataset schemas.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

from app.models.dataset_2d import AttackType
from app.core.config import settings


# Dataset 2D
class Dataset2DBase(BaseModel):
    """Base 2D dataset schema."""

    name: str
    description: Optional[str] = None
    storage_path: str
    metadata: Optional[Dict[str, Any]] = None


class Dataset2DCreate(Dataset2DBase):
    """Schema for creating 2D dataset."""

    pass


class Dataset2DUpdate(BaseModel):
    """Schema for updating 2D dataset."""

    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Dataset2DResponse(BaseModel):
    """Schema for 2D dataset response."""

    id: UUID
    name: str
    description: Optional[str] = None
    storage_path: str
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")
    owner_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class DatasetSummaryResponse(BaseModel):
    """Summary schema for dataset listings."""

    id: UUID
    name: str
    description: Optional[str] = None
    image_count: int
    storage_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetStatisticsResponse(BaseModel):
    """Detailed dataset statistics schema."""

    dataset_id: UUID
    name: str
    image_count: int
    storage_path: str
    total_size_bytes: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None



# Image
class ImageBase(BaseModel):
    """Base image schema."""

    file_name: str
    storage_key: str
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ImageCreate(ImageBase):
    """Schema for creating image."""

    dataset_id: UUID
    uploaded_by: Optional[UUID] = None


class ImageResponse(BaseModel):
    """Schema for image response."""

    id: UUID
    dataset_id: UUID
    file_name: str
    storage_key: str
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")
    uploaded_by: Optional[UUID]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# Patch 2D
class Patch2DBase(BaseModel):
    """Base 2D patch schema."""
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: Optional[str] = None
    target_class: Optional[str] = None
    method: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    patch_metadata: Optional[Dict[str, Any]] = None


class Patch2DCreate(Patch2DBase):
    """Schema for creating 2D patch (aligned with DB model field names)."""
    model_config = ConfigDict(protected_namespaces=())

    target_model_id: Optional[UUID] = None  # DB: nullable, field name matches model
    source_dataset_id: Optional[UUID] = None
    storage_key: Optional[str] = None
    file_name: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None


class Patch2DResponse(Patch2DBase):
    """Schema for 2D patch response (aligned with DB model field names)."""
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    target_model_id: Optional[UUID]  # DB: nullable, field name matches model
    source_dataset_id: Optional[UUID]
    storage_key: Optional[str] = None
    file_name: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    created_by: Optional[UUID]
    created_at: datetime


class PatchGenerationRequest(BaseModel):
    """Schema for adversarial patch generation request."""
    model_config = ConfigDict(protected_namespaces=())

    patch_name: str = Field(..., min_length=1, max_length=200)
    model_id: UUID
    dataset_id: UUID
    target_class: str = Field(..., min_length=1)
    plugin_name: str = Field(default=settings.DEFAULT_PATCH_PLUGIN)
    patch_size: int = Field(default=100, ge=10, le=500)
    area_ratio: float = Field(default=0.3, ge=0.05, le=1.0)
    epsilon: float = Field(default=0.6, ge=0.0, le=1.0)
    alpha: float = Field(default=0.03, ge=0.001, le=0.5)
    iterations: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=8, ge=1, le=64)
    description: Optional[str] = Field(default=None)
    created_by: Optional[UUID] = Field(default=None)
    session_id: Optional[str] = Field(default=None, description="SSE session ID for real-time event streaming")


# Attack Dataset 2D
class AttackDataset2DBase(BaseModel):
    """Base 2D attack dataset schema."""
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: Optional[str] = None
    attack_type: AttackType
    target_class: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class AttackDataset2DCreate(AttackDataset2DBase):
    """Schema for creating 2D attack dataset (aligned with DB model)."""
    model_config = ConfigDict(protected_namespaces=())

    target_model_id: Optional[UUID] = None  # DB: nullable, field name matches model
    base_dataset_id: UUID  # DB: NOT NULL - REQUIRED
    output_dataset_id: Optional[UUID] = None  # NEW: Output dataset reference
    patch_id: Optional[UUID] = None
    created_by: Optional[UUID] = None


class AttackDataset2DResponse(AttackDataset2DBase):
    """Schema for 2D attack dataset response (aligned with DB model)."""
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    target_model_id: Optional[UUID]  # DB: nullable, field name matches model
    base_dataset_id: UUID  # DB: NOT NULL
    output_dataset_id: Optional[UUID] = None  # NEW: Output dataset reference
    patch_id: Optional[UUID]
    created_by: Optional[UUID]
    created_at: datetime
