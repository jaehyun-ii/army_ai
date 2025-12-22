"""
3D Dataset / Patch / Attack schemas.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.dataset_2d import AttackType


# Dataset 3D
class Dataset3DBase(BaseModel):
    """Base 3D dataset schema."""

    name: str
    description: Optional[str] = None
    storage_path: str
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")


class Dataset3DCreate(Dataset3DBase):
    """Schema for creating 3D dataset."""
    pass


class Dataset3DUpdate(BaseModel):
    """Schema for updating 3D dataset."""

    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")


class Dataset3DResponse(BaseModel):
    """Schema for 3D dataset response."""

    id: UUID
    name: str
    description: Optional[str] = None
    storage_path: str
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")
    owner_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# Image 3D
class Image3DBase(BaseModel):
    """Base 3D image schema."""

    file_name: str
    storage_key: str
    width: Optional[int] = None
    height: Optional[int] = None
    depth: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")


class Image3DCreate(Image3DBase):
    """Schema for creating 3D image."""

    dataset_id: UUID
    uploaded_by: Optional[UUID] = None


class Image3DResponse(BaseModel):
    """Schema for 3D image response."""

    id: UUID
    dataset_id: UUID
    file_name: str
    storage_key: str
    width: Optional[int] = None
    height: Optional[int] = None
    depth: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="metadata_")
    uploaded_by: Optional[UUID]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class Image3DListResponse(BaseModel):
    """Paginated list of 3D images."""

    total: int
    items: List[Image3DResponse]


# Patch 3D
class Patch3DBase(BaseModel):
    """Base 3D patch schema."""
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: Optional[str] = None
    target_class: Optional[str] = None
    method: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    patch_metadata: Optional[Dict[str, Any]] = None


class Patch3DCreate(Patch3DBase):
    """Schema for creating 3D patch."""
    model_config = ConfigDict(protected_namespaces=())

    target_model_id: Optional[UUID] = None
    source_dataset_id: Optional[UUID] = None
    storage_key: Optional[str] = None
    file_name: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None


class Patch3DResponse(Patch3DBase):
    """Schema for 3D patch response."""
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    target_model_id: Optional[UUID]
    source_dataset_id: Optional[UUID]
    storage_key: Optional[str] = None
    file_name: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    created_by: Optional[UUID]
    created_at: datetime


# Attack Dataset 3D
class AttackDataset3DBase(BaseModel):
    """Base 3D attack dataset schema."""
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: Optional[str] = None
    attack_type: AttackType
    target_class: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class AttackDataset3DCreate(AttackDataset3DBase):
    """Schema for creating 3D attack dataset (base dataset optional)."""
    model_config = ConfigDict(protected_namespaces=())

    target_model_id: Optional[UUID] = None
    output_dataset_id: Optional[UUID] = None
    patch_id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None
    created_by: Optional[UUID] = None


class AttackDataset3DResponse(AttackDataset3DBase):
    """Schema for 3D attack dataset response."""
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    target_model_id: Optional[UUID]
    output_dataset_id: Optional[UUID]
    patch_id: Optional[UUID]
    experiment_id: Optional[UUID]
    created_by: Optional[UUID]
    created_at: datetime
    updated_at: datetime

