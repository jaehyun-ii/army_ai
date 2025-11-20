"""
Model repository schemas.
"""
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

from app.models.model_repo import ModelFramework, ModelStage, ArtifactType


# OD Model (merged with version fields)
class ODModelBase(BaseModel):
    """Base OD model schema."""
    model_config = ConfigDict(protected_namespaces=())

    name: str
    task: str = "object-detection"
    description: Optional[str] = None
    # Merged version fields
    version: str
    framework: ModelFramework
    framework_version: Optional[str] = None
    input_spec: Optional[Dict[str, Any]] = None
    labelmap: Optional[Dict[str, Any]] = None
    inference_params: Optional[Dict[str, Any]] = None
    stage: ModelStage = ModelStage.DRAFT

    @field_validator('input_spec', 'labelmap', 'inference_params', mode='before')
    @classmethod
    def convert_null_string_to_none(cls, v):
        """Convert string 'null' to None for JSON fields.

        This handles edge cases where frontend might send string "null"
        instead of actual null or omitting the field.
        """
        if v == 'null' or v == '':
            return None
        return v


class ODModelCreate(ODModelBase):
    """Schema for creating OD model."""

    pass


class ODModelResponse(ODModelBase):
    """Schema for OD model response."""

    id: UUID
    owner_id: Optional[UUID]
    created_by: Optional[UUID]
    published_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


# OD Model Artifact
class ODModelArtifactBase(BaseModel):
    """Base OD model artifact schema."""
    model_config = ConfigDict(protected_namespaces=())

    artifact_type: ArtifactType
    storage_key: str
    file_name: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    content_type: Optional[str] = None


class ODModelArtifactCreate(ODModelArtifactBase):
    """Schema for creating OD model artifact (aligned with DB model)."""
    model_config = ConfigDict(protected_namespaces=())

    model_id: UUID


class ODModelArtifactResponse(ODModelArtifactBase):
    """Schema for OD model artifact response (aligned with DB model)."""
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    model_id: UUID
    created_at: datetime
