"""
Real-time performance measurement schemas (aligned with database schema).
Removed: Camera, RTInference (tables do not exist in DB schema)
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from decimal import Decimal

from app.models.realtime import RTRunStatus


# RT Capture Run
class RTCaptureRunBase(BaseModel):
    """Base RT capture run schema (aligned with DB schema - no camera_id)."""

    window_seconds: int = 5
    frames_expected: int = 10
    fps_target: Optional[Decimal] = None
    notes: Optional[str] = None


class RTCaptureRunCreate(RTCaptureRunBase):
    """Schema for creating RT capture run (no camera_id in DB schema)."""
    model_config = ConfigDict(protected_namespaces=())

    model_id: Optional[UUID] = None


class RTCaptureRunUpdate(BaseModel):
    """Schema for updating RT capture run."""

    ended_at: Optional[datetime] = None
    status: Optional[RTRunStatus] = None
    notes: Optional[str] = None


class RTCaptureRunResponse(RTCaptureRunBase):
    """Schema for RT capture run response (no camera_id in DB schema)."""
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    model_id: Optional[UUID] = None
    started_at: datetime
    ended_at: Optional[datetime]
    status: RTRunStatus
    created_by: Optional[UUID]
    created_at: datetime
    updated_at: datetime


# RT Frame
class RTFrameBase(BaseModel):
    """Base RT frame schema."""

    seq_no: int
    storage_key: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias="metadata_",
        serialization_alias="metadata_"
    )


class RTFrameCreate(RTFrameBase):
    """Schema for creating RT frame."""

    run_id: UUID
    captured_at: Optional[datetime] = None


class RTFrameUpdate(BaseModel):
    """Schema for updating RT frame."""

    storage_key: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RTFrameResponse(RTFrameBase):
    """Schema for RT frame response."""

    id: UUID
    run_id: UUID
    captured_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


