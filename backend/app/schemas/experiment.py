"""
Experiment schemas.

PHASE 2 UPDATE: Tags are now normalized in separate tables.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

from app.models.experiment import ExperimentStatus


# ========== Experiment Tag Schemas ==========

class ExperimentTagBase(BaseModel):
    """Base experiment tag schema."""
    name: str = Field(..., min_length=1, max_length=100)
    color: Optional[str] = Field(None, max_length=20)
    description: Optional[str] = None


class ExperimentTagCreate(ExperimentTagBase):
    """Schema for creating experiment tag."""
    pass


class ExperimentTagUpdate(BaseModel):
    """Schema for updating experiment tag."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    color: Optional[str] = Field(None, max_length=20)
    description: Optional[str] = None


class ExperimentTagResponse(ExperimentTagBase):
    """Schema for experiment tag response."""
    id: UUID
    created_at: datetime
    deleted_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ExperimentTagListResponse(BaseModel):
    """Schema for paginated experiment tag list."""
    items: List[ExperimentTagResponse]
    total: int


# ========== Experiment Schemas ==========

class ExperimentBase(BaseModel):
    """
    Base experiment schema.

    PHASE 2 UPDATE: tags field now accepts tag names (will be converted to tag objects).
    """
    name: str
    description: Optional[str] = None
    objective: Optional[str] = None
    hypothesis: Optional[str] = None
    tag_names: Optional[List[str]] = Field(None, description="List of tag names to assign")
    config: Optional[Dict[str, Any]] = None


class ExperimentCreate(ExperimentBase):
    """Schema for creating experiment."""
    pass


class ExperimentUpdate(BaseModel):
    """Schema for updating experiment."""
    name: Optional[str] = None
    description: Optional[str] = None
    objective: Optional[str] = None
    hypothesis: Optional[str] = None
    status: Optional[ExperimentStatus] = None
    tag_names: Optional[List[str]] = Field(None, description="List of tag names to assign (replaces existing)")
    config: Optional[Dict[str, Any]] = None
    results_summary: Optional[Dict[str, Any]] = None


class ExperimentResponse(BaseModel):
    """
    Schema for experiment response.

    PHASE 2 UPDATE: tags field returns tag objects instead of strings.
    """
    id: UUID
    name: str
    description: Optional[str] = None
    objective: Optional[str] = None
    hypothesis: Optional[str] = None
    status: ExperimentStatus
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    config: Optional[Dict[str, Any]]
    results_summary: Optional[Dict[str, Any]]
    created_by: Optional[UUID]
    created_at: datetime
    updated_at: datetime

    # Tags (computed from tag_assignments relationship)
    tags: List[ExperimentTagResponse] = Field(default_factory=list, description="Experiment tags")

    model_config = ConfigDict(from_attributes=True)
