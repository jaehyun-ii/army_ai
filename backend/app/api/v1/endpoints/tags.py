"""
API endpoints for experiment tags.

PHASE 2 NEW: Tag management endpoints.
"""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.crud.experiment_tag import experiment_tag
from app.schemas.experiment import (
    ExperimentTagCreate,
    ExperimentTagUpdate,
    ExperimentTagResponse,
    ExperimentTagListResponse,
)

router = APIRouter()


@router.get("/", response_model=ExperimentTagListResponse)
async def get_tags(
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: str = Query(None, description="Search tags by name"),
) -> ExperimentTagListResponse:
    """
    Get all experiment tags.

    Optional search parameter for autocomplete.
    """
    if search:
        tags = await experiment_tag.search_by_name(db, query=search, skip=skip, limit=limit)
        total = len(tags)  # Approximation for search
    else:
        tags = await experiment_tag.get_multi(db, skip=skip, limit=limit)
        from sqlalchemy import select, func as sql_func
        from app.models.experiment import ExperimentTag
        result = await db.execute(
            select(sql_func.count()).select_from(ExperimentTag).where(ExperimentTag.deleted_at.is_(None))
        )
        total = result.scalar()

    return ExperimentTagListResponse(items=tags, total=total)


@router.post("/", response_model=ExperimentTagResponse, status_code=201)
async def create_tag(
    *,
    db: AsyncSession = Depends(get_db),
    tag_in: ExperimentTagCreate,
) -> ExperimentTagResponse:
    """
    Create new experiment tag.

    Prevents duplicate tag names (case-insensitive).
    """
    # Check for duplicate
    existing = await experiment_tag.get_by_name(db, name=tag_in.name)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Tag with name '{tag_in.name}' already exists"
        )

    tag = await experiment_tag.create(db, obj_in=tag_in)
    return tag


@router.get("/{tag_id}", response_model=ExperimentTagResponse)
async def get_tag(
    tag_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentTagResponse:
    """Get experiment tag by ID."""
    tag = await experiment_tag.get(db, id=tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    return tag


@router.put("/{tag_id}", response_model=ExperimentTagResponse)
async def update_tag(
    tag_id: UUID,
    tag_in: ExperimentTagUpdate,
    db: AsyncSession = Depends(get_db),
) -> ExperimentTagResponse:
    """Update experiment tag."""
    tag = await experiment_tag.get(db, id=tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    # Check for duplicate name if name is being changed
    if tag_in.name and tag_in.name != tag.name:
        existing = await experiment_tag.get_by_name(db, name=tag_in.name)
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Tag with name '{tag_in.name}' already exists"
            )

    tag = await experiment_tag.update(db, db_obj=tag, obj_in=tag_in)
    return tag


@router.delete("/{tag_id}", status_code=204)
async def delete_tag(
    tag_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete experiment tag (soft delete).

    Also removes all tag assignments.
    """
    tag = await experiment_tag.get(db, id=tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    await experiment_tag.remove(db, id=tag_id)
