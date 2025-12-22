"""Dataset metadata endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_db
from app.core.exceptions import NotFoundError
from app.services.dataset_management_service import dataset_statistics_service

router = APIRouter()


@router.get("/{dataset_id}/top-classes")
async def get_top_classes(
    dataset_id: UUID,
    limit: int = Query(5, ge=1, le=20, description="Number of top classes to return"),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await dataset_statistics_service.get_top_classes(
            db=db,
            dataset_id=dataset_id,
            limit=limit,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
