"""Dataset statistics endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_db
from app import schemas
from app.core.exceptions import NotFoundError
from app.services.dataset_management_service import dataset_statistics_service

router = APIRouter()


@router.get("/{dataset_id}/stats", response_model=schemas.DatasetStatisticsResponse)
async def get_dataset_stats(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> schemas.DatasetStatisticsResponse:
    try:
        stats = await dataset_statistics_service.get_dataset_statistics(db=db, dataset_id=dataset_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Coerce to schema-friendly types
    stats["dataset_id"] = UUID(stats["dataset_id"])
    return schemas.DatasetStatisticsResponse.model_validate(stats)
