"""Dataset image endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from uuid import UUID

from app.database import get_db
from app import schemas
from app.core.exceptions import NotFoundError
from app.services.dataset_management_service import dataset_statistics_service

router = APIRouter()


@router.get("/{dataset_id}/images", response_model=List[schemas.ImageResponse])
async def list_dataset_images(
    dataset_id: UUID,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> List[schemas.ImageResponse]:
    try:
        images = await dataset_statistics_service.list_dataset_images(
            db=db,
            dataset_id=dataset_id,
            skip=skip,
            limit=limit,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [schemas.ImageResponse.model_validate(image) for image in images]
