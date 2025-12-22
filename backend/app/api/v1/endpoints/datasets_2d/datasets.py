"""Dataset CRUD endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from uuid import UUID

from app.database import get_db
from app import schemas, crud
from app.services.dataset_management_service import dataset_statistics_service

router = APIRouter()


@router.post("/", response_model=schemas.Dataset2DResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    *,
    db: AsyncSession = Depends(get_db),
    dataset_in: schemas.Dataset2DCreate,
) -> schemas.Dataset2DResponse:
    """Create a new 2D dataset (metadata only)."""
    dataset = await crud.dataset_2d.create(db, obj_in=dataset_in)
    return dataset


@router.get("/{dataset_id}", response_model=schemas.Dataset2DResponse)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> schemas.Dataset2DResponse:
    """Get a 2D dataset by ID."""
    dataset = await crud.dataset_2d.get(db, id=dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found",
        )
    return dataset


@router.get("/", response_model=List[schemas.DatasetSummaryResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> List[schemas.DatasetSummaryResponse]:
    """List 2D datasets with image counts."""
    summaries = await dataset_statistics_service.list_datasets(
        db=db,
        skip=skip,
        limit=limit,
    )
    return [schemas.DatasetSummaryResponse.model_validate(summary) for summary in summaries]


@router.patch("/{dataset_id}", response_model=schemas.Dataset2DResponse)
async def update_dataset(
    dataset_id: UUID,
    dataset_in: schemas.Dataset2DUpdate,
    db: AsyncSession = Depends(get_db),
) -> schemas.Dataset2DResponse:
    dataset = await crud.dataset_2d.get(db, id=dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found",
        )

    updated = await crud.dataset_2d.update(db, db_obj=dataset, obj_in=dataset_in)
    return updated


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    dataset = await crud.dataset_2d.get(db, id=dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found",
        )

    await crud.dataset_2d.soft_delete(db, id=dataset_id)


@router.get("/{dataset_id}/images", response_model=List[schemas.ImageResponse])
async def get_dataset_images(
    dataset_id: UUID,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> List[schemas.ImageResponse]:
    """Get images from a dataset."""
    # Check if dataset exists
    dataset = await crud.dataset_2d.get(db, id=dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found",
        )

    # Get images
    images = await crud.image_2d.get_by_dataset(db, dataset_id=dataset_id, skip=skip, limit=limit)
    return images


@router.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image(
    image_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft delete an image by ID."""
    image = await crud.image_2d.get(db, id=image_id)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID {image_id} not found",
        )

    await crud.image_2d.soft_delete(db, id=image_id)
