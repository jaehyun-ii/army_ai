"""
Annotation endpoints for retrieving and managing detection/inference results.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from uuid import UUID

from app.database import get_db
from app import schemas, models
from app.core.exceptions import NotFoundError, InternalServerError

router = APIRouter()


@router.get("/image/{image_id}", response_model=List[schemas.AnnotationResponse])
async def get_image_annotations(
    image_id: UUID,
    db: AsyncSession = Depends(get_db),
    annotation_type: Optional[models.AnnotationType] = Query(None, description="Filter by annotation type"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
) -> List[schemas.AnnotationResponse]:
    """Get all annotations for a specific image.

    Args:
        image_id: UUID of the image (2D)
        annotation_type: Optional filter by annotation type (bbox, polygon, keypoint, segmentation)
        min_confidence: Optional minimum confidence threshold

    Returns:
        List of annotations with detection information (class, bbox, confidence, etc.)
    """
    try:
        # Build query
        query = select(models.Annotation).where(
            models.Annotation.image_2d_id == image_id,
            models.Annotation.deleted_at.is_(None)
        )

        # Apply filters
        if annotation_type:
            query = query.where(models.Annotation.annotation_type == annotation_type)

        if min_confidence is not None:
            query = query.where(models.Annotation.confidence >= min_confidence)

        # Order by confidence descending
        query = query.order_by(models.Annotation.confidence.desc())

        result = await db.execute(query)
        annotations = result.scalars().all()

        return [schemas.AnnotationResponse.model_validate(ann) for ann in annotations]

    except Exception as e:
        raise InternalServerError(detail=f"Error retrieving annotations: {str(e)}")


@router.get("/dataset/{dataset_id}", response_model=dict)
async def get_dataset_annotations_summary(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
) -> dict:
    """Get annotation summary for all images in a dataset.

    Returns a summary including:
    - Total annotation count
    - Annotations per image
    - Class distribution
    - Average confidence per class

    Args:
        dataset_id: UUID of the dataset
        min_confidence: Optional minimum confidence threshold

    Returns:
        Dictionary with annotation summary statistics
    """
    try:
        # First verify dataset exists
        dataset_query = select(models.Dataset2D).where(
            models.Dataset2D.id == dataset_id,
            models.Dataset2D.deleted_at.is_(None)
        )
        dataset_result = await db.execute(dataset_query)
        dataset = dataset_result.scalar_one_or_none()

        if not dataset:
            raise NotFoundError(detail=f"Dataset {dataset_id} not found")

        # Get all images in dataset
        images_query = select(models.Image2D).where(
            models.Image2D.dataset_id == dataset_id,
            models.Image2D.deleted_at.is_(None)
        )
        images_result = await db.execute(images_query)
        images = images_result.scalars().all()
        image_ids = [img.id for img in images]

        # Get all annotations for these images
        annotations_query = select(models.Annotation).where(
            models.Annotation.image_2d_id.in_(image_ids),
            models.Annotation.deleted_at.is_(None)
        )

        if min_confidence is not None:
            annotations_query = annotations_query.where(models.Annotation.confidence >= min_confidence)

        annotations_result = await db.execute(annotations_query)
        annotations = annotations_result.scalars().all()

        # Calculate statistics
        total_annotations = len(annotations)

        # Annotations per image
        image_annotation_counts = {}
        for ann in annotations:
            image_id = str(ann.image_2d_id)
            image_annotation_counts[image_id] = image_annotation_counts.get(image_id, 0) + 1

        # Class distribution
        class_counts = {}
        class_confidences = {}
        for ann in annotations:
            class_name = ann.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            if class_name not in class_confidences:
                class_confidences[class_name] = []
            class_confidences[class_name].append(float(ann.confidence or 0.0))

        # Average confidence per class
        avg_confidence_per_class = {}
        for class_name, confidences in class_confidences.items():
            avg_confidence_per_class[class_name] = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "dataset_id": str(dataset_id),
            "dataset_name": dataset.name,
            "total_images": len(images),
            "total_annotations": total_annotations,
            "images_with_annotations": len(image_annotation_counts),
            "images_without_annotations": len(images) - len(image_annotation_counts),
            "avg_annotations_per_image": total_annotations / len(images) if images else 0.0,
            "class_distribution": class_counts,
            "avg_confidence_per_class": avg_confidence_per_class,
            "unique_classes": len(class_counts),
        }

    except NotFoundError:
        raise
    except Exception as e:
        raise InternalServerError(detail=f"Error retrieving dataset annotations: {str(e)}")


@router.post("/bulk", response_model=List[schemas.AnnotationResponse], status_code=status.HTTP_201_CREATED)
async def create_annotations_bulk(
    annotations: List[schemas.AnnotationCreate],
    image_id: UUID = Query(..., description="Image ID for all annotations"),
    db: AsyncSession = Depends(get_db),
) -> List[schemas.AnnotationResponse]:
    """Create multiple annotations for an image at once.

    Useful for batch uploading detection results.

    Args:
        annotations: List of annotation data to create
        image_id: UUID of the image these annotations belong to

    Returns:
        List of created annotation records
    """
    try:
        # Verify image exists
        image_query = select(models.Image2D).where(
            models.Image2D.id == image_id,
            models.Image2D.deleted_at.is_(None)
        )
        image_result = await db.execute(image_query)
        image = image_result.scalar_one_or_none()

        if not image:
            raise NotFoundError(detail=f"Image {image_id} not found")

        # Create annotation records
        created_annotations = []
        for ann_data in annotations:
            ann_dict = ann_data.model_dump(exclude_unset=True)
            ann_dict["image_2d_id"] = image_id

            annotation = models.Annotation(**ann_dict)
            db.add(annotation)
            created_annotations.append(annotation)

        await db.commit()

        # Refresh to get generated IDs
        for ann in created_annotations:
            await db.refresh(ann)

        return [schemas.AnnotationResponse.model_validate(ann) for ann in created_annotations]

    except NotFoundError:
        raise
    except Exception as e:
        await db.rollback()
        raise InternalServerError(detail=f"Error creating annotations: {str(e)}")


@router.delete("/image/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image_annotations(
    image_id: UUID,
    db: AsyncSession = Depends(get_db),
    annotation_type: Optional[models.AnnotationType] = Query(None, description="Only delete annotations of this type"),
) -> None:
    """Soft delete all annotations for an image.

    Args:
        image_id: UUID of the image
        annotation_type: Optional filter to only delete specific annotation type
    """
    try:
        # Build query
        query = select(models.Annotation).where(
            models.Annotation.image_2d_id == image_id,
            models.Annotation.deleted_at.is_(None)
        )

        if annotation_type:
            query = query.where(models.Annotation.annotation_type == annotation_type)

        result = await db.execute(query)
        annotations = result.scalars().all()

        if not annotations:
            raise NotFoundError(detail=f"No annotations found for image {image_id}")

        # Soft delete by setting deleted_at
        from datetime import datetime, timezone
        for ann in annotations:
            ann.deleted_at = datetime.now(timezone.utc)

        await db.commit()

    except NotFoundError:
        raise
    except Exception as e:
        await db.rollback()
        raise InternalServerError(detail=f"Error deleting annotations: {str(e)}")
