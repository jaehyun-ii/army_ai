"""
Dataset service endpoints for upload and management operations.
These endpoints handle business logic and file operations.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Body, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional, List
from uuid import UUID
import os
import shutil
from pathlib import Path

from app.database import get_db
from app import schemas
from app.services.dataset_upload_service import dataset_upload_service
from app.services.yolo_dataset_upload_service import yolo_dataset_upload_service
from app.services.dataset_management_service import dataset_statistics_service
from app.core.exceptions import (
    NotFoundError,
    ValidationError,
    ConflictError,
    InternalServerError
)

router = APIRouter()


@router.post("/upload-folder", status_code=status.HTTP_201_CREATED)
async def upload_dataset_folder(
    *,
    db: AsyncSession = Depends(get_db),
    source_folder: str = Body(..., description="Path to the folder to upload"),
    dataset_name: str = Body(..., description="Name for the dataset"),
    description: Optional[str] = Body(None, description="Optional dataset description"),
    owner_id: Optional[UUID] = Body(None, description="Optional owner UUID"),
    inference_metadata_path: Optional[str] = Body(None, description="Optional path to YOLO inference metadata JSON"),
) -> Dict[str, Any]:
    """Upload a dataset from a local folder.

    This endpoint accepts a folder path and uploads all images found in that folder,
    creating dataset and image records in the database.

    The folder should contain:
    - Image files (.jpg, .jpeg, .png, .bmp, .tiff, .webp)
    - Optional metadata.json file with dataset metadata

    If inference_metadata_path is provided, the YOLO inference results will be parsed
    and stored in the database for fast class distribution queries.

    Returns:
        Dictionary with dataset info, image count, list of uploaded images, and optional metadata stats
    """
    try:
        dataset, images, metadata_stats = await dataset_upload_service.upload_dataset_from_folder(
            db=db,
            source_folder=source_folder,
            dataset_name=dataset_name,
            description=description,
            owner_id=owner_id,
            inference_metadata_path=inference_metadata_path
        )

        response = {
            "dataset": schemas.Dataset2DResponse.model_validate(dataset),
            "image_count": len(images),
            "images": [schemas.ImageResponse.model_validate(img) for img in images],
            "message": f"Successfully uploaded dataset with {len(images)} images"
        }

        if metadata_stats:
            response["metadata_processed"] = True
            response["metadata_stats"] = metadata_stats
        else:
            response["metadata_processed"] = False

        return response
    except (NotFoundError, ValidationError, ConflictError) as e:
        # Custom exceptions are already HTTPExceptions, re-raise as-is
        raise
    except Exception as e:
        # Catch unexpected errors and convert to InternalServerError
        raise InternalServerError(detail=f"Error uploading dataset: {str(e)}")


@router.get("/datasets", response_model=List[schemas.Dataset2DResponse])
async def get_all_datasets(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> List[schemas.Dataset2DResponse]:
    """Get list of all datasets.

    Returns:
        List of datasets with basic information
    """
    try:
        from app.crud.dataset_2d import dataset_2d_crud
        datasets = await dataset_2d_crud.get_multi(db, skip=skip, limit=limit)
        return [schemas.Dataset2DResponse.model_validate(ds) for ds in datasets]
    except Exception as e:
        raise InternalServerError(detail=f"Error getting datasets: {str(e)}")


@router.get("/{dataset_id}/stats")
async def get_dataset_statistics(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get statistics for a dataset including storage info.

    Returns:
        - dataset_id: Dataset UUID
        - name: Dataset name
        - image_count: Number of images
        - storage_path: Physical storage location
        - total_size_bytes: Total storage size
        - created_at: Creation timestamp
        - metadata: Dataset metadata
    """
    try:
        stats = await dataset_statistics_service.get_dataset_statistics(db, dataset_id)
        return stats
    except (NotFoundError, ValidationError, ConflictError) as e:
        # Custom exceptions are already HTTPExceptions, re-raise as-is
        raise
    except Exception as e:
        # Catch unexpected errors
        raise InternalServerError(detail=f"Error getting dataset statistics: {str(e)}")


@router.post("/{dataset_id}/detection-stats")
async def get_dataset_detection_statistics(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    model_id: UUID = Body(..., description="Model ID for detection"),
    conf_threshold: float = Body(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
) -> Dict[str, Any]:
    """Get detection statistics for a dataset.

    Runs object detection on all images in the dataset and returns:
        - Basic dataset statistics (image count, storage size, etc.)
        - Detection statistics:
            - total_detections: Total number of detections across all images
            - images_with_detections: Number of images with at least one detection
            - images_without_detections: Number of images with no detections
            - class_counts: Dictionary of {class_name: count} for all detected classes
            - unique_classes: Number of unique classes detected
            - average_detections_per_image: Average detections per image

    Args:
        dataset_id: Dataset UUID
        model_id: Model version to use for detection
        conf_threshold: Confidence threshold for detections (0.0 - 1.0)

    Returns:
        Complete dataset statistics with detection information
    """
    # Define a unique estimator ID for this request
    estimator_id = f"stats__{dataset_id}__{model_id}"

    # Import services here to avoid circular dependencies at startup if needed
    from app.services.estimator_loader_service import estimator_loader
    from app.services.model_inference_service import model_inference_service
    import logging
    logger = logging.getLogger(__name__)

    try:
        # 1. Load the model as a temporary estimator
        await estimator_loader.load_estimator_from_db(
            db=db,
            model_id=model_id,
            estimator_id=estimator_id
        )

        # 2. Run the statistics service
        stats = await dataset_statistics_service.get_dataset_detection_statistics(
            db=db,
            dataset_id=dataset_id,
            estimator_id=estimator_id, # Pass estimator_id now
            conf_threshold=conf_threshold
        )
        return stats
    except (NotFoundError, ValidationError, ConflictError) as e:
        raise
    except Exception as e:
        raise InternalServerError(detail=f"Error getting detection statistics: {str(e)}")
    finally:
        # 3. Unload the temporary estimator
        if model_inference_service.is_loaded(estimator_id):
            model_inference_service.unregister_estimator(estimator_id)
            logger.info(f"Unloaded temporary estimator '{estimator_id}'")


@router.post("/upload-multipart", status_code=status.HTTP_201_CREATED)
async def upload_dataset_multipart(
    *,
    db: AsyncSession = Depends(get_db),
    dataset_name: str = Form(..., description="Name for the dataset"),
    description: Optional[str] = Form(None, description="Optional dataset description"),
    images: List[UploadFile] = File(..., description="Image files to upload"),
    metadata: Optional[UploadFile] = File(None, description="Optional YOLO inference metadata JSON"),
) -> Dict[str, Any]:
    """Upload a dataset via multipart form data (browser file upload).

    This endpoint accepts files uploaded from a browser form and creates
    a dataset with the uploaded images.

    Returns:
        Dictionary with dataset info, image count, and list of uploaded images
    """
    try:
        # 임시 폴더 생성
        import uuid
        temp_dir = Path("storage/temp") / f"upload_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 이미지 파일 저장
        for image in images:
            # 파일명에서 경로 제거 (basename만 사용)
            filename = os.path.basename(image.filename)
            file_path = temp_dir / filename

            # 파일 저장
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

        # 메타데이터 파일 저장 (선택사항)
        metadata_path = None
        if metadata:
            # 파일명에서 경로 제거
            metadata_filename = os.path.basename(metadata.filename)
            metadata_path = temp_dir / metadata_filename
            with open(metadata_path, "wb") as buffer:
                shutil.copyfileobj(metadata.file, buffer)

        # dataset_service를 사용해서 데이터셋 생성
        dataset, image_records, metadata_stats = await dataset_upload_service.upload_dataset_from_folder(
            db=db,
            source_folder=str(temp_dir),
            dataset_name=dataset_name,
            description=description,
            owner_id=None,
            inference_metadata_path=str(metadata_path) if metadata_path else None
        )

        response = {
            "dataset": schemas.Dataset2DResponse.model_validate(dataset),
            "image_count": len(image_records),
            "images": [schemas.ImageResponse.model_validate(img) for img in image_records],
            "message": f"Successfully uploaded dataset with {len(image_records)} images"
        }

        if metadata_stats:
            response["metadata_processed"] = True
            response["metadata_stats"] = metadata_stats
        else:
            response["metadata_processed"] = False

        # 임시 폴더 삭제
        shutil.rmtree(temp_dir, ignore_errors=True)

        return response

    except (NotFoundError, ValidationError, ConflictError) as e:
        # Custom exceptions are already HTTPExceptions, re-raise as-is
        raise
    except Exception as e:
        # Catch unexpected errors and convert to InternalServerError
        # 임시 폴더 정리
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise InternalServerError(detail=f"Error uploading dataset: {str(e)}")


@router.post("/upload-yolo", status_code=status.HTTP_201_CREATED)
async def upload_yolo_dataset(
    *,
    db: AsyncSession = Depends(get_db),
    images_folder: str = Body(..., description="Path to folder containing images"),
    labels_folder: str = Body(..., description="Path to folder containing YOLO label txt files"),
    dataset_name: str = Body(..., description="Name for the dataset"),
    classes_file: Optional[str] = Body(None, description="Path to classes.txt file (one class per line)"),
    description: Optional[str] = Body(None, description="Optional dataset description"),
    owner_id: Optional[UUID] = Body(None, description="Optional owner UUID"),
) -> Dict[str, Any]:
    """Upload a YOLO format dataset from image and label folders.

    This endpoint accepts paths to separate image and label folders,
    matches them by filename, and creates dataset/image/annotation records.

    **Workflow:**
    1. Match images with labels by filename (without extension)
    2. Report unmatched images (images without labels)
    3. Save images to storage
    4. Parse YOLO labels and create Annotation records
    5. Create Dataset2D and Image2D records

    **YOLO Label Format:**
    - Each .txt file contains one bounding box per line
    - Format: `class_id x_center y_center width height`
    - All values normalized to [0, 1]

    **Example:**
    ```json
    {
      "images_folder": "/home/user/datasets/coco_person_100/images",
      "labels_folder": "/home/user/datasets/coco_person_100/labels",
      "dataset_name": "COCO Person 100",
      "classes_file": "/home/user/datasets/coco_person_100/classes.txt",
      "description": "COCO dataset with 100 person images"
    }
    ```

    **Returns:**
    - dataset: Dataset2D record
    - image_count: Number of images uploaded
    - images: List of Image2D records
    - upload_stats:
        - matched_images: Number of images with labels
        - unmatched_images: Number of images without labels
        - unmatched_image_names: List of image filenames without labels
        - total_annotations: Total annotations created
        - class_distribution: Dict[class_name, count]
    """
    try:
        dataset, images, upload_stats = await yolo_dataset_upload_service.upload_yolo_dataset(
            db=db,
            images_folder=images_folder,
            labels_folder=labels_folder,
            dataset_name=dataset_name,
            classes_file=classes_file,
            description=description,
            owner_id=owner_id,
        )

        return {
            "dataset": dataset,
            "image_count": len(images),
            "images": images,
            "upload_stats": upload_stats,
            "message": f"Successfully uploaded YOLO dataset with {len(images)} images and {upload_stats['total_annotations']} annotations"
        }

    except (NotFoundError, ValidationError, ConflictError) as e:
        raise
    except Exception as e:
        raise InternalServerError(detail=f"Error uploading YOLO dataset: {str(e)}")


@router.post("/upload-yolo-files", status_code=status.HTTP_201_CREATED)
async def upload_yolo_dataset_files(
    *,
    db: AsyncSession = Depends(get_db),
    name: str = Form(..., description="Dataset name"),
    description: Optional[str] = Form(None, description="Optional dataset description"),
    image_files: List[UploadFile] = File(..., description="Image files"),
    label_files: List[UploadFile] = File(..., description="YOLO label txt files"),
    classes_file: UploadFile = File(..., description="classes.txt file"),
) -> Dict[str, Any]:
    """Upload a YOLO format dataset via multipart file upload.

    This endpoint accepts uploaded image files, label files, and a classes file,
    then creates dataset/image/annotation records.

    **Returns:**
    - dataset: Dataset2D record
    - image_count: Number of images uploaded
    - images: List of Image2D records
    - upload_stats: Upload statistics
    """
    import uuid
    import tempfile

    # Create temporary directories for processing
    temp_base = Path(tempfile.gettempdir()) / f"yolo_upload_{uuid.uuid4().hex[:8]}"
    temp_images_dir = temp_base / "images"
    temp_labels_dir = temp_base / "labels"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded image files
        for image_file in image_files:
            filename = os.path.basename(image_file.filename)
            file_path = temp_images_dir / filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)

        # Save uploaded label files
        for label_file in label_files:
            filename = os.path.basename(label_file.filename)
            file_path = temp_labels_dir / filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(label_file.file, buffer)

        # Save classes file
        classes_path = temp_base / "classes.txt"
        with open(classes_path, "wb") as buffer:
            shutil.copyfileobj(classes_file.file, buffer)

        # Use yolo_dataset_upload_service to process
        dataset, images, upload_stats = await yolo_dataset_upload_service.upload_yolo_dataset(
            db=db,
            images_folder=str(temp_images_dir),
            labels_folder=str(temp_labels_dir),
            dataset_name=name,
            classes_file=str(classes_path),
            description=description,
            owner_id=None,
        )

        return {
            "dataset": dataset,
            "image_count": len(images),
            "images": images,
            "upload_stats": upload_stats,
            "message": f"Successfully uploaded YOLO dataset with {len(images)} images and {upload_stats['total_annotations']} annotations"
        }

    except (NotFoundError, ValidationError, ConflictError) as e:
        raise
    except Exception as e:
        raise InternalServerError(detail=f"Error uploading YOLO dataset files: {str(e)}")
    finally:
        # Clean up temporary directories
        if temp_base.exists():
            shutil.rmtree(temp_base, ignore_errors=True)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset_with_files(
    dataset_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete a 2D dataset and its files from storage.

    This endpoint performs:
    1. Physical file deletion from storage
    2. Soft delete of all images in DB
    3. Soft delete of dataset in DB
    """
    try:
        dataset_statistics_service.delete_dataset_with_files(db, dataset_id)
    except (NotFoundError, ValidationError, ConflictError) as e:
        # Custom exceptions are already HTTPExceptions, re-raise as-is
        raise
    except Exception as e:
        # Catch unexpected errors
        raise InternalServerError(detail=f"Error deleting dataset: {str(e)}")
