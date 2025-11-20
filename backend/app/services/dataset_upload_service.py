"""
Dataset upload service and helpers.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import (
    ConflictError,
    InternalServerError,
    NotFoundError,
    ValidationError,
)
from app.models.dataset_2d import Dataset2D, Image2D
from app.schemas.dataset_2d import Dataset2DCreate, ImageCreate
from app.utils.storage import storage_manager

logger = logging.getLogger(__name__)

DEFAULT_CLASS_NAME = "unknown"
DEFAULT_CLASS_ID = 0
DEFAULT_CONFIDENCE = 0.0
DEFAULT_BBOX_COORD = 0.0
DEFAULT_model_name = "unknown"
BATCH_SIZE = 1000


@dataclass
class DatasetUploadContext:
    source_folder: str
    dataset_name: str
    description: Optional[str]
    owner_id: Optional[UUID]
    inference_metadata_path: Optional[str]


@dataclass
class DatasetIngestionResult:
    storage_path: str
    metadata: Dict[str, Any]
    image_list: List[Dict[str, Any]]


class DatasetUploadValidator:
    """Validate dataset upload inputs."""

    @staticmethod
    def validate_metadata_path(metadata_path: Optional[str]) -> Optional[Path]:
        if not metadata_path:
            return None

        metadata_path_obj = Path(metadata_path).resolve()
        if not metadata_path_obj.exists():
            raise NotFoundError(
                resource=f"Inference metadata file at {metadata_path}"
            )

        try:
            if str(metadata_path_obj).startswith(("/etc", "/sys", "/proc", "/dev")):
                raise ValidationError(
                    detail="Access to system directories is not allowed"
                )
            if not metadata_path_obj.is_file():
                raise ValidationError(detail="Metadata path must be a regular file")
        except (OSError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(detail=f"Invalid metadata path: {str(exc)}") from exc

        return metadata_path_obj


class DatasetFolderIngestor:
    """Handle copying uploaded dataset folders into managed storage."""

    def __init__(self, storage):
        self._storage = storage

    def ingest(self, context: DatasetUploadContext) -> DatasetIngestionResult:
        storage_path, metadata, image_list = self._storage.save_dataset_folder(
            source_folder=context.source_folder,
            dataset_name=context.dataset_name,
            owner_id=context.owner_id,
        )
        logger.info("Saved %s images to storage at %s", len(image_list), storage_path)

        return DatasetIngestionResult(
            storage_path=storage_path,
            metadata=metadata,
            image_list=image_list,
        )

    def cleanup(self, storage_path: str) -> None:
        try:
            self._storage.delete_dataset_folder(storage_path)
            logger.info("Successfully cleaned up storage at %s", storage_path)
        except Exception as cleanup_error:
            logger.error(
                "Failed to cleanup storage: %s",
                cleanup_error,
                exc_info=True,
            )


class DatasetPersistenceManager:
    """Persist dataset and image records."""

    async def create_dataset_records(
        self,
        db: AsyncSession,
        context: DatasetUploadContext,
        ingestion: DatasetIngestionResult,
    ) -> Tuple[Dataset2D, List[Image2D]]:
        dataset_data = Dataset2DCreate(
            name=context.dataset_name,
            description=context.description or ingestion.metadata.get("description", ""),
            storage_path=ingestion.storage_path,
            metadata=ingestion.metadata,
        )

        dataset = Dataset2D(
            name=dataset_data.name,
            description=dataset_data.description,
            owner_id=context.owner_id,
            storage_path=dataset_data.storage_path,
            metadata_=dataset_data.metadata,
        )
        db.add(dataset)
        await db.flush()

        images: List[Image2D] = []
        for image_info in ingestion.image_list:
            image = Image2D(
                dataset_id=dataset.id,
                file_name=image_info["file_name"],
                storage_key=image_info["storage_key"],
                width=image_info["width"],
                height=image_info["height"],
                mime_type=image_info["mime_type"],
                uploaded_by=context.owner_id,
                metadata_={
                    "size_bytes": image_info["size_bytes"],
                    "relative_path": image_info["relative_path"],
                },
            )
            db.add(image)
            images.append(image)

        await db.flush()
        return dataset, images


class InferenceMetadataProcessor:
    """Process inference metadata files and store in annotations table."""

    async def process(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        metadata_path: str,
        uploaded_images: List[Image2D],
    ) -> Dict[str, Any]:
        from datetime import datetime, timezone
        from collections import defaultdict
        from app.models.annotation import Annotation, AnnotationType
        from decimal import Decimal

        # Read and parse metadata file
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_json = json.load(f)
        except FileNotFoundError as exc:
            raise NotFoundError(resource=f"Metadata file at {metadata_path}") from exc
        except PermissionError as exc:
            raise ValidationError(
                detail="Permission denied: Cannot read metadata file"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValidationError(
                detail=f"Invalid JSON format at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise ValidationError(detail="Invalid file encoding. Expected UTF-8.") from exc
        except OSError as exc:
            raise InternalServerError(detail=f"Error reading metadata file: {str(exc)}") from exc

        # Parse inference timestamp
        timestamp_str = metadata_json.get("timestamp", "")
        try:
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str.replace("Z", "+00:00")
            inference_timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now(timezone.utc)
            if inference_timestamp.tzinfo is not None:
                inference_timestamp = inference_timestamp.replace(tzinfo=None)
        except (ValueError, AttributeError):
            inference_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

        # Build filename -> image mapping
        image_map = {}
        for img in uploaded_images:
            image_map[img.file_name] = img

        # Track statistics
        total_detections = 0
        total_images_processed = 0
        class_counts = defaultdict(int)
        skipped_images = []
        annotations = []

        # Process each image in metadata
        for img_data in metadata_json.get("images", []):
            filename = img_data.get("filename", "")

            # Find matching image in database
            image_db = image_map.get(filename)
            if not image_db:
                skipped_images.append(filename)
                continue

            detections_list = img_data.get("detections", [])

            # Create Annotation records for each detection
            for det in detections_list:
                bbox = det.get("bbox", {})

                # Check if YOLO format (x_center, y_center, width, height) or xyxy format
                if "x_center" in bbox and "y_center" in bbox:
                    # YOLO normalized format (0-1)
                    x_center = bbox.get("x_center", 0.0)
                    y_center = bbox.get("y_center", 0.0)
                    width = bbox.get("width", 0.0)
                    height = bbox.get("height", 0.0)

                    # Store as-is (database uses same format)
                    bbox_x = x_center
                    bbox_y = y_center
                    bbox_width = width
                    bbox_height = height
                else:
                    # Legacy xyxy format
                    x1 = bbox.get("x1", 0.0)
                    y1 = bbox.get("y1", 0.0)
                    x2 = bbox.get("x2", 0.0)
                    y2 = bbox.get("y2", 0.0)

                    # Check if normalized (0-1 range)
                    if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                        # Convert to pixel coordinates
                        x1 = x1 * image_db.width
                        y1 = y1 * image_db.height
                        x2 = x2 * image_db.width
                        y2 = y2 * image_db.height

                    # Convert xyxy to xywh and normalize
                    width_px = x2 - x1
                    height_px = y2 - y1
                    x_center_px = (x1 + x2) / 2.0
                    y_center_px = (y1 + y2) / 2.0

                    # Normalize to 0-1
                    bbox_x = x_center_px / image_db.width
                    bbox_y = y_center_px / image_db.height
                    bbox_width = width_px / image_db.width
                    bbox_height = height_px / image_db.height

                annotation = Annotation(
                    image_2d_id=image_db.id,
                    annotation_type=AnnotationType.BBOX,
                    class_name=det.get("class", DEFAULT_CLASS_NAME),
                    class_index=det.get("class_id", DEFAULT_CLASS_ID),
                    bbox_x=Decimal(str(bbox_x)),
                    bbox_y=Decimal(str(bbox_y)),
                    bbox_width=Decimal(str(bbox_width)),
                    bbox_height=Decimal(str(bbox_height)),
                    confidence=Decimal(str(det.get("confidence", DEFAULT_CONFIDENCE))),
                    metadata_={
                        "model": metadata_json.get("model", DEFAULT_model_name),
                        "inference_timestamp": timestamp_str,
                    }
                )
                annotations.append(annotation)

                total_detections += 1
                class_name = det.get("class", DEFAULT_CLASS_NAME)
                class_counts[class_name] += 1

            if detections_list:
                total_images_processed += 1

        # Batch insert annotations
        if annotations:
            for i in range(0, len(annotations), BATCH_SIZE):
                batch = annotations[i : i + BATCH_SIZE]
                db.add_all(batch)
                await db.flush()
        else:
            logger.warning("No detections found in metadata for dataset %s", dataset_id)

        if skipped_images:
            logger.warning(
                "Skipped %d images not found in uploaded dataset: %s",
                len(skipped_images),
                skipped_images[:10],
            )

        # Update dataset metadata with model info
        result = await db.execute(
            select(Dataset2D).filter(Dataset2D.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        if dataset:
            if dataset.metadata_ is None:
                dataset.metadata_ = {}
            dataset.metadata_["model"] = metadata_json.get("model", DEFAULT_model_name)
            dataset.metadata_["inference_timestamp"] = timestamp_str
            dataset.metadata_["total_detections"] = total_detections
            dataset.metadata_["total_images_with_detections"] = total_images_processed

        await db.flush()

        # Prepare statistics for response
        class_distribution = dict(class_counts)
        top_classes = sorted(
            class_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "total_detections": total_detections,
            "total_images": total_images_processed,
            "class_distribution": class_distribution,
            "top_classes": [
                {"class": name, "count": count} for name, count in top_classes
            ],
        }


class DatasetUploadService:
    """Service for uploading and managing 2D datasets."""

    def __init__(
        self,
        *,
        storage=storage_manager,
        validator: Optional[DatasetUploadValidator] = None,
        folder_ingestor: Optional[DatasetFolderIngestor] = None,
        persistence_manager: Optional[DatasetPersistenceManager] = None,
        metadata_processor: Optional[InferenceMetadataProcessor] = None,
    ):
        self.storage = storage
        self._validator = validator or DatasetUploadValidator()
        self._folder_ingestor = folder_ingestor or DatasetFolderIngestor(self.storage)
        self._persistence_manager = persistence_manager or DatasetPersistenceManager()
        self._metadata_processor = metadata_processor or InferenceMetadataProcessor()

    async def upload_dataset_from_folder(
        self,
        db: AsyncSession,
        source_folder: str,
        dataset_name: str,
        description: Optional[str] = None,
        owner_id: Optional[UUID] = None,
        inference_metadata_path: Optional[str] = None,
    ) -> Tuple[Dataset2D, List[Image2D], Optional[Dict[str, Any]]]:
        logger.info("Starting dataset upload: %s from %s", dataset_name, source_folder)

        context = DatasetUploadContext(
            source_folder=source_folder,
            dataset_name=dataset_name,
            description=description,
            owner_id=owner_id,
            inference_metadata_path=inference_metadata_path,
        )

        metadata_path_obj = self._validator.validate_metadata_path(
            context.inference_metadata_path
        )

        ingestion = self._folder_ingestor.ingest(context)
        if not ingestion.image_list:
            self._folder_ingestor.cleanup(ingestion.storage_path)
            raise ValidationError(
                detail=(
                    "No images found in the source folder. Dataset must contain at least one image."
                )
            )

        try:
            dataset, images = await self._persistence_manager.create_dataset_records(
                db=db,
                context=context,
                ingestion=ingestion,
            )

            metadata_stats = None
            if metadata_path_obj:
                metadata_stats = await self._metadata_processor.process(
                    db=db,
                    dataset_id=dataset.id,
                    metadata_path=str(metadata_path_obj),
                    uploaded_images=images,
                )

            await db.commit()

            await db.refresh(dataset)
            for image in images:
                await db.refresh(image)

            return dataset, images, metadata_stats

        except (ValidationError, ConflictError, NotFoundError):
            await db.rollback()
            self._folder_ingestor.cleanup(ingestion.storage_path)
            raise
        except InternalServerError:
            await db.rollback()
            self._folder_ingestor.cleanup(ingestion.storage_path)
            raise
        except Exception as exc:
            await db.rollback()
            logger.error(
                "Upload failed, cleaning up storage at %s",
                ingestion.storage_path,
                exc_info=True,
            )
            self._folder_ingestor.cleanup(ingestion.storage_path)
            raise exc

    async def _process_inference_metadata_async(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        metadata_path: str,
        uploaded_images: List[Image2D],
    ) -> Dict[str, Any]:
        """Backwards compatible wrapper for existing integrations."""
        return await self._metadata_processor.process(
            db=db,
            dataset_id=dataset_id,
            metadata_path=metadata_path,
            uploaded_images=uploaded_images,
        )


dataset_upload_service = DatasetUploadService()
