"""
YOLO format dataset upload service.

Handles upload of datasets with separate image and label folders.
Label format: YOLO txt format (class_id x_center y_center width height, normalized 0-1)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
from decimal import Decimal
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.models.dataset_2d import Dataset2D, Image2D
from app.models.annotation import Annotation, AnnotationType
from app import crud, schemas

import cv2

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_LABEL_EXTENSION = ".txt"


class YoloDatasetUploadService:
    """
    Service for uploading YOLO format datasets.

    Workflow:
        1. User selects image folder and label folder
        2. Match images with labels by filename (without extension)
        3. Report unmatched images (images without labels)
        4. Save images to storage
        5. Parse YOLO labels and create Annotation records
        6. Create Dataset2D and Image2D records
    """

    def __init__(self):
        self.storage_root = Path(settings.STORAGE_ROOT)
        self.datasets_dir = self.storage_root / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    async def upload_yolo_dataset(
        self,
        db: AsyncSession,
        images_folder: str,
        labels_folder: str,
        dataset_name: str,
        classes_file: Optional[str] = None,
        description: Optional[str] = None,
        owner_id: Optional[UUID] = None,
    ) -> Tuple[schemas.Dataset2DResponse, List[schemas.ImageResponse], Dict[str, Any]]:
        """
        Upload YOLO format dataset from image and label folders.

        Args:
            db: Database session
            images_folder: Path to folder containing images
            labels_folder: Path to folder containing YOLO label txt files
            dataset_name: Name for the dataset
            classes_file: Optional path to classes.txt file (one class per line)
            description: Optional dataset description
            owner_id: Optional owner UUID

        Returns:
            Tuple of (Dataset2D, List[Image2D], upload_stats)
            upload_stats contains:
                - matched_images: Number of images with labels
                - unmatched_images: Number of images without labels
                - unmatched_image_names: List of image filenames without labels
                - total_annotations: Total number of annotations created
                - class_distribution: Dict[class_name, count]
        """
        logger.info(f"Starting YOLO dataset upload: {dataset_name}")

        # Step 1: Validate folders
        images_path = Path(images_folder).resolve()
        labels_path = Path(labels_folder).resolve()

        if not images_path.exists() or not images_path.is_dir():
            raise NotFoundError(f"Images folder not found: {images_folder}")

        if not labels_path.exists() or not labels_path.is_dir():
            raise NotFoundError(f"Labels folder not found: {labels_folder}")

        # Step 2: Load classes
        class_names = self._load_classes(classes_file) if classes_file else None

        # Step 3: Match images with labels
        matched_files, unmatched_images = self._match_images_with_labels(
            images_path, labels_path
        )

        if not matched_files:
            raise ValidationError(
                "No matching image-label pairs found. "
                "Ensure image and label files have the same base name."
            )

        logger.info(
            f"Found {len(matched_files)} matched pairs, "
            f"{len(unmatched_images)} unmatched images"
        )

        # Step 4: Create dataset storage directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_storage_path = self.datasets_dir / f"{dataset_name}_{timestamp}"
        dataset_storage_path.mkdir(parents=True, exist_ok=True)

        # Step 5: Create Dataset2D record
        metadata = {
            "format": "YOLO",
            "matched_images": len(matched_files),
            "unmatched_images": len(unmatched_images),
        }

        # Store class names in metadata if provided
        if class_names:
            metadata["classes"] = class_names

        dataset = await crud.dataset_2d.create(
            db,
            obj_in=schemas.Dataset2DCreate(
                name=dataset_name,
                description=description or f"YOLO dataset with {len(matched_files)} images",
                storage_path=str(dataset_storage_path.relative_to(self.storage_root)),
                metadata=metadata
            ),
            owner_id=owner_id,
        )

        # Step 6: Process each matched image-label pair
        image_records = []
        annotation_records = []
        class_counts = defaultdict(int)
        failed_images = []

        for img_path, label_path in matched_files:
            try:
                # Copy image to storage
                import shutil
                dest_img_path = dataset_storage_path / img_path.name
                shutil.copy2(img_path, dest_img_path)

                # Get image dimensions using cv2
                img = cv2.imread(str(dest_img_path))
                if img is None:
                    raise ValidationError(f"Failed to read image: {img_path.name}")
                height, width = img.shape[:2]

                # Create Image2D record
                image_record = await crud.image_2d.create(
                    db,
                    obj_in=schemas.ImageCreate(
                        dataset_id=dataset.id,
                        file_name=img_path.name,
                        storage_key=str(dest_img_path.relative_to(self.storage_root)),
                        width=width,
                        height=height,
                        mime_type=self._get_mime_type(img_path.suffix),
                    ),
                    owner_id=owner_id,
                )
                image_records.append(image_record)

                # Parse YOLO labels
                annotations = self._parse_yolo_labels(
                    label_path,
                    image_record.id,
                    class_names
                )

                # Create Annotation records
                for ann_data in annotations:
                    ann_record = Annotation(
                        image_2d_id=image_record.id,
                        annotation_type=AnnotationType.BBOX,
                        class_name=ann_data["class_name"],
                        class_index=ann_data["class_index"],
                        bbox_x=Decimal(str(ann_data["bbox_x"])),
                        bbox_y=Decimal(str(ann_data["bbox_y"])),
                        bbox_width=Decimal(str(ann_data["bbox_width"])),
                        bbox_height=Decimal(str(ann_data["bbox_height"])),
                        confidence=Decimal("1.0"),  # Ground truth has confidence 1.0
                        metadata_={
                            "source": "YOLO label",
                            "format": "normalized_xywh",
                        }
                    )
                    annotation_records.append(ann_record)
                    class_counts[ann_data["class_name"]] += 1

            except Exception as e:
                logger.error(f"Failed to process image {img_path.name}: {e}", exc_info=True)
                failed_images.append(img_path.name)

        # Batch insert annotations
        if annotation_records:
            db.add_all(annotation_records)

        await db.commit()

        # Refresh records
        await db.refresh(dataset)
        for img in image_records:
            await db.refresh(img)

        # Prepare statistics
        upload_stats = {
            "matched_images": len(matched_files),
            "unmatched_images": len(unmatched_images),
            "unmatched_image_names": [img.name for img in unmatched_images],
            "failed_images": failed_images,
            "total_annotations": len(annotation_records),
            "class_distribution": dict(class_counts),
        }

        logger.info(
            f"Upload complete: {len(image_records)} images, "
            f"{len(annotation_records)} annotations"
        )

        return (
            schemas.Dataset2DResponse.model_validate(dataset),
            [schemas.ImageResponse.model_validate(img) for img in image_records],
            upload_stats,
        )

    def _load_classes(self, classes_file: str) -> List[str]:
        """
        Load class names from classes.txt file.

        Format: One class name per line (index = line number)
        """
        classes_path = Path(classes_file).resolve()
        if not classes_path.exists():
            raise NotFoundError(f"Classes file not found: {classes_file}")

        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                # Read lines and strip whitespace
                classes = [line.strip() for line in f if line.strip()]

            logger.info(f"Loaded {len(classes)} classes from {classes_file}")
            return classes
        except Exception as e:
            raise ValidationError(f"Failed to read classes file: {str(e)}")

    def _match_images_with_labels(
        self,
        images_path: Path,
        labels_path: Path,
    ) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
        """
        Match images with their corresponding label files.

        Returns:
            Tuple of (matched_pairs, unmatched_images)
            matched_pairs: List of (image_path, label_path) tuples
            unmatched_images: List of image paths without labels
        """
        # Get all images
        image_files = {}
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            for img_path in images_path.glob(f"*{ext}"):
                if img_path.is_file():
                    stem = img_path.stem  # filename without extension
                    image_files[stem] = img_path

        # Get all label files
        label_files = {}
        for label_path in labels_path.glob(f"*{SUPPORTED_LABEL_EXTENSION}"):
            if label_path.is_file():
                stem = label_path.stem
                label_files[stem] = label_path

        # Match by filename stem
        matched_pairs = []
        unmatched_images = []

        for stem, img_path in image_files.items():
            if stem in label_files:
                matched_pairs.append((img_path, label_files[stem]))
            else:
                unmatched_images.append(img_path)

        return matched_pairs, unmatched_images

    def _parse_yolo_labels(
        self,
        label_path: Path,
        image_id: UUID,
        class_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse YOLO format label file.

        Format: class_id x_center y_center width height (normalized 0-1)
        Each line is one bounding box.

        Returns:
            List of annotation data dicts
        """
        annotations = []

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        logger.warning(
                            f"Invalid YOLO format at {label_path.name}:{line_num} - "
                            f"expected 5 values, got {len(parts)}"
                        )
                        continue

                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Validate normalized coordinates (0-1 range)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                0 <= width <= 1 and 0 <= height <= 1):
                            logger.warning(
                                f"YOLO coordinates out of range at {label_path.name}:{line_num}"
                            )
                            continue

                        # Skip invalid boxes (too small or zero size)
                        # Database constraint requires bbox_width > 0 and bbox_height > 0
                        if width <= 0.01 or height <= 0.01:
                            logger.warning(
                                f"Skipping too small bbox at {label_path.name}:{line_num} "
                                f"(width={width}, height={height})"
                            )
                            continue

                        # Get class name
                        if class_names and 0 <= class_id < len(class_names):
                            class_name = class_names[class_id]
                        else:
                            class_name = f"class_{class_id}"

                        annotations.append({
                            "class_index": class_id,
                            "class_name": class_name,
                            "bbox_x": x_center,
                            "bbox_y": y_center,
                            "bbox_width": width,
                            "bbox_height": height,
                        })

                    except ValueError as e:
                        logger.warning(
                            f"Failed to parse line at {label_path.name}:{line_num} - {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"Error reading label file {label_path}: {e}", exc_info=True)

        return annotations

    def _get_mime_type(self, extension: str) -> str:
        """Get MIME type from file extension."""
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }
        return mime_types.get(extension.lower(), "image/jpeg")


# Global instance
yolo_dataset_upload_service = YoloDatasetUploadService()
