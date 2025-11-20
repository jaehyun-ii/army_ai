"""
Patch application service for applying existing patches to datasets.

This service implements Step 2 of patch attack workflow:
    patch + base_dataset → apply patch → attacked_dataset
"""
from __future__ import annotations

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, schemas
from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.models.dataset_2d import AttackType
from app.services.sse_support import SSELogger, SSEManager

logger = logging.getLogger(__name__)

# Shared SSE manager instance for all attack services
_shared_sse_manager = SSEManager()


class PatchAttackService:
    """
    Service for applying adversarial patches to datasets (Step 2 of patch attack).

    Workflow:
        1. Load Patch2D record and patch file
        2. Load base_dataset images
        3. Apply patch to all images (center placement)
        4. Save patched images to new dataset
        5. Create AttackDataset2D record
    """

    def __init__(self):
        self.storage_root = Path(settings.STORAGE_ROOT)
        self.attack_datasets_dir = self.storage_root / "attack_datasets"
        self.attack_datasets_dir.mkdir(parents=True, exist_ok=True)

        # Use shared SSE manager
        self.sse_manager = _shared_sse_manager

    async def apply_patch_to_dataset(
        self,
        db: AsyncSession,
        attack_name: str,
        patch_id: UUID,
        base_dataset_id: UUID,
        patch_scale: float = 30.0,
        session_id: Optional[str] = None,
        current_user_id: Optional[UUID] = None,
    ) -> Tuple[schemas.AttackDataset2DResponse, UUID]:
        """
        Apply an existing patch to a dataset.

        Args:
            db: Database session
            attack_name: Name for the attacked dataset
            patch_id: ID of the patch to apply
            base_dataset_id: Dataset to apply the patch to
            patch_scale: Patch size as percentage of bbox area (1-100%)
            session_id: SSE session ID for progress updates
            current_user_id: User ID for ownership

        Returns:
            Tuple of (AttackDataset2D response, output_dataset_id)
        """
        # Create SSE session if session_id provided and not already created
        if session_id and session_id not in self.sse_manager._event_queues:
            self.sse_manager.create_session(session_id)
            logger.info(f"Service: Created SSE session: {session_id}")
        elif session_id:
            logger.info(f"Service: SSE session already exists: {session_id}")

        # Initialize SSE logger
        sse_logger = SSELogger(logger, self.sse_manager, session_id)

        try:
            await sse_logger.status("패치 적용 시작...")

            # Step 1: Load patch record
            await sse_logger.status("패치 로딩 중...")
            patch_record = await crud.patch_2d.get(db, id=patch_id)
            if not patch_record:
                raise NotFoundError(f"Patch {patch_id} not found")

            # Load patch file
            patch_path = self.storage_root / patch_record.storage_key
            if not patch_path.exists():
                raise NotFoundError(f"Patch file not found: {patch_path}")

            patch_bgr = cv2.imread(str(patch_path))
            if patch_bgr is None:
                raise ValidationError(f"Failed to load patch image: {patch_path}")

            # Convert BGR to RGB
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

            await sse_logger.info(
                f"패치 로드 완료: {patch_record.name}, shape={patch_rgb.shape}, target_class={patch_record.target_class}"
            )

            # Step 2: Load base dataset
            await sse_logger.status("베이스 데이터셋 로딩 중...")
            base_dataset = await crud.dataset_2d.get(db, id=base_dataset_id)
            if not base_dataset:
                raise NotFoundError(f"Dataset {base_dataset_id} not found")

            images = await self._load_dataset_images(db, base_dataset_id)
            await sse_logger.info(f"이미지 로드 완료: {len(images)}개")

            # Step 2.5: Load annotations for bbox-based patch placement
            await sse_logger.status("어노테이션 로딩 중...")
            annotations = await self._load_dataset_annotations(db, base_dataset_id)
            await sse_logger.info(f"어노테이션 로드 완료: {len(annotations)}개")

            # Step 3: Apply patch to all images with bbox-based placement
            await sse_logger.status("패치 적용 중...")
            patched_images = []
            failed_count = 0
            total_patches_applied = 0
            images_with_patches = 0

            for idx, img_data in enumerate(images):
                try:
                    # Progress update - Send every image for smooth progress
                    await sse_logger.progress(
                        f"패치 적용 중... ({idx + 1}/{len(images)})",
                        processed=idx + 1,
                        total=len(images),
                        successful=len(patched_images),
                        failed=failed_count,
                    )

                    # Get annotations for this image
                    img_annotations = [ann for ann in annotations if ann["image_id"] == img_data["id"]]

                    # Filter annotations by target class
                    target_annotations = [
                        ann for ann in img_annotations
                        if ann["class_name"] == patch_record.target_class
                    ] if patch_record.target_class else img_annotations

                    # Apply patch to bboxes
                    img = img_data["image"]
                    if target_annotations:
                        patched_img, num_patches = self._apply_patch_to_bboxes(
                            img, patch_rgb, target_annotations, patch_scale
                        )
                        total_patches_applied += num_patches
                        if num_patches > 0:
                            images_with_patches += 1
                    else:
                        # No matching bboxes, keep original image
                        patched_img = img

                    patched_images.append({
                        "image": patched_img,
                        "original_file_name": img_data["file_name"],
                        "original_id": img_data["id"],
                        "num_patches": len(target_annotations) if target_annotations else 0,
                    })

                except Exception as e:
                    logger.error(f"Failed to apply patch to image {idx}: {e}", exc_info=True)
                    failed_count += 1
                    await sse_logger.warning(f"이미지 {idx} 패치 적용 실패: {str(e)}")

            if not patched_images:
                raise ValidationError("All images failed to be patched")

            await sse_logger.info(
                f"패치 적용 완료: 성공 {len(patched_images)}, 실패 {failed_count}, "
                f"총 패치 수: {total_patches_applied}, 패치 적용 이미지: {images_with_patches}"
            )

            # Step 4: Create output dataset
            await sse_logger.status("패치 적용된 이미지 저장 중...")

            output_dataset_name = f"{attack_name}_output"
            output_dataset_path = self.attack_datasets_dir / f"{attack_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dataset_path.mkdir(parents=True, exist_ok=True)

            output_dataset = await crud.dataset_2d.create(
                db,
                obj_in=schemas.Dataset2DCreate(
                    name=output_dataset_name,
                    description=f"Output dataset from patch attack (patch: {patch_record.name})",
                    storage_path=str(output_dataset_path.relative_to(self.storage_root)),
                ),
                owner_id=current_user_id,
            )

            # Save images and create image records
            for img_data in patched_images:
                # Save image
                file_name = f"patched_{img_data['original_file_name']}"
                img_path = output_dataset_path / file_name

                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_data["image"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_path), img_bgr)

                # Create Image2D record
                height, width = img_data["image"].shape[:2]
                await crud.image_2d.create(
                    db,
                    obj_in=schemas.ImageCreate(
                        dataset_id=output_dataset.id,
                        file_name=file_name,
                        storage_key=str(img_path.relative_to(self.storage_root)),
                        width=width,
                        height=height,
                        mime_type="image/png",
                        uploaded_by=current_user_id,
                    ),
                )

            await db.commit()
            await sse_logger.info(f"저장 완료: {len(patched_images)}개 이미지")

            # Step 5: Create AttackDataset2D record
            await sse_logger.status("공격 데이터셋 레코드 생성 중...")

            attack_dataset = await crud.attack_dataset_2d.create(
                db,
                obj_in=schemas.AttackDataset2DCreate(
                    name=attack_name,
                    description=f"Patch attack using patch '{patch_record.name}'",
                    attack_type=AttackType.PATCH,
                    target_model_id=patch_record.target_model_id,
                    base_dataset_id=base_dataset_id,
                    output_dataset_id=output_dataset.id,  # NEW: Use dedicated column
                    target_class=patch_record.target_class,
                    patch_id=patch_id,
                    parameters={
                        "patch_name": patch_record.name,
                        "patch_method": patch_record.method,
                        "patch_scale": patch_scale,
                        "processed_images": len(patched_images),
                        "failed_images": failed_count,
                        "total_patches_applied": total_patches_applied,
                        "images_with_patches": images_with_patches,
                        "storage_path": str(output_dataset_path),
                        # Note: output_dataset_id moved to dedicated column for better schema design
                    },
                    created_by=current_user_id,
                ),
            )

            await db.commit()

            await sse_logger.success(
                "패치 적용 완료!",
                attack_dataset_id=str(attack_dataset.id),
                output_dataset_id=str(output_dataset.id),
                processed=len(patched_images),
                failed=failed_count,
            )

            # Send complete event to close SSE stream
            if session_id:
                logger.info(f"Sending 'complete' event to session: {session_id}")
                await self.sse_manager.send_event(session_id, {
                    "type": "complete",
                    "message": "패치 적용 완료!",
                    "attack_dataset_id": str(attack_dataset.id),
                    "output_dataset_id": str(output_dataset.id),
                    "processed": len(patched_images),
                    "failed": failed_count,
                })
                logger.info(f"'complete' event sent successfully to session: {session_id}")

            return schemas.AttackDataset2DResponse.model_validate(attack_dataset), output_dataset.id

        except NotFoundError as e:
            # Resource not found (patch, dataset, etc.)
            error_message = f"리소스를 찾을 수 없습니다: {str(e)}"
            await sse_logger.error(error_message)
            logger.error(f"Resource not found: {e}", exc_info=True)
            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "error",
                    "message": error_message,
                })
            raise
        except ValidationError as e:
            # Invalid parameters or data
            error_message = f"잘못된 입력입니다: {str(e)}"
            await sse_logger.error(error_message)
            logger.error(f"Validation error: {e}", exc_info=True)
            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "error",
                    "message": error_message,
                })
            raise
        except Exception as e:
            # Check for database unique constraint violation
            error_str = str(e).lower()
            if "unique" in error_str or "duplicate" in error_str:
                # Extract dataset name from error if possible
                if "already exists" in error_str or "duplicate key" in error_str:
                    error_message = f"이미 존재하는 데이터셋 이름입니다. 다른 이름을 사용해주세요."
                else:
                    error_message = "데이터셋 이름이 중복되었습니다. 다른 이름을 사용해주세요."
            elif "foreign key" in error_str or "constraint" in error_str:
                error_message = "데이터베이스 제약 조건 위반: 참조된 리소스가 존재하지 않습니다."
            elif "connection" in error_str or "timeout" in error_str:
                error_message = "데이터베이스 연결 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            else:
                error_message = f"패치 적용 중 오류가 발생했습니다: {str(e)}"

            await sse_logger.error(error_message)
            logger.error(f"Error applying patch to dataset: {e}", exc_info=True)

            # Send error event to close SSE stream
            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "error",
                    "message": error_message,
                })
            raise

    def _apply_patch_to_bboxes(
        self,
        image: np.ndarray,
        patch: np.ndarray,
        annotations: List[Dict[str, Any]],
        patch_scale: float,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply patch to detection bboxes in the image (matches frontend logic).

        Args:
            image: Original image (H, W, C) RGB uint8
            patch: Patch to apply (H_p, W_p, C) RGB uint8
            annotations: List of bbox annotations with keys: bbox_x, bbox_y, bbox_width, bbox_height
            patch_scale: Patch size as percentage of bbox area (1-100%)

        Returns:
            Tuple of (patched image, number of patches applied)
        """
        patched_img = image.copy()
        img_h, img_w = image.shape[:2]
        patches_applied = 0

        for ann in annotations:
            try:
                # Get normalized bbox coordinates (0-1 range)
                bbox_x = float(ann.get("bbox_x", 0))
                bbox_y = float(ann.get("bbox_y", 0))
                bbox_width = float(ann.get("bbox_width", 0))
                bbox_height = float(ann.get("bbox_height", 0))

                # Skip invalid bboxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                # Convert to pixel coordinates
                # bbox_x, bbox_y are center coordinates in normalized space
                px_width = bbox_width * img_w
                px_height = bbox_height * img_h
                px_center_x = bbox_x * img_w
                px_center_y = bbox_y * img_h

                # Calculate patch size based on bbox area and patch_scale
                # This matches frontend logic: patchArea = bboxArea * (patchScale / 100)
                bbox_area = px_width * px_height
                patch_area = bbox_area * (patch_scale / 100.0)
                patch_size = int(np.sqrt(patch_area))

                # Skip if patch is too small
                if patch_size < 5:
                    continue

                # Resize patch to desired size
                resized_patch = cv2.resize(patch, (patch_size, patch_size))

                # Calculate patch position (center of bbox)
                # This matches frontend logic: patchX = bboxCenterX - patchSize / 2
                patch_x = int(px_center_x - patch_size / 2)
                patch_y = int(px_center_y - patch_size / 2)

                # Ensure patch stays within image bounds
                patch_x = max(0, min(patch_x, img_w - patch_size))
                patch_y = max(0, min(patch_y, img_h - patch_size))

                # Apply patch (opaque replacement, matches frontend drawImage)
                x_end = min(patch_x + patch_size, img_w)
                y_end = min(patch_y + patch_size, img_h)
                actual_patch_w = x_end - patch_x
                actual_patch_h = y_end - patch_y

                if actual_patch_w > 0 and actual_patch_h > 0:
                    patched_img[patch_y:y_end, patch_x:x_end] = resized_patch[:actual_patch_h, :actual_patch_w]
                    patches_applied += 1

            except Exception as e:
                logger.warning(f"Failed to apply patch to bbox: {e}")
                continue

        return patched_img, patches_applied

    async def _load_dataset_annotations(
        self,
        db: AsyncSession,
        dataset_id: UUID,
    ) -> List[Dict[str, Any]]:
        """
        Load all bbox annotations from a dataset.

        Returns:
            List of dicts with keys: image_id, class_name, bbox_x, bbox_y, bbox_width, bbox_height
        """
        # Get all images from dataset
        images_db = await crud.image_2d.get_by_dataset(db, dataset_id=dataset_id)

        if not images_db:
            return []

        annotations = []
        for img_record in images_db:
            # Get annotations for this image
            img_annotations = await crud.annotation.get_by_image(db, image_2d_id=img_record.id)

            for ann in img_annotations:
                # Only process bbox annotations
                if ann.annotation_type != "bbox":
                    continue

                # Skip annotations with missing bbox data
                if (ann.bbox_x is None or ann.bbox_y is None or
                    ann.bbox_width is None or ann.bbox_height is None):
                    continue

                annotations.append({
                    "image_id": img_record.id,
                    "class_name": ann.class_name,
                    "bbox_x": ann.bbox_x,
                    "bbox_y": ann.bbox_y,
                    "bbox_width": ann.bbox_width,
                    "bbox_height": ann.bbox_height,
                })

        return annotations

    async def _load_dataset_images(
        self,
        db: AsyncSession,
        dataset_id: UUID,
    ) -> List[Dict[str, Any]]:
        """
        Load all images from a dataset.

        Returns:
            List of dicts with keys: id, file_name, image (numpy array RGB)
        """
        # Get all images from dataset
        images_db = await crud.image_2d.get_by_dataset(db, dataset_id=dataset_id)

        if not images_db:
            raise ValidationError(f"Dataset {dataset_id} has no images")

        loaded_images = []
        for img_record in images_db:
            # Construct full path
            img_path = self.storage_root / img_record.storage_key

            if not img_path.exists():
                logger.warning(f"Image file not found: {img_path}")
                continue

            # Load image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            loaded_images.append({
                "id": img_record.id,
                "file_name": img_record.file_name,
                "image": img_rgb,  # (H, W, C) RGB uint8
            })

        return loaded_images


# Global instance
patch_attack_service = PatchAttackService()
