"""
Patch generation service for creating adversarial patches.

This service implements Step 1 of patch attack workflow:
    source_dataset → train patch → Patch2D record + patch file
"""
from __future__ import annotations

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, schemas
from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.services.sse_support import SSELogger, SSEManager
from app.services.estimator_loader_service import EstimatorLoaderService
from app.services.model_inference_service import model_inference_service

# ART imports - use real ART library for patch attacks
from app.ai.attacks.evasion import AdversarialPatchPyTorch, RobustDPatch, DPatch
from app.ai.estimators.object_detection import PyTorchYolo as ARTPyTorchYolo

logger = logging.getLogger(__name__)


class PatchService:
    """
    Service for generating adversarial patches (Step 1 of patch attack).

    Workflow:
        1. Load source_dataset images with target_class
        2. Load model as estimator
        3. Train adversarial patch using ART
        4. Save patch file
        5. Create Patch2D record
    """

    def __init__(self):
        self.storage_root = Path(settings.STORAGE_ROOT)
        self.patches_dir = self.storage_root / "patches"
        self.patches_dir.mkdir(parents=True, exist_ok=True)

        self.estimator_loader = EstimatorLoaderService()
        self.sse_manager = SSEManager()

    async def generate_patch(
        self,
        db: AsyncSession,
        patch_name: str,
        attack_method: str,
        source_dataset_id: UUID,
        model_id: UUID,
        target_class: str,
        patch_size: int,
        learning_rate: float,
        iterations: int,
        session_id: Optional[str] = None,
        current_user_id: Optional[UUID] = None,
    ) -> schemas.Patch2DResponse:
        """
        Generate an adversarial patch.

        Args:
            db: Database session
            patch_name: Name for the patch
            attack_method: "patch", "dpatch", or "robust_dpatch"
            source_dataset_id: Dataset to train the patch on
            model_id: Target model for attack
            target_class: Target class name (e.g., "person")
            patch_size: Size of the patch (height/width)
            learning_rate: Learning rate for optimization
            iterations: Number of optimization iterations
            session_id: SSE session ID for progress updates
            current_user_id: User ID for ownership

        Returns:
            Patch2D response schema
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
            # Validate attack method
            if attack_method not in ["patch", "dpatch", "robust_dpatch"]:
                raise ValidationError(
                    f"Invalid attack method: {attack_method}. Must be 'patch', 'dpatch', or 'robust_dpatch'"
                )

            await sse_logger.status("패치 생성 시작...")

            # Step 1: Load model
            await sse_logger.status("모델 로딩 중...")
            model = await crud.od_model.get(db, id=model_id)
            if not model:
                raise NotFoundError(f"Model {model_id} not found")

            # Get target class ID from labelmap
            if not model.labelmap:
                raise ValidationError(f"Model {model_id} has no labelmap")

            target_class_id = None
            for class_id_str, class_name in model.labelmap.items():
                if class_name.lower() == target_class.lower():
                    target_class_id = int(class_id_str)
                    break

            if target_class_id is None:
                raise ValidationError(
                    f"Target class '{target_class}' not found in model labelmap. "
                    f"Available: {list(model.labelmap.values())}"
                )

            await sse_logger.info(f"타겟 클래스: {target_class} (ID: {target_class_id})")

            # Load estimator using real ART library
            # For DPatch/RobustDPatch, we need channels_first=False
            channels_first = (attack_method == "patch")  # True for AdversarialPatchPyTorch, False for DPatch/RobustDPatch
            estimator, input_size = await self._load_art_estimator(db, model_id, channels_first=channels_first)
            await sse_logger.info("모델 로딩 완료")

            # Step 2: Collect training images with target_class
            await sse_logger.status(f"'{target_class}' 이미지 수집 중...")
            training_images = await self._collect_target_images(
                db, source_dataset_id, target_class_id
            )

            if not training_images:
                raise ValidationError(
                    f"No images found with target class '{target_class}' in dataset {source_dataset_id}"
                )

            await sse_logger.info(f"수집 완료: {len(training_images)}개 이미지")

            # Use input_size from _load_art_estimator
            model_height, model_width = input_size[0], input_size[1]
            await sse_logger.info(f"모델 입력 크기: {model_width}x{model_height}")

            # Step 3: Prepare training data (resize all images to model input size)
            await sse_logger.status("이미지 전처리 중...")
            x_train_list_hwc = []  # For DPatch/RobustDPatch (HWC format)
            x_train_list_chw = []  # For AdversarialPatchPyTorch (CHW format)

            for img_data in training_images:
                img = img_data["image"]  # (H, W, C) RGB

                # Resize to model input size if needed (following notebook and noise_attack_service approach)
                if img.shape[0] != model_height or img.shape[1] != model_width:
                    from PIL import Image
                    img_pil = Image.fromarray(img.astype(np.uint8))
                    img_pil = img_pil.resize((model_width, model_height), Image.BICUBIC)
                    img = np.array(img_pil).astype(np.float32)

                # Store both formats
                x_train_list_hwc.append(img)  # (H, W, C) for DPatch/RobustDPatch
                x_train_list_chw.append(img.transpose(2, 0, 1))  # (C, H, W) for AdversarialPatchPyTorch

            # Prepare data based on attack method
            # AdversarialPatchPyTorch expects NCHW, DPatch/RobustDPatch expect NHWC
            if attack_method == "patch":
                x_train = np.stack(x_train_list_chw, axis=0).astype(np.float32)  # (N, C, H, W)
            else:  # dpatch or robust_dpatch
                x_train = np.stack(x_train_list_hwc, axis=0).astype(np.float32)  # (N, H, W, C)

            await sse_logger.info(f"학습 데이터 준비 완료: {x_train.shape}")

            # Prepare labels (y) for targeted attacks
            # Convert normalized bbox to pixel coordinates for resized images
            y_train = []
            for img_data in training_images:
                # Get annotations for this image
                annotations = img_data.get("annotations", [])

                # Filter for target_class
                target_boxes = []
                target_labels = []
                for ann in annotations:
                    if ann["category_id"] == target_class_id:
                        # bbox is [x1, y1, x2, y2] in normalized coordinates (0-1)
                        # Convert to pixel coordinates for resized image
                        x1_norm, y1_norm, x2_norm, y2_norm = ann["bbox"]
                        x1_pixel = x1_norm * model_width
                        y1_pixel = y1_norm * model_height
                        x2_pixel = x2_norm * model_width
                        y2_pixel = y2_norm * model_height

                        target_boxes.append([x1_pixel, y1_pixel, x2_pixel, y2_pixel])
                        target_labels.append(target_class_id)

                if target_boxes:
                    y_train.append({
                        "boxes": np.array(target_boxes, dtype=np.float32),
                        "labels": np.array(target_labels, dtype=np.int64),
                        "scores": np.ones(len(target_boxes), dtype=np.float32),  # Add scores for ART
                    })
                else:
                    # No target class in this image, add empty
                    y_train.append({
                        "boxes": np.array([], dtype=np.float32).reshape(0, 4),
                        "labels": np.array([], dtype=np.int64),
                        "scores": np.array([], dtype=np.float32),
                    })

            # Step 4: Create ART patch attack
            await sse_logger.status(f"{attack_method.upper()} 패치 생성 중...")

            if attack_method == "patch":
                # AdversarialPatchPyTorch
                # batch_size=1 to handle variable number of targets per image (from notebook)
                attack = AdversarialPatchPyTorch(
                    estimator=estimator,
                    rotation_max=22.5,
                    scale_min=0.2,
                    scale_max=0.4,
                    learning_rate=learning_rate,
                    max_iter=iterations,
                    batch_size=1,  # REQUIRED: different images have different number of target boxes
                    patch_shape=(3, patch_size, patch_size),  # CHW
                    patch_type="circle",
                    optimizer="Adam",
                    targeted=False,  # Untargeted: evade detection (from notebook)
                    verbose=True,  # Enable verbose logging to see iteration progress
                )
            elif attack_method == "dpatch":
                # DPatch (does not support 'targeted' parameter)
                attack = DPatch(
                    estimator=estimator,
                    patch_shape=(patch_size, patch_size, 3),  # HWC
                    learning_rate=learning_rate,
                    max_iter=iterations,
                    batch_size=1,  # REQUIRED: different images have different number of target boxes
                    verbose=True,  # Enable verbose logging
                )
            else:  # robust_dpatch
                # RobustDPatch with targeted attack on specific class
                attack = RobustDPatch(
                    estimator=estimator,
                    patch_shape=(patch_size, patch_size, 3),  # HWC
                    learning_rate=learning_rate,
                    max_iter=iterations,
                    batch_size=1,  # REQUIRED: different images have different number of target boxes
                    sample_size=5,  # EOT samples for robustness
                    targeted=True,  # Enable targeted attack on specific class
                    verbose=True,  # Enable verbose logging
                )

            await sse_logger.info(
                f"패치 훈련 시작: {iterations} iterations, learning_rate={learning_rate}"
            )
            await sse_logger.info("각 이터레이션 진행 상황은 서버 콘솔 로그에서 확인할 수 있습니다")

            # Step 5: Generate patch with progress monitoring
            # Note: ART's generate() method trains the patch
            # The verbose=True flag will output progress to stdout/logger
            import asyncio
            import concurrent.futures
            import sys

            # Run patch generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Shared state for progress tracking
            progress_data = {
                "current_iteration": 0,
                "total_iterations": iterations,
            }

            # Monkey-patch tqdm to capture progress
            original_tqdm = None
            try:
                from tqdm import tqdm as original_tqdm_cls
                original_tqdm = original_tqdm_cls

                class TqdmSSE(original_tqdm_cls):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._sse_n = 0

                    def update(self, n=1):
                        result = super().update(n)
                        self._sse_n += n
                        progress_data["current_iteration"] = self._sse_n
                        return result

                # Replace tqdm in all relevant modules
                sys.modules['tqdm'].tqdm = TqdmSSE
                sys.modules['tqdm.auto'].tqdm = TqdmSSE
                logger.info("Successfully patched tqdm for progress tracking")
            except Exception as e:
                logger.warning(f"Failed to monkey-patch tqdm: {e}")

            # Create a wrapper function to run in thread pool
            def generate_patch_with_logging():
                logger.info(f"Starting patch generation with {iterations} iterations...")

                try:
                    # All attack methods use y_train for targeted attack on specific class
                    result = attack.generate(x=x_train, y=y_train)
                    return result
                finally:
                    logger.info("Patch generation completed")
                    # Restore original tqdm
                    if original_tqdm:
                        try:
                            sys.modules['tqdm'].tqdm = original_tqdm
                            sys.modules['tqdm.auto'].tqdm = original_tqdm
                            logger.info("Restored original tqdm")
                        except Exception as e:
                            logger.warning(f"Failed to restore tqdm: {e}")

            # Run in executor and send progress updates
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = loop.run_in_executor(executor, generate_patch_with_logging)

                # Send progress updates while patch is being generated
                last_iteration = 0
                update_interval = 1  # Update every 1 second

                while not future.done():
                    await asyncio.sleep(update_interval)

                    current_iter = progress_data["current_iteration"]
                    if current_iter > last_iteration:
                        # Send progress update with actual iteration count
                        progress_percent = int((current_iter / iterations) * 100)

                        await sse_logger.progress(
                            f"패치 생성 중... ({current_iter}/{iterations})",
                            iteration=current_iter,
                            total_iterations=iterations,
                            progress=progress_percent,
                        )
                        last_iteration = current_iter

                # Get the result
                result = await future

            # Handle different return types
            # AdversarialPatchPyTorch returns (patch, mask) tuple
            # DPatch and RobustDPatch return patch only
            if isinstance(result, tuple):
                patch = result[0]  # Extract patch from tuple
                logger.info(f"Patch generated as tuple, extracted patch with shape: {patch.shape}")
            else:
                patch = result

            await sse_logger.success(f"패치 생성 완료: shape={patch.shape}")

            # Step 6: Save patch file
            await sse_logger.status("패치 파일 저장 중...")

            # Convert patch to image format
            # AdversarialPatchPyTorch returns CHW, DPatch/RobustDPatch return HWC
            if attack_method == "patch":
                # CHW → HWC
                patch_img = np.transpose(patch, (1, 2, 0))
            else:
                patch_img = patch

            # Clip and convert to uint8
            patch_img = np.clip(patch_img, 0, 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            patch_bgr = cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            patch_filename = f"{patch_name}_{timestamp}.png"
            patch_path = self.patches_dir / patch_filename

            # Save
            cv2.imwrite(str(patch_path), patch_bgr)
            file_size = patch_path.stat().st_size

            await sse_logger.info(f"패치 저장 완료: {patch_filename}")

            # Step 7: Create Patch2D record
            await sse_logger.status("DB 레코드 생성 중...")

            patch_record = await crud.patch_2d.create(
                db,
                obj_in=schemas.Patch2DCreate(
                    name=patch_name,
                    description=f"{attack_method.upper()} patch targeting '{target_class}'",
                    target_model_id=model_id,
                    source_dataset_id=source_dataset_id,
                    target_class=target_class,
                    method=attack_method,
                    hyperparameters={
                        "patch_size": patch_size,
                        "learning_rate": learning_rate,
                        "iterations": iterations,
                        "training_images": len(training_images),
                    },
                    patch_metadata={
                        "shape": list(patch.shape),
                        "attack_method": attack_method,
                    },
                    storage_key=str(patch_path.relative_to(self.storage_root)),
                    file_name=patch_filename,
                    size_bytes=file_size,
                ),
            )

            await db.commit()

            await sse_logger.success(
                f"패치 생성 완료: shape={patch.shape}",
                patch_id=str(patch_record.id),
                file_path=str(patch_path),
            )

            # Send complete event to close SSE stream
            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "complete",
                    "message": "패치 생성 완료!",
                    "patch_id": str(patch_record.id),
                    "file_path": str(patch_path),
                })

            return schemas.Patch2DResponse.model_validate(patch_record)

        except NotFoundError as e:
            # Resource not found (dataset, model, etc.)
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
                # Extract patch name from error if possible
                if "already exists" in error_str or "duplicate key" in error_str:
                    error_message = f"이미 존재하는 패치 이름입니다. 다른 이름을 사용해주세요."
                else:
                    error_message = "패치 이름이 중복되었습니다. 다른 이름을 사용해주세요."
            elif "foreign key" in error_str or "constraint" in error_str:
                error_message = "데이터베이스 제약 조건 위반: 참조된 리소스가 존재하지 않습니다."
            elif "connection" in error_str or "timeout" in error_str:
                error_message = "데이터베이스 연결 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            elif "cuda" in error_str or "gpu" in error_str or "device" in error_str:
                error_message = "GPU 오류가 발생했습니다. CPU 모드로 재시도하거나 시스템 관리자에게 문의하세요."
            elif "memory" in error_str or "out of memory" in error_str:
                error_message = "메모리 부족 오류가 발생했습니다. 이미지 수를 줄이거나 배치 크기를 줄여주세요."
            else:
                error_message = f"패치 생성 중 오류가 발생했습니다: {str(e)}"

            await sse_logger.error(error_message)
            logger.error(f"Error generating patch: {e}", exc_info=True)

            # Send error event to close SSE stream
            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "error",
                    "message": error_message,
                })
            raise

    async def _collect_target_images(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        target_class_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Collect images containing the target class.

        Returns:
            List of dicts with keys: id, file_name, image (numpy RGB), annotations
        """
        # Get all images from dataset
        images_db = await crud.image_2d.get_by_dataset(db, dataset_id=dataset_id)

        if not images_db:
            raise ValidationError(f"Dataset {dataset_id} has no images")

        target_images = []
        for img_record in images_db:
            # Get annotations for this image
            annotations_db = await crud.annotation.get_by_image(db, image_2d_id=img_record.id)

            # Check if any annotation has target_class_id
            has_target = False
            ann_list = []
            for ann in annotations_db:
                if ann.class_index is not None:
                    category_id = ann.class_index
                    if category_id == target_class_id:
                        has_target = True

                    # Get bbox (convert from normalized xywh to xyxy)
                    if ann.bbox_x is not None and ann.bbox_y is not None and ann.bbox_width is not None and ann.bbox_height is not None:
                        # bbox is stored as normalized (0-1) x_center, y_center, width, height
                        # Convert to [x1, y1, x2, y2] in normalized coordinates
                        x_center = float(ann.bbox_x)
                        y_center = float(ann.bbox_y)
                        width = float(ann.bbox_width)
                        height = float(ann.bbox_height)

                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2

                        ann_list.append({
                            "category_id": category_id,
                            "bbox": [x1, y1, x2, y2],  # [x1, y1, x2, y2] normalized
                        })

            if not has_target:
                continue

            # Load image
            img_path = self.storage_root / img_record.storage_key

            if not img_path.exists():
                logger.warning(f"Image file not found: {img_path}")
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            target_images.append({
                "id": img_record.id,
                "file_name": img_record.file_name,
                "image": img_rgb,  # (H, W, C) RGB uint8
                "annotations": ann_list,
            })

        return target_images

    async def _load_art_estimator(
        self,
        db: AsyncSession,
        model_id: UUID,
        channels_first: bool = True,
    ):
        """
        Load model from DB and create a real ART estimator (for attacks).

        Args:
            db: Database session
            model_id: Model ID
            channels_first: True for NCHW (AdversarialPatchPyTorch), False for NHWC (DPatch/RobustDPatch)

        Returns:
            Tuple of (ART-compatible estimator, input_size)
        """
        import torch
        from ultralytics import YOLO

        # Get model from DB
        model = await crud.od_model.get(db, id=model_id)
        if not model:
            raise ValidationError(f"Model {model_id} not found")

        # Get weights artifact
        if not model.artifacts:
            raise ValidationError(f"Model {model_id} has no artifacts")

        weights_artifact = next((a for a in model.artifacts if a.artifact_type == "weights"), None)
        if not weights_artifact:
            raise ValidationError(f"Model {model_id} has no weights artifact")

        # Get model path
        from pathlib import Path
        model_path = Path(weights_artifact.storage_key)
        if not model_path.exists():
            model_path = Path(weights_artifact.storage_path)
            if not model_path.exists():
                raise ValidationError(f"Model file not found: {model_path}")

        # Get class names from labelmap
        class_names = ["person"]  # Default
        if model.labelmap:
            class_names = [model.labelmap[str(i)] for i in sorted([int(k) for k in model.labelmap.keys()])]

        # Get input size
        input_size = [640, 640]  # Default
        if model.input_spec and "shape" in model.input_spec:
            input_size = model.input_spec["shape"][:2]

        # Load YOLO model using ultralytics
        yolo_model = YOLO(str(model_path))

        # Detect model name from path
        filename = str(model_path).lower()
        if 'yolo11' in filename or 'yolov11' in filename:
            model_name = 'yolov11'
        elif 'yolo10' in filename or 'yolov10' in filename:
            model_name = 'yolov10'
        elif 'yolo9' in filename or 'yolov9' in filename:
            model_name = 'yolov9'
        elif 'yolo8' in filename or 'yolov8' in filename:
            model_name = 'yolov8'
        else:
            model_name = 'yolov8'  # Default

        # Create ART PyTorchYolo estimator
        # This is the REAL ART estimator, not our custom one
        # is_ultralytics=True is REQUIRED for YOLOv8+
        # model_name is also required when using is_ultralytics=True
        # channels_first=True AND provide NCHW input (C, H, W format)

        # Detect available device (GPU significantly faster for patch generation)
        import torch
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # Move model to device for faster processing
        if device_type == "cuda":
            yolo_model.model.to("cuda")
            logger.info("Using CUDA for patch generation (faster)")
        else:
            logger.info("Using CPU for patch generation (slower)")

        # Configure input shape and channels based on attack method
        if channels_first:
            # AdversarialPatchPyTorch expects NCHW format
            input_shape = (3, *input_size)  # (C, H, W)
            logger.info(f"Creating estimator with channels_first=True (NCHW format)")
        else:
            # DPatch/RobustDPatch expect NHWC format
            input_shape = (*input_size, 3)  # (H, W, C)
            logger.info(f"Creating estimator with channels_first=False (NHWC format)")

        estimator = ARTPyTorchYolo(
            model=yolo_model.model,
            input_shape=input_shape,
            channels_first=channels_first,
            clip_values=(0, 255),
            attack_losses=("loss_total", "loss_cls", "loss_box", "loss_dfl"),  # Multiple losses like notebook
            device_type=device_type,  # Use GPU if available for 10-50x speedup
            is_ultralytics=True,  # REQUIRED for YOLOv8+
            model_name=model_name,  # Required with is_ultralytics
        )

        logger.info(f"ART estimator loaded: {type(estimator)}")
        return estimator, input_size


# Global instance
patch_service = PatchService()
