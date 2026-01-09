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
from app.utils.system_monitor import system_monitor

# ART imports - use real ART library for patch attacks
from app.ai.attacks.evasion import AdversarialPatchPyTorch, RobustDPatchPyTorch, DPatchPyTorch
from app.ai.estimators.object_detection import PyTorchYolo as ARTPyTorchYolo

logger = logging.getLogger(__name__)


def get_patch_intensity_preset(attack_method: str, intensity_level: str) -> dict:
    """
    패치 공격 강도 레벨에 따른 iterations와 patch_size 프리셋을 반환합니다.

    Args:
        attack_method: 공격 방법 ("patch", "dpatch", "robust_dpatch")
        intensity_level: 강도 레벨 ("weak", "medium", "strong")

    Returns:
        iterations, patch_size, description을 포함한 딕셔너리
    """
    presets = {
        "patch": {
            "weak": {"iterations": 100, "patch_size": 100, "description": "약한 패치 (은폐성 우선)"},
            "medium": {"iterations": 300, "patch_size": 150, "description": "보통 패치 (균형)"},
            "strong": {"iterations": 500, "patch_size": 200, "description": "강한 패치 (효과성 우선)"},
        },
        "dpatch": {
            "weak": {"iterations": 100, "patch_size": 100, "description": "약한 D-Patch (은폐성 우선)"},
            "medium": {"iterations": 300, "patch_size": 150, "description": "보통 D-Patch (균형)"},
            "strong": {"iterations": 500, "patch_size": 200, "description": "강한 D-Patch (효과성 우선)"},
        },
        "robust_dpatch": {
            "weak": {"iterations": 100, "patch_size": 100, "description": "약한 Robust D-Patch (은폐성 우선)"},
            "medium": {"iterations": 300, "patch_size": 150, "description": "보통 Robust D-Patch (균형)"},
            "strong": {"iterations": 500, "patch_size": 200, "description": "강한 Robust D-Patch (효과성 우선)"},
        },
        "naturalistic": {
            "weak": {"iterations": 100, "patch_size": 128, "description": "약한 자연스러운 패치 (높은 은폐성)"},
            "medium": {"iterations": 1000, "patch_size": 128, "description": "보통 자연스러운 패치 (균형)"},
            "strong": {"iterations": 2000, "patch_size": 128, "description": "강한 자연스러운 패치 (효과성 우선)"},
        },
    }

    method_presets = presets.get(attack_method, presets["patch"])
    return method_presets.get(intensity_level, method_presets["medium"])


def get_attack_config(attack_method: str, iterations: int) -> dict:
    """
    공격 타입별 최적의 학습률과 스케줄러 설정을 반환합니다.

    Args:
        attack_method: 공격 방법 ("patch", "dpatch", "robust_dpatch")
        iterations: 반복 횟수

    Returns:
        학습률, 스케줄러 타입, 스케줄러 파라미터를 포함한 딕셔너리
    """
    configs = {
        "patch": {
            # AdversarialPatchPyTorch: 다양한 변환에 강한 일반 패치
            "learning_rate": 5.0,
            "scheduler_type": "cosine",
            "scheduler_params": {
                "T_max": iterations,
                "eta_min": 0.5,  # 최소 학습률
            },
            "description": "일반 적대적 패치 (다양한 변환 지원)",
        },
        "dpatch": {
            # DPatchPyTorch: 객체 탐지 모델에 특화
            "learning_rate": 5.0,
            "scheduler_type": "step",
            "scheduler_params": {
                "step_size": max(1, iterations // 4),  # 25%마다 감소
                "gamma": 0.5,  # 절반으로 감소
            },
            "description": "객체 탐지 특화 패치",
        },
        "robust_dpatch": {
            # RobustDPatchPyTorch: EoT로 크롭/회전/밝기 변화에 강함
            "learning_rate": 3.0,  # EoT 노이즈 완화를 위해 낮춤
            "scheduler_type": "cosine",
            "scheduler_params": {
                "T_max": iterations,
                "eta_min": 0.3,  # 최소 학습률
            },
            "description": "강건한 패치 (크롭/회전/밝기 변화 대응)",
        },
        "naturalistic": {
            # NaturalisticPatchPyTorch: BigGAN 기반 자연스러운 패치
            "learning_rate": 0.02,  # NAP 논문 설정
            "scheduler_type": "constant",  # NAP는 스케줄링 사용 안함
            "scheduler_params": {},
            "description": "GAN 기반 자연스러운 패치",
        },
    }

    return configs.get(attack_method, configs["patch"])


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
        # Use 2D storage root for patches (patches are 2D-only)
        self.storage_root = Path(settings.STORAGE_2D_ROOT)
        self.storage_root_main = Path(settings.STORAGE_ROOT)  # For storage_key resolution
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
        patch_size: Optional[int],
        learning_rate: Optional[float],
        iterations: Optional[int],
        intensity_level: Optional[str] = None,
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
            patch_size: Size of the patch (height/width, auto-configured if None)
            learning_rate: Learning rate for optimization (None for auto-config)
            iterations: Number of optimization iterations (auto-configured if None)
            intensity_level: Attack intensity level: "weak", "medium", or "strong" (auto-configures patch_size and iterations)
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

        attack = None
        estimator = None
        x_train = None
        y_train = None

        try:
            # Validate attack method
            if attack_method not in ["patch", "dpatch", "robust_dpatch", "naturalistic"]:
                raise ValidationError(
                    f"Invalid attack method: {attack_method}. Must be 'patch', 'dpatch', 'robust_dpatch', or 'naturalistic'"
                )

            # Apply intensity level preset if provided
            if intensity_level is not None:
                if intensity_level not in ["weak", "medium", "strong"]:
                    raise ValidationError(
                        f"Invalid intensity_level: {intensity_level}. Must be 'weak', 'medium', or 'strong'"
                    )

                preset = get_patch_intensity_preset(attack_method, intensity_level)
                iterations = preset["iterations"]
                patch_size = preset["patch_size"]
                await sse_logger.info(
                    f"공격 강도: {intensity_level} - {preset['description']}"
                )
                await sse_logger.info(
                    f"자동 설정: 반복 {iterations}회, 패치 크기 {patch_size}px"
                )
            else:
                # Validate required parameters when intensity_level not provided
                if iterations is None:
                    raise ValidationError("iterations is required when intensity_level is not provided")
                if patch_size is None:
                    raise ValidationError("patch_size is required when intensity_level is not provided")
                await sse_logger.info(f"운용자 지정: 반복 {iterations}회, 패치 크기 {patch_size}px")

            # Get optimal configuration for attack method
            attack_config = get_attack_config(attack_method, iterations)

            # Use auto-configured learning rate if not provided
            if learning_rate is None:
                learning_rate = attack_config["learning_rate"]
                await sse_logger.info(
                    f"자동 학습률 설정: {learning_rate} ({attack_config['description']})"
                )
            else:
                await sse_logger.info(f"운용자 지정 학습률: {learning_rate}")

            scheduler_type = attack_config["scheduler_type"]
            scheduler_params = attack_config["scheduler_params"]
            await sse_logger.info(
                f"스케줄러: {scheduler_type} (공격 특성에 최적화)"
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
            # All PyTorch-based attacks use NCHW format internally
            channels_first = True  # Always use NCHW for PyTorch models
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

            # Ensure patch size fits into the resized model input
            patch_size_px = min(patch_size, model_height, model_width)
            if patch_size_px != patch_size:
                await sse_logger.warning(
                    f"요청한 패치 크기({patch_size}px)가 입력 크기 {model_width}x{model_height}을 초과하여 "
                    f"{patch_size_px}px로 조정합니다."
                )

            # Step 3: Prepare training data (resize all images to model input size)
            await sse_logger.status("이미지 전처리 중...")
            x_train_list_chw = []  # All PyTorch attacks use CHW format

            for img_data in training_images:
                img = img_data["image"]  # (H, W, C) RGB

                # Resize to model input size if needed (following notebook and noise_attack_service approach)
                if img.shape[0] != model_height or img.shape[1] != model_width:
                    from PIL import Image
                    img_pil = Image.fromarray(img.astype(np.uint8))
                    img_pil = img_pil.resize((model_width, model_height), Image.BICUBIC)
                    img = np.array(img_pil).astype(np.float32)

                # Convert to CHW format for PyTorch
                x_train_list_chw.append(img.transpose(2, 0, 1))  # (C, H, W)

            # Prepare data in NCHW format for all PyTorch-based attacks
            x_train = np.ascontiguousarray(np.stack(x_train_list_chw, axis=0).astype(np.float32))  # (N, C, H, W)

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

            # Step 4: Calculate optimal batch size based on GPU memory
            # Use conservative settings to prevent OOM
            optimal_batch_size = system_monitor.get_optimal_batch_size(
                image_size=(model_height, model_width),
                model_type="patch",  # Very memory intensive (×15 multiplier)
                # Use default safety_margin=0.7 and default_batch_size=32
            )
            await sse_logger.info(f"GPU 메모리 기반 최적 배치 사이즈: {optimal_batch_size}")

            # Step 5: Create ART patch attack
            # Note: Attack loss (AdversarialPatchAttackLoss) is automatically configured
            # by each attack class's __init__() method via _configure_attack_loss()
            await sse_logger.status(f"{attack_method.upper()} 패치 생성 중...")

            if attack_method == "patch":
                # AdversarialPatchPyTorch - uses AdversarialPatchAttackLoss (MaxProbExtractor)
                attack = AdversarialPatchPyTorch(
                    estimator=estimator,
                    rotation_max=22.5,
                    scale_min=0.2,
                    scale_max=0.4,
                    learning_rate=learning_rate,
                    max_iter=iterations,
                    batch_size=optimal_batch_size,  # Dynamic batch size based on GPU memory
                    patch_shape=(3, patch_size_px, patch_size_px),  # CHW
                    patch_type="square",
                    optimizer="Adam",
                    targeted=False,  # Untargeted: evade detection (from notebook)
                    verbose=True,  # Enable verbose logging to see iteration progress
                    scheduler_type=scheduler_type,  # Auto-configured scheduler
                    scheduler_params=scheduler_params,
                )
            elif attack_method == "dpatch":
                # DPatch (does not support 'targeted' parameter)
                attack = DPatchPyTorch(
                    estimator=estimator,
                    patch_shape=(patch_size_px, patch_size_px, 3),  # HWC
                    learning_rate=learning_rate,
                    max_iter=iterations,
                    batch_size=optimal_batch_size,  # Dynamic batch size based on GPU memory
                    verbose=True,  # Enable verbose logging
                    scheduler_type=scheduler_type,  # Auto-configured scheduler
                    scheduler_params=scheduler_params,
                )
            elif attack_method == "robust_dpatch":
                # RobustDPatch with targeted attack on specific class
                patch_location_center = (
                    max(0, (model_height - patch_size_px) // 2),
                    max(0, (model_width - patch_size_px) // 2),
                )
                attack = RobustDPatchPyTorch(
                    estimator=estimator,
                    patch_shape=(patch_size_px, patch_size_px, 3),  # HWC
                    patch_location=patch_location_center,
                    learning_rate=learning_rate,
                    max_iter=iterations,
                    batch_size=optimal_batch_size,  # Dynamic batch size based on GPU memory
                    sample_size=5,  # EOT samples for robustness
                    targeted=True,  # Enable targeted attack on specific class
                    verbose=True,  # Enable verbose logging
                    scheduler_type=scheduler_type,  # Auto-configured scheduler
                    scheduler_params=scheduler_params,
                )
            else:  # naturalistic
                # Naturalistic patch using BigGAN
                from app.ai.attacks.evasion import NaturalisticPatchPyTorch

                # Deformator is required for naturalistic patches
                # Use storage path (works both in container and host with volume mount)
                deformator_path = Path("/storage/gan_weights/deformators/BigGAN/models/deformator_0.pt")

                if not deformator_path.exists():
                    error_msg = (
                        f"BigGAN deformator가 필요합니다: {deformator_path}\n"
                        f"파일이 존재하지 않습니다. GAN 가중치를 다운로드하세요."
                    )
                    await sse_logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

                enable_deformator = True
                await sse_logger.info(f"BigGAN deformator 사용: {deformator_path}")

                attack = NaturalisticPatchPyTorch(
                    estimator=estimator,
                    gan_class_id=259,  # Default to Pomeranian (can be parameterized later)
                    tv_weight=0.0,  # Start with no TV loss for better attack effectiveness
                    alpha_latent=1.0,  # Use full latent optimization
                    patch_scale=0.2,  # 20% of person bbox
                    learning_rate=learning_rate,
                    max_iter=iterations,
                    batch_size=optimal_batch_size,
                    max_latent_value=8.0,
                    enable_deformator=enable_deformator,
                    deformator_path=str(deformator_path) if deformator_path else None,
                    enable_shift_deformator=True,
                    enable_latent_rounding=True,
                    enable_appearance_aug=True,
                    summary_writer=False,  # Disable TensorBoard (use tqdm for progress instead)
                    verbose=True,
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
            # AdversarialPatchPyTorch returns (patch, mask) tuple, others return patch only
            # All patches are in CHW format
            if isinstance(result, tuple):
                patch = result[0]  # Extract patch from tuple
                logger.info(f"Patch generated as tuple, extracted patch with shape: {patch.shape}")
            else:
                patch = result

            await sse_logger.success(f"패치 생성 완료: shape={patch.shape}")

            # Step 6: Save patch file
            await sse_logger.status("패치 파일 저장 중...")

            # Convert patch to image format (handle CHW or HWC)
            if patch.ndim != 3:
                raise ValueError(f"Unexpected patch shape: {patch.shape}")

            if patch.shape[0] in (1, 3) and patch.shape[2] not in (1, 3):
                patch_img = np.transpose(patch, (1, 2, 0))  # CHW → HWC
            elif patch.shape[2] in (1, 3):
                patch_img = patch  # already HWC
            else:
                patch_img = np.transpose(patch, (1, 2, 0))  # fallback to CHW → HWC

            # Clip and convert to uint8 (auto-scale if patch is in 0-1 range)
            patch_img = patch_img.astype(np.float32)
            if patch_img.max() <= 1.5:
                patch_img = patch_img * 255.0
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
                        "patch_size": patch_size_px,
                        "requested_patch_size": patch_size,
                        "learning_rate": learning_rate,
                        "iterations": iterations,
                        "training_images": len(training_images),
                    },
                    patch_metadata={
                        "shape": list(patch.shape),
                        "attack_method": attack_method,
                    },
                    storage_key=str(patch_path.relative_to(self.storage_root_main)),
                    file_name=patch_filename,
                    size_bytes=file_size,
                ),
            )

            await db.commit()

            await sse_logger.success(
                f"패치 생성 완료: shape={patch.shape}",
                patch_id=str(patch_record.id),
                file_path=str(patch_path),
                storage_key=str(patch_path.relative_to(self.storage_root_main)),
            )

            # Send complete event to close SSE stream
            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "complete",
                    "message": "패치 생성 완료!",
                    "patch_id": str(patch_record.id),
                    "file_path": str(patch_path),
                    "storage_key": str(patch_path.relative_to(self.storage_root_main)),
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
        finally:
            try:
                import gc
                import torch

                attack = None
                estimator = None
                x_train = None
                y_train = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup GPU memory: {cleanup_error}")

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
            img_path = self.storage_root_main / img_record.storage_key

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
            channels_first: True for NCHW (all PyTorch attacks)

        Returns:
            Tuple of (ART-compatible estimator, input_size)
        """
        import torch
        from ultralytics import YOLO, RTDETR
        from app.ai.estimators.object_detection import PyTorchRTDETR

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

        # Detect model type from database metadata
        model_type = None
        if (model.inference_params and
            "metadata" in model.inference_params and
            "yaml_config" in model.inference_params["metadata"] and
            "model_type" in model.inference_params["metadata"]["yaml_config"]):
            model_type = model.inference_params["metadata"]["yaml_config"]["model_type"]
            logger.info(f"Detected model type from yaml_config: {model_type}")
        elif model.inference_params and "estimator_type" in model.inference_params:
            model_type = model.inference_params["estimator_type"]
            logger.info(f"Detected model type from estimator_type: {model_type}")

        # Fallback: detect from filename
        if not model_type:
            filename_lower = str(model_path).lower()
            if 'rtdetr' in filename_lower or 'rt-detr' in filename_lower:
                model_type = 'rtdetr'
            elif 'efficientdet' in filename_lower or 'effdet' in filename_lower:
                model_type = 'efficientdet'
            else:
                model_type = 'yolo'
            logger.info(f"Detected model type from filename: {model_type}")

        # Normalize model type
        model_type = model_type.lower()
        is_rtdetr = model_type in ['rtdetr', 'rt-detr']
        is_efficientdet = 'efficientdet' in model_type

        # Detect available device (GPU significantly faster for patch generation)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # Configure input shape for PyTorch (always NCHW)
        if channels_first:
            # All PyTorch-based attacks use NCHW format
            input_shape = (3, *input_size)  # (C, H, W)
            logger.info(f"Creating estimator with channels_first=True (NCHW format)")
        else:
            # Legacy NHWC format (not used for PyTorch attacks)
            input_shape = (*input_size, 3)  # (H, W, C)
            logger.info(f"Creating estimator with channels_first=False (NHWC format)")

        if is_rtdetr:
            # Load RT-DETR model
            logger.info("Loading RT-DETR model for adversarial patch generation")
            rtdetr_model = RTDETR(str(model_path))

            # Move model to device for faster processing
            if device_type == "cuda":
                rtdetr_model.model.to("cuda")
                logger.info("Using CUDA for patch generation (faster)")
            else:
                logger.info("Using CPU for patch generation (slower)")

            # Create ART PyTorchRTDETR estimator
            estimator = PyTorchRTDETR(
                model=rtdetr_model.model,
                input_shape=input_shape,
                channels_first=channels_first,
                clip_values=(0, 255),
                attack_losses=("loss_total", "loss_giou", "loss_cls", "loss_l1"),
                device_type=device_type,
                is_ultralytics=True,
            )
            logger.info(f"RT-DETR ART estimator loaded: {type(estimator)}")

        elif is_efficientdet:
            # Load EfficientDet model
            logger.info("Loading EfficientDet model for adversarial patch generation")
            from app.ai.estimators.object_detection.model_factory import model_factory

            factory_device_type = "gpu" if device_type == "cuda" else "cpu"
            estimator = model_factory.load_model(
                model_path=str(model_path),
                model_type="efficientdet",
                class_names=class_names,
                input_size=input_size,
                device_type=factory_device_type,
                clip_values=(0, 255),
            )

            # Align input_size with estimator configuration (may come from MMDetection config)
            if estimator.input_shape and len(estimator.input_shape) >= 3:
                if channels_first:
                    input_size = [estimator.input_shape[1], estimator.input_shape[2]]
                else:
                    input_size = [estimator.input_shape[0], estimator.input_shape[1]]

            logger.info(f"EfficientDet ART estimator loaded: {type(estimator)}")

        else:
            # Load YOLO model
            logger.info("Loading YOLO model for adversarial patch generation")
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

            # Move model to device for faster processing
            if device_type == "cuda":
                yolo_model.model.to("cuda")
                logger.info("Using CUDA for patch generation (faster)")
            else:
                logger.info("Using CPU for patch generation (slower)")

            # Create ART PyTorchYolo estimator
            estimator = ARTPyTorchYolo(
                model=yolo_model.model,
                input_shape=input_shape,
                channels_first=channels_first,
                clip_values=(0, 255),
                attack_losses=("loss_total", "loss_cls", "loss_box", "loss_dfl"),
                device_type=device_type,
                is_ultralytics=True,
                model_name=model_name,
            )
            logger.info(f"YOLO ART estimator loaded: {type(estimator)}")

        return estimator, input_size


# Global instance
patch_service = PatchService()
