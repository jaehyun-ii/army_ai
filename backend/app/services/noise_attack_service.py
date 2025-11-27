"""
Noise attack service for creating FGSM/PGD attacked datasets.

This service implements single-step workflow:
    base_dataset → apply noise → attacked_dataset
"""
from __future__ import annotations

import logging
import numpy as np
import cv2
import torch
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
from app.services.estimator_loader_service import EstimatorLoaderService
from app.services.model_inference_service import model_inference_service

# All attacks from app.ai (custom fork of ART)
from app.ai.attacks.evasion import (
    # PyTorch versions (primary, feature-complete implementations)
    UniversalNoiseAttackPyTorch,
    NoiseOSFDPyTorch,
)
from app.ai.attacks.evasion.fast_gradient_pytorch import FastGradientMethodPyTorch
from app.ai.attacks.evasion.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch

logger = logging.getLogger(__name__)

# Shared SSE manager instance for all attack services
from app.services.patch_attack_service import _shared_sse_manager


class NoiseAttackService:
    """
    Service for creating noise-based attacked datasets (FGSM, PGD).

    Workflow:
        1. Load base_dataset images
        2. Load model as estimator
        3. Apply ART noise attack (FastGradientMethod or ProjectedGradientDescent)
        4. Save attacked images to new dataset
        5. Create AttackDataset2D record
    """

    def __init__(self):
        self.storage_root = Path(settings.STORAGE_ROOT)
        self.attack_datasets_dir = self.storage_root / "attack_datasets"
        self.attack_datasets_dir.mkdir(parents=True, exist_ok=True)

        self.estimator_loader = EstimatorLoaderService()
        # Use shared SSE manager
        self.sse_manager = _shared_sse_manager

    async def create_noise_attack_dataset(
        self,
        db: AsyncSession,
        attack_name: str,
        attack_method: str,
        base_dataset_id: UUID,
        model_id: UUID,
        epsilon: float,
        alpha: Optional[float] = None,
        iterations: Optional[int] = None,
        target_class: Optional[str] = None,
        session_id: Optional[str] = None,
        current_user_id: Optional[UUID] = None,
    ) -> Tuple[schemas.AttackDataset2DResponse, UUID]:
        """
        Create noise-based attacked dataset.

        Args:
            db: Database session
            attack_name: Name for the attacked dataset
            attack_method: "fgsm", "pgd", "universal_noise", or "noise_osfd"
            base_dataset_id: Source dataset to attack
            model_id: Target model for attack
            epsilon: Maximum perturbation (in [0, 255] scale)
            alpha: Step size for PGD (in [0, 255] scale, optional)
            iterations: Number of iterations for PGD (optional)
            target_class: Target class name to attack (e.g., 'person'). If None, attack all objects.
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
            # Validate attack method
            valid_methods = ["fgsm", "pgd", "universal_noise", "noise_osfd"]
            if attack_method not in valid_methods:
                raise ValidationError(
                    f"Invalid attack method: {attack_method}. "
                    f"Must be one of: {', '.join(valid_methods)}"
                )

            # Validate PGD parameters
            if attack_method == "pgd":
                if alpha is None:
                    alpha = epsilon / 4  # Default: 1/4 of epsilon
                if iterations is None:
                    iterations = 10  # Default iterations

            await sse_logger.status("공격 데이터셋 생성 시작...")

            # Step 1: Load base dataset
            await sse_logger.status("베이스 데이터셋 로딩 중...")
            base_dataset = await crud.dataset_2d.get(db, id=base_dataset_id)
            if not base_dataset:
                raise NotFoundError(f"Dataset {base_dataset_id} not found")

            images = await self._load_dataset_images(db, base_dataset_id)
            await sse_logger.info(f"이미지 로드 완료: {len(images)}개")

            # Step 1.5: Get target class ID if target_class is specified
            target_class_id = None
            if target_class:
                await sse_logger.status(f"타겟 클래스 '{target_class}' ID 확인 중...")
                # Get model to access labelmap
                model = await crud.od_model.get(db, id=model_id)
                if not model:
                    raise NotFoundError(f"Model {model_id} not found")

                # Find class ID from labelmap
                if model.labelmap:
                    for class_id_str, class_name in model.labelmap.items():
                        if class_name.lower() == target_class.lower():
                            target_class_id = int(class_id_str)
                            break

                if target_class_id is None:
                    raise ValidationError(
                        f"Target class '{target_class}' not found in model labelmap. "
                        f"Available classes: {list(model.labelmap.values()) if model.labelmap else []}"
                    )

                await sse_logger.info(f"타겟 클래스: {target_class} (ID: {target_class_id})")

                # Load annotations for target class filtering
                await sse_logger.status("어노테이션 로딩 중...")
                annotations = await self._load_dataset_annotations(db, base_dataset_id)
                await sse_logger.info(f"어노테이션 로드 완료: {len(annotations)}개")
            else:
                annotations = []
                await sse_logger.info("타겟 클래스 미지정: 전체 이미지 공격")

            # Step 2: Load model as estimator (use custom PyTorchYolo from app.ai)
            await sse_logger.status("모델 로딩 중...")
            estimator, input_size = await self._load_custom_estimator(db, model_id)
            await sse_logger.info("모델 로딩 완료")

            # Step 3: Create ART attack object
            await sse_logger.status(f"{attack_method.upper()} 공격 객체 생성 중...")

            # Normalize epsilon to [0, 1] scale (ART expects clip_values=(0, 255))
            # Since estimator has clip_values=(0, 255), eps should be in [0, 255]
            eps_normalized = epsilon  # Keep in [0, 255] scale

            if attack_method == "fgsm":
                attack = FastGradientMethodPyTorch(
                    estimator=estimator,
                    norm=np.inf,
                    eps=eps_normalized,
                    targeted=False,
                    batch_size=16,  # Process images in batches for better performance
                )
                await sse_logger.info(f"FGSM (PyTorch) 생성: epsilon={epsilon}, batch_size=16")

            elif attack_method == "pgd":
                # Normalize alpha as well
                alpha_normalized = alpha

                attack = ProjectedGradientDescentPyTorch(
                    estimator=estimator,
                    norm=np.inf,
                    eps=eps_normalized,
                    eps_step=alpha_normalized,
                    max_iter=iterations,
                    targeted=False,
                    num_random_init=0,
                    batch_size=16,  # Process images in batches for better performance
                    verbose=False,
                )
                await sse_logger.info(f"PGD 생성: epsilon={epsilon}, alpha={alpha}, iterations={iterations}, batch_size=16")

            elif attack_method == "universal_noise":
                # Universal Noise Attack - PyTorch version with pseudo-GT (feature-complete)
                if alpha is None:
                    alpha = epsilon / 10  # Default step size
                if iterations is None:
                    iterations = 50  # Default iterations for training

                # Use target_class_id if specified, otherwise default to 0 (person in COCO)
                universal_target_class_id = target_class_id if target_class_id is not None else 0

                attack = UniversalNoiseAttackPyTorch(
                    estimator=estimator,
                    eps=eps_normalized / 255.0,
                    eps_step=alpha / 255.0,
                    max_iter=iterations,
                    batch_size=4,
                    apply_mask=True,  # Apply perturbation to detected regions
                    target_class_id=universal_target_class_id,
                    verbose=True,
                )
                await sse_logger.info(
                    f"Universal Noise Attack 생성 (PyTorch): epsilon={epsilon}, "
                    f"step_size={alpha}, iterations={iterations}, "
                    f"target_class_id={universal_target_class_id}, pseudo-GT enabled"
                )

            elif attack_method == "noise_osfd":
                # Noise OSFD Attack - PyTorch version with full features (RRB augmentation)
                if alpha is None:
                    alpha = epsilon / 10
                if iterations is None:
                    iterations = 30

                # Get feature layer indices (can be customized)
                feature_layers = [10, 15, 20]  # Default YOLO backbone layers
                amplification_factor = 10.0  # Default amplification

                attack = NoiseOSFDPyTorch(
                    estimator=estimator,
                    eps=eps_normalized / 255.0,
                    eps_step=alpha / 255.0,
                    max_iter=iterations,
                    batch_size=4,
                    feature_layer_indices=feature_layers,
                    amplification_factor=amplification_factor,
                    apply_augmentation=True,
                    verbose=True,
                )
                await sse_logger.info(
                    f"Noise OSFD Attack 생성 (PyTorch): epsilon={epsilon}, "
                    f"step_size={alpha}, iterations={iterations}, "
                    f"layers={feature_layers}, RRB augmentation enabled"
                )

            # Step 4: Apply attack to all images
            await sse_logger.status("이미지에 공격 적용 중...")
            attacked_images = []
            failed_count = 0

            # Use thread pool to avoid blocking event loop during attack generation
            import asyncio
            import concurrent.futures

            loop = asyncio.get_event_loop()

            # For Universal Noise and OSFD, train perturbation once on all images
            universal_perturbation = None
            if attack_method in ["universal_noise", "noise_osfd"]:
                await sse_logger.status(f"{attack_method.upper()}: 범용 perturbation 학습 중...")

                # Select training images
                if target_class_id is not None and annotations:
                    # Filter images that contain target class
                    await sse_logger.info(f"타겟 클래스 {target_class}가 포함된 이미지 선택 중...")
                    images_with_target = []
                    for img_data in images:
                        # Check if this image has target class annotations
                        img_anns = [ann for ann in annotations if ann["image_id"] == img_data["id"]]
                        target_anns = [ann for ann in img_anns if ann.get("category_id") == target_class_id]
                        if target_anns:
                            images_with_target.append(img_data)

                    if not images_with_target:
                        raise ValidationError(
                            f"타겟 클래스 '{target_class}'가 포함된 이미지가 없습니다. "
                            f"다른 클래스를 선택하거나 타겟 클래스 없이 전체 공격을 수행하세요."
                        )

                    # Use up to 50 images with target class
                    training_images = images_with_target[:min(len(images_with_target), 50)]
                    await sse_logger.info(
                        f"타겟 클래스 포함 이미지: {len(images_with_target)}개 중 {len(training_images)}개 선택"
                    )
                else:
                    # No target class specified, use first 50 images
                    training_images = images[:min(len(images), 50)]
                    await sse_logger.info(f"전체 이미지 중 {len(training_images)}개로 학습")

                # Prepare all images for batch training
                all_images_batch = []
                for img_data in training_images:
                    img = img_data["image"]
                    model_height, model_width = input_size[0], input_size[1]
                    if img.shape[0] != model_height or img.shape[1] != model_width:
                        from PIL import Image
                        img_pil = Image.fromarray(img.astype(np.uint8))
                        img_pil = img_pil.resize((model_width, model_height), Image.BICUBIC)
                        img = np.array(img_pil).astype(np.float32)
                    x = img.transpose(2, 0, 1).astype(np.float32)
                    all_images_batch.append(x)

                x_batch = np.array(all_images_batch)  # (N, C, H, W)

                # Prepare y batch if target_class is specified
                y_batch = None
                if target_class_id is not None and annotations:
                    await sse_logger.info(f"타겟 클래스 {target_class_id}의 annotations 준비 중...")
                    y_batch = []
                    for img_data in training_images:
                        # Get original and resized sizes
                        original_h = img_data.get("original_height", img_data["image"].shape[0])
                        original_w = img_data.get("original_width", img_data["image"].shape[1])
                        model_height, model_width = input_size[0], input_size[1]

                        # Adjust bbox if image was/will be resized
                        needs_resize = (original_h != model_height or original_w != model_width)

                        y_img = self._convert_annotations_to_art_format(
                            annotations,
                            img_data["id"],
                            target_class_id,
                            original_size=(original_h, original_w) if needs_resize else None,
                            resized_size=(model_height, model_width) if needs_resize else None,
                        )
                        y_batch.append(y_img if y_img else {"boxes": np.array([]), "labels": np.array([]), "scores": np.array([])})
                    await sse_logger.info(f"타겟 클래스 annotations 준비 완료")

                # Train universal perturbation
                def train_universal():
                    result = attack.generate(x=x_batch, y=y_batch)
                    if isinstance(result, tuple):
                        return result[1]  # Return perturbation
                    return None

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    universal_perturbation = await loop.run_in_executor(executor, train_universal)

                await sse_logger.info(f"범용 perturbation 학습 완료")

            # For FGSM/PGD, process all images in batch mode
            if attack_method in ["fgsm", "pgd"]:
                await sse_logger.status(f"{attack_method.upper()} 공격을 배치 모드로 적용 중...")

                # Prepare all images as batch
                x_batch_list = []
                y_batch_list = []
                resize_info = []  # Track resize info for each image

                for img_data in images:
                    img = img_data["image"]
                    original_height = img_data.get("original_height", img.shape[0])
                    original_width = img_data.get("original_width", img.shape[1])

                    # Resize to model input size if needed
                    model_height, model_width = input_size[0], input_size[1]
                    resized = False
                    if img.shape[0] != model_height or img.shape[1] != model_width:
                        from PIL import Image
                        img_pil = Image.fromarray(img.astype(np.uint8))
                        img_pil = img_pil.resize((model_width, model_height), Image.BICUBIC)
                        img = np.array(img_pil).astype(np.float32)
                        resized = True

                    # Convert to CHW format
                    x = img.transpose(2, 0, 1).astype(np.float32)  # (H, W, C) -> (C, H, W)
                    x_batch_list.append(x)

                    # Prepare target labels if specified
                    y_target = None
                    if target_class_id is not None and annotations:
                        y_target = self._convert_annotations_to_art_format(
                            annotations,
                            img_data["id"],
                            target_class_id,
                            original_size=(original_height, original_width) if resized else None,
                            resized_size=(model_height, model_width) if resized else None,
                        )
                    y_batch_list.append(y_target if y_target else {"boxes": np.array([]), "labels": np.array([]), "scores": np.array([])})

                    resize_info.append({
                        "resized": resized,
                        "original_height": original_height,
                        "original_width": original_width
                    })

                # Stack into batch (N, C, H, W)
                x_batch = np.array(x_batch_list)

                # Generate adversarial examples for entire batch
                def generate_batch():
                    y_arg = y_batch_list if target_class_id is not None else None
                    result = attack.generate(x=x_batch, y=y_arg)
                    if isinstance(result, tuple):
                        return result[0]
                    return result

                await sse_logger.progress(
                    f"{attack_method.upper()} 배치 공격 생성 중... (0/{len(images)})",
                    processed=0,
                    total=len(images),
                    successful=0,
                    failed=0
                )

                # Execute attack in background thread (single execution for all images)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    x_adv_batch = await loop.run_in_executor(executor, generate_batch)

                await sse_logger.info(f"{attack_method.upper()} 배치 공격 완료")

                # Process results
                for idx, (x_adv, img_data, resize_info_item) in enumerate(zip(x_adv_batch, images, resize_info)):
                    try:
                        await sse_logger.progress(
                            f"결과 처리 중... ({idx + 1}/{len(images)})",
                            processed=idx + 1,
                            total=len(images),
                            successful=len(attacked_images),
                            failed=failed_count
                        )

                        # Convert back to HWC format
                        # x_adv is (C, H, W) from batch
                        adv_img = x_adv.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)

                        attacked_images.append({
                            "image": adv_img,
                            "original_file_name": img_data["file_name"],
                            "original_id": img_data["id"],
                        })

                        # Send progress update AFTER completing the image
                        await sse_logger.progress(
                            f"노이즈 공격 완료... ({idx + 1}/{len(images)})",
                            processed=idx + 1,
                            total=len(images),
                            successful=len(attacked_images),
                            failed=failed_count,
                        )

                    except Exception as e:
                        logger.error(f"Failed to attack image {idx} in batch: {e}", exc_info=True)
                        failed_count += 1
                        await sse_logger.warning(f"이미지 {idx} 공격 실패: {str(e)}")

            # For Universal Noise/OSFD, apply pre-trained perturbation to each image in a batch
            elif attack_method in ["universal_noise", "noise_osfd"] and universal_perturbation is not None:
                await sse_logger.status(f"{attack_method.upper()}: 학습된 perturbation을 이미지에 적용 중...")

                # Prepare all images for batch application
                x_batch_list = []
                for img_data in images:
                    img = img_data["image"]
                    # Resize to model input size if needed
                    model_height, model_width = input_size[0], input_size[1]
                    if img.shape[0] != model_height or img.shape[1] != model_width:
                        from PIL import Image
                        img_pil = Image.fromarray(img.astype(np.uint8))
                        img_pil = img_pil.resize((model_width, model_height), Image.BICUBIC)
                        img = np.array(img_pil).astype(np.float32)
                    
                    # Convert to CHW format
                    x = img.transpose(2, 0, 1).astype(np.float32)
                    x_batch_list.append(x)
                
                x_batch = np.array(x_batch_list)

                # Apply universal perturbation to the entire batch via broadcasting
                x_adv_batch = x_batch + universal_perturbation
                if estimator.clip_values is not None:
                    x_adv_batch = np.clip(x_adv_batch, estimator.clip_values[0], estimator.clip_values[1])

                await sse_logger.info(f"Perturbation applied to batch of {len(x_adv_batch)} images.")

                # Process the results
                for idx, (x_adv, img_data) in enumerate(zip(x_adv_batch, images)):
                    try:
                        # Convert back to HWC
                        adv_img = x_adv.transpose(1, 2, 0)
                        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)

                        attacked_images.append({
                            "image": adv_img,
                            "original_file_name": img_data["file_name"],
                            "original_id": img_data["id"],
                        })

                        await sse_logger.progress(
                            f"결과 처리 중... ({idx + 1}/{len(images)})",
                            processed=idx + 1,
                            total=len(images),
                            successful=len(attacked_images),
                            failed=failed_count
                        )
                    except Exception as e:
                        logger.error(f"Failed to process perturbed image {idx}: {e}", exc_info=True)
                        failed_count += 1
                        await sse_logger.warning(f"이미지 {idx} 처리 실패: {str(e)}")

            if not attacked_images:
                raise ValidationError("All images failed to be attacked")

            await sse_logger.info(f"공격 완료: 성공 {len(attacked_images)}, 실패 {failed_count}")

            # Step 5: Create output dataset
            await sse_logger.status("공격된 이미지 저장 중...")

            output_dataset_name = f"{attack_name}_output"
            output_dataset_path = self.attack_datasets_dir / f"{attack_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dataset_path.mkdir(parents=True, exist_ok=True)

            output_dataset = await crud.dataset_2d.create(
                db,
                obj_in=schemas.Dataset2DCreate(
                    name=output_dataset_name,
                    description=f"Output dataset from {attack_method.upper()} attack",
                    storage_path=str(output_dataset_path.relative_to(self.storage_root)),
                ),
                owner_id=current_user_id,
            )

            # Save images and create image records
            for img_data in attacked_images:
                # Save image
                file_name = f"adv_{img_data['original_file_name']}"
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
                    ),
                    owner_id=current_user_id,
                )

            await db.commit()
            await sse_logger.info(f"저장 완료: {len(attacked_images)}개 이미지")

            # Step 6: Create AttackDataset2D record
            await sse_logger.status("공격 데이터셋 레코드 생성 중...")

            attack_dataset = await crud.attack_dataset_2d.create(
                db,
                obj_in=schemas.AttackDataset2DCreate(
                    name=attack_name,
                    description=f"{attack_method.upper()} attack with epsilon={epsilon}"
                                + (f", target_class={target_class}" if target_class else ""),
                    attack_type=AttackType.NOISE,
                    target_model_id=model_id,
                    base_dataset_id=base_dataset_id,
                    output_dataset_id=output_dataset.id,  # NEW: Use dedicated column
                    target_class=target_class,  # Store target class if specified
                    patch_id=None,
                    parameters={
                        "attack_method": attack_method,
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "iterations": iterations,
                        "target_class_id": target_class_id,  # Store class ID for reference
                        "processed_images": len(attacked_images),
                        "failed_images": failed_count,
                        "storage_path": str(output_dataset_path),
                        # Note: output_dataset_id moved to dedicated column for better schema design
                    },
                ),
                owner_id=current_user_id,
            )

            await db.commit()

            # Note: No need to cleanup estimator since we created it directly with ART, not via model_inference_service

            await sse_logger.success(
                "공격 데이터셋 생성 완료!",
                attack_dataset_id=str(attack_dataset.id),
                output_dataset_id=str(output_dataset.id),
                processed=len(attacked_images),
                failed=failed_count,
            )

            # Send complete event to close SSE stream
            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "complete",
                    "message": "공격 데이터셋 생성 완료!",
                    "attack_dataset_id": str(attack_dataset.id),
                    "output_dataset_id": str(output_dataset.id),
                    "processed": len(attacked_images),
                    "failed": failed_count,
                })

            return schemas.AttackDataset2DResponse.model_validate(attack_dataset), output_dataset.id

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
                error_message = f"공격 데이터셋 생성 중 오류가 발생했습니다: {str(e)}"

            await sse_logger.error(error_message)
            logger.error(f"Error creating noise attack dataset: {e}", exc_info=True)

            # Send error event to close SSE stream
            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "error",
                    "message": error_message,
                })
            raise

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

            # Store original size for bbox adjustment
            height, width = img_rgb.shape[:2]

            loaded_images.append({
                "id": img_record.id,
                "file_name": img_record.file_name,
                "image": img_rgb,  # (H, W, C) RGB uint8
                "original_height": height,
                "original_width": width,
            })

        return loaded_images

    async def _load_art_estimator(
        self,
        db: AsyncSession,
        model_id: UUID,
    ):
        """
        Load model from DB and create a real ART estimator (for attacks).

        Returns:
            ART-compatible estimator (PyTorchYolo)
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
        estimator = ARTPyTorchYolo(
            model=yolo_model.model,
            input_shape=(3, *input_size),  # (C, H, W)
            channels_first=True,  # PyTorch uses NCHW format
            clip_values=(0, 255),
            attack_losses=("loss_total",),
            device_type="cpu",  # Use CPU for consistency
            is_ultralytics=True,  # REQUIRED for YOLOv8+
            model_name=model_name,  # Required with is_ultralytics
        )

        logger.info(f"ART estimator loaded: {type(estimator)}")
        return estimator, input_size

    async def _load_custom_estimator(
        self,
        db: AsyncSession,
        model_id: UUID,
    ):
        """
        Load model from DB and create a custom estimator (for new attacks).
        Uses model_factory to support all model types (YOLO, EfficientDet, etc.).

        Returns:
            Custom PyTorchObjectDetector estimator and input_size
        """
        from pathlib import Path
        from app.ai.estimators.object_detection.model_factory import model_factory
        from app import schemas

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

        # Get estimator type from inference_params
        if not model.inference_params or "estimator_type" not in model.inference_params:
            raise ValueError("Model is missing 'estimator_type' in inference_params")

        estimator_type = schemas.EstimatorType(model.inference_params["estimator_type"])

        # Map estimator_type to model_factory type
        model_type_map = {
            schemas.EstimatorType.YOLO: 'yolo',
            schemas.EstimatorType.RT_DETR: 'rtdetr',
            schemas.EstimatorType.FASTER_RCNN: 'faster_rcnn',
            schemas.EstimatorType.EFFICIENTDET: 'efficientdet',
        }
        model_type = model_type_map.get(estimator_type)
        if not model_type:
            raise ValueError(f"Unsupported estimator type: {estimator_type}")

        # Get config file path for MMDetection models (EfficientDet)
        config_path = None
        if model_type == 'efficientdet':
            config_artifact = next((a for a in model.artifacts if a.artifact_type == "config"), None)
            if config_artifact:
                config_path = Path(config_artifact.storage_key)
                if not config_path.exists():
                    config_path = Path(config_artifact.storage_path)

        # Use model_factory to load the appropriate estimator
        # Note: EfficientDet models will automatically find config files in the same directory
        estimator = model_factory.load_model(
            model_path=str(model_path),
            model_type=model_type,
            class_names=class_names,
            input_size=input_size,
            device_type="gpu",  # Use GPU for faster attacks
            clip_values=(0, 255),
        )

        logger.info(f"Custom estimator loaded: {type(estimator)}")
        return estimator, input_size

    async def _load_dataset_annotations(
        self,
        db: AsyncSession,
        dataset_id: UUID,
    ) -> List[Dict[str, Any]]:
        """
        Load all bbox annotations from a dataset.

        Returns:
            List of dicts with keys: image_id, class_name, category_id, bbox (xyxy format)
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

                # Convert to xyxy format (ART expects this)
                x1 = ann.bbox_x
                y1 = ann.bbox_y
                x2 = ann.bbox_x + ann.bbox_width
                y2 = ann.bbox_y + ann.bbox_height

                annotations.append({
                    "image_id": img_record.id,
                    "class_name": ann.class_name,
                    "category_id": ann.class_index,  # Use class_index from model
                    "bbox": [x1, y1, x2, y2],  # xyxy format
                })

        return annotations

    def _convert_annotations_to_art_format(
        self,
        annotations: List[Dict[str, Any]],
        image_id: UUID,
        target_class_id: Optional[int] = None,
        original_size: Optional[Tuple[int, int]] = None,
        resized_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Convert annotations to ART format for a specific image.

        Args:
            annotations: List of annotation dicts (bbox in absolute coordinates)
            image_id: Image ID to filter for
            target_class_id: If specified, only include this class
            original_size: (height, width) of original image
            resized_size: (height, width) of resized image for model input

        Returns:
            Dict with 'boxes', 'labels', 'scores' or None if no matching annotations
            Note: bbox coordinates are adjusted if image was resized
        """
        # Filter annotations for this image
        img_annotations = [ann for ann in annotations if ann["image_id"] == image_id]

        if not img_annotations:
            return None

        # Filter by target class if specified
        if target_class_id is not None:
            img_annotations = [
                ann for ann in img_annotations
                if ann.get("category_id") == target_class_id
            ]

        if not img_annotations:
            return None

        # Convert to ART format
        boxes = np.array([ann["bbox"] for ann in img_annotations], dtype=np.float32)

        # Adjust bbox coordinates if image was resized
        if original_size is not None and resized_size is not None:
            orig_h, orig_w = original_size
            resized_h, resized_w = resized_size

            if orig_h != resized_h or orig_w != resized_w:
                # Calculate scale factors
                scale_x = resized_w / orig_w
                scale_y = resized_h / orig_h

                # Resize bboxes: [x1, y1, x2, y2]
                boxes[:, 0] *= scale_x  # x1
                boxes[:, 1] *= scale_y  # y1
                boxes[:, 2] *= scale_x  # x2
                boxes[:, 3] *= scale_y  # y2

                logger.debug(
                    f"Resized {len(boxes)} bboxes from {orig_w}x{orig_h} to {resized_w}x{resized_h}"
                )

        labels = np.array([ann.get("category_id", 0) for ann in img_annotations], dtype=np.int64)
        scores = np.ones(len(img_annotations), dtype=np.float32)  # Confidence = 1.0 for GT

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }


# Global instance
noise_attack_service = NoiseAttackService()
