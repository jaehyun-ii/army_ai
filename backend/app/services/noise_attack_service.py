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
from app.utils.system_monitor import system_monitor

# All attacks from app.ai (custom fork of ART)
from app.ai.attacks.evasion import (
    # PyTorch versions (primary, feature-complete implementations)
    UniversalNoiseAttackPyTorch,
    NoiseOSFDPyTorch,
    DistortionAwareAttackPyTorch,
)
from app.ai.attacks.evasion.fast_gradient_pytorch import FastGradientMethodPyTorch
from app.ai.attacks.evasion.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch

logger = logging.getLogger(__name__)

# Shared SSE manager instance for all attack services
from app.services.patch_attack_service import _shared_sse_manager


def get_attack_intensity_preset(attack_method: str, intensity_level: str) -> dict:
    """
    공격 강도 레벨에 따른 epsilon과 iterations 프리셋을 반환합니다.

    Args:
        attack_method: 공격 방법 ("fgsm", "pgd", "universal_noise", "noise_osfd", "distortion_aware")
        intensity_level: 강도 레벨 ("weak", "medium", "strong")

    Returns:
        epsilon과 iterations를 포함한 딕셔너리
    """
    presets = {
        "fgsm": {
            "weak": {"epsilon": 4.0, "iterations": 1, "description": "약한 공격 (은폐성 우선)"},
            "medium": {"epsilon": 8.0, "iterations": 1, "description": "보통 공격 (균형)"},
            "strong": {"epsilon": 16.0, "iterations": 1, "description": "강한 공격 (효과성 우선)"},
        },
        "pgd": {
            "weak": {"epsilon": 4.0, "iterations": 100, "description": "약한 공격 (은폐성 우선)"},
            "medium": {"epsilon": 8.0, "iterations": 300, "description": "보통 공격 (균형)"},
            "strong": {"epsilon": 16.0, "iterations": 500, "description": "강한 공격 (효과성 우선)"},
        },
        "universal_noise": {
            "weak": {"epsilon": 4.0, "iterations": 100, "description": "약한 범용 노이즈 (은폐성 우선)"},
            "medium": {"epsilon": 8.0, "iterations": 300, "description": "보통 범용 노이즈 (균형)"},
            "strong": {"epsilon": 12.0, "iterations": 500, "description": "강한 범용 노이즈 (효과성 우선)"},
        },
        "noise_osfd": {
            "weak": {"epsilon": 4.0, "iterations": 100 , "description": "약한 OSFD (은폐성 우선)"},
            "medium": {"epsilon": 8.0, "iterations": 300, "description": "보통 OSFD (AEGIS 논문 기본값)"},
            "strong": {"epsilon": 12.0, "iterations": 500, "description": "강한 OSFD (효과성 우선)"},
        },
        "distortion_aware": {
            "weak": {"epsilon": 255.0, "iterations": 1000, "description": "약한 Distortion-Aware (NCC=0.7, 높은 은폐성)"},
            "medium": {"epsilon": 255.0, "iterations": 3000, "description": "보통 Distortion-Aware (NCC=0.6, 논문 기본값)"},
            "strong": {"epsilon": 255.0, "iterations": 5000, "description": "강한 Distortion-Aware (NCC=0.5, 최대 효과)"},
        },
    }

    method_presets = presets.get(attack_method, presets["pgd"])
    return method_presets.get(intensity_level, method_presets["medium"])


def get_noise_attack_config(attack_method: str, iterations: int, epsilon: float) -> dict:
    """
    노이즈 공격 타입별 최적의 스텝 크기(alpha)와 스케줄러 설정을 반환합니다.

    Args:
        attack_method: 공격 방법 ("fgsm", "pgd", "universal_noise", "noise_osfd", "distortion_aware")
        iterations: 반복 횟수
        epsilon: 최대 perturbation 크기

    Returns:
        alpha(스텝 크기), 스케줄러 타입, 스케줄러 파라미터를 포함한 딕셔너리
    """
    configs = {
        "fgsm": {
            # FGSM: 단일 스텝 (alpha 불필요, 스케줄러 불필요)
            "description": "Fast Gradient Sign Method (단일 스텝)",
        },
        "pgd": {
            # PGD: 다중 스텝 반복 공격
            "alpha": epsilon / 4,  # epsilon의 1/4로 안정적 수렴
            "scheduler_type": "step",
            "scheduler_params": {
                "step_size": max(1, iterations // 3),  # 1/3 지점마다 감소
                "gamma": 0.5,  # 절반으로 감소
            },
            "description": "Projected Gradient Descent (다중 스텝)",
        },
        "universal_noise": {
            # Universal Noise: 범용 perturbation 학습
            "alpha": 0.255,  # Fixed learning rate: 0.001 in [0,1] scale = 0.255 in [0,255] scale
            "scheduler_type": "cosine",
            "scheduler_params": {
                "T_max": iterations,
                "eta_min": 0.0,  # 최소 학습률 0 (완전히 감소)
            },
            "description": "Universal Noise (범용 perturbation)",
        },
        "noise_osfd": {
            # Noise OSFD: 특징 증폭 기반 공격 (AEGIS paper defaults)
            "alpha": 0.255,  # Fixed learning rate: 0.001 in [0,1] scale = 0.255 in [0,255] scale
            "scheduler_type": "cosine",
            "scheduler_params": {
                "T_max": iterations,
                "eta_min": 0.0,  # 최소 학습률 0
            },
            "description": "Object-aware Spatial Feature Distortion (OSFD)",
        },
        "distortion_aware": {
            # Distortion-Aware: NCC 기반 왜곡 제어 (attack_detector paper)
            # attack_detector uses step=100 in [0,255] scale
            # Convert to [0,1] scale: 100/255 ≈ 0.392
            "alpha": 0.392,  # Fixed step size matching attack_detector
            "scheduler_type": None,  # No scheduler (fixed step like attack_detector)
            "scheduler_params": {},
            "attack_mode": "image_specific",  # Image-specific attack
            "use_model_loss": True,  # Use YOLO training loss (L_box + L_cls + L_dfl)
            "apply_mask": True,  # Apply object masking
            "ncc_threshold": 0.6,  # Default NCC threshold (configurable per preset)
            "description": "Distortion-Aware Attack (NCC-based, Image-Specific)",
        },
    }

    return configs.get(attack_method, {})


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
        # Use 2D storage root for 2D noise attacks
        self.storage_root = Path(settings.STORAGE_2D_ROOT)
        self.storage_root_main = Path(settings.STORAGE_ROOT)  # For storage_key resolution
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
        epsilon: Optional[float] = None,
        alpha: Optional[float] = None,
        iterations: Optional[int] = None,
        intensity_level: Optional[str] = None,
        target_class: Optional[str] = None,
        top_k_detections: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
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
            epsilon: Maximum perturbation (in [0, 255] scale, optional if intensity_level provided)
            alpha: Step size for PGD (in [0, 255] scale, optional, auto-configured if None)
            iterations: Number of iterations for PGD (optional, auto-configured if None)
            intensity_level: Attack intensity level ("weak", "medium", "strong", optional)
            target_class: Target class name to attack (e.g., 'person'). If None, attack all objects.
            top_k_detections: For universal_noise/noise_osfd - number of detections per image:
                             -1 = all detections above threshold (multi-target, default)
                              1 = single highest confidence detection (AEGIS-style)
                              N = top N detections (only when y=None/pseudo-GT mode)
            confidence_threshold: Minimum confidence for pseudo-GT detections (0.0-1.0, default 0.0)
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
            valid_methods = ["fgsm", "pgd", "universal_noise", "noise_osfd", "distortion_aware"]
            if attack_method not in valid_methods:
                raise ValidationError(
                    f"Invalid attack method: {attack_method}. "
                    f"Must be one of: {', '.join(valid_methods)}"
                )

            # Apply intensity level preset if provided
            if intensity_level is not None:
                valid_levels = ["weak", "medium", "strong"]
                if intensity_level not in valid_levels:
                    raise ValidationError(
                        f"Invalid intensity level: {intensity_level}. "
                        f"Must be one of: {', '.join(valid_levels)}"
                    )

                preset = get_attack_intensity_preset(attack_method, intensity_level)

                # Use preset values if not manually specified
                if epsilon is None:
                    epsilon = preset["epsilon"]
                if iterations is None:
                    iterations = preset["iterations"]

                await sse_logger.info(
                    f"공격 강도 레벨: {preset['description']}"
                )
                await sse_logger.info(
                    f"자동 설정 - Epsilon: {epsilon}, 반복 횟수: {iterations}"
                )
            else:
                # No intensity level specified, use defaults or provided values
                if epsilon is None:
                    epsilon = 8.0  # Default medium intensity
                    await sse_logger.info("Epsilon 기본값 사용: 8.0")

                # Set default iterations based on attack method
                if iterations is None:
                    iterations = 50 if attack_method in ["universal_noise", "noise_osfd"] else 10
                    await sse_logger.info(f"반복 횟수 기본값 사용: {iterations}")

            # Get optimal configuration for attack method (includes auto alpha)
            attack_config = get_noise_attack_config(attack_method, iterations, epsilon)

            # Use auto-configured alpha if not provided
            if alpha is None and "alpha" in attack_config:
                alpha = attack_config["alpha"]
                await sse_logger.info(
                    f"자동 스텝 크기 설정: {alpha:.2f} (epsilon/{epsilon/alpha:.1f})"
                )
            elif alpha is not None:
                await sse_logger.info(f"운용자 지정 스텝 크기: {alpha}")

            # Log attack configuration
            if attack_config:
                await sse_logger.info(
                    f"공격 방법: {attack_config.get('description', attack_method)}"
                )
                if "scheduler_type" in attack_config:
                    await sse_logger.info(
                        f"스케줄러: {attack_config['scheduler_type']} (공격 특성에 최적화)"
                    )

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

            # Step 3: Calculate optimal batch size based on GPU memory
            # Use conservative settings to prevent OOM
            model_height, model_width = input_size
            optimal_batch_size = system_monitor.get_optimal_batch_size(
                image_size=(model_height, model_width),
                model_type="noise",  # Moderately memory intensive (×6 multiplier)
                # Use default safety_margin=0.7 and default_batch_size=32
            )
            await sse_logger.info(f"GPU 메모리 기반 최적 배치 사이즈: {optimal_batch_size}")

            # Step 4: Create ART attack object
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
                    batch_size=optimal_batch_size,  # Dynamic batch size based on GPU memory
                )
                await sse_logger.info(f"FGSM (PyTorch) 생성: epsilon={epsilon}, batch_size={optimal_batch_size}")

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
                    batch_size=optimal_batch_size,  # Dynamic batch size based on GPU memory
                    verbose=False,
                )
                await sse_logger.info(f"PGD 생성: epsilon={epsilon}, alpha={alpha}, iterations={iterations}, batch_size={optimal_batch_size}")

            elif attack_method == "universal_noise":
                # Universal Noise Attack - PyTorch version with pseudo-GT (feature-complete)
                # alpha is already set by attack_config above

                # Use target_class_id if specified, otherwise default to 0 (person in COCO)
                universal_target_class_id = target_class_id if target_class_id is not None else 0

                # Set defaults for multi-target parameters
                if top_k_detections is None:
                    top_k_detections = -1  # Default: all detections (multi-target)
                if confidence_threshold is None:
                    confidence_threshold = 0.0  # Default: no confidence filtering

                # Get scheduler configuration
                scheduler_type = attack_config.get("scheduler_type", "constant")
                scheduler_params = attack_config.get("scheduler_params", {})

                # Store the main event loop reference
                import asyncio
                main_loop = asyncio.get_event_loop()

                # Define progress callback for SSE updates
                def training_progress_callback(current_iter: int, total_iter: int, loss_value: float, current_lr: float):
                    """Callback to send training progress via SSE (called from training thread)"""
                    try:
                        # Schedule coroutine in main event loop from worker thread
                        asyncio.run_coroutine_threadsafe(
                            sse_logger.progress(
                                f"Universal Noise 학습 중... ({current_iter}/{total_iter}) LR: {current_lr:.6f}",
                                processed=current_iter,
                                total=total_iter,
                                successful=current_iter,
                                failed=0
                            ),
                            main_loop
                        )
                    except Exception as e:
                        logger.warning(f"Failed to send training progress via SSE: {e}")

                # Attack loss configuration is now handled automatically by
                # UniversalNoiseAttackPyTorch.__init__() via _configure_attack_loss()
                # No need to mutate estimator's private fields here!
                await sse_logger.info(
                    "Attack Loss가 자동으로 활성화됩니다 (DetectionAttackLoss)"
                )

                # Log multi-target configuration
                if top_k_detections == -1:
                    multi_target_mode = "모든 detection (다중 타겟)"
                elif top_k_detections == 1:
                    multi_target_mode = "단일 타겟 (AEGIS 스타일)"
                else:
                    multi_target_mode = f"상위 {top_k_detections}개 detection"
                await sse_logger.info(
                    f"Pseudo-GT 모드: {multi_target_mode}, confidence threshold: {confidence_threshold}"
                )

                attack = UniversalNoiseAttackPyTorch(
                    estimator=estimator,
                    eps=eps_normalized / 255.0,
                    eps_step=alpha / 255.0,
                    max_iter=iterations,
                    batch_size=optimal_batch_size,  # Dynamic batch size based on GPU memory
                    apply_mask=True,  # Apply perturbation to detected regions
                    target_class_id=universal_target_class_id,
                    top_k_detections=top_k_detections,  # Multi-target support
                    confidence_threshold=confidence_threshold,  # Confidence filtering
                    verbose=True,
                    scheduler_type=scheduler_type,  # Auto-configured scheduler
                    scheduler_params=scheduler_params,
                    progress_callback=training_progress_callback,  # Send progress via SSE
                )
                await sse_logger.info(
                    f"Universal Noise Attack 생성 (PyTorch): epsilon={epsilon}, "
                    f"step_size={alpha}, iterations={iterations}, "
                    f"target_class_id={universal_target_class_id}, "
                    f"multi_target_mode={multi_target_mode}, "
                    f"pseudo-GT enabled, Attack Loss enabled"
                )

            elif attack_method == "noise_osfd":
                # Noise OSFD Attack - PyTorch version with full features (RRB augmentation)
                # alpha is already set by attack_config above

                # Get feature layer indices (can be customized)
                feature_layers = [10, 15, 20]  # Default YOLO backbone layers
                amplification_factor = 3.0  # AEGIS paper default (NOT 10.0!)

                # Get scheduler configuration
                scheduler_type = attack_config.get("scheduler_type", "constant")
                scheduler_params = attack_config.get("scheduler_params", {})

                # Store the main event loop reference
                import asyncio
                main_loop = asyncio.get_event_loop()

                # Define progress callback for SSE updates
                def training_progress_callback(current_iter: int, total_iter: int, loss_value: float, current_lr: float):
                    """Callback to send training progress via SSE (called from training thread)"""
                    try:
                        # Schedule coroutine in main event loop from worker thread
                        asyncio.run_coroutine_threadsafe(
                            sse_logger.progress(
                                f"OSFD 학습 중... ({current_iter}/{total_iter}) Loss: {loss_value:.2f}, LR: {current_lr:.6f}",
                                processed=current_iter,
                                total=total_iter,
                                successful=current_iter,
                                failed=0
                            ),
                            main_loop
                        )
                    except Exception as e:
                        logger.warning(f"Failed to send training progress via SSE: {e}")

                attack = NoiseOSFDPyTorch(
                    estimator=estimator,
                    eps=eps_normalized / 255.0,
                    eps_step=alpha / 255.0,
                    max_iter=iterations,
                    batch_size=optimal_batch_size,  # Dynamic batch size based on GPU memory
                    feature_layer_indices=feature_layers,
                    amplification_factor=amplification_factor,
                    apply_augmentation=True,
                    verbose=True,
                    lr_scheduler_type=scheduler_type,  # Auto-configured scheduler
                    lr_scheduler_params=scheduler_params,
                    progress_callback=training_progress_callback,  # Send progress via SSE
                )
                await sse_logger.info(
                    f"Noise OSFD Attack 생성 (PyTorch): epsilon={epsilon}, "
                    f"step_size={alpha}, iterations={iterations}, "
                    f"layers={feature_layers}, RRB augmentation enabled"
                )

            elif attack_method == "distortion_aware":
                # Distortion-Aware Attack - Image-specific with NCC-based distortion control
                # Matches attack_detector paper implementation

                # Get NCC threshold from preset (varies by intensity level)
                ncc_threshold_map = {
                    "weak": 0.7,    # High similarity (less distortion)
                    "medium": 0.6,  # Paper default
                    "strong": 0.5,  # Lower similarity (more distortion)
                }
                ncc_threshold = ncc_threshold_map.get(intensity_level, 0.6)

                # Get configuration from attack_config
                attack_mode = attack_config.get("attack_mode", "image_specific")
                use_model_loss = attack_config.get("use_model_loss", True)
                apply_mask = attack_config.get("apply_mask", True)

                await sse_logger.info(
                    f"Distortion-Aware Attack 설정: NCC threshold={ncc_threshold}, "
                    f"mode={attack_mode}, model_loss={use_model_loss}, "
                    f"masking={apply_mask}"
                )

                # Use target_class_id if specified, otherwise default to 0 (person in COCO)
                distortion_target_class_id = target_class_id if target_class_id is not None else 0

                # Store the main event loop reference
                import asyncio
                main_loop = asyncio.get_event_loop()

                # Define progress callback for SSE updates
                def distortion_progress_callback(current_iter: int, total_iter: int, loss_value: float, current_lr: float):
                    """Callback to send training progress via SSE (called from training thread)"""
                    try:
                        # Schedule coroutine in main event loop from worker thread
                        asyncio.run_coroutine_threadsafe(
                            sse_logger.progress(
                                f"Distortion-Aware 공격 중... ({current_iter}/{total_iter}) NCC: {ncc_threshold:.2f}",
                                processed=current_iter,
                                total=total_iter,
                                successful=current_iter,
                                failed=0
                            ),
                            main_loop
                        )
                    except Exception as e:
                        logger.warning(f"Failed to send distortion progress via SSE: {e}")

                # CRITICAL FIX: alpha is ALREADY normalized (0.392 = 100/255)
                # DO NOT divide by 255 again!
                await sse_logger.info(
                    f"Step size 확인: alpha={alpha:.6f} (이미 정규화됨, 다시 나누지 않음!)"
                )

                attack = DistortionAwareAttackPyTorch(
                    estimator=estimator,
                    eps=eps_normalized / 255.0,  # 255.0 / 255.0 = 1.0 (no epsilon constraint)
                    eps_step=alpha,  # ✅ FIXED: alpha is already normalized (0.392)
                    max_iter=iterations,  # 1000-5000 iterations based on intensity
                    ncc_threshold=ncc_threshold,  # NCC-based distortion control
                    target_class_id=distortion_target_class_id,
                    verbose=True,
                )
                await sse_logger.info(
                    f"Distortion-Aware Attack 생성 완료"
                )
                await sse_logger.info(
                    f"  - Epsilon: {epsilon} → 정규화: {eps_normalized/255.0:.6f}"
                )
                await sse_logger.info(
                    f"  - Step size: {alpha:.6f} (원본: 100/255 = 0.392)"
                )
                await sse_logger.info(
                    f"  - Iterations: {iterations}"
                )
                await sse_logger.info(
                    f"  - NCC threshold: {ncc_threshold} ({intensity_level} 강도)"
                )
                await sse_logger.info(
                    f"  - 예상 최대 perturbation: ~{alpha * iterations:.1f} (clipping 전)"
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
            # For Distortion-Aware, attack each image individually (no universal perturbation)
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

                    # Use all images with target class for training
                    training_images = images_with_target
                    await sse_logger.info(
                        f"타겟 클래스 포함 이미지: {len(images_with_target)}개 전체 사용"
                    )
                else:
                    # No target class specified, use all images
                    training_images = images
                    await sse_logger.info(f"전체 이미지 {len(training_images)}개로 학습")

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
                    result = attack.generate(x=x_batch, y=y_batch, return_perturbation=True)
                    if isinstance(result, tuple):
                        return result[1]  # Return perturbation
                    return None

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    universal_perturbation = await loop.run_in_executor(executor, train_universal)

                # Validate perturbation was generated
                if universal_perturbation is None:
                    raise ValidationError("Failed to generate universal perturbation")

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

                        # CRITICAL FIX: Resize back to original size if image was resized
                        if resize_info_item["resized"]:
                            from PIL import Image
                            original_h = resize_info_item["original_height"]
                            original_w = resize_info_item["original_width"]
                            adv_img_pil = Image.fromarray(adv_img.astype(np.uint8))
                            adv_img_pil = adv_img_pil.resize((original_w, original_h), Image.BICUBIC)
                            adv_img = np.array(adv_img_pil).astype(np.uint8)

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

                # Prepare all images for batch application (keep in HWC format)
                x_batch_list = []
                resize_info_list = []
                y_batch_list = [] if target_class_id is not None and annotations else None
                for img_data in images:
                    img = img_data["image"]
                    original_h, original_w = img.shape[:2]

                    # Resize to model input size if needed
                    model_height, model_width = input_size[0], input_size[1]
                    resized = False
                    if img.shape[0] != model_height or img.shape[1] != model_width:
                        from PIL import Image
                        img_pil = Image.fromarray(img.astype(np.uint8))
                        img_pil = img_pil.resize((model_width, model_height), Image.BICUBIC)
                        img = np.array(img_pil).astype(np.float32)
                        resized = True

                    # Store resize info
                    resize_info_list.append({
                        "resized": resized,
                        "original_height": original_h,
                        "original_width": original_w
                    })

                    # CRITICAL FIX: Convert to CHW format (C, H, W) to match estimator's channels_first=True
                    # Estimator expects NCHW input when channels_first=True
                    x = img.transpose(2, 0, 1).astype(np.float32)  # (H, W, C) -> (C, H, W)
                    x_batch_list.append(x)

                    if y_batch_list is not None:
                        y_target = self._convert_annotations_to_art_format(
                            annotations,
                            img_data["id"],
                            target_class_id,
                            original_size=(original_h, original_w) if resized else None,
                            resized_size=(model_height, model_width) if resized else None,
                        )
                        y_batch_list.append(
                            y_target if y_target else {"boxes": np.array([]), "labels": np.array([]), "scores": np.array([])}
                        )

                x_batch = np.array(x_batch_list)  # (N, C, H, W) - channels first

                # Use attack.apply_perturbation() with masking; prefer GT when available.
                def apply_with_mask():
                    # Set the pre-trained perturbation
                    attack.set_perturbation(universal_perturbation)
                    # Apply perturbation with masking (GT if provided; otherwise pseudo-GT)
                    return attack.apply_perturbation(x=x_batch, y=y_batch_list, apply_mask=True)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    x_adv_batch = await loop.run_in_executor(executor, apply_with_mask)

                await sse_logger.info(f"Perturbation applied to batch of {len(x_adv_batch)} images with masking.")

                # Process the results (x_adv_batch is in CHW format from channels_first=True estimator)
                for idx, (x_adv, img_data, resize_info_item) in enumerate(zip(x_adv_batch, images, resize_info_list)):
                    try:
                        # CRITICAL FIX: Convert from CHW (C, H, W) to HWC (H, W, C) for PIL operations
                        # estimator.channels_first=True means output is also in CHW format
                        x_adv_hwc = x_adv.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        adv_img = np.clip(x_adv_hwc, 0, 255).astype(np.uint8)

                        # CRITICAL FIX: Resize back to original size if image was resized
                        if resize_info_item["resized"]:
                            from PIL import Image
                            original_h = resize_info_item["original_height"]
                            original_w = resize_info_item["original_width"]
                            adv_img_pil = Image.fromarray(adv_img.astype(np.uint8))
                            adv_img_pil = adv_img_pil.resize((original_w, original_h), Image.BICUBIC)
                            adv_img = np.array(adv_img_pil).astype(np.uint8)

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

            # For Distortion-Aware, attack each image individually (image-specific attack)
            elif attack_method == "distortion_aware":
                await sse_logger.status(f"Distortion-Aware: 이미지별 독립 공격 중...")

                for idx, img_data in enumerate(images):
                    try:
                        await sse_logger.progress(
                            f"이미지 {idx + 1}/{len(images)} 공격 중...",
                            processed=idx,
                            total=len(images),
                            successful=len(attacked_images),
                            failed=failed_count
                        )

                        img = img_data["image"]
                        original_h, original_w = img.shape[:2]

                        # Resize to model input size if needed
                        model_height, model_width = input_size[0], input_size[1]
                        resized = False
                        if img.shape[0] != model_height or img.shape[1] != model_width:
                            from PIL import Image
                            img_pil = Image.fromarray(img.astype(np.uint8))
                            img_pil = img_pil.resize((model_width, model_height), Image.BICUBIC)
                            img = np.array(img_pil).astype(np.float32)
                            resized = True

                        # Convert to CHW format and add batch dimension
                        x = img.transpose(2, 0, 1).astype(np.float32)  # (H, W, C) -> (C, H, W)
                        x_batch = np.expand_dims(x, axis=0)  # (1, C, H, W)

                        # Prepare y if target_class is specified
                        y_batch = None
                        if target_class_id is not None and annotations:
                            y_target = self._convert_annotations_to_art_format(
                                annotations,
                                img_data["id"],
                                target_class_id,
                                original_size=(original_h, original_w) if resized else None,
                                resized_size=(model_height, model_width) if resized else None,
                            )
                            y_batch = [y_target] if y_target else None

                        # Attack this single image (y=None means pseudo-GT will be generated)
                        def attack_single():
                            return attack.generate(x=x_batch, y=y_batch)

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            x_adv_batch = await loop.run_in_executor(executor, attack_single)

                        # Extract result (remove batch dimension)
                        x_adv = x_adv_batch[0]  # (C, H, W)

                        # Convert from CHW to HWC
                        x_adv_hwc = x_adv.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        adv_img = np.clip(x_adv_hwc, 0, 255).astype(np.uint8)

                        # Resize back to original size if needed
                        if resized:
                            from PIL import Image
                            adv_img_pil = Image.fromarray(adv_img.astype(np.uint8))
                            adv_img_pil = adv_img_pil.resize((original_w, original_h), Image.BICUBIC)
                            adv_img = np.array(adv_img_pil).astype(np.uint8)

                        attacked_images.append({
                            "image": adv_img,
                            "original_file_name": img_data["file_name"],
                            "original_id": img_data["id"],
                        })

                        await sse_logger.progress(
                            f"이미지 {idx + 1}/{len(images)} 완료",
                            processed=idx + 1,
                            total=len(images),
                            successful=len(attacked_images),
                            failed=failed_count
                        )

                    except Exception as e:
                        logger.error(f"Failed to attack image {idx} with Distortion-Aware: {e}", exc_info=True)
                        failed_count += 1
                        await sse_logger.warning(f"이미지 {idx} 공격 실패: {str(e)}")

                await sse_logger.info(f"Distortion-Aware 공격 완료: 성공 {len(attacked_images)}, 실패 {failed_count}")

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
                        storage_key=str(img_path.relative_to(self.storage_root_main)),
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
                        "top_k_detections": top_k_detections,  # Multi-target config
                        "confidence_threshold": confidence_threshold,  # Pseudo-GT filtering
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
            img_path = self.storage_root_main / img_record.storage_key

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

        # Detect available device (GPU if available, otherwise CPU)
        import torch
        device_type = "gpu" if torch.cuda.is_available() else "cpu"

        # Log device selection
        if device_type == "gpu":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU for attack: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("GPU not available, using CPU (will be slower)")

        # Use model_factory to load the appropriate estimator
        # Note: EfficientDet models will automatically find config files in the same directory
        estimator = model_factory.load_model(
            model_path=str(model_path),
            model_type=model_type,
            class_names=class_names,
            input_size=input_size,
            device_type=device_type,  # Use GPU if available for better performance
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
