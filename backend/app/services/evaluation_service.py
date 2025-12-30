"""
New evaluation execution service - redesigned logic.

Evaluation modes:
1. Base dataset: Simple evaluation with GT
2. Attack dataset: Dual evaluation (original + attack) then compare
"""
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.models.evaluation import EvalStatus
from app.schemas.evaluation import (
    EvalRunUpdate,
    EvalItemCreate,
    EvalDatasetType,
)
from app.services.inference_service import InferenceService
from app.services.sse_support import SSEManager, SSELogger
from app.services.metrics_calculator import (
    BoundingBox,
    parse_detection_to_bbox,
    calculate_overall_metrics,
    calculate_robustness_metrics,
)
from app.core.exceptions import NotFoundError, ValidationError
from app.ai.estimators.object_detection.class_mapper import COCO_CLASSES
from app.ai.evaluates import Evaluator

logger = logging.getLogger(__name__)


class EvaluationService:
    """Evaluation service with redesigned logic."""

    def __init__(self):
        self.inference_service = InferenceService()
        self.sse_manager = SSEManager()
        self.evaluator = Evaluator(iou_threshold=0.5)

    def _get_class_names_from_dataset(self, dataset: Any) -> Optional[List[str]]:
        """
        Get class names from dataset metadata.

        Returns:
            List of class names if found in metadata, otherwise None
        """
        if not dataset or not dataset.metadata_:
            return None

        classes = dataset.metadata_.get("classes", [])
        if not classes or not isinstance(classes, list):
            return None

        return classes

    def _map_class_name(self, class_name: str, class_names: Optional[List[str]] = None) -> str:
        """
        Map class_X format to actual class name.

        Args:
            class_name: The class name (e.g., "class_0", "class_18", or "person")
            class_names: List of class names from dataset metadata (if available)

        Returns:
            Mapped class name
        """
        if not class_name.startswith('class_'):
            return class_name

        try:
            class_idx = int(class_name.split('_')[1])

            # First, try to use dataset-specific class names
            if class_names and 0 <= class_idx < len(class_names):
                return class_names[class_idx]

            # Fallback to COCO classes (for backward compatibility)
            if 0 <= class_idx < len(COCO_CLASSES):
                return COCO_CLASSES[class_idx]

        except (ValueError, IndexError):
            pass

        # If all else fails, return original class name
        return class_name

    async def execute_evaluation(
        self,
        db: AsyncSession,
        eval_run_id: UUID,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,  # Fixed at 0.5 for precision/recall (AP is calculated for all IoUs 0.5~0.95)
        target_class: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Execute evaluation with new logic:
        - Base dataset: Single evaluation with GT
        - Attack dataset: Dual evaluation (original + attack) then compare

        Args:
            target_class: If provided, only evaluate this specific class
        """
        eval_logger = SSELogger(logger, self.sse_manager, session_id)
        await eval_logger.status("평가 시작 중...", eval_run_id=str(eval_run_id))

        # Log evaluation parameters
        logger.info(f"[Evaluation Service] Starting evaluation with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}, target_class={target_class}")

        if target_class:
            await eval_logger.info(f"🎯 타겟 클래스 필터링 활성화: {target_class}")
            logger.info(f"🎯 Target class filtering enabled for evaluation {eval_run_id}: {target_class}")

        # Get evaluation run
        eval_run = await crud.evaluation.get_eval_run(db, eval_run_id)
        if not eval_run:
            await eval_logger.error("평가 실행을 찾을 수 없습니다")
            raise NotFoundError(resource=f"Evaluation run {eval_run_id}")

        dimension = getattr(eval_run, "dimension", "2d") or "2d"
        await eval_logger.info(f"평가 이름: {eval_run.name}")
        await eval_logger.info(f"데이터 차원: {dimension}")

        # Update status to running
        await crud.evaluation.update_eval_run(
            db, eval_run_id,
            EvalRunUpdate(status=EvalStatus.RUNNING, started_at=datetime.utcnow())
        )
        await db.commit()

        model_id = eval_run.model_id

        try:
            await self.inference_service.load_model(db, model_id)

            if eval_run.attack_dataset_id or getattr(eval_run, "attack_dataset_3d_id", None):
                # Mode 2: Attack dataset evaluation
                await self._evaluate_attack_dataset(
                    db, eval_run, eval_logger, conf_threshold, iou_threshold, target_class, dimension
                )
            else:
                # Mode 1: Base dataset evaluation
                await self._evaluate_base_dataset(
                    db, eval_run, eval_logger, conf_threshold, iou_threshold, target_class, dimension
                )

            # Mark as completed
            await crud.evaluation.update_eval_run(
                db, eval_run_id,
                EvalRunUpdate(status=EvalStatus.COMPLETED, ended_at=datetime.utcnow())
            )
            await db.commit()

            await eval_logger.success("평가 완료!")

            # Generate PDF report (for all evaluations)
            report_path = None
            try:
                    await eval_logger.status("PDF 보고서 생성 중...")
                    from app.services.report_generation_service import report_generation_service

                    # Check dependencies
                    deps = report_generation_service.check_dependencies()
                    if deps.get("weasyprint"):
                        report_path = await report_generation_service.generate_evaluation_report(
                            db=db,
                            evaluation_id=eval_run_id,
                            include_charts=deps.get("matplotlib", False)
                        )
                        await eval_logger.success(f"보고서 생성 완료: {report_path}")
                        logger.info(f"PDF 보고서 생성 완료: {report_path}")
                    else:
                        await eval_logger.warning("WeasyPrint 미설치로 보고서 생성 건너뜀")
                        logger.warning("WeasyPrint not available, skipping report generation")
            except Exception as e:
                # 보고서 생성 실패는 평가 전체를 실패시키지 않음
                await eval_logger.warning(f"보고서 생성 실패: {str(e)}")
                logger.error(f"Report generation failed for evaluation {eval_run_id}: {e}", exc_info=True)

            if session_id:
                event_data = {
                    "type": "complete",
                    "message": "평가 완료",
                    "eval_run_id": str(eval_run_id),
                    "timestamp": datetime.now().isoformat()
                }
                if report_path:
                    event_data["report_path"] = report_path
                await self.sse_manager.send_event(session_id, event_data)

        except Exception as e:
            logger.error(f"Evaluation {eval_run_id} failed: {e}", exc_info=True)
            await eval_logger.error(f"평가 실패: {str(e)}")

            await crud.evaluation.update_eval_run(
                db, eval_run_id,
                EvalRunUpdate(status=EvalStatus.FAILED, ended_at=datetime.utcnow())
            )
            await db.commit()

            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "error",
                    "message": f"평가 실패: {str(e)}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            raise
        finally:
            await self.inference_service.unload_model(model_id)

    async def _evaluate_base_dataset(
        self,
        db: AsyncSession,
        eval_run: Any,
        eval_logger: SSELogger,
        conf_threshold: float,
        iou_threshold: float,
        target_class: Optional[str] = None,
        dimension: str = "2d",
    ) -> None:
        """
        Mode 1: Base dataset evaluation
        - Load base dataset
        - Run inference
        - Compare with GT
        - Calculate metrics

        Args:
            target_class: If provided, only evaluate this specific class
        """
        await eval_logger.info("기준 데이터셋 평가 시작")

        # Get base dataset
        if dimension == "3d":
            from sqlalchemy import select
            from app.models.dataset_3d import Dataset3D, Image3D
            dataset_row = await db.execute(
                select(Dataset3D).where(Dataset3D.id == eval_run.base_dataset_3d_id, Dataset3D.deleted_at.is_(None))
            )
            dataset = dataset_row.scalar_one_or_none()
            if not dataset:
                raise NotFoundError(resource=f"Dataset {eval_run.base_dataset_3d_id}")

            images_result = await db.execute(
                select(Image3D).where(Image3D.dataset_id == eval_run.base_dataset_3d_id, Image3D.deleted_at.is_(None))
            )
            images = list(images_result.scalars().all())
        else:
            dataset = await crud.dataset_2d.get(db, id=eval_run.base_dataset_id)
            if not dataset:
                raise NotFoundError(resource=f"Dataset {eval_run.base_dataset_id}")
            images = await crud.image_2d.get_by_dataset(db, dataset_id=eval_run.base_dataset_id)
        if not images:
            raise ValidationError(detail=f"No images in dataset {eval_run.base_dataset_id or eval_run.base_dataset_3d_id}")

        await eval_logger.info(f"총 {len(images)}개 이미지 발견")

        # Run inference
        await eval_logger.status("모델 추론 시작...", total_images=len(images))
        image_ids = [img.id for img in images]

        inference_results = await self.inference_service.run_inference(
            db=db,
            model_id=eval_run.model_id,
            image_ids=image_ids,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            target_class=target_class,
            dimension=dimension,
        )

        await eval_logger.status("모델 추론 완료", completed_images=len(inference_results))
        await eval_logger.status("메트릭 계산 중...")

        # Calculate metrics with GT
        metrics = await self._calculate_metrics_with_gt(
            db=db,
            inference_results=inference_results,
            images=images,
            dataset=dataset,
            target_class=target_class,
            iou_threshold=iou_threshold,
            dimension=dimension,
        )

        # Log metrics based on whether target class is set
        if target_class and 'ap50' in metrics['overall']:
            await eval_logger.info(f"타겟 클래스 '{target_class}' AP@50: {metrics['overall']['ap50']:.3f}")
        elif 'map50' in metrics['overall']:
            await eval_logger.info(f"mAP@50: {metrics['overall']['map50']:.3f}")
        await eval_logger.info(f"Precision: {metrics['overall']['precision']:.3f}")
        await eval_logger.info(f"Recall: {metrics['overall']['recall']:.3f}")

        # Save results
        await self._save_evaluation_results(
            db=db,
            eval_run_id=eval_run.id,
            inference_results=inference_results,
            images=images,
            metrics=metrics,
            eval_logger=eval_logger,
            dataset=dataset,
            dataset_type=EvalDatasetType.BASE,  # NEW: pre_attack evaluations are always base
            target_class=target_class,  # Pass target class for filtering
            iou_threshold=iou_threshold,
            dimension=dimension,
        )

    async def _evaluate_attack_dataset(
        self,
        db: AsyncSession,
        eval_run: Any,
        eval_logger: SSELogger,
        conf_threshold: float,
        iou_threshold: float,
        target_class: Optional[str] = None,
        dimension: str = "2d",
    ) -> None:
        """
        Mode 2: Attack dataset evaluation
        - Find original dataset from attack dataset
        - Evaluate original dataset (Step 1)
        - Evaluate attack dataset (Step 2)
        - Compare results (Step 3)

        Args:
            target_class: If provided, only evaluate this specific class
        """
        await eval_logger.info("공격 데이터셋 평가 시작 (Dual Evaluation)")

        if dimension == "3d":
            from sqlalchemy import select
            from app.models.dataset_3d import Dataset3D, Image3D, AttackDataset3D
            # Note: 3D attack datasets do NOT have base_dataset_id field
            # User must manually select base_dataset_3d_id in UI if comparison is needed
            attack_dataset_row = await db.execute(
                select(AttackDataset3D).where(AttackDataset3D.id == eval_run.attack_dataset_3d_id, AttackDataset3D.deleted_at.is_(None))
            )
            attack_dataset = attack_dataset_row.scalar_one_or_none()
            if not attack_dataset:
                raise NotFoundError(resource=f"Attack dataset {eval_run.attack_dataset_3d_id}")

            # For 3D: base_dataset_3d_id is OPTIONAL
            # - If provided: Compare original vs attack (with robustness metrics)
            # - If not provided: Evaluate attack dataset only (no comparison)
            original_dataset_id = eval_run.base_dataset_3d_id  # May be None
            attack_output_dataset_id = attack_dataset.output_dataset_id
        else:
            # Get attack dataset info (2D)
            attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
            if not attack_dataset:
                raise NotFoundError(resource=f"Attack dataset {eval_run.attack_dataset_id}")

            # For 2D: base_dataset_id is auto-populated from attack_dataset
            original_dataset_id = attack_dataset.base_dataset_id
            attack_output_dataset_id = attack_dataset.output_dataset_id

        if not attack_output_dataset_id:
            raise ValidationError(
                detail=f"Attack dataset {eval_run.attack_dataset_id or eval_run.attack_dataset_3d_id} is missing output_dataset_id. "
                "Please ensure attack dataset creation populates output_dataset_id."
            )

        if original_dataset_id:
            await eval_logger.info(f"원본 데이터셋 ID: {original_dataset_id}")
        await eval_logger.info(f"공격 데이터셋 ID: {attack_output_dataset_id}")

        # ========== Step 1: Evaluate Original Dataset (if provided) ==========
        original_dataset = None
        original_images = None
        original_metrics = None

        if original_dataset_id:
            await eval_logger.info("Step 1/3: 원본 데이터셋 평가")

            if dimension == "3d":
                dataset_row = await db.execute(
                    select(Dataset3D).where(Dataset3D.id == original_dataset_id, Dataset3D.deleted_at.is_(None))
                )
                original_dataset = dataset_row.scalar_one_or_none()
                if not original_dataset:
                    raise NotFoundError(resource=f"Original dataset {original_dataset_id}")
                images_row = await db.execute(
                    select(Image3D).where(Image3D.dataset_id == original_dataset_id, Image3D.deleted_at.is_(None))
                )
                original_images = list(images_row.scalars().all())
            else:
                original_dataset = await crud.dataset_2d.get(db, id=original_dataset_id)
                if not original_dataset:
                    raise NotFoundError(resource=f"Original dataset {original_dataset_id}")
                original_images = await crud.image_2d.get_by_dataset(db, dataset_id=original_dataset_id)
            await eval_logger.info(f"원본 데이터셋: {len(original_images)}개 이미지")

            # Run inference on original
            original_inference_results = await self.inference_service.run_inference(
                db=db,
                model_id=eval_run.model_id,
                image_ids=[img.id for img in original_images],
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                target_class=target_class,
                dimension=dimension,
            )

            await eval_logger.info(f"원본 데이터셋 추론 완료: {len(original_inference_results)}개")

            # Calculate metrics for original
            original_metrics = await self._calculate_metrics_with_gt(
                db=db,
                inference_results=original_inference_results,
                images=original_images,
                dataset=original_dataset,
                target_class=target_class,
                iou_threshold=iou_threshold,
                dimension=dimension,
            )

            # Log metrics based on whether target class is set
            if target_class and 'ap50' in original_metrics['overall']:
                await eval_logger.info(f"원본 타겟 클래스 '{target_class}' AP@50: {original_metrics['overall']['ap50']:.3f}")
            elif 'map50' in original_metrics['overall']:
                await eval_logger.info(f"원본 mAP@50: {original_metrics['overall']['map50']:.3f}")
        else:
            await eval_logger.info("Step 1/3: 원본 데이터셋 평가 생략 (base_dataset_id 미제공)")

        # ========== Step 2: Evaluate Attack Dataset ==========
        await eval_logger.info("Step 2/3: 공격 데이터셋 평가")

        if dimension == "3d":
            attack_ds_row = await db.execute(
                select(Dataset3D).where(Dataset3D.id == attack_output_dataset_id, Dataset3D.deleted_at.is_(None))
            )
            attack_dataset_obj = attack_ds_row.scalar_one_or_none()
            if not attack_dataset_obj:
                raise NotFoundError(resource=f"Attack output dataset {attack_output_dataset_id}")
            attack_images_row = await db.execute(
                select(Image3D).where(Image3D.dataset_id == attack_output_dataset_id, Image3D.deleted_at.is_(None))
            )
            attack_images = list(attack_images_row.scalars().all())
        else:
            attack_dataset_obj = await crud.dataset_2d.get(db, id=attack_output_dataset_id)
            if not attack_dataset_obj:
                raise NotFoundError(resource=f"Attack output dataset {attack_output_dataset_id}")
            attack_images = await crud.image_2d.get_by_dataset(db, dataset_id=attack_output_dataset_id)
        await eval_logger.info(f"공격 데이터셋: {len(attack_images)}개 이미지")

        # Run inference on attack
        attack_inference_results = await self.inference_service.run_inference(
            db=db,
            model_id=eval_run.model_id,
            image_ids=[img.id for img in attack_images],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            target_class=target_class,
            dimension=dimension,
        )

        await eval_logger.info(f"공격 데이터셋 추론 완료: {len(attack_inference_results)}개")

        # Calculate metrics for attack
        # GT selection logic differs by dimension:
        # - 2D: Always use original GT (filename mapping) - objects don't move, only perturbed
        # - 3D: Always use attack dataset's own GT - CARLA generates new GT for each attack
        if dimension == "3d":
            # 3D: Attack dataset has its own GT generated by CARLA
            await eval_logger.info("3D 공격 데이터셋 자체 GT 사용 (CARLA 생성)")
            attack_metrics = await self._calculate_metrics_with_gt(
                db=db,
                inference_results=attack_inference_results,
                images=attack_images,  # Use attack dataset's own images for GT
                dataset=attack_dataset_obj,  # Use attack dataset for class names
                use_filename_mapping=False,  # Direct image_id matching
                target_class=target_class,
                iou_threshold=iou_threshold,
                dimension=dimension,
            )
        else:
            # 2D: Use original GT (object positions unchanged, only perturbed)
            if original_images:
                await eval_logger.info("2D 공격 이미지 평가 (원본 GT 사용, filename mapping)")
                attack_metrics = await self._calculate_metrics_with_gt(
                    db=db,
                    inference_results=attack_inference_results,
                    images=original_images,  # Use original images for GT
                    dataset=original_dataset,  # Use original dataset for class names
                    use_filename_mapping=True,  # Map by filename instead of image_id
                    target_class=target_class,
                    iou_threshold=iou_threshold,
                    dimension=dimension,
                )
                # Check if mapping was successful
                has_metrics = (
                    attack_metrics.get('overall') and (
                        'map50' in attack_metrics['overall'] or 'ap50' in attack_metrics['overall']
                    )
                )
                if not has_metrics:
                    await eval_logger.error("공격 데이터셋 메트릭 계산 실패 - GT 매핑 실패")
                    raise ValidationError(detail="공격 이미지와 원본 이미지 매핑 실패. 파일명 패턴을 확인하세요.")
            else:
                # Should not happen for 2D (base_dataset_id is auto-populated)
                raise ValidationError(detail="2D 공격 데이터셋 평가에는 base_dataset_id가 필요합니다.")

        # Log metrics based on whether target class is set
        if target_class and 'ap50' in attack_metrics['overall']:
            await eval_logger.info(f"공격 타겟 클래스 '{target_class}' AP@50: {attack_metrics['overall']['ap50']:.3f}")
        elif 'map50' in attack_metrics['overall']:
            await eval_logger.info(f"공격 mAP@50: {attack_metrics['overall']['map50']:.3f}")

        # ========== Step 3: Compare Results (if original provided) ==========
        robustness_metrics = None
        if original_metrics:
            await eval_logger.info("Step 3/3: 결과 비교")

            robustness_metrics = calculate_robustness_metrics(
                original_metrics['overall'],
                attack_metrics['overall']
            )

            # Log metrics based on whether target class is set
            metric_label = f"타겟 클래스 '{target_class}' AP" if target_class else "mAP"
            await eval_logger.info(f"{metric_label} 하락: {robustness_metrics['delta_map']:.3f}")
            await eval_logger.info(f"하락률: {robustness_metrics['drop_percentage']:.1f}%")
            await eval_logger.info(f"Robustness ratio: {robustness_metrics['robustness_ratio']:.3f}")
        else:
            await eval_logger.info("Step 3/3: 결과 비교 생략 (원본 데이터셋 미제공)")

        # Save results
        # Case 1: Original dataset provided → Save BASE + ATTACK with robustness
        # Case 2: No original dataset → Save ATTACK only (no robustness)

        if original_metrics:
            # Combined metrics with robustness (for eval_runs backward compatibility)
            combined_metrics = {
                'overall': attack_metrics['overall'],
                'per_class': attack_metrics['per_class'],
                'robustness': robustness_metrics,
                'original_metrics': original_metrics['overall'],
            }

            # Save base evaluation results
            await self._save_evaluation_results(
                db=db,
                eval_run_id=eval_run.id,
                inference_results=original_inference_results,
                images=original_images,
                metrics=original_metrics,
                eval_logger=eval_logger,
                dataset=original_dataset,
                dataset_type=EvalDatasetType.BASE,
                target_class=target_class,
                iou_threshold=iou_threshold,
                dimension=dimension,
            )

            # Save attack evaluation results with robustness
            # NOTE: For 2D attacks, GT is in original_images (use filename mapping)
            # For 3D attacks, GT is in attack_images (direct image_id matching)
            gt_images = original_images if dimension == "2d" else attack_images
            await self._save_evaluation_results(
                db=db,
                eval_run_id=eval_run.id,
                inference_results=attack_inference_results,
                images=gt_images,  # ✅ Use original_images for 2D (GT location), attack_images for 3D
                metrics=combined_metrics,
                eval_logger=eval_logger,
                dataset=attack_dataset_obj,  # ✅ FIXED: Use attack dataset for dataset_id
                dataset_type=EvalDatasetType.ATTACK,
                target_class=target_class,
                iou_threshold=iou_threshold,
                dimension=dimension,
                class_names_dataset=original_dataset,  # ✅ Use original dataset for class names
            )
        else:
            # No original dataset - Save only attack results without robustness
            await self._save_evaluation_results(
                db=db,
                eval_run_id=eval_run.id,
                inference_results=attack_inference_results,
                images=attack_images,
                metrics=attack_metrics,  # No robustness metrics
                eval_logger=eval_logger,
                dataset=attack_dataset_obj,
                dataset_type=EvalDatasetType.ATTACK,
                target_class=target_class,
                iou_threshold=iou_threshold,
                dimension=dimension,
            )

    def _find_matching_image(
        self,
        attack_filename: str,
        original_images: List[Any],
        remove_suffix: bool = True,
    ) -> Optional[Any]:
        """
        Find matching original image for an attack image by filename.

        Args:
            attack_filename: Filename of attack image
            original_images: List of original images to search
            remove_suffix: If True, try to remove attack-related prefixes/suffixes

        Returns:
            Matching original image or None if not found
        """
        # Create filename mapping
        filename_to_image = {}
        for img in original_images:
            filename_to_image[img.file_name] = img

        if not remove_suffix:
            # Direct match only
            return filename_to_image.get(attack_filename)

        # Try multiple mapping strategies
        base_filename = attack_filename

        # Strategy 1: Remove attack prefixes
        for prefix in ['adv_', 'patched_', 'noise_', 'attacked_', 'adversarial_', 'attack_']:
            if base_filename.startswith(prefix):
                base_filename = base_filename[len(prefix):]
                break

        # Strategy 2: Remove suffixes (before extension)
        name_without_ext, ext = base_filename.rsplit('.', 1) if '.' in base_filename else (base_filename, '')
        for suffix in ['_adv', '_attacked', '_adversarial', '_attack']:
            if name_without_ext.endswith(suffix):
                name_without_ext = name_without_ext[:-len(suffix)]
                break
        base_filename = f"{name_without_ext}.{ext}" if ext else name_without_ext

        # Try exact match first
        image = filename_to_image.get(base_filename)
        if image:
            return image

        # Try original attack filename
        image = filename_to_image.get(attack_filename)
        if image:
            return image

        # Try fuzzy matching - find any filename containing the base name
        base_name_no_ext = name_without_ext
        for orig_filename, orig_img in filename_to_image.items():
            orig_name_no_ext = orig_filename.rsplit('.', 1)[0] if '.' in orig_filename else orig_filename
            if base_name_no_ext in orig_name_no_ext or orig_name_no_ext in base_name_no_ext:
                return orig_img

        return None

    async def _calculate_metrics_with_gt(
        self,
        db: AsyncSession,
        inference_results: List[Dict[str, Any]],
        images: List[Any],
        dataset: Any = None,
        use_filename_mapping: bool = False,
        target_class: Optional[str] = None,
        iou_threshold: float = 0.5,
        dimension: str = "2d",
    ) -> Dict[str, Any]:
        """
        Calculate metrics by comparing predictions with GT.

        Args:
            dataset: Dataset object to get class names from metadata
            use_filename_mapping: If True, map attack images to original images by filename
            target_class: If provided, only evaluate this specific class
        """
        # DEBUG: Log target class parameter
        logger.info(f"🔍 _calculate_metrics_with_gt called with target_class={repr(target_class)}")
        logger.info(f"🔍 target_class type: {type(target_class)}, bool: {bool(target_class)}")

        # Get class names from dataset metadata
        class_names = self._get_class_names_from_dataset(dataset) if dataset else None
        if class_names:
            logger.info(f"Using {len(class_names)} class names from dataset metadata: {class_names[:5]}")
        else:
            logger.info("Using COCO class names as fallback")

        # DEBUG: Check coordinate system alignment for first image
        if len(inference_results) > 0:
            first_result = inference_results[0]
            image_id = first_result["image_id"]
            image = next((img for img in images if str(img.id) == image_id), None)

            if image:
                logger.info("=" * 80)
                logger.info("COORDINATE SYSTEM DEBUG (First Image)")
                logger.info("=" * 80)
                logger.info(f"Image: {image.file_name}")
                logger.info(f"DB size: {getattr(image, 'width', 'N/A')}x{getattr(image, 'height', 'N/A')}")

                # Check actual file size
                from pathlib import Path
                from app.core.config import settings
                try:
                    image_path = Path(settings.STORAGE_ROOT) / image.storage_key
                    if image_path.exists():
                        import cv2
                        img = cv2.imread(str(image_path))
                        if img is not None:
                            actual_h, actual_w = img.shape[:2]
                            logger.info(f"Actual file size: {actual_w}x{actual_h}")

                            if actual_w != getattr(image, 'width', 0) or actual_h != getattr(image, 'height', 0):
                                logger.error("⚠️⚠️⚠️ SIZE MISMATCH DETECTED! ⚠️⚠️⚠️")
                                logger.error(f"DB size: {image.width}x{image.height}")
                                logger.error(f"File size: {actual_w}x{actual_h}")
                                logger.error("This WILL cause coordinate misalignment and IoU=0!")
                        else:
                            logger.warning("Could not read image file with cv2")
                    else:
                        logger.warning(f"Image file not found: {image_path}")
                except Exception as e:
                    logger.warning(f"Error checking image size: {e}")

                # Log first prediction
                if first_result.get("detections"):
                    first_pred = first_result["detections"][0]
                    logger.info(f"First Prediction: class={first_pred.get('class_name')}, "
                              f"bbox={first_pred.get('bbox')}, "
                              f"conf={first_pred.get('confidence'):.3f}")

                # Log first GT
                try:
                    if dimension == "3d":
                        annotations = await crud.annotation.get_by_image_3d(db, image_3d_id=UUID(image_id))
                    else:
                        annotations = await crud.annotation.get_by_image(db, image_2d_id=UUID(image_id))
                    if annotations:
                        first_ann = annotations[0]
                        logger.info(f"First GT (raw DB): class={first_ann.class_name}, "
                                  f"x={first_ann.bbox_x}, y={first_ann.bbox_y}, "
                                  f"w={first_ann.bbox_width}, h={first_ann.bbox_height}")
                        logger.info(f"Is normalized? {float(first_ann.bbox_x) <= 1.0 and float(first_ann.bbox_y) <= 1.0}")
                    else:
                        logger.warning("No GT annotations found for first image")
                except Exception as e:
                    logger.warning(f"Error loading GT for debug: {e}")

                logger.info("=" * 80)

        all_predictions: List[BoundingBox] = []
        all_ground_truths: List[BoundingBox] = []

        # Create filename to image mapping if needed
        if use_filename_mapping:
            filename_to_image = {}
            for img in images:
                filename = img.file_name
                filename_to_image[filename] = img

            logger.info(f"Original images filename mapping created: {len(filename_to_image)} entries")
            # Log first few for debugging
            if filename_to_image:
                sample_filenames = list(filename_to_image.keys())[:10]
                logger.info(f"Sample original filenames (first 10): {sample_filenames}")

        for result in inference_results:
            image_id = str(result["image_id"])

            # Find matching original image
            if use_filename_mapping:
                # Get attack image to extract filename
                attack_img = None
                if dimension == "3d":
                    from sqlalchemy import select
                    from app.models.dataset_3d import Image3D
                    attack_row = await db.execute(
                        select(Image3D).where(Image3D.id == UUID(image_id), Image3D.deleted_at.is_(None))
                    )
                    attack_img = attack_row.scalar_one_or_none()
                else:
                    attack_img = await crud.image_2d.get(db, id=UUID(image_id))
                if not attack_img:
                    logger.warning(f"Attack image not found: {image_id}")
                    continue

                # Try to map to original image by filename
                attack_filename = attack_img.file_name

                logger.info(f"Attempting to map attack image: {attack_filename}")

                # Try multiple mapping strategies
                base_filename = attack_filename
                original_found = False

                # Strategy 1: Remove attack prefixes
                for prefix in ['adv_', 'attacked_', 'adversarial_', 'attack_']:
                    if base_filename.startswith(prefix):
                        base_filename = base_filename[len(prefix):]
                        logger.info(f"  Removed prefix '{prefix}': {base_filename}")
                        break

                # Strategy 2: Remove suffixes (before extension)
                name_without_ext, ext = base_filename.rsplit('.', 1) if '.' in base_filename else (base_filename, '')
                for suffix in ['_adv', '_attacked', '_adversarial', '_attack']:
                    if name_without_ext.endswith(suffix):
                        name_without_ext = name_without_ext[:-len(suffix)]
                        logger.info(f"  Removed suffix '{suffix}': {name_without_ext}")
                        break
                base_filename = f"{name_without_ext}.{ext}" if ext else name_without_ext

                # Try exact match first
                image = filename_to_image.get(base_filename)
                if image:
                    logger.info(f"  ✓ Found match: {base_filename}")
                    original_found = True
                else:
                    # Try original attack filename
                    image = filename_to_image.get(attack_filename)
                    if image:
                        logger.info(f"  ✓ Found match with original filename: {attack_filename}")
                        original_found = True
                    else:
                        # Try fuzzy matching - find any filename containing the base name
                        base_name_no_ext = name_without_ext
                        for orig_filename, orig_img in filename_to_image.items():
                            orig_name_no_ext = orig_filename.rsplit('.', 1)[0] if '.' in orig_filename else orig_filename
                            if base_name_no_ext in orig_name_no_ext or orig_name_no_ext in base_name_no_ext:
                                image = orig_img
                                logger.info(f"  ✓ Found fuzzy match: {attack_filename} ≈ {orig_filename}")
                                original_found = True
                                break

                if not image:
                    logger.error(f"✗ No original image found for attack image: {attack_filename}")
                    logger.error(f"  Tried: {base_filename}, {attack_filename}, fuzzy matching")
                    # Show some examples of available original filenames
                    available_samples = list(filename_to_image.keys())[:5]
                    logger.error(f"  Available original filenames (first 5): {available_samples}")
                    continue

                # Use original image ID for GT lookup
                gt_image_id = str(image.id)
            else:
                image = next((img for img in images if str(img.id) == image_id), None)
                if not image:
                    continue
                gt_image_id = image_id

            pred_image_width = result.get("image_width") or getattr(image, 'width', 640)
            pred_image_height = result.get("image_height") or getattr(image, 'height', 640)
            gt_image_width = getattr(image, 'width', pred_image_width)
            gt_image_height = getattr(image, 'height', pred_image_height)

            # Parse predictions
            pred_boxes = result.get("detections", [])
            for detection in pred_boxes:
                try:
                    bbox = parse_detection_to_bbox(
                        detection,
                        image_id=image_id,
                        image_width=pred_image_width,
                        image_height=pred_image_height
                    )
                    # Filter by target class if specified
                    if target_class is None or bbox.class_name == target_class:
                        all_predictions.append(bbox)
                except Exception as e:
                    logger.warning(f"Failed to parse prediction bbox: {e}")

            # Load GT annotations (using gt_image_id which may be different from image_id for attack datasets)
            try:
                if dimension == "3d":
                    annotations = await crud.annotation.get_by_image_3d(db, image_3d_id=UUID(gt_image_id))
                else:
                    annotations = await crud.annotation.get_by_image(db, image_2d_id=UUID(gt_image_id))

                for ann in annotations:
                    try:
                        if ann.bbox_x is not None and ann.bbox_y is not None:
                            x_center_val = float(ann.bbox_x)
                            y_center_val = float(ann.bbox_y)
                            w_val = float(ann.bbox_width) if ann.bbox_width else 0
                            h_val = float(ann.bbox_height) if ann.bbox_height else 0

                            # Debug: Log raw bbox values from database
                            logger.debug(
                                "Raw DB bbox - image_id=%s, class=%s, x=%s, y=%s, w=%s, h=%s (image=%sx%s)",
                                gt_image_id,
                                ann.class_name,
                                x_center_val,
                                y_center_val,
                                w_val,
                                h_val,
                                gt_image_width,
                                gt_image_height,
                            )

                            is_normalized = (
                                0.0 <= x_center_val <= 1.0 and
                                0.0 <= y_center_val <= 1.0 and
                                0.0 <= w_val <= 1.0 and
                                0.0 <= h_val <= 1.0
                            )

                            if not is_normalized and gt_image_width and gt_image_height:
                                x_center_norm = x_center_val / gt_image_width
                                y_center_norm = y_center_val / gt_image_height
                                w_norm = w_val / gt_image_width
                                h_norm = h_val / gt_image_height
                            else:
                                x_center_norm = x_center_val
                                y_center_norm = y_center_val
                                w_norm = w_val
                                h_norm = h_val

                            x1 = max(0.0, min(1.0, x_center_norm - w_norm / 2))
                            y1 = max(0.0, min(1.0, y_center_norm - h_norm / 2))
                            x2 = max(0.0, min(1.0, x_center_norm + w_norm / 2))
                            y2 = max(0.0, min(1.0, y_center_norm + h_norm / 2))

                            # Skip invalid boxes
                            if x2 <= x1 or y2 <= y1:
                                logger.warning(f"Invalid bbox after conversion: ({x1}, {y1}, {x2}, {y2})")
                                continue

                            # Map class_X to actual class name using dataset metadata
                            class_name = self._map_class_name(ann.class_name, class_names)

                            logger.debug(f"Final GT bbox - class={class_name}, bbox=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

                            # Filter by target class if specified
                            if target_class is None or class_name == target_class:
                                bbox_obj = BoundingBox(
                                    x1=x1, y1=y1, x2=x2, y2=y2,
                                    class_name=class_name,
                                    confidence=1.0,
                                    image_id=image_id
                                )
                                all_ground_truths.append(bbox_obj)
                    except Exception as e:
                        logger.warning(f"Failed to parse GT bbox: {e}")
            except Exception as e:
                logger.warning(f"Failed to load annotations for image {image_id}: {e}")

        # Calculate metrics
        if len(all_ground_truths) == 0:
            logger.error("No ground truth annotations found - cannot calculate metrics")
            if use_filename_mapping:
                logger.error("This is likely due to filename mapping failure between attack and original images")
                logger.error("Please check that attack image filenames can be mapped to original filenames")
            return {'overall': {}, 'per_class': {}, 'iou_distribution': None}

        logger.info(f"Calculating metrics: {len(all_predictions)} predictions, {len(all_ground_truths)} GTs")

        # DEBUG: Log class distribution for target class filtering
        if target_class:
            pred_classes = {}
            gt_classes = {}
            for p in all_predictions:
                pred_classes[p.class_name] = pred_classes.get(p.class_name, 0) + 1
            for g in all_ground_truths:
                gt_classes[g.class_name] = gt_classes.get(g.class_name, 0) + 1

            logger.info(f"🎯 TARGET CLASS FILTERING ACTIVE: '{target_class}'")
            logger.info(f"📊 Prediction class distribution: {pred_classes}")
            logger.info(f"📊 GT class distribution: {gt_classes}")

            if target_class not in pred_classes and target_class not in gt_classes:
                logger.warning(f"⚠️ Target class '{target_class}' not found in predictions or GTs!")

            # Verify all predictions/GTs are target class
            non_target_preds = [p.class_name for p in all_predictions if p.class_name != target_class]
            non_target_gts = [g.class_name for g in all_ground_truths if g.class_name != target_class]

            if non_target_preds:
                logger.error(f"❌ FILTER FAILED: Found {len(non_target_preds)} predictions with non-target classes: {set(non_target_preds)}")
            if non_target_gts:
                logger.error(f"❌ FILTER FAILED: Found {len(non_target_gts)} GTs with non-target classes: {set(non_target_gts)}")

        # ============================================================
        # EVALUATION: Detailed metrics with per-class analysis
        # ============================================================
        logger.info("🚀 Using Evaluator for detailed metrics calculation")

        # Update evaluator IoU threshold
        self.evaluator.iou_threshold = iou_threshold

        # Perform comprehensive evaluation
        eval_results = self.evaluator.evaluate(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            target_class=target_class,
            include_iou_distribution=True,
        )

        overall_metrics = eval_results['overall']
        per_class_metrics = eval_results['per_class']
        iou_distribution = eval_results['iou_distribution']

        logger.info(f"✅ Evaluation complete: "
                   f"mAP@0.5:0.95={overall_metrics.get('map', 0.0):.4f}, "
                   f"mAP@0.5={overall_metrics.get('map50', 0.0):.4f}, "
                   f"{len(per_class_metrics)} classes evaluated")

        # Calculate PR curves for IoU 0.5, 0.75, 0.95
        from app.services.metrics_calculator import calculate_pr_curves_multiple_ious

        logger.info("📈 Calculating PR curves for IoU thresholds: 0.5, 0.75, 0.95...")
        pr_curves = calculate_pr_curves_multiple_ious(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            iou_thresholds=[0.5, 0.75, 0.95],
            num_points=101
        )

        logger.info(f"✅ PR curves calculated: "
                   f"IoU 0.5 AP={pr_curves['iou_0.50']['ap']:.4f}, "
                   f"IoU 0.75 AP={pr_curves['iou_0.75']['ap']:.4f}, "
                   f"IoU 0.95 AP={pr_curves['iou_0.95']['ap']:.4f}")

        # Rename metrics based on target class setting
        if target_class:
            # Single class evaluation: use "ap" instead of "map"
            logger.info(f"🎯 TARGET CLASS MODE: Converting 'map' to 'ap' for target_class='{target_class}'")
            logger.info(f"🎯 Original overall_metrics keys: {list(overall_metrics.keys())}")
            logger.info(f"🎯 map={overall_metrics.get('map'):.4f}, map50={overall_metrics.get('map50'):.4f}")

            result = {
                'overall': {
                    'ap': overall_metrics['map'],        # map → ap (IoU 평균)
                    'ap50': overall_metrics['map50'],    # AP@50
                    'ap75': overall_metrics['map75'],    # AP@75
                    'precision': overall_metrics['precision'],
                    'recall': overall_metrics['recall'],
                    'f1': overall_metrics['f1'],
                    'total_gt': overall_metrics.get('total_gt', 0),
                    'total_pred': overall_metrics.get('total_pred', 0),
                    'target_class': target_class,  # Explicitly mark target class
                },
                'per_class': per_class_metrics,
                'iou_distribution': iou_distribution,
                'pr_curves': pr_curves,  # NEW: PR curves for IoU 0.5, 0.75, 0.95
            }
            logger.info(f"🎯 Converted overall keys: {list(result['overall'].keys())}")
            logger.info(f"🎯 ap={result['overall']['ap']:.4f}, ap50={result['overall']['ap50']:.4f}")
            return result
        else:
            # Multi-class evaluation: keep "map" naming
            logger.info(f"📊 MULTI-CLASS MODE: Keeping 'map' naming (no target class)")
            return {
                'overall': overall_metrics,
                'per_class': per_class_metrics,
                'iou_distribution': iou_distribution,
                'pr_curves': pr_curves,  # NEW: PR curves for IoU 0.5, 0.75, 0.95
            }

    async def _save_evaluation_results(
        self,
        db: AsyncSession,
        eval_run_id: UUID,
        inference_results: List[Dict[str, Any]],
        images: List[Any],
        metrics: Dict[str, Any],
        eval_logger: SSELogger,
        dataset: Any = None,
        dataset_type: Optional[EvalDatasetType] = None,  # NEW: EvalDatasetType enum
        target_class: Optional[str] = None,  # NEW: Target class for filtering
        iou_threshold: float = 0.5,  # IoU threshold for matching
        dimension: str = "2d",
        class_names_dataset: Optional[Any] = None,  # NEW: For class name resolution (useful for attack datasets)
    ) -> None:
        """
        Save evaluation results to database using new structure.

        Creates eval_dataset_result first, then links eval_items and eval_class_metrics to it.

        Args:
            dataset: The dataset being evaluated (determines dataset_id in eval_dataset_results)
            class_names_dataset: Optional dataset to use for class name resolution.
                                 Useful when evaluating attack datasets (dataset=attack, class_names_dataset=original)
        """
        await eval_logger.status("평가 결과 저장 중...")

        # Get class names from class_names_dataset if provided, otherwise from dataset
        class_names = self._get_class_names_from_dataset(
            class_names_dataset or dataset
        ) if (class_names_dataset or dataset) else None

        # Extract target_class from metrics if not provided
        if not target_class and 'overall' in metrics:
            target_class = metrics['overall'].get('target_class')

        # ============================================================
        # STEP 1: Create eval_dataset_result
        # ============================================================
        from app.schemas.evaluation import EvalDatasetResultCreate, DatasetDimension

        # Determine dataset_id from dataset object
        dataset_id = dataset.id if dataset else None
        if not dataset_id:
            raise ValueError("dataset_id is required for saving evaluation results")

        # Determine dimension enum
        dimension_enum = DatasetDimension.THREE_D if dimension == "3d" else DatasetDimension.TWO_D

        # Extract metrics for this dataset (without robustness/original_metrics)
        # Note: robustness and original_metrics are at the top level of metrics dict,
        # not inside metrics['overall'], so we extract them separately
        dataset_metrics_summary = metrics.get('overall', {})

        dataset_iou_distribution = metrics.get('iou_distribution')
        dataset_pr_curves = metrics.get('pr_curves')  # NEW: PR curves for IoU 0.5, 0.75, 0.95

        # Create eval_dataset_result
        dataset_result_create = EvalDatasetResultCreate(
            eval_run_id=eval_run_id,
            dataset_type=dataset_type,
            dataset_id=dataset_id,
            dataset_dimension=dimension_enum,
            metrics_summary=dataset_metrics_summary,
            iou_distribution=dataset_iou_distribution,
            pr_curves=dataset_pr_curves,  # NEW: Store PR curves
        )

        dataset_result = await crud.evaluation.create_eval_dataset_result(db, dataset_result_create)
        eval_dataset_result_id = dataset_result.id

        logger.info(f"✅ Created eval_dataset_result with PR curves: {eval_dataset_result_id} (type={dataset_type}, dataset_id={dataset_id})")

        # Save evaluation items
        for result in inference_results:
            image_id = UUID(result["image_id"])

            # Load GT for this image
            # For 2D attack datasets: image_id is attack image, but GT is in original images
            # Need to map by filename
            ground_truth = []
            image = None
            image_width = 640  # Default value
            image_height = 640  # Default value
            gt_image_id = image_id  # Default: same as inference image_id

            # For 2D attack datasets: Find original image by filename mapping
            if dataset_type == EvalDatasetType.ATTACK and dimension == "2d":
                # Find attack image by ID to get its filename
                attack_image = await crud.image_2d.get(db, id=image_id)
                if attack_image:
                    attack_filename = attack_image.file_name
                    # Map to original image by filename (strip attack suffix)
                    original_image = self._find_matching_image(
                        attack_filename, images, remove_suffix=True
                    )
                    if original_image:
                        gt_image_id = original_image.id
                        image = original_image
                        logger.debug(f"Mapped attack image {attack_filename} -> original {original_image.file_name}")
                    else:
                        logger.warning(f"Could not find original image for attack file {attack_filename}")
                else:
                    logger.warning(f"Could not find attack image with ID {image_id}")
            else:
                # Direct mapping: image_id is the same for both inference and GT
                image = next((img for img in images if img.id == image_id), None)

            try:
                if dimension == "3d":
                    annotations = await crud.annotation.get_by_image_3d(db, image_3d_id=gt_image_id)
                else:
                    annotations = await crud.annotation.get_by_image(db, image_2d_id=gt_image_id)

                if image and annotations:
                    image_width = getattr(image, 'width', 640)
                    image_height = getattr(image, 'height', 640)

                    for ann in annotations:
                        if ann.bbox_x is not None and ann.bbox_y is not None:
                            x_center_val = float(ann.bbox_x)
                            y_center_val = float(ann.bbox_y)
                            w_val = float(ann.bbox_width) if ann.bbox_width else 0
                            h_val = float(ann.bbox_height) if ann.bbox_height else 0

                            is_normalized = (
                                0.0 <= x_center_val <= 1.0 and
                                0.0 <= y_center_val <= 1.0 and
                                0.0 <= w_val <= 1.0 and
                                0.0 <= h_val <= 1.0
                            )

                            if not is_normalized and image_width and image_height:
                                x_center_norm = x_center_val / image_width
                                y_center_norm = y_center_val / image_height
                                w_norm = w_val / image_width
                                h_norm = h_val / image_height
                            else:
                                x_center_norm = x_center_val
                                y_center_norm = y_center_val
                                w_norm = w_val
                                h_norm = h_val

                            x1 = max(0.0, min(1.0, x_center_norm - w_norm / 2))
                            y1 = max(0.0, min(1.0, y_center_norm - h_norm / 2))
                            x2 = max(0.0, min(1.0, x_center_norm + w_norm / 2))
                            y2 = max(0.0, min(1.0, y_center_norm + h_norm / 2))

                            if x2 <= x1 or y2 <= y1:
                                logger.warning(f"Invalid bbox in save: ({x1}, {y1}, {x2}, {y2})")
                                continue

                            class_name = self._map_class_name(ann.class_name, class_names)

                            # Filter by target class if specified
                            if target_class is None or class_name == target_class:
                                ground_truth.append({
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "class_name": class_name,
                                    "class_id": ann.class_index,
                                    "confidence": float(ann.confidence) if ann.confidence else 1.0
                                })
            except Exception as e:
                logger.warning(f"Failed to load ground truth for image {image_id}: {e}")

            # Filter predictions by target class if specified
            predictions = result.get("detections", [])
            if target_class:
                predictions = [p for p in predictions if p.get('class_name') == target_class]

            # Calculate per-image metrics (AP, precision, recall)
            image_metrics = self._calculate_image_metrics(
                predictions=predictions,
                ground_truths=ground_truth,
                image_id=str(image_id),
                image_width=image_width,
                image_height=image_height,
                iou_threshold=iou_threshold
            )
            image_metrics["inference_time_ms"] = result.get("inference_time_ms", 0)
            image_metrics["status"] = result.get("status", "unknown")

            eval_item = EvalItemCreate(
                run_id=eval_run_id,
                eval_dataset_result_id=eval_dataset_result_id,  # NEW: Link to dataset result
                image_2d_id=image_id if dimension != "3d" else None,
                image_3d_id=image_id if dimension == "3d" else None,
                dataset_type=dataset_type,  # DEPRECATED: Kept for backward compatibility
                ground_truth=ground_truth,
                prediction=result.get("detections", []),
                metrics=image_metrics,
            )
            await crud.evaluation.create_eval_item(db, eval_item)

        # Save per-class metrics
        per_class_metrics = metrics.get('per_class', {})
        per_class_iou_dist = metrics.get('per_class_iou_distribution', {})

        if per_class_metrics:
            await eval_logger.status(f"클래스별 메트릭 저장 중... ({len(per_class_metrics)}개 클래스)")
            logger.info(f"Saving per-class metrics for {len(per_class_metrics)} classes")

            from app.schemas.evaluation import EvalClassMetricsCreate

            for class_name, class_metrics in per_class_metrics.items():
                # Remove iou_distribution from class_metrics if it exists (it's stored separately)
                class_metrics_copy = {k: v for k, v in class_metrics.items() if k != 'iou_distribution'}

                class_metrics_create = EvalClassMetricsCreate(
                    run_id=eval_run_id,
                    eval_dataset_result_id=eval_dataset_result_id,  # NEW: Link to dataset result
                    class_name=class_name,
                    dataset_type=dataset_type,  # DEPRECATED: Kept for backward compatibility
                    metrics=class_metrics_copy,
                    iou_distribution=per_class_iou_dist.get(class_name)
                )
                await crud.evaluation.create_eval_class_metrics(db, class_metrics_create)

            logger.info(f"✅ Saved {len(per_class_metrics)} per-class metrics records")

        # ============================================================
        # STEP 3: Update evaluation run (DEPRECATED fields for backward compatibility)
        # ============================================================
        # NOTE: metrics_summary and iou_distribution in eval_runs are DEPRECATED.
        # The actual data is in eval_dataset_results.
        # However, we still update eval_runs for backward compatibility and to store
        # robustness information (which is a comparison between base and attack).

        if 'robustness' in metrics and 'original_metrics' in metrics:
            # Post-attack: save robustness and original_metrics for comparison
            # The actual dataset-specific metrics are in eval_dataset_results
            metrics_to_save = {
                'robustness': metrics['robustness'],
                'original_metrics': metrics['original_metrics']
            }
        else:
            # Pre-attack or single dataset: save overall metrics for backward compatibility
            metrics_to_save = metrics.get('overall', {})

        # Extract overall IoU distribution (for backward compatibility)
        overall_iou_dist = metrics.get('iou_distribution')

        await crud.evaluation.update_eval_run(
            db, eval_run_id,
            EvalRunUpdate(
                metrics_summary=metrics_to_save,
                iou_distribution=overall_iou_dist
            )
        )

        await db.commit()
        logger.info(f"✅ Evaluation results saved successfully (run_id={eval_run_id}, dataset_result_id={eval_dataset_result_id})")

    def _calculate_image_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        image_id: str,
        image_width: int = 640,
        image_height: int = 640,
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculate per-image metrics including AP, precision, recall.

        Args:
            predictions: List of prediction dicts with bbox and class_name
            ground_truths: List of GT dicts with bbox and class_name
            image_id: Image identifier
            image_width: Image width for coordinate normalization
            image_height: Image height for coordinate normalization

        Returns:
            Dictionary containing:
            - ap: Average Precision (mean of AP@0.5:0.95)
            - ap50: AP at IoU=0.5
            - ap75: AP at IoU=0.75
            - precision: Precision at IoU=0.5
            - recall: Recall at IoU=0.5
            - f1: F1 score at IoU=0.5
            - tp_count: True Positive count
            - fp_count: False Positive count
            - fn_count: False Negative count
        """
        from app.services.metrics_calculator import (
            BoundingBox,
            parse_detection_to_bbox,
            calculate_ap_ar_at_iou,
            calculate_ap_ar_all_ious,
        )

        # Convert predictions to BoundingBox objects
        pred_boxes = []
        for pred in predictions:
            try:
                bbox = parse_detection_to_bbox(
                    pred,
                    image_id=image_id,
                    image_width=image_width,
                    image_height=image_height
                )
                pred_boxes.append(bbox)
            except Exception as e:
                logger.warning(f"Failed to parse prediction bbox: {e}")

        # Convert ground truths to BoundingBox objects
        gt_boxes = []
        for gt in ground_truths:
            try:
                # GT format: {"bbox": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}, "class_name": ...}
                gt_detection = {
                    "bbox": gt.get("bbox", {}),
                    "class_name": gt.get("class_name", "unknown"),
                    "confidence": 1.0  # GT always has confidence 1.0
                }
                bbox = parse_detection_to_bbox(
                    gt_detection,
                    image_id=image_id,
                    image_width=image_width,
                    image_height=image_height
                )
                gt_boxes.append(bbox)
            except Exception as e:
                logger.warning(f"Failed to parse GT bbox: {e}")

        # If no GT, return empty metrics
        if len(gt_boxes) == 0:
            return {
                "ap": 0.0,
                "ap50": 0.0,
                "ap75": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp_count": 0,
                "fp_count": len(pred_boxes),
                "fn_count": 0,
            }

        # If no predictions, return empty metrics with FN count
        if len(pred_boxes) == 0:
            return {
                "ap": 0.0,
                "ap50": 0.0,
                "ap75": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp_count": 0,
                "fp_count": 0,
                "fn_count": len(gt_boxes),
            }

        # Calculate AP across IoU thresholds
        ap_metrics = calculate_ap_ar_all_ious(pred_boxes, gt_boxes)

        # Calculate metrics at specified IoU threshold for precision/recall
        metrics_50 = calculate_ap_ar_at_iou(pred_boxes, gt_boxes, iou_threshold=iou_threshold)

        # Calculate TP, FP, FN counts at specified IoU threshold
        # TP: correctly matched predictions
        # FP: predictions without match
        # FN: ground truths without match
        from app.services.metrics_calculator import calculate_iou

        gt_matched = [False] * len(gt_boxes)
        tp_count = 0
        fp_count = 0

        # Sort predictions by confidence descending
        sorted_preds = sorted(pred_boxes, key=lambda x: x.confidence, reverse=True)

        for pred in sorted_preds:
            # Find best matching GT
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if gt.class_name != pred.class_name:
                    continue
                if gt_matched[gt_idx]:
                    continue

                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Match if IoU >= threshold
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_count += 1
                gt_matched[best_gt_idx] = True
            else:
                fp_count += 1

        # FN: unmatched ground truths
        fn_count = sum(1 for matched in gt_matched if not matched)

        # Calculate F1 score
        precision = metrics_50["precision"]
        recall = metrics_50["recall"]
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "ap": float(ap_metrics["map"]),  # Mean AP across IoU 0.5:0.95
            "ap50": float(ap_metrics["map50"]),
            "ap75": float(ap_metrics["map75"]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
        }


# Global instance
evaluation_service = EvaluationService()
