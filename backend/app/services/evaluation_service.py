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

logger = logging.getLogger(__name__)


class EvaluationService:
    """Evaluation service with redesigned logic."""

    def __init__(self):
        self.inference_service = InferenceService()
        self.sse_manager = SSEManager()

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
        iou_threshold: float = 0.45,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Execute evaluation with new logic:
        - Base dataset: Single evaluation with GT
        - Attack dataset: Dual evaluation (original + attack) then compare
        """
        eval_logger = SSELogger(logger, self.sse_manager, session_id)
        await eval_logger.status("평가 시작 중...", eval_run_id=str(eval_run_id))

        # Get evaluation run
        eval_run = await crud.evaluation.get_eval_run(db, eval_run_id)
        if not eval_run:
            await eval_logger.error("평가 실행을 찾을 수 없습니다")
            raise NotFoundError(resource=f"Evaluation run {eval_run_id}")

        await eval_logger.info(f"평가 이름: {eval_run.name}")

        # Update status to running
        await crud.evaluation.update_eval_run(
            db, eval_run_id,
            EvalRunUpdate(status=EvalStatus.RUNNING, started_at=datetime.utcnow())
        )
        await db.commit()

        model_id = eval_run.model_id

        try:
            await self.inference_service.load_model(db, model_id)

            if eval_run.attack_dataset_id:
                # Mode 2: Attack dataset evaluation
                await self._evaluate_attack_dataset(
                    db, eval_run, eval_logger, conf_threshold, iou_threshold
                )
            else:
                # Mode 1: Base dataset evaluation
                await self._evaluate_base_dataset(
                    db, eval_run, eval_logger, conf_threshold, iou_threshold
                )

            # Mark as completed
            await crud.evaluation.update_eval_run(
                db, eval_run_id,
                EvalRunUpdate(status=EvalStatus.COMPLETED, ended_at=datetime.utcnow())
            )
            await db.commit()

            await eval_logger.success("평가 완료!")

            if session_id:
                await self.sse_manager.send_event(session_id, {
                    "type": "complete",
                    "message": "평가 완료",
                    "eval_run_id": str(eval_run_id),
                    "timestamp": datetime.now().isoformat()
                })

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
    ) -> None:
        """
        Mode 1: Base dataset evaluation
        - Load base dataset
        - Run inference
        - Compare with GT
        - Calculate metrics
        """
        await eval_logger.info("기준 데이터셋 평가 시작")

        # Get base dataset
        dataset = await crud.dataset_2d.get(db, id=eval_run.base_dataset_id)
        if not dataset:
            raise NotFoundError(resource=f"Dataset {eval_run.base_dataset_id}")

        images = await crud.image_2d.get_by_dataset(db, dataset_id=eval_run.base_dataset_id)
        if not images:
            raise ValidationError(detail=f"No images in dataset {eval_run.base_dataset_id}")

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
        )

        await eval_logger.status("모델 추론 완료", completed_images=len(inference_results))
        await eval_logger.status("메트릭 계산 중...")

        # Calculate metrics with GT
        metrics = await self._calculate_metrics_with_gt(
            db=db,
            inference_results=inference_results,
            images=images,
            dataset=dataset,
        )

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
        )

    async def _evaluate_attack_dataset(
        self,
        db: AsyncSession,
        eval_run: Any,
        eval_logger: SSELogger,
        conf_threshold: float,
        iou_threshold: float,
    ) -> None:
        """
        Mode 2: Attack dataset evaluation
        - Find original dataset from attack dataset
        - Evaluate original dataset (Step 1)
        - Evaluate attack dataset (Step 2)
        - Compare results (Step 3)
        """
        await eval_logger.info("공격 데이터셋 평가 시작 (Dual Evaluation)")

        # Get attack dataset info
        attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
        if not attack_dataset:
            raise NotFoundError(resource=f"Attack dataset {eval_run.attack_dataset_id}")

        # Get original dataset ID and attack output dataset ID
        original_dataset_id = attack_dataset.base_dataset_id
        attack_output_dataset_id = attack_dataset.output_dataset_id

        if not attack_output_dataset_id:
            raise ValidationError(
                detail=f"Attack dataset {eval_run.attack_dataset_id} is missing output_dataset_id. "
                "Please run Phase 1 migration to fix legacy data."
            )

        await eval_logger.info(f"원본 데이터셋 ID: {original_dataset_id}")
        await eval_logger.info(f"공격 데이터셋 ID: {attack_output_dataset_id}")

        # ========== Step 1: Evaluate Original Dataset ==========
        await eval_logger.info("Step 1/3: 원본 데이터셋 평가")

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
        )

        await eval_logger.info(f"원본 데이터셋 추론 완료: {len(original_inference_results)}개")

        # Calculate metrics for original
        original_metrics = await self._calculate_metrics_with_gt(
            db=db,
            inference_results=original_inference_results,
            images=original_images,
            dataset=original_dataset,
        )

        await eval_logger.info(f"원본 mAP@50: {original_metrics['overall']['map50']:.3f}")

        # ========== Step 2: Evaluate Attack Dataset ==========
        await eval_logger.info("Step 2/3: 공격 데이터셋 평가")

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
        )

        await eval_logger.info(f"공격 데이터셋 추론 완료: {len(attack_inference_results)}개")

        # Calculate metrics for attack (using ORIGINAL GT)
        # Map attack images to original images by filename
        await eval_logger.info("공격 이미지와 원본 이미지 매핑 중...")

        attack_metrics = await self._calculate_metrics_with_gt(
            db=db,
            inference_results=attack_inference_results,
            images=original_images,  # Use original images for GT
            dataset=original_dataset,  # Use original dataset for class names
            use_filename_mapping=True,  # Map by filename instead of image_id
        )

        # Check if metrics were calculated successfully
        if not attack_metrics.get('overall') or 'map50' not in attack_metrics['overall']:
            await eval_logger.error("공격 데이터셋 메트릭 계산 실패 - GT를 찾을 수 없습니다")
            raise ValidationError(detail="공격 이미지와 원본 이미지 매핑 실패. 파일명 패턴을 확인하세요.")

        await eval_logger.info(f"공격 mAP@50: {attack_metrics['overall']['map50']:.3f}")

        # ========== Step 3: Compare Results ==========
        await eval_logger.info("Step 3/3: 결과 비교")

        robustness_metrics = calculate_robustness_metrics(
            original_metrics['overall'],
            attack_metrics['overall']
        )

        await eval_logger.info(f"mAP 하락: {robustness_metrics['delta_map']:.3f}")
        await eval_logger.info(f"하락률: {robustness_metrics['drop_percentage']:.1f}%")
        await eval_logger.info(f"Robustness ratio: {robustness_metrics['robustness_ratio']:.3f}")

        # Save results with robustness metrics
        combined_metrics = {
            'overall': attack_metrics['overall'],
            'per_class': attack_metrics['per_class'],
            'robustness': robustness_metrics,
            'original_metrics': original_metrics['overall'],
        }

        # NEW: Save BOTH base and attack evaluation results
        # Step 1: Save base evaluation results
        await self._save_evaluation_results(
            db=db,
            eval_run_id=eval_run.id,
            inference_results=original_inference_results,
            images=original_images,
            metrics=original_metrics,
            eval_logger=eval_logger,
            dataset=original_dataset,
            dataset_type=EvalDatasetType.BASE,  # Mark as base dataset
        )

        # Step 2: Save attack evaluation results
        await self._save_evaluation_results(
            db=db,
            eval_run_id=eval_run.id,
            inference_results=attack_inference_results,
            images=attack_images,
            metrics=combined_metrics,
            eval_logger=eval_logger,
            dataset=original_dataset,  # Use original dataset for class names
            dataset_type=EvalDatasetType.ATTACK,  # Mark as attack dataset
        )

    async def _calculate_metrics_with_gt(
        self,
        db: AsyncSession,
        inference_results: List[Dict[str, Any]],
        images: List[Any],
        dataset: Any = None,
        use_filename_mapping: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate metrics by comparing predictions with GT.

        Args:
            dataset: Dataset object to get class names from metadata
            use_filename_mapping: If True, map attack images to original images by filename
        """
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
                sample_filenames = list(filename_to_image.keys())[:5]
                logger.info(f"Sample original filenames: {sample_filenames}")

        for result in inference_results:
            image_id = str(result["image_id"])

            # Find matching original image
            if use_filename_mapping:
                # Get attack image to extract filename
                attack_img = await crud.image_2d.get(db, id=UUID(image_id))
                if not attack_img:
                    logger.warning(f"Attack image not found: {image_id}")
                    continue

                # Try to map to original image by filename
                attack_filename = attack_img.file_name

                # Remove attack prefixes and suffixes
                base_filename = attack_filename
                # Remove prefixes: adv_, attacked_, adversarial_
                for prefix in ['adv_', 'attacked_', 'adversarial_']:
                    if base_filename.startswith(prefix):
                        base_filename = base_filename[len(prefix):]
                        break

                # Remove suffixes: _adv, _attacked, _adversarial (before extension)
                name_without_ext, ext = base_filename.rsplit('.', 1) if '.' in base_filename else (base_filename, '')
                for suffix in ['_adv', '_attacked', '_adversarial']:
                    if name_without_ext.endswith(suffix):
                        name_without_ext = name_without_ext[:-len(suffix)]
                        break
                base_filename = f"{name_without_ext}.{ext}" if ext else name_without_ext

                logger.info(f"Mapping: {attack_filename} → {base_filename}")
                image = filename_to_image.get(base_filename) or filename_to_image.get(attack_filename)

                if not image:
                    logger.warning(f"No original image found for attack image: {attack_filename}")
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
                    all_predictions.append(bbox)
                except Exception as e:
                    logger.warning(f"Failed to parse prediction bbox: {e}")

            # Load GT annotations (using gt_image_id which may be different from image_id for attack datasets)
            try:
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
            logger.warning("No ground truth annotations found")
            return {'overall': {}, 'per_class': {}}

        logger.info(f"Calculating metrics: {len(all_predictions)} predictions, {len(all_ground_truths)} GTs")

        overall_metrics = calculate_overall_metrics(all_predictions, all_ground_truths)

        # Skip per-class metrics calculation - not needed
        # class_names = set([gt.class_name for gt in all_ground_truths])
        # class_names.update([pred.class_name for pred in all_predictions])
        # per_class_metrics = {}
        # for class_name in class_names:
        #     per_class_metrics[class_name] = calculate_class_metrics(
        #         all_predictions, all_ground_truths, class_name
        #     )

        return {
            'overall': overall_metrics,
            'per_class': {},  # Empty dict
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
    ) -> None:
        """Save evaluation results to database."""
        await eval_logger.status("평가 결과 저장 중...")

        # Get class names from dataset metadata
        class_names = self._get_class_names_from_dataset(dataset) if dataset else None

        # Save evaluation items
        for result in inference_results:
            image_id = UUID(result["image_id"])

            # Load GT for this image
            ground_truth = []
            try:
                annotations = await crud.annotation.get_by_image(db, image_2d_id=image_id)
                image = next((img for img in images if img.id == image_id), None)

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

                            ground_truth.append({
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "class_name": class_name,
                                "class_id": ann.class_index,
                                "confidence": float(ann.confidence) if ann.confidence else 1.0
                            })
            except Exception as e:
                logger.warning(f"Failed to load ground truth for image {image_id}: {e}")

            eval_item = EvalItemCreate(
                run_id=eval_run_id,
                image_2d_id=image_id,
                dataset_type=dataset_type,  # NEW: Set dataset_type
                ground_truth=ground_truth,
                prediction=result.get("detections", []),
                metrics={
                    "inference_time_ms": result.get("inference_time_ms", 0),
                    "status": result.get("status", "unknown"),
                },
            )
            await crud.evaluation.create_eval_item(db, eval_item)

        # Skip class metrics - not needed
        # await eval_logger.status(f"클래스별 메트릭 저장 중... ({len(metrics.get('per_class', {}))}개 클래스)")
        # for class_name, class_metrics in metrics.get('per_class', {}).items():
        #     class_metrics_create = EvalClassMetricsCreate(
        #         run_id=eval_run_id,
        #         class_name=class_name,
        #         metrics=class_metrics,
        #     )
        #     await crud.evaluation.create_eval_class_metrics(db, class_metrics_create)

        # Update evaluation run with results
        # For post_attack phase, save the full combined_metrics including original_metrics and robustness
        # For other phases, just save overall metrics
        if 'robustness' in metrics and 'original_metrics' in metrics:
            # Post-attack: save combined metrics with robustness info
            metrics_to_save = {
                **metrics['overall'],
                'robustness': metrics['robustness'],
                'original_metrics': metrics['original_metrics']
            }
        else:
            # Pre-attack or other: just save overall metrics
            metrics_to_save = metrics.get('overall', {})

        await crud.evaluation.update_eval_run(
            db, eval_run_id,
            EvalRunUpdate(metrics_summary=metrics_to_save)
        )
        await db.commit()


# Global instance
evaluation_service = EvaluationService()
