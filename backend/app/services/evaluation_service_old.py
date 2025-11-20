"""
Evaluation execution service for running model evaluations.
"""
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from app import crud
from app.models.evaluation import EvalStatus
from app.schemas.evaluation import (
    EvalRunUpdate,
    EvalItemCreate,
    EvalClassMetricsCreate,
)
from app.services.inference_service import InferenceService
from app.services.sse_support import SSEManager, SSELogger
from app.services.metrics_calculator import (
    BoundingBox,
    parse_detection_to_bbox,
    calculate_overall_metrics,
    calculate_class_metrics,
    calculate_robustness_metrics,
)
from app.core.exceptions import NotFoundError, ValidationError
from app.ai.estimators.object_detection.class_mapper import COCO_CLASSES

logger = logging.getLogger(__name__)


def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    Boxes are in format: {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    """
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_ap(
    precisions: List[float],
    recalls: List[float],
    use_07_metric: bool = False
) -> float:
    """Calculate Average Precision (AP)."""
    if not precisions or not recalls:
        return 0.0

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = [recalls[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]

    if use_07_metric:
        # VOC 2007 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(np.array(recalls) >= t) == 0:
                p = 0
            else:
                p = np.max(np.array(precisions)[np.array(recalls) >= t])
            ap += p / 11.0
    else:
        # VOC 2010+ all-point interpolation
        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)

        ap = 0.0
        for i in i_list:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


class EvaluationService:
    """Service for running model evaluations."""

    def __init__(self):
        self.inference_service = InferenceService()
        self.sse_manager = SSEManager()

    async def execute_evaluation(
        self,
        db: AsyncSession,
        eval_run_id: UUID,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Execute an evaluation run.
        This is the main entry point for running evaluations.
        """
        # Create SSE logger
        eval_logger = SSELogger(logger, self.sse_manager, session_id)

        await eval_logger.status("평가 시작 중...", eval_run_id=str(eval_run_id))

        # Get evaluation run
        eval_run = await crud.evaluation.get_eval_run(db, eval_run_id)
        if not eval_run:
            await eval_logger.error("평가 실행을 찾을 수 없습니다")
            raise NotFoundError(resource=f"Evaluation run {eval_run_id}")

        await eval_logger.info(f"평가 이름: {eval_run.name}")
        await eval_logger.info(f"평가 단계: {'정상 데이터' if eval_run.phase == 'pre_attack' else '공격 데이터'}")

        # Update status to running
        await crud.evaluation.update_eval_run(
            db,
            eval_run_id,
            EvalRunUpdate(
                status=EvalStatus.RUNNING,
                started_at=datetime.utcnow(),
            ),
        )
        await db.commit()

        await eval_logger.status("평가 실행 중...", status="running")

        model_id = eval_run.model_id
        try:
            # Load the model explicitly for this evaluation
            await self.inference_service.load_model(db, model_id)

            # Determine evaluation mode
            # Mode 1: Base dataset only - simple evaluation with GT
            # Mode 2: Attack dataset - dual evaluation (original + attack) then compare

            if eval_run.attack_dataset_id:
                # Post-attack: get output_dataset_id from attack_dataset parameters
                attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
                if not attack_dataset:
                    await eval_logger.error(f"공격 데이터셋을 찾을 수 없습니다: {eval_run.attack_dataset_id}")
                    raise NotFoundError(resource=f"Attack dataset {eval_run.attack_dataset_id}")

                # Extract output_dataset_id from parameters
                output_dataset_id = attack_dataset.parameters.get('output_dataset_id')
                if not output_dataset_id:
                    await eval_logger.error("공격 데이터셋에 output_dataset_id가 없습니다")
                    raise ValidationError(detail="No output_dataset_id in attack dataset parameters")

                dataset_id_for_images = UUID(output_dataset_id)
                await eval_logger.info(f"Post-attack 평가: 공격된 이미지 사용 (output_dataset: {output_dataset_id})")

                # Load pre-attack evaluation results for comparison
                await eval_logger.info("Pre-attack 평가 결과 로딩 중...")
                await eval_logger.info(f"검색 조건 - model_id: {model_id}, base_dataset_id: {eval_run.base_dataset_id}")

                pre_attack_runs = await crud.evaluation.get_eval_runs(
                    db=db,
                    model_id=model_id,
                    base_dataset_id=eval_run.base_dataset_id,
                    attack_dataset_id=None,  # Pre-attack has no attack_dataset_id
                    limit=1
                )

                await eval_logger.info(f"검색된 pre-attack runs: {len(pre_attack_runs[0]) if pre_attack_runs and pre_attack_runs[0] else 0}개")

                if pre_attack_runs and len(pre_attack_runs[0]) > 0:
                    pre_attack_run = pre_attack_runs[0][0]
                    await eval_logger.info(f"Pre-attack run 발견: ID={pre_attack_run.id}, status={pre_attack_run.status}")

                    if pre_attack_run.status == EvalStatus.COMPLETED:
                        # Load pre-attack evaluation items (predictions)
                        pre_attack_items, total = await crud.evaluation.get_eval_items(
                            db=db,
                            run_id=pre_attack_run.id,
                            skip=0,
                            limit=10000  # Load all
                        )
                        await eval_logger.info(f"Pre-attack items 로딩: {len(pre_attack_items)}개 (total: {total})")

                        baseline_predictions = {str(item.image_2d_id): item.prediction for item in pre_attack_items}
                        await eval_logger.info(f"Pre-attack 결과 로딩 완료: {len(baseline_predictions)}개 이미지")

                        # Log sample baseline predictions
                        if baseline_predictions:
                            sample_id = list(baseline_predictions.keys())[0]
                            sample_pred = baseline_predictions[sample_id]
                            await eval_logger.info(f"Sample baseline - image_id: {sample_id}, predictions: {len(sample_pred) if isinstance(sample_pred, list) else 'not a list'}")
                    else:
                        await eval_logger.warning(f"Pre-attack 평가가 완료되지 않음 (status: {pre_attack_run.status})")
                else:
                    await eval_logger.warning("Pre-attack 평가 결과를 찾을 수 없음. GT와 비교합니다.")
                    await eval_logger.warning(f"힌트: base_dataset_id={eval_run.base_dataset_id}에 대한 완료된 pre-attack 평가를 먼저 실행하세요.")
            else:
                # Pre-attack: evaluate on clean images
                dataset_id_for_images = eval_run.base_dataset_id
                await eval_logger.info("Pre-attack 평가: 클린 이미지 사용 (GT와 비교)")

            if not dataset_id_for_images:
                await eval_logger.error("평가할 데이터셋이 지정되지 않았습니다")
                raise ValidationError(detail="No dataset specified for evaluation")

            await eval_logger.status("데이터셋 로딩 중...")

            # Get dataset and images
            dataset = await crud.dataset_2d.get(db, id=dataset_id_for_images)
            if not dataset:
                await eval_logger.error(f"데이터셋을 찾을 수 없습니다: {dataset_id_for_images}")
                raise NotFoundError(resource=f"Dataset {dataset_id_for_images}")

            images = await crud.image_2d.get_by_dataset(db, dataset_id=dataset_id_for_images)
            if not images:
                await eval_logger.error("데이터셋에 이미지가 없습니다")
                raise ValidationError(detail=f"No images in dataset {dataset_id_for_images}")

            await eval_logger.info(f"총 {len(images)}개 이미지 발견")
            logger.info(
                f"Running evaluation {eval_run_id} on {len(images)} images"
            )

            # Run inference on all images
            await eval_logger.status("모델 추론 시작...", total_images=len(images))
            image_ids = [img.id for img in images]

            inference_results = await self.inference_service.run_inference(
                db=db,
                model_id=model_id,
                image_ids=image_ids,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

            await eval_logger.status("모델 추론 완료", completed_images=len(inference_results))
            await eval_logger.status("메트릭 계산 중...")

            # Calculate metrics
            metrics = await self._calculate_metrics(
                db=db,
                eval_run=eval_run,
                inference_results=inference_results,
                images=images,
                iou_threshold=0.5,  # Standard IoU threshold for mAP@50
                baseline_predictions=baseline_predictions,  # For post-attack comparison
            )

            await eval_logger.info(f"mAP@50: {metrics['overall']['map50']:.3f}")
            await eval_logger.info(f"Precision: {metrics['overall']['precision']:.3f}")
            await eval_logger.info(f"Recall: {metrics['overall']['recall']:.3f}")

            # Save evaluation items
            await eval_logger.status("평가 결과 저장 중...")
            for result in inference_results:
                image_id = UUID(result["image_id"])
                image = next((img for img in images if img.id == image_id), None)

                # Load ground truth annotations for this image
                ground_truth = []
                try:
                    annotations = await crud.annotation.get_by_image(db, image_2d_id=image_id)
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

                                class_name = ann.class_name
                                if class_name.startswith('class_'):
                                    try:
                                        class_idx = int(class_name.split('_')[1])
                                        if 0 <= class_idx < len(COCO_CLASSES):
                                            class_name = COCO_CLASSES[class_idx]
                                    except (ValueError, IndexError):
                                        pass  # Keep original class name if conversion fails

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
                    file_name=result.get("file_name"),
                    ground_truth=ground_truth,
                    prediction=result.get("detections", []),
                    metrics={
                        "inference_time_ms": result.get("inference_time_ms", 0),
                        "status": result.get("status", "unknown"),
                    },
                )
                await crud.evaluation.create_eval_item(db, eval_item)

            # Save class metrics
            await eval_logger.status(f"클래스별 메트릭 저장 중... ({len(metrics['per_class'])}개 클래스)")
            for class_name, class_metrics in metrics["per_class"].items():
                class_metrics_create = EvalClassMetricsCreate(
                    run_id=eval_run_id,
                    class_name=class_name,
                    metrics=class_metrics,
                )
                await crud.evaluation.create_eval_class_metrics(
                    db, class_metrics_create
                )

            # Update evaluation run with results
            await crud.evaluation.update_eval_run(
                db,
                eval_run_id,
                EvalRunUpdate(
                    status=EvalStatus.COMPLETED,
                    ended_at=datetime.utcnow(),
                    metrics_summary=metrics["overall"],
                ),
            )
            await db.commit()

            logger.info(f"Evaluation {eval_run_id} completed successfully")
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
                db,
                eval_run_id,
                EvalRunUpdate(
                    status=EvalStatus.FAILED,
                    ended_at=datetime.utcnow(),
                ),
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
            # Unload the model
            await self.inference_service.unload_model(model_id)

    async def _calculate_metrics(
        self,
        db: AsyncSession,
        eval_run: Any,
        inference_results: List[Dict[str, Any]],
        images: List[Any],
        iou_threshold: float = 0.5,
        baseline_predictions: Optional[Dict[str, List]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics using the advanced metrics calculator.

        Args:
            baseline_predictions: For post-attack evaluation, dict mapping image_id to pre-attack predictions
                                 If None, uses GT for comparison (pre-attack evaluation)

        Returns:
            Dictionary containing:
            - overall: Overall metrics (mAP, AR, size-specific, etc.)
            - per_class: Per-class metrics
            - has_ground_truth: Boolean indicating if GT was available
        """
        # Collect all predictions and ground truths as BoundingBox objects
        all_predictions: List[BoundingBox] = []
        all_ground_truths: List[BoundingBox] = []

        for result in inference_results:
            image_id = str(result["image_id"])
            image = next((img for img in images if str(img.id) == image_id), None)

            if not image:
                continue

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
                        image_height=pred_image_height,
                    )
                    all_predictions.append(bbox)
                except Exception as e:
                    logger.warning(f"Failed to parse prediction bbox: {e}")
                    continue

            # Load ground truth or baseline predictions for comparison
            # If baseline_predictions exists (post-attack), use pre-attack predictions
            # Otherwise, use GT annotations (pre-attack)
            try:
                if baseline_predictions and image_id in baseline_predictions:
                    # Post-attack: compare with pre-attack predictions
                    baseline_dets = baseline_predictions[image_id]
                    if not isinstance(baseline_dets, list):
                        baseline_dets = []

                    for det in baseline_dets:
                        try:
                            bbox = parse_detection_to_bbox(
                                det,
                                image_id=image_id,
                                image_width=pred_image_width,
                                image_height=pred_image_height,
                            )
                            # Use confidence 1.0 for baseline (treat as "ground truth")
                            bbox.confidence = 1.0
                            all_ground_truths.append(bbox)
                        except Exception as e:
                            logger.warning(f"Failed to parse baseline bbox: {e}")
                            continue
                else:
                    # Pre-attack: load GT annotations from database
                    annotations = await crud.annotation.get_by_image(db, image_2d_id=UUID(image_id))

                    for ann in annotations:
                        try:
                            # Annotations are stored in YOLO format (normalized center coordinates):
                            # bbox_x = normalized x_center (0-1)
                            # bbox_y = normalized y_center (0-1)
                            # bbox_width = normalized width (0-1)
                            # bbox_height = normalized height (0-1)
                            if ann.bbox_x is not None and ann.bbox_y is not None:
                                # Convert to float
                                x_center_norm = float(ann.bbox_x)
                                y_center_norm = float(ann.bbox_y)
                                w_norm = float(ann.bbox_width) if ann.bbox_width else 0
                                h_norm = float(ann.bbox_height) if ann.bbox_height else 0

                                # Check if normalized (0-1) or absolute coordinates
                                if x_center_norm <= 1.0 and y_center_norm <= 1.0 and w_norm <= 1.0 and h_norm <= 1.0:
                                    # Normalized center coordinates - convert to absolute corners
                                x_center_norm = x_center_norm
                                y_center_norm = y_center_norm
                                w_norm = w_norm
                                h_norm = h_norm
                            else:
                                if gt_image_width and gt_image_height:
                                    x_center_norm = x_center_norm / gt_image_width
                                    y_center_norm = y_center_norm / gt_image_height
                                    w_norm = w_norm / gt_image_width
                                    h_norm = h_norm / gt_image_height

                            x1 = max(0.0, min(1.0, x_center_norm - w_norm / 2))
                            y1 = max(0.0, min(1.0, y_center_norm - h_norm / 2))
                            x2 = max(0.0, min(1.0, x_center_norm + w_norm / 2))
                            y2 = max(0.0, min(1.0, y_center_norm + h_norm / 2))

                                # Handle class name: Convert "class_N" to actual COCO class name
                                class_name = ann.class_name
                                if class_name.startswith('class_'):
                                    try:
                                        class_idx = int(class_name.split('_')[1])
                                        if 0 <= class_idx < len(COCO_CLASSES):
                                            class_name = COCO_CLASSES[class_idx]
                                            logger.debug(f"Converted GT class '{ann.class_name}' to '{class_name}'")
                                        else:
                                            logger.warning(f"GT class index {class_idx} out of range for COCO classes")
                                    except (ValueError, IndexError) as e:
                                        logger.warning(f"Failed to parse GT class name '{ann.class_name}': {e}")

                                bbox_obj = BoundingBox(
                                    x1=x1,
                                    y1=y1,
                                    x2=x2,
                                    y2=y2,
                                    class_name=class_name,
                                    confidence=1.0,  # GT has confidence 1.0
                                    image_id=image_id,
                                )
                                all_ground_truths.append(bbox_obj)
                        except Exception as e:
                            logger.warning(f"Failed to parse ground truth bbox for annotation {ann.id}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Failed to load annotations for image {image_id}: {e}")
                pass

        # Check if we have ground truth
        has_ground_truth = len(all_ground_truths) > 0
        comparison_mode = "baseline_predictions" if baseline_predictions else "ground_truth"

        # DEBUG: Log class names for debugging
        logger.info(f"Evaluation mode: comparing with {comparison_mode}")
        logger.info(f"Total predictions: {len(all_predictions)}, Total baseline/GT: {len(all_ground_truths)}")

        if has_ground_truth and len(all_predictions) > 0:
            pred_classes = set([p.class_name for p in all_predictions])
            gt_classes = set([g.class_name for g in all_ground_truths])
            logger.info(f"DEBUG - Prediction classes: {sorted(pred_classes)}")
            logger.info(f"DEBUG - Ground truth classes: {sorted(gt_classes)}")
            logger.info(f"DEBUG - Common classes: {sorted(pred_classes & gt_classes)}")
            logger.info(f"DEBUG - Missing in predictions: {sorted(gt_classes - pred_classes)}")
            logger.info(f"DEBUG - Extra in predictions: {sorted(pred_classes - gt_classes)}")

            # Sample first prediction and GT for format verification
            if all_predictions:
                sample_pred = all_predictions[0]
                logger.info(f"DEBUG - Sample prediction: class='{sample_pred.class_name}', bbox=({sample_pred.x1:.2f}, {sample_pred.y1:.2f}, {sample_pred.x2:.2f}, {sample_pred.y2:.2f}), conf={sample_pred.confidence:.3f}")
            if all_ground_truths:
                sample_gt = all_ground_truths[0]
                logger.info(f"DEBUG - Sample GT: class='{sample_gt.class_name}', bbox=({sample_gt.x1:.2f}, {sample_gt.y1:.2f}, {sample_gt.x2:.2f}, {sample_gt.y2:.2f})")

        if not has_ground_truth:
            logger.warning("No ground truth annotations found. Metrics will be limited.")
            # Return limited metrics without GT
            class_names = set([p.class_name for p in all_predictions])
            per_class_metrics = {}

            for class_name in class_names:
                class_preds = [p for p in all_predictions if p.class_name == class_name]
                avg_conf = np.mean([p.confidence for p in class_preds]) if class_preds else 0.0

                per_class_metrics[class_name] = {
                    "map": 0.0,
                    "map50": 0.0,
                    "map75": 0.0,
                    "ap_small": 0.0,
                    "ap_medium": 0.0,
                    "ap_large": 0.0,
                    "ar_1": 0.0,
                    "ar_10": 0.0,
                    "ar_100": 0.0,
                    "ar_small": 0.0,
                    "ar_medium": 0.0,
                    "ar_large": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "pred_count": len(class_preds),
                    "avg_confidence": float(avg_conf),
                }

            overall_metrics = {
                "map": 0.0,
                "map50": 0.0,
                "map75": 0.0,
                "ap_small": 0.0,
                "ap_medium": 0.0,
                "ap_large": 0.0,
                "ar_1": 0.0,
                "ar_10": 0.0,
                "ar_100": 0.0,
                "ar_small": 0.0,
                "ar_medium": 0.0,
                "ar_large": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "total_pred": len(all_predictions),
                "total_gt": 0,
                "has_ground_truth": False,
            }
        else:
            # Calculate comprehensive metrics with ground truth
            logger.info(f"Calculating metrics with {len(all_predictions)} predictions and {len(all_ground_truths)} ground truths")

            # Overall metrics
            overall_metrics = calculate_overall_metrics(all_predictions, all_ground_truths)
            overall_metrics["has_ground_truth"] = True

            # Per-class metrics
            class_names = set([gt.class_name for gt in all_ground_truths])
            class_names.update([pred.class_name for pred in all_predictions])

            per_class_metrics = {}
            for class_name in class_names:
                per_class_metrics[class_name] = calculate_class_metrics(
                    all_predictions,
                    all_ground_truths,
                    class_name,
                )

        return {
            "overall": overall_metrics,
            "per_class": per_class_metrics,
        }

    def _calculate_class_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        class_name: str,
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate metrics for a specific class."""

        # Filter by class
        class_preds = []
        class_gts = []

        for pred in predictions:
            image_id = pred["image_id"]
            boxes = [b for b in pred["boxes"] if b.get("class_name") == class_name]
            for box in boxes:
                class_preds.append({
                    "image_id": image_id,
                    "bbox": box.get("bbox", box),
                    "confidence": box.get("confidence", 1.0),
                })

        for gt in ground_truths:
            image_id = gt["image_id"]
            boxes = [b for b in gt["boxes"] if b.get("class_name") == class_name]
            for box in boxes:
                class_gts.append({
                    "image_id": image_id,
                    "bbox": box.get("bbox", box),
                    "matched": False,
                })

        if not class_preds:
            return {
                "ap": 0.0,
                "ap50": 0.0,
                "ap75": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        # Sort predictions by confidence
        class_preds.sort(key=lambda x: x["confidence"], reverse=True)

        # Calculate TP, FP for each prediction
        tp = []
        fp = []

        for pred in class_preds:
            # Find matching ground truth
            best_iou = 0.0
            best_gt_idx = -1

            for i, gt in enumerate(class_gts):
                if gt["image_id"] != pred["image_id"]:
                    continue

                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            # Check if match
            if best_iou >= iou_threshold and not class_gts[best_gt_idx]["matched"]:
                tp.append(1)
                fp.append(0)
                class_gts[best_gt_idx]["matched"] = True
            else:
                tp.append(0)
                fp.append(1)

        # Calculate cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Calculate precision and recall
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / max(len(class_gts), 1)

        # Calculate AP
        ap50 = calculate_ap(precisions.tolist(), recalls.tolist())

        # Calculate AP@75 (using IoU threshold 0.75)
        # For simplicity, we'll approximate this
        ap75 = ap50 * 0.8  # Rough approximation

        # Final precision and recall
        final_precision = precisions[-1] if len(precisions) > 0 else 0.0
        final_recall = recalls[-1] if len(recalls) > 0 else 0.0

        return {
            "ap": ap50,
            "ap50": ap50,
            "ap75": ap75,
            "precision": float(final_precision),
            "recall": float(final_recall),
        }


# Global instance
evaluation_service = EvaluationService()
