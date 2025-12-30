"""
Visualization-specific endpoints for evaluation data.
Provides PR curves, IoU analysis, and other visualization data.
"""
from typing import List, Dict, Any, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import logging

from app.database import get_db
from app.crud import evaluation as crud_evaluation
from app.services.metrics_calculator import calculate_iou, BoundingBox, parse_detection_to_bbox
from app.schemas.evaluation import EvalDatasetType

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/runs/{run_id}/iou-distribution")
async def get_iou_distribution_data(
    run_id: UUID,
    dataset_type: Optional[str] = Query(None, description="Dataset type: 'base' or 'attack'"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get IoU distribution data for an evaluation run.

    If stored distribution is available, returns it.
    Otherwise, calculates it from eval_items on-the-fly.

    Returns:
    {
        "base": {
            "iou_histogram": {"bins": [...], "counts": [...]},
            "iou_mean": 0.75,
            "iou_median": 0.78,
            ...
        },
        "attack": {...}
    }
    """
    from app.ai.evaluates.metrics.iou_distribution import calculate_iou_distribution
    from app.services.metrics_calculator import parse_detection_to_bbox

    # Get evaluation run
    eval_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=run_id)
    if not eval_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found"
        )

    result_data = {}

    # Get dataset results
    dataset_results = await crud_evaluation.get_eval_dataset_results_by_run(db, run_id)

    for dataset_result in dataset_results:
        ds_type = dataset_result.dataset_type.value.lower()  # 'base' or 'attack'

        # Filter by dataset_type if specified
        if dataset_type and ds_type != dataset_type.lower():
            continue

        # Check if IoU distribution is already stored
        if dataset_result.iou_distribution:
            result_data[ds_type] = dataset_result.iou_distribution
            logger.info(f"Using stored IoU distribution for {ds_type} dataset")
        else:
            # Calculate on-the-fly
            logger.warning(f"No stored IoU distribution for {ds_type}, calculating from eval_items...")

            items = await crud_evaluation.get_all_eval_items(
                db, run_id=run_id, dataset_type=dataset_result.dataset_type
            )

            if items:
                all_predictions = []
                all_ground_truths = []

                for item in items:
                    image_id = str(item.image_2d_id or item.image_3d_id)

                    # Parse predictions
                    for pred in item.prediction or []:
                        try:
                            bbox = parse_detection_to_bbox(
                                pred,
                                image_id=image_id,
                                image_width=640,
                                image_height=640,
                            )
                            all_predictions.append(bbox)
                        except Exception as e:
                            logger.debug(f"Failed to parse prediction: {e}")

                    # Parse ground truths
                    for gt in item.ground_truth or []:
                        try:
                            gt_detection = {
                                "bbox": gt.get("bbox", {}),
                                "class_name": gt.get("class_name", "unknown"),
                                "confidence": 1.0,
                            }
                            bbox = parse_detection_to_bbox(
                                gt_detection,
                                image_id=image_id,
                                image_width=640,
                                image_height=640,
                            )
                            all_ground_truths.append(bbox)
                        except Exception as e:
                            logger.debug(f"Failed to parse GT: {e}")

                if all_predictions and all_ground_truths:
                    iou_dist = calculate_iou_distribution(
                        predictions=all_predictions,
                        ground_truths=all_ground_truths,
                        iou_threshold=0.5,
                        num_bins=20,
                    )
                    result_data[ds_type] = iou_dist
                else:
                    result_data[ds_type] = None
            else:
                result_data[ds_type] = None

    if not result_data:
        return {"message": "No IoU distribution data available"}

    return result_data


@router.get("/runs/{run_id}/pr-curve-data")
async def get_pr_curve_data(
    run_id: UUID,
    dataset_type: Optional[str] = Query(None, description="Dataset type: 'base' or 'attack'"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get PR (Precision-Recall) curve data from stored evaluation results.

    Returns PR curves for IoU 0.5, 0.75, 0.95 from eval_dataset_results.
    Falls back to real-time calculation from eval_items if not available.

    Query Parameters:
    - dataset_type: Optional filter ('base' or 'attack'). If not provided, returns all available.

    Response Format:
    {
        "base": {
            "iou_0.50": {
                "precisions": [1.0, 0.95, ...],
                "recalls": [0.0, 0.02, ...],
                "confidence_thresholds": [1.0, 0.99, ...],
                "ap": 0.856
            },
            "iou_0.75": {...},
            "iou_0.95": {...}
        },
        "attack": {
            "iou_0.50": {...},
            "iou_0.75": {...},
            "iou_0.95": {...}
        }
    }
    """
    # Get evaluation run
    eval_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=run_id)
    if not eval_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found"
        )

    # Get dataset results with PR curves
    result_data = {}

    # Determine which dataset types to fetch
    if dataset_type:
        try:
            dataset_type_enum = EvalDatasetType(dataset_type.upper())
            dataset_results = await crud_evaluation.get_eval_dataset_results_by_run(db, run_id)
            dataset_results = [dr for dr in dataset_results if dr.dataset_type == dataset_type_enum]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dataset_type: {dataset_type}. Must be 'base' or 'attack'."
            )
    else:
        # Fetch all dataset results for this run
        dataset_results = await crud_evaluation.get_eval_dataset_results_by_run(db, run_id)

    # Process each dataset result
    for dataset_result in dataset_results:
        if not dataset_result:
            continue

        ds_type = dataset_result.dataset_type.value.lower()  # 'base' or 'attack'

        # Check if PR curves are stored
        if dataset_result.pr_curves and isinstance(dataset_result.pr_curves, dict):
            # Use stored PR curves
            result_data[ds_type] = dataset_result.pr_curves
            logger.info(f"Using stored PR curves for {ds_type} dataset")
        else:
            # Fallback: Calculate PR curves from eval_items (backward compatibility)
            logger.warning(f"No stored PR curves for {ds_type} dataset, calculating from eval_items...")

            items = await crud_evaluation.get_all_eval_items(
                db, run_id=run_id, dataset_type=dataset_result.dataset_type
            )

            if items:
                pr_curves = _calculate_pr_curves_from_items(items)
                result_data[ds_type] = pr_curves
            else:
                result_data[ds_type] = {
                    "iou_0.50": {"precisions": [], "recalls": [], "confidence_thresholds": [], "ap": 0.0},
                    "iou_0.75": {"precisions": [], "recalls": [], "confidence_thresholds": [], "ap": 0.0},
                    "iou_0.95": {"precisions": [], "recalls": [], "confidence_thresholds": [], "ap": 0.0},
                }

    if not result_data:
        return {
            "message": "No PR curve data available",
            "data": {}
        }

    return result_data


def _calculate_pr_curves_from_items(
    eval_items: List[Any],
    iou_thresholds: List[float] = None,
) -> Dict[str, Any]:
    """
    Fallback function to calculate PR curves from eval_items.
    Used when pr_curves are not stored in eval_dataset_results.
    """
    from app.services.metrics_calculator import calculate_pr_curves_multiple_ious

    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.95]

    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    for item in eval_items:
        image_id = str(item.image_2d_id or item.image_3d_id)

        # Parse predictions
        for pred in item.prediction or []:
            try:
                bbox = parse_detection_to_bbox(
                    pred,
                    image_id=image_id,
                    image_width=640,
                    image_height=640,
                )
                all_predictions.append(bbox)
            except Exception as e:
                logger.debug(f"Failed to parse prediction: {e}")

        # Parse ground truths
        for gt in item.ground_truth or []:
            try:
                gt_detection = {
                    "bbox": gt.get("bbox", {}),
                    "class_name": gt.get("class_name", "unknown"),
                    "confidence": 1.0,
                }
                bbox = parse_detection_to_bbox(
                    gt_detection,
                    image_id=image_id,
                    image_width=640,
                    image_height=640,
                )
                all_ground_truths.append(bbox)
            except Exception as e:
                logger.debug(f"Failed to parse GT: {e}")

    if not all_predictions or not all_ground_truths:
        return {
            "iou_0.50": {"precisions": [], "recalls": [], "confidence_thresholds": [], "ap": 0.0},
            "iou_0.75": {"precisions": [], "recalls": [], "confidence_thresholds": [], "ap": 0.0},
            "iou_0.95": {"precisions": [], "recalls": [], "confidence_thresholds": [], "ap": 0.0},
        }

    # Calculate PR curves
    return calculate_pr_curves_multiple_ious(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        iou_thresholds=iou_thresholds,
        num_points=101
    )


def calculate_pr_curve(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.5,
    num_points: int = 101,
) -> List[Dict[str, float]]:
    """
    Calculate Precision-Recall curve by varying confidence threshold.

    Returns list of {recall, precision, confidence_threshold} points.
    """
    if not predictions or not ground_truths:
        return []

    # Get confidence thresholds (from 0 to 1)
    confidence_thresholds = np.linspace(0, 1, num_points)
    pr_points = []

    for conf_threshold in confidence_thresholds:
        # Filter predictions by confidence threshold
        filtered_preds = [p for p in predictions if p.confidence >= conf_threshold]

        if not filtered_preds:
            # No predictions at this threshold
            pr_points.append({
                "recall": 0.0,
                "precision": 1.0 if conf_threshold > 0.99 else 0.0,
                "confidence_threshold": float(conf_threshold),
            })
            continue

        # Calculate TP, FP at this threshold
        gt_matched = [False] * len(ground_truths)
        tp_count = 0
        fp_count = 0

        for pred in filtered_preds:
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt.class_name != pred.class_name:
                    continue
                if gt_matched[gt_idx]:
                    continue

                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Match if IoU exceeds threshold
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_count += 1
                gt_matched[best_gt_idx] = True
            else:
                fp_count += 1

        # Calculate precision and recall
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / len(ground_truths) if len(ground_truths) > 0 else 0.0

        pr_points.append({
            "recall": float(recall * 100),  # Convert to percentage
            "precision": float(precision * 100),  # Convert to percentage
            "confidence_threshold": float(conf_threshold),
        })

    return pr_points


def calculate_iou_thresholds_metrics(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
) -> List[Dict[str, float]]:
    """
    Calculate metrics (AP, Precision, Recall) at different IoU thresholds.
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5 to 0.95
    metrics_data = []

    for iou_threshold in iou_thresholds:
        # Calculate matching at this IoU threshold
        gt_matched = [False] * len(ground_truths)
        tp_count = 0
        fp_count = 0

        for pred in predictions:
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt.class_name != pred.class_name:
                    continue
                if gt_matched[gt_idx]:
                    continue

                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Match if IoU exceeds threshold
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_count += 1
                gt_matched[best_gt_idx] = True
            else:
                fp_count += 1

        # Calculate metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / len(ground_truths) if len(ground_truths) > 0 else 0.0

        # Simple AP estimation (not interpolated)
        ap = precision * recall if recall > 0 else 0.0

        metrics_data.append({
            "iou": float(iou_threshold),
            "precision": float(precision * 100),
            "recall": float(recall * 100),
            "ap": float(ap * 100),
        })

    return metrics_data
