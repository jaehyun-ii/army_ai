"""
Visualization-specific endpoints for evaluation data.
Provides PR curves, IoU analysis, and other visualization data.
"""
from typing import List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from app.database import get_db
from app.crud import evaluation as crud_evaluation
from app.services.metrics_calculator import calculate_iou, BoundingBox

router = APIRouter()


@router.get("/runs/{run_id}/pr-curve-data")
async def get_pr_curve_data(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Generate PR (Precision-Recall) curve data from evaluation items.

    Returns actual precision and recall values at different confidence thresholds.
    """
    # Get evaluation run
    eval_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=run_id)
    if not eval_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found"
        )

    # Get all evaluation items (predictions and ground truths)
    items_result = await crud_evaluation.get_eval_items(db=db, run_id=run_id)

    # Handle tuple return (items, total) or list return
    if isinstance(items_result, tuple):
        items, _ = items_result
    else:
        items = items_result

    if not items:
        return {
            "pr_curve": [],
            "iou_thresholds": [],
            "message": "No evaluation items found"
        }

    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    for item in items:
        # Ground truths
        ground_truth = item.ground_truth if hasattr(item, 'ground_truth') else []
        if ground_truth:
            if isinstance(ground_truth, list):
                for gt_box in ground_truth:
                    if isinstance(gt_box, dict):
                        all_ground_truths.append(BoundingBox(
                            class_name=gt_box.get("class_name", "unknown"),
                            x1=float(gt_box.get("x1", 0)),
                            y1=float(gt_box.get("y1", 0)),
                            x2=float(gt_box.get("x2", 0)),
                            y2=float(gt_box.get("y2", 0)),
                            confidence=1.0,
                        ))

        # Predictions
        prediction = item.prediction if hasattr(item, 'prediction') else []
        if prediction:
            if isinstance(prediction, list):
                for pred_box in prediction:
                    if isinstance(pred_box, dict):
                        all_predictions.append(BoundingBox(
                            class_name=pred_box.get("class_name", "unknown"),
                            x1=float(pred_box.get("x1", 0)),
                            y1=float(pred_box.get("y1", 0)),
                            x2=float(pred_box.get("x2", 0)),
                            y2=float(pred_box.get("y2", 0)),
                            confidence=float(pred_box.get("confidence", 0)),
                        ))

    if not all_predictions or not all_ground_truths:
        return {
            "pr_curve": [],
            "iou_thresholds": [],
            "message": "Insufficient data for PR curve"
        }

    # Sort predictions by confidence (descending)
    all_predictions.sort(key=lambda x: x.confidence, reverse=True)

    # Calculate PR curve at IoU threshold 0.5
    iou_threshold = 0.5
    pr_curve_data = calculate_pr_curve(all_predictions, all_ground_truths, iou_threshold)

    # Calculate metrics at different IoU thresholds
    iou_thresholds_data = calculate_iou_thresholds_metrics(
        all_predictions,
        all_ground_truths
    )

    return {
        "pr_curve": pr_curve_data,
        "iou_thresholds": iou_thresholds_data,
        "total_predictions": len(all_predictions),
        "total_ground_truths": len(all_ground_truths),
    }


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
