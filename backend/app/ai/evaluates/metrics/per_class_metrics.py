"""
Per-Class Detailed Metrics.

Calculates comprehensive per-class metrics including:
- mAP, mAP@0.5, mAP@0.75, mAP@0.5:0.95
- Precision, Recall, F1 Score
- AR (Average Recall) at various detection limits
- Size-specific metrics (small, medium, large)
- IoU distribution per class
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

from app.services.metrics_calculator import (
    BoundingBox,
    calculate_iou,
    calculate_ap_ar_at_iou,
    calculate_ap_ar_all_ious,
    calculate_ap_ar_by_size,
)
from app.ai.evaluates.metrics.iou_distribution import (
    calculate_iou_distribution,
)

logger = logging.getLogger(__name__)


def calculate_enhanced_class_metrics(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    class_name: str,
    iou_threshold: float = 0.5,
    include_iou_distribution: bool = True,
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a single class with enhanced details.

    Args:
        predictions: All predicted bounding boxes
        ground_truths: All ground truth bounding boxes
        class_name: Class to calculate metrics for
        iou_threshold: IoU threshold for precision/recall (default: 0.5)
        include_iou_distribution: Whether to include IoU distribution stats

    Returns:
        Dictionary containing:
        - AP metrics: map, map50, map75 (mAP@0.5:0.95, mAP@0.5, mAP@0.75)
        - AR metrics: ar_1, ar_10, ar_100
        - Size-specific AP: ap_small, ap_medium, ap_large
        - Size-specific AR: ar_small, ar_medium, ar_large
        - Basic metrics: precision, recall, f1
        - Count metrics: gt_count, pred_count, tp_count, fp_count, fn_count
        - IoU distribution: iou_mean, iou_median, iou_std, iou_histogram
    """
    # Filter by class
    preds_class = [p for p in predictions if p.class_name == class_name]
    gts_class = [g for g in ground_truths if g.class_name == class_name]

    # If no GT for this class, return zero metrics
    if len(gts_class) == 0:
        return {
            # AP metrics
            "map": 0.0,
            "map50": 0.0,
            "map75": 0.0,
            "ap_small": 0.0,
            "ap_medium": 0.0,
            "ap_large": 0.0,
            # AR metrics
            "ar_1": 0.0,
            "ar_10": 0.0,
            "ar_100": 0.0,
            "ar_small": 0.0,
            "ar_medium": 0.0,
            "ar_large": 0.0,
            # Basic metrics
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            # Count metrics
            "gt_count": 0,
            "pred_count": len(preds_class),
            "tp_count": 0,
            "fp_count": len(preds_class),
            "fn_count": 0,
            # IoU distribution
            "iou_distribution": None,
        }

    # Calculate overall AP/AR metrics across IoU thresholds
    overall = calculate_ap_ar_all_ious(preds_class, gts_class)

    # Calculate size-specific metrics
    by_size = calculate_ap_ar_by_size(preds_class, gts_class)

    # Calculate basic precision/recall at specified IoU threshold
    basic = calculate_ap_ar_at_iou(preds_class, gts_class, iou_threshold)

    # Calculate TP/FP/FN counts
    tp_count, fp_count, fn_count = _calculate_tp_fp_fn(
        preds_class, gts_class, iou_threshold
    )

    # Calculate F1 score
    precision = basic["precision"]
    recall = basic["recall"]
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate IoU distribution
    iou_dist = None
    if include_iou_distribution:
        iou_dist = calculate_iou_distribution(
            preds_class, gts_class, iou_threshold=iou_threshold
        )

    return {
        # AP metrics
        "map": float(overall["map"]),        # mAP@0.5:0.95
        "map50": float(overall["map50"]),    # mAP@0.5
        "map75": float(overall["map75"]),    # mAP@0.75
        "ap_small": float(by_size["small"]["map"]),
        "ap_medium": float(by_size["medium"]["map"]),
        "ap_large": float(by_size["large"]["map"]),
        # AR metrics
        "ar_1": float(overall["ar_1"]),
        "ar_10": float(overall["ar_10"]),
        "ar_100": float(overall["ar_100"]),
        "ar_small": float(by_size["small"]["ar_100"]),
        "ar_medium": float(by_size["medium"]["ar_100"]),
        "ar_large": float(by_size["large"]["ar_100"]),
        # Basic metrics
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        # Count metrics
        "gt_count": len(gts_class),
        "pred_count": len(preds_class),
        "tp_count": tp_count,
        "fp_count": fp_count,
        "fn_count": fn_count,
        # IoU distribution
        "iou_distribution": iou_dist,
    }


def calculate_all_classes_metrics(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.5,
    include_iou_distribution: bool = True,
    target_class: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate enhanced metrics for all classes.

    Args:
        predictions: All predicted bounding boxes
        ground_truths: All ground truth bounding boxes
        iou_threshold: IoU threshold for precision/recall
        include_iou_distribution: Whether to include IoU distribution stats
        target_class: If provided, only calculate metrics for this class

    Returns:
        Dictionary mapping class_name to enhanced metrics
    """
    # Get all unique classes
    if target_class:
        classes = {target_class}
    else:
        classes = set([gt.class_name for gt in ground_truths])
        classes.update([pred.class_name for pred in predictions])

    per_class_metrics = {}

    for class_name in classes:
        metrics = calculate_enhanced_class_metrics(
            predictions,
            ground_truths,
            class_name,
            iou_threshold,
            include_iou_distribution,
        )
        per_class_metrics[class_name] = metrics

    return per_class_metrics


def _calculate_tp_fp_fn(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """
    Calculate True Positive, False Positive, False Negative counts.

    Args:
        predictions: Predicted bounding boxes
        ground_truths: Ground truth bounding boxes
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (tp_count, fp_count, fn_count)
    """
    if len(ground_truths) == 0:
        return 0, len(predictions), 0

    if len(predictions) == 0:
        return 0, 0, len(ground_truths)

    # Sort predictions by confidence
    sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    gt_matched = [False] * len(ground_truths)
    tp_count = 0
    fp_count = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            # Must be the same class and same image
            if gt.class_name != pred.class_name or gt.image_id != pred.image_id:
                continue
            
            if gt_matched[gt_idx]:
                continue

            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_count += 1
            gt_matched[best_gt_idx] = True
        else:
            fp_count += 1

    fn_count = sum(1 for matched in gt_matched if not matched)

    return tp_count, fp_count, fn_count


def aggregate_class_metrics(
    per_class_metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate per-class metrics into overall metrics (macro-average).

    Args:
        per_class_metrics: Dictionary mapping class_name to metrics

    Returns:
        Aggregated metrics across all classes
    """
    if not per_class_metrics:
        return {
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
            "f1": 0.0,
            "total_gt": 0,
            "total_pred": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
        }

    # Calculate macro-averages
    num_classes = len(per_class_metrics)

    aggregated = {
        # AP metrics (macro-average)
        "map": np.mean([m["map"] for m in per_class_metrics.values()]),
        "map50": np.mean([m["map50"] for m in per_class_metrics.values()]),
        "map75": np.mean([m["map75"] for m in per_class_metrics.values()]),
        "ap_small": np.mean([m["ap_small"] for m in per_class_metrics.values()]),
        "ap_medium": np.mean([m["ap_medium"] for m in per_class_metrics.values()]),
        "ap_large": np.mean([m["ap_large"] for m in per_class_metrics.values()]),
        # AR metrics (macro-average)
        "ar_1": np.mean([m["ar_1"] for m in per_class_metrics.values()]),
        "ar_10": np.mean([m["ar_10"] for m in per_class_metrics.values()]),
        "ar_100": np.mean([m["ar_100"] for m in per_class_metrics.values()]),
        "ar_small": np.mean([m["ar_small"] for m in per_class_metrics.values()]),
        "ar_medium": np.mean([m["ar_medium"] for m in per_class_metrics.values()]),
        "ar_large": np.mean([m["ar_large"] for m in per_class_metrics.values()]),
        # Basic metrics (macro-average)
        "precision": np.mean([m["precision"] for m in per_class_metrics.values()]),
        "recall": np.mean([m["recall"] for m in per_class_metrics.values()]),
        "f1": np.mean([m["f1"] for m in per_class_metrics.values()]),
        # Count metrics (sum across classes)
        "total_gt": sum([m["gt_count"] for m in per_class_metrics.values()]),
        "total_pred": sum([m["pred_count"] for m in per_class_metrics.values()]),
        "total_tp": sum([m["tp_count"] for m in per_class_metrics.values()]),
        "total_fp": sum([m["fp_count"] for m in per_class_metrics.values()]),
        "total_fn": sum([m["fn_count"] for m in per_class_metrics.values()]),
    }

    return {k: float(v) for k, v in aggregated.items()}
