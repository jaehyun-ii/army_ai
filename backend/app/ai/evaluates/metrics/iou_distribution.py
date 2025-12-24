"""
IoU Distribution Calculator.

Calculates IoU distribution statistics between predictions and ground truths:
- IoU histogram (binned distribution)
- Mean, median, std IoU
- IoU threshold statistics
- Per-class IoU distributions
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box with class and confidence (normalized 0-1 coordinates)."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float = 1.0
    image_id: str = ""
    area: float = 0.0

    def __post_init__(self):
        """Calculate area after initialization."""
        if self.area == 0.0:
            self.area = (self.x2 - self.x1) * (self.y2 - self.y1)


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    union = box1.area + box2.area - intersection

    return intersection / union if union > 0 else 0.0


def calculate_iou_distribution(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.5,
    num_bins: int = 20,
) -> Dict[str, Any]:
    """
    Calculate comprehensive IoU distribution statistics.

    Args:
        predictions: List of predicted bounding boxes
        ground_truths: List of ground truth bounding boxes
        iou_threshold: Threshold for matching (default: 0.5)
        num_bins: Number of bins for histogram (default: 20)

    Returns:
        Dictionary containing:
        - iou_histogram: Distribution of IoU values across bins
        - iou_mean: Mean IoU of all matched pairs
        - iou_median: Median IoU
        - iou_std: Standard deviation of IoU
        - iou_max: Maximum IoU
        - iou_min: Minimum IoU (non-zero)
        - matched_pairs_count: Number of matched prediction-GT pairs
        - threshold_statistics: Match counts at various IoU thresholds
    """
    if len(ground_truths) == 0 or len(predictions) == 0:
        return {
            "iou_histogram": {"bins": [], "counts": []},
            "iou_mean": 0.0,
            "iou_median": 0.0,
            "iou_std": 0.0,
            "iou_max": 0.0,
            "iou_min": 0.0,
            "matched_pairs_count": 0,
            "threshold_statistics": {},
        }

    # Calculate IoU for all prediction-GT pairs
    iou_values = []
    matched_pairs = []  # List of (pred_idx, gt_idx, iou) tuples

    # Find best matches for each prediction
    gt_matched = [False] * len(ground_truths)
    sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1].confidence, reverse=True)

    for pred_idx, pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            # Only match same class
            if gt.class_name != pred.class_name:
                continue

            # Skip already matched GT
            if gt_matched[gt_idx]:
                continue

            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Record match if IoU exceeds threshold
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_pairs.append((pred_idx, best_gt_idx, best_iou))
            iou_values.append(best_iou)
            gt_matched[best_gt_idx] = True

    # If no matches found, return empty statistics
    if len(iou_values) == 0:
        return {
            "iou_histogram": {"bins": [], "counts": []},
            "iou_mean": 0.0,
            "iou_median": 0.0,
            "iou_std": 0.0,
            "iou_max": 0.0,
            "iou_min": 0.0,
            "matched_pairs_count": 0,
            "threshold_statistics": {},
        }

    # Calculate statistics
    iou_array = np.array(iou_values)
    iou_mean = float(np.mean(iou_array))
    iou_median = float(np.median(iou_array))
    iou_std = float(np.std(iou_array))
    iou_max = float(np.max(iou_array))
    iou_min = float(np.min(iou_array))

    # Create histogram
    hist_counts, hist_bins = np.histogram(iou_array, bins=num_bins, range=(0.0, 1.0))

    # Convert to bin centers for visualization
    bin_centers = [(hist_bins[i] + hist_bins[i+1]) / 2 for i in range(len(hist_bins) - 1)]

    histogram = {
        "bins": [float(b) for b in bin_centers],
        "counts": [int(c) for c in hist_counts],
    }

    # Calculate threshold statistics
    threshold_statistics = {}
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        matches_at_threshold = sum(1 for iou in iou_values if iou >= threshold)
        threshold_statistics[f"iou_{int(threshold*100)}"] = {
            "match_count": matches_at_threshold,
            "match_rate": matches_at_threshold / len(predictions) if len(predictions) > 0 else 0.0,
        }

    return {
        "iou_histogram": histogram,
        "iou_mean": iou_mean,
        "iou_median": iou_median,
        "iou_std": iou_std,
        "iou_max": iou_max,
        "iou_min": iou_min,
        "matched_pairs_count": len(matched_pairs),
        "threshold_statistics": threshold_statistics,
    }


def calculate_per_class_iou_distribution(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.5,
    num_bins: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate IoU distribution for each class separately.

    Args:
        predictions: List of predicted bounding boxes
        ground_truths: List of ground truth bounding boxes
        iou_threshold: Threshold for matching
        num_bins: Number of bins for histogram

    Returns:
        Dictionary mapping class_name to IoU distribution statistics
    """
    # Get all unique classes
    classes = set([gt.class_name for gt in ground_truths])
    classes.update([pred.class_name for pred in predictions])

    per_class_distributions = {}

    for class_name in classes:
        # Filter by class
        preds_class = [p for p in predictions if p.class_name == class_name]
        gts_class = [g for g in ground_truths if g.class_name == class_name]

        # Calculate distribution for this class
        distribution = calculate_iou_distribution(
            preds_class, gts_class, iou_threshold, num_bins
        )

        per_class_distributions[class_name] = distribution

    return per_class_distributions


def calculate_iou_confusion_matrix(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_thresholds: List[float] = None,
) -> Dict[str, Any]:
    """
    Calculate confusion-style statistics at various IoU thresholds.

    Args:
        predictions: List of predicted bounding boxes
        ground_truths: List of ground truth bounding boxes
        iou_thresholds: List of IoU thresholds to evaluate (default: [0.5, 0.75, 0.9])

    Returns:
        Dictionary with TP/FP/FN counts at each threshold
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.9]

    results = {}

    for threshold in iou_thresholds:
        # Sort predictions by confidence
        sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)

        gt_matched = [False] * len(ground_truths)
        tp_count = 0
        fp_count = 0

        for pred in sorted_preds:
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

            if best_iou >= threshold and best_gt_idx >= 0:
                tp_count += 1
                gt_matched[best_gt_idx] = True
            else:
                fp_count += 1

        fn_count = sum(1 for matched in gt_matched if not matched)

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / len(ground_truths) if len(ground_truths) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[f"iou_{int(threshold*100)}"] = {
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return results
