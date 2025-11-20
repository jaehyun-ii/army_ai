"""
Comprehensive metrics calculator for object detection evaluation.

Calculates:
1. Core Performance Metrics:
   - AP (Average Precision): mAP@[.5:.95], AP50, AP75, AP_S/M/L
   - AR (Average Recall): AR@1/10/100, AR_S/M/L

2. Robustness Metrics (when both clean and adversarial datasets are evaluated):
   - ΔAP = AP_clean − AP_adv
   - Drop% = ΔAP / AP_clean × 100
   - Robustness ratio (R_AP) = AP_adv / AP_clean
   - ΔRecall@IoU: Recall drop at fixed IoU thresholds
   - Precision@Recall change: Precision change at fixed recall levels
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

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

    @property
    def size_category(self) -> str:
        """Categorize box by size (COCO standard)."""
        if self.area < 32**2:
            return "small"
        elif self.area < 96**2:
            return "medium"
        else:
            return "large"


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


def calculate_ap_ar_at_iou(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float,
    recall_thresholds: List[int] = [1, 10, 100],
) -> Dict[str, float]:
    """
    Calculate AP and AR at a specific IoU threshold.

    Args:
        predictions: List of predicted boxes (sorted by confidence descending)
        ground_truths: List of ground truth boxes
        iou_threshold: IoU threshold for matching
        recall_thresholds: Max detections for AR calculation (1, 10, 100)

    Returns:
        Dictionary with AP and AR metrics
    """
    if len(ground_truths) == 0:
        return {"ap": 0.0, "ar_1": 0.0, "ar_10": 0.0, "ar_100": 0.0, "precision": 0.0, "recall": 0.0}

    if len(predictions) == 0:
        return {"ap": 0.0, "ar_1": 0.0, "ar_10": 0.0, "ar_100": 0.0, "precision": 0.0, "recall": 0.0}

    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truths)

    # Calculate TP, FP for each prediction
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    # DEBUG: Track matching statistics
    class_mismatches = 0
    iou_too_low = 0
    successful_matches = 0

    for pred_idx, pred in enumerate(predictions):
        # Find best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1
        same_class_gt_count = 0

        for gt_idx, gt in enumerate(ground_truths):
            if gt.class_name != pred.class_name:
                class_mismatches += 1
                continue
            same_class_gt_count += 1
            if gt_matched[gt_idx]:
                continue

            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Match if IoU exceeds threshold
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
            successful_matches += 1
        else:
            fp[pred_idx] = 1
            if same_class_gt_count > 0 and best_iou > 0 and best_iou < iou_threshold:
                iou_too_low += 1

    # DEBUG: Log matching statistics (only for first few calls to avoid spam)
    if len(predictions) > 0 and tp.sum() == 0 and len(predictions) <= 3 and len(ground_truths) <= 3:
        logger.warning(f"DEBUG - No TP matches! IoU threshold: {iou_threshold}, Predictions: {len(predictions)}, GTs: {len(ground_truths)}")
        logger.warning(f"DEBUG - Successful matches: {successful_matches}, IoU too low: {iou_too_low}, Class mismatches: {class_mismatches}")

        # Log actual bbox coordinates for debugging
        for i, pred in enumerate(predictions[:2]):  # First 2 predictions
            logger.warning(f"DEBUG - Pred[{i}]: class='{pred.class_name}', bbox=({pred.x1:.2f}, {pred.y1:.2f}, {pred.x2:.2f}, {pred.y2:.2f}), area={pred.area:.2f}")
        for i, gt in enumerate(ground_truths[:2]):  # First 2 GTs
            logger.warning(f"DEBUG - GT[{i}]: class='{gt.class_name}', bbox=({gt.x1:.2f}, {gt.y1:.2f}, {gt.x2:.2f}, {gt.y2:.2f}), area={gt.area:.2f}")

        # Calculate and log IoU for first prediction
        if len(predictions) > 0 and len(ground_truths) > 0:
            test_iou = calculate_iou(predictions[0], ground_truths[0])
            logger.warning(f"DEBUG - IoU between Pred[0] and GT[0]: {test_iou:.4f}")

    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Calculate precision and recall at each threshold
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / len(ground_truths)

    # Calculate AP using all-point interpolation (VOC 2010+)
    ap = calculate_interpolated_ap(precisions, recalls)

    # Calculate AR at different max detection thresholds
    ar_metrics = {}
    for max_dets in recall_thresholds:
        if max_dets < len(predictions):
            ar = recalls[max_dets - 1]
        else:
            ar = recalls[-1] if len(recalls) > 0 else 0.0
        ar_metrics[f"ar_{max_dets}"] = float(ar)

    # Overall precision and recall
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0

    return {
        "ap": float(ap),
        **ar_metrics,
        "precision": float(final_precision),
        "recall": float(final_recall),
    }


def calculate_interpolated_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Calculate AP using all-point interpolation (COCO/VOC 2010+ style).
    """
    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0

    # Add sentinel values at the end
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    # Calculate AP as area under curve
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def calculate_ap_ar_all_ious(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_thresholds: List[float] = None,
) -> Dict[str, float]:
    """
    Calculate AP and AR across multiple IoU thresholds.

    Returns mAP@[.5:.95] (COCO style: average over IoU 0.5 to 0.95 in 0.05 steps)
    """
    if iou_thresholds is None:
        # COCO standard: 0.5 to 0.95 in steps of 0.05
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    ap_per_iou = []
    ar_1_per_iou = []
    ar_10_per_iou = []
    ar_100_per_iou = []

    for iou_thresh in iou_thresholds:
        metrics = calculate_ap_ar_at_iou(predictions, ground_truths, iou_thresh)
        ap_per_iou.append(metrics["ap"])
        ar_1_per_iou.append(metrics["ar_1"])
        ar_10_per_iou.append(metrics["ar_10"])
        ar_100_per_iou.append(metrics["ar_100"])

    # Calculate specific IoU thresholds
    ap_50 = calculate_ap_ar_at_iou(predictions, ground_truths, 0.5)["ap"]
    ap_75 = calculate_ap_ar_at_iou(predictions, ground_truths, 0.75)["ap"]

    return {
        "map": float(np.mean(ap_per_iou)),  # mAP@[.5:.95]
        "map50": float(ap_50),  # mAP@50
        "map75": float(ap_75),  # mAP@75
        "ar_1": float(np.mean(ar_1_per_iou)),
        "ar_10": float(np.mean(ar_10_per_iou)),
        "ar_100": float(np.mean(ar_100_per_iou)),
    }


def calculate_ap_ar_by_size(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate AP and AR broken down by object size (small, medium, large).

    COCO size categories:
    - Small: area < 32^2 pixels
    - Medium: 32^2 <= area < 96^2 pixels
    - Large: area >= 96^2 pixels
    """
    size_categories = ["small", "medium", "large"]
    results = {}

    for size_cat in size_categories:
        # Filter predictions and ground truths by size
        preds_size = [p for p in predictions if p.size_category == size_cat]
        gts_size = [g for g in ground_truths if g.size_category == size_cat]

        if len(gts_size) == 0:
            results[size_cat] = {
                "map": 0.0,
                "map50": 0.0,
                "map75": 0.0,
                "ar_1": 0.0,
                "ar_10": 0.0,
                "ar_100": 0.0,
            }
        else:
            metrics = calculate_ap_ar_all_ious(preds_size, gts_size)
            results[size_cat] = metrics

    return results


def calculate_class_metrics(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    class_name: str,
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a single class.

    Returns:
        Dictionary containing:
        - AP metrics: map, map50, map75, ap_s, ap_m, ap_l
        - AR metrics: ar_1, ar_10, ar_100, ar_s, ar_m, ar_l
        - Basic metrics: precision, recall
    """
    # Filter by class
    preds_class = [p for p in predictions if p.class_name == class_name]
    gts_class = [g for g in ground_truths if g.class_name == class_name]

    if len(gts_class) == 0:
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
            "gt_count": 0,
            "pred_count": len(preds_class),
        }

    # Overall metrics
    overall = calculate_ap_ar_all_ious(preds_class, gts_class)

    # Size-specific metrics
    by_size = calculate_ap_ar_by_size(preds_class, gts_class)

    # Basic precision/recall at IoU=0.5
    basic = calculate_ap_ar_at_iou(preds_class, gts_class, 0.5)

    return {
        # AP metrics
        "map": overall["map"],
        "map50": overall["map50"],
        "map75": overall["map75"],
        "ap_small": by_size["small"]["map"],
        "ap_medium": by_size["medium"]["map"],
        "ap_large": by_size["large"]["map"],
        # AR metrics
        "ar_1": overall["ar_1"],
        "ar_10": overall["ar_10"],
        "ar_100": overall["ar_100"],
        "ar_small": by_size["small"]["ar_100"],
        "ar_medium": by_size["medium"]["ar_100"],
        "ar_large": by_size["large"]["ar_100"],
        # Basic metrics
        "precision": basic["precision"],
        "recall": basic["recall"],
        "gt_count": len(gts_class),
        "pred_count": len(preds_class),
    }


def calculate_overall_metrics(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
) -> Dict[str, Any]:
    """
    Calculate overall metrics across all classes.

    Returns comprehensive metrics including AP, AR, and size-specific metrics.
    """
    if len(ground_truths) == 0:
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
            "total_pred": len(predictions),
        }

    # Get all unique classes
    classes = set([gt.class_name for gt in ground_truths])
    classes.update([pred.class_name for pred in predictions])

    # Calculate per-class metrics
    per_class = {}
    for class_name in classes:
        per_class[class_name] = calculate_class_metrics(
            predictions, ground_truths, class_name
        )

    # Aggregate across classes (macro-average)
    if per_class:
        precision = np.mean([m["precision"] for m in per_class.values()])
        recall = np.mean([m["recall"] for m in per_class.values()])

        # Calculate F1 Score: 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        overall = {
            "map": np.mean([m["map"] for m in per_class.values()]),
            "map50": np.mean([m["map50"] for m in per_class.values()]),
            "map75": np.mean([m["map75"] for m in per_class.values()]),
            "ap_small": np.mean([m["ap_small"] for m in per_class.values()]),
            "ap_medium": np.mean([m["ap_medium"] for m in per_class.values()]),
            "ap_large": np.mean([m["ap_large"] for m in per_class.values()]),
            "ar_1": np.mean([m["ar_1"] for m in per_class.values()]),
            "ar_10": np.mean([m["ar_10"] for m in per_class.values()]),
            "ar_100": np.mean([m["ar_100"] for m in per_class.values()]),
            "ar_small": np.mean([m["ar_small"] for m in per_class.values()]),
            "ar_medium": np.mean([m["ar_medium"] for m in per_class.values()]),
            "ar_large": np.mean([m["ar_large"] for m in per_class.values()]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_gt": len(ground_truths),
            "total_pred": len(predictions),
        }
    else:
        overall = {
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
            "total_gt": len(ground_truths),
            "total_pred": len(predictions),
        }

    return overall


def calculate_robustness_metrics(
    clean_metrics: Dict[str, float],
    adv_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Calculate robustness/degradation metrics by comparing clean vs adversarial performance.

    Args:
        clean_metrics: Metrics from clean/baseline dataset
        adv_metrics: Metrics from adversarial dataset

    Returns:
        Dictionary containing:
        - delta_map: Absolute drop in mAP
        - delta_map50: Absolute drop in mAP@50
        - drop_percentage: Percentage drop in mAP
        - robustness_ratio: AP_adv / AP_clean (1.0 = fully robust)
        - delta_recall_50: Recall drop at IoU=0.5
        - delta_recall_75: Recall drop at IoU=0.75
        - delta_precision: Precision drop
    """
    ap_clean = clean_metrics.get("map", 0.0)
    ap_adv = adv_metrics.get("map", 0.0)

    ap50_clean = clean_metrics.get("map50", 0.0)
    ap50_adv = adv_metrics.get("map50", 0.0)

    # Calculate deltas
    delta_map = ap_clean - ap_adv
    delta_map50 = ap50_clean - ap50_adv

    # Calculate drop percentage
    drop_percentage = (delta_map / ap_clean * 100) if ap_clean > 0 else 0.0

    # Calculate robustness ratio (closer to 1.0 = more robust)
    robustness_ratio = (ap_adv / ap_clean) if ap_clean > 0 else 0.0

    # Recall and precision deltas
    recall_clean = clean_metrics.get("recall", 0.0)
    recall_adv = adv_metrics.get("recall", 0.0)
    delta_recall = recall_clean - recall_adv

    precision_clean = clean_metrics.get("precision", 0.0)
    precision_adv = adv_metrics.get("precision", 0.0)
    delta_precision = precision_clean - precision_adv

    # F1 Score deltas
    f1_clean = clean_metrics.get("f1", 0.0)
    f1_adv = adv_metrics.get("f1", 0.0)
    delta_f1 = f1_clean - f1_adv

    return {
        "delta_map": float(delta_map),
        "delta_map50": float(delta_map50),
        "drop_percentage": float(drop_percentage),
        "robustness_ratio": float(robustness_ratio),
        "delta_recall": float(delta_recall),
        "delta_precision": float(delta_precision),
        "delta_f1": float(delta_f1),
        # Clean vs Adv comparison
        "ap_clean": float(ap_clean),
        "ap_adv": float(ap_adv),
        "ap50_clean": float(ap50_clean),
        "ap50_adv": float(ap50_adv),
        "recall_clean": float(recall_clean),
        "recall_adv": float(recall_adv),
        "f1_clean": float(f1_clean),
        "f1_adv": float(f1_adv),
        "precision_clean": float(precision_clean),
        "precision_adv": float(precision_adv),
    }


def parse_detection_to_bbox(
    detection: Dict[str, Any],
    image_id: str,
    image_width: int = 640,
    image_height: int = 640,
) -> BoundingBox:
    """
    Parse detection dictionary to BoundingBox object.

    Handles multiple bbox formats:
    - Absolute coordinates: {x1, y1, x2, y2}
    - Normalized coordinates: {x_center, y_center, width, height} (0-1)
    - YOLO format: {x_center, y_center, width, height} (0-1)
    """
    bbox = detection.get("bbox", detection)
    class_name = detection.get("class_name", "unknown")
    confidence = detection.get("confidence", 1.0)

    def clamp01(value: float) -> float:
        return min(1.0, max(0.0, value))

    def normalize_xy(value: float, dimension: float) -> float:
        if dimension and dimension > 0:
            return value / dimension
        return value

    # Handle absolute coordinates
    if "x1" in bbox and "y1" in bbox and "x2" in bbox and "y2" in bbox:
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])

        is_already_normalized = (
            0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and
            0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0
        )

        if not is_already_normalized:
            x1 = normalize_xy(x1, image_width)
            y1 = normalize_xy(y1, image_height)
            x2 = normalize_xy(x2, image_width)
            y2 = normalize_xy(y2, image_height)

        x1 = clamp01(x1)
        y1 = clamp01(y1)
        x2 = clamp01(x2)
        y2 = clamp01(y2)

        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            class_name=class_name,
            confidence=float(confidence),
            image_id=image_id,
        )

    # Handle normalized/YOLO format
    elif "x_center" in bbox or "x" in bbox:
        x_center = float(bbox.get("x_center", bbox.get("x", 0.5)))
        y_center = float(bbox.get("y_center", bbox.get("y", 0.5)))
        width = float(bbox.get("width", bbox.get("w", 0.1)))
        height = float(bbox.get("height", bbox.get("h", 0.1)))

        is_already_normalized = (
            0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and
            0.0 <= width <= 1.0 and 0.0 <= height <= 1.0
        )

        if not is_already_normalized:
            x_center = normalize_xy(x_center, image_width)
            y_center = normalize_xy(y_center, image_height)
            width = normalize_xy(width, image_width)
            height = normalize_xy(height, image_height)

        x1 = clamp01(x_center - width / 2)
        y1 = clamp01(y_center - height / 2)
        x2 = clamp01(x_center + width / 2)
        y2 = clamp01(y_center + height / 2)

        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            class_name=class_name,
            confidence=float(confidence),
            image_id=image_id,
        )

    else:
        raise ValueError(f"Unsupported bbox format: {bbox}")
