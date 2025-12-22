"""
Post-processing utilities for object detection predictions.

This module provides utilities for filtering and refining object detection results:
- IoU (Intersection over Union) calculation
- NMS (Non-Maximum Suppression)
- Class-specific filtering
- Confidence thresholding
"""
from typing import Dict, List
import numpy as np
import torch
from torchvision.ops import nms


def calculate_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0


def apply_nms(
    predictions: List[Dict[str, np.ndarray]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> List[Dict[str, np.ndarray]]:
    """
    Apply Non-Maximum Suppression to predictions.

    Args:
        predictions: List of prediction dicts from estimator.predict()
                    Each dict contains 'boxes', 'labels', 'scores'
        iou_threshold: IoU threshold for NMS (default: 0.5)
        score_threshold: Minimum confidence score (default: 0.0)

    Returns:
        Filtered predictions after NMS

    Example:
        >>> predictions = estimator.predict(images)
        >>> filtered = apply_nms(predictions, iou_threshold=0.5, score_threshold=0.3)
    """
    filtered_predictions = []

    for pred in predictions:
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]

        if len(boxes) == 0:
            filtered_predictions.append({
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
            })
            continue

        # Convert to torch tensors
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()

        # Apply score threshold first
        score_mask = scores_tensor >= score_threshold
        boxes_tensor = boxes_tensor[score_mask]
        labels_filtered = labels[score_mask]
        scores_tensor = scores_tensor[score_mask]

        if len(boxes_tensor) == 0:
            filtered_predictions.append({
                "boxes": np.array([]).reshape(0, 4),
                "labels": np.array([]),
                "scores": np.array([]),
            })
            continue

        # Apply NMS per class (important for multi-class detection)
        keep_indices = []
        for class_id in np.unique(labels_filtered):
            class_mask = labels_filtered == class_id
            class_indices = np.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            # Apply NMS for this class
            class_boxes = boxes_tensor[class_mask]
            class_scores = scores_tensor[class_mask]

            # torchvision.ops.nms expects boxes in [x1, y1, x2, y2] format
            keep = nms(class_boxes, class_scores, iou_threshold)

            # Map back to original indices
            keep_indices.extend(class_indices[keep.cpu().numpy()])

        # Sort by original order
        keep_indices = sorted(keep_indices)

        # Filter results
        filtered_predictions.append({
            "boxes": boxes[keep_indices],
            "labels": labels[keep_indices],
            "scores": scores[keep_indices],
        })

    return filtered_predictions


def filter_by_class(
    predictions: List[Dict[str, np.ndarray]],
    target_class_ids: List[int],
) -> List[Dict[str, np.ndarray]]:
    """
    Filter predictions to only include specific classes.

    Args:
        predictions: List of prediction dicts
        target_class_ids: List of class IDs to keep

    Returns:
        Filtered predictions

    Example:
        >>> # Only keep person (0) and car (2)
        >>> filtered = filter_by_class(predictions, target_class_ids=[0, 2])
    """
    filtered_predictions = []

    for pred in predictions:
        mask = np.isin(pred["labels"], target_class_ids)

        filtered_predictions.append({
            "boxes": pred["boxes"][mask],
            "labels": pred["labels"][mask],
            "scores": pred["scores"][mask],
        })

    return filtered_predictions


def filter_by_confidence(
    predictions: List[Dict[str, np.ndarray]],
    min_confidence: float,
) -> List[Dict[str, np.ndarray]]:
    """
    Filter predictions by minimum confidence score.

    Args:
        predictions: List of prediction dicts
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        Filtered predictions

    Example:
        >>> filtered = filter_by_confidence(predictions, min_confidence=0.5)
    """
    filtered_predictions = []

    for pred in predictions:
        mask = pred["scores"] >= min_confidence

        filtered_predictions.append({
            "boxes": pred["boxes"][mask],
            "labels": pred["labels"][mask],
            "scores": pred["scores"][mask],
        })

    return filtered_predictions


def filter_by_iou_with_gt(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    min_iou: float,
) -> List[Dict[str, np.ndarray]]:
    """
    Filter predictions that have sufficient IoU with ground truth boxes.

    Useful for:
    - Keeping only predictions that align with known objects
    - Targeted attacks on specific object instances

    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth annotation dicts
        min_iou: Minimum IoU threshold with any GT box

    Returns:
        Filtered predictions

    Example:
        >>> # Only keep predictions that overlap with GT objects
        >>> filtered = filter_by_iou_with_gt(predictions, ground_truths, min_iou=0.3)
    """
    filtered_predictions = []

    for pred, gt in zip(predictions, ground_truths):
        if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
            filtered_predictions.append({
                "boxes": np.array([]).reshape(0, 4),
                "labels": np.array([]),
                "scores": np.array([]),
            })
            continue

        keep_mask = np.zeros(len(pred["boxes"]), dtype=bool)

        # Check each prediction against all GT boxes
        for pred_idx, pred_box in enumerate(pred["boxes"]):
            for gt_box in gt["boxes"]:
                iou = calculate_iou_numpy(pred_box, gt_box)
                if iou >= min_iou:
                    keep_mask[pred_idx] = True
                    break

        filtered_predictions.append({
            "boxes": pred["boxes"][keep_mask],
            "labels": pred["labels"][keep_mask],
            "scores": pred["scores"][keep_mask],
        })

    return filtered_predictions


def apply_combined_filtering(
    predictions: List[Dict[str, np.ndarray]],
    target_class_ids: List[int] | None = None,
    min_confidence: float = 0.0,
    nms_iou_threshold: float | None = None,
    ground_truths: List[Dict[str, np.ndarray]] | None = None,
    min_iou_with_gt: float | None = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Apply multiple filtering operations in sequence.

    Processing order:
    1. Confidence filtering (if min_confidence > 0)
    2. Class filtering (if target_class_ids specified)
    3. NMS (if nms_iou_threshold specified)
    4. IoU with GT filtering (if ground_truths and min_iou_with_gt specified)

    Args:
        predictions: List of prediction dicts
        target_class_ids: List of class IDs to keep (optional)
        min_confidence: Minimum confidence threshold (optional)
        nms_iou_threshold: IoU threshold for NMS (optional)
        ground_truths: Ground truth annotations (optional)
        min_iou_with_gt: Minimum IoU with GT boxes (optional)

    Returns:
        Filtered predictions

    Example:
        >>> filtered = apply_combined_filtering(
        ...     predictions,
        ...     target_class_ids=[0],  # Only person
        ...     min_confidence=0.3,
        ...     nms_iou_threshold=0.5,
        ... )
    """
    result = predictions

    # 1. Confidence filtering
    if min_confidence > 0:
        result = filter_by_confidence(result, min_confidence)

    # 2. Class filtering
    if target_class_ids is not None:
        result = filter_by_class(result, target_class_ids)

    # 3. NMS
    if nms_iou_threshold is not None:
        result = apply_nms(result, iou_threshold=nms_iou_threshold)

    # 4. IoU with GT filtering
    if ground_truths is not None and min_iou_with_gt is not None:
        result = filter_by_iou_with_gt(result, ground_truths, min_iou_with_gt)

    return result
