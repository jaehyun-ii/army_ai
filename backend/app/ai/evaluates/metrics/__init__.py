"""
Evaluation Metrics Package.

Provides comprehensive evaluation metrics for object detection:
- Per-class detailed metrics (mAP, precision, recall, F1)
- IoU distribution analysis
- Size-specific performance metrics
"""
from app.ai.evaluates.metrics.per_class_metrics import (
    calculate_enhanced_class_metrics,
    calculate_all_classes_metrics,
    aggregate_class_metrics,
)
from app.ai.evaluates.metrics.iou_distribution import (
    calculate_iou_distribution,
    calculate_per_class_iou_distribution,
    calculate_iou_confusion_matrix,
)

__all__ = [
    # Per-class metrics
    "calculate_enhanced_class_metrics",
    "calculate_all_classes_metrics",
    "aggregate_class_metrics",
    # IoU distribution
    "calculate_iou_distribution",
    "calculate_per_class_iou_distribution",
    "calculate_iou_confusion_matrix",
]
