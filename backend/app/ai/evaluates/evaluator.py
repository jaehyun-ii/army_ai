"""
Object Detection Evaluation System.

Provides comprehensive evaluation metrics including:
- Per-class detailed metrics (mAP, precision, recall, F1)
- IoU distribution analysis
- Size-specific performance metrics
- Robustness analysis for adversarial evaluation
"""
import logging
from typing import Dict, Any, List, Optional

from app.services.metrics_calculator import (
    BoundingBox,
    parse_detection_to_bbox,
    calculate_robustness_metrics,
)
from app.ai.evaluates.metrics.per_class_metrics import (
    calculate_all_classes_metrics,
    aggregate_class_metrics,
)
from app.ai.evaluates.metrics.iou_distribution import (
    calculate_iou_distribution,
    calculate_per_class_iou_distribution,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive object detection evaluator.

    Features:
    - Per-class metrics: mAP, mAP@0.5, mAP@0.75, mAP@0.5:0.95
    - Precision, Recall, F1 Score per class
    - IoU distribution analysis (overall and per-class)
    - Size-specific metrics (small, medium, large objects)
    - Robustness metrics for adversarial evaluation
    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize Evaluator.

        Args:
            iou_threshold: IoU threshold for matching predictions to GT (default: 0.5)
        """
        self.iou_threshold = iou_threshold
        logger.info(f"Evaluator initialized with IoU threshold: {iou_threshold}")

    def evaluate(
        self,
        predictions: List[BoundingBox],
        ground_truths: List[BoundingBox],
        target_class: Optional[str] = None,
        include_iou_distribution: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.

        Args:
            predictions: List of predicted bounding boxes
            ground_truths: List of ground truth bounding boxes
            target_class: If provided, only evaluate this specific class
            include_iou_distribution: Whether to include IoU distribution analysis

        Returns:
            Dictionary containing:
            - overall: Aggregated metrics across all classes
            - per_class: Detailed metrics for each class
            - iou_distribution: Overall IoU distribution statistics
            - per_class_iou_distribution: IoU distribution per class
        """
        logger.info(f"Starting evaluation: {len(predictions)} predictions, {len(ground_truths)} GTs")
        if target_class:
            logger.info(f"Target class filtering enabled: {target_class}")

        # Calculate per-class metrics
        per_class_metrics = calculate_all_classes_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_threshold=self.iou_threshold,
            include_iou_distribution=include_iou_distribution,
            target_class=target_class,
        )

        # Aggregate metrics
        overall_metrics = aggregate_class_metrics(per_class_metrics)

        # Calculate overall IoU distribution
        overall_iou_dist = None
        per_class_iou_dist = None
        if include_iou_distribution:
            overall_iou_dist = calculate_iou_distribution(
                predictions=predictions,
                ground_truths=ground_truths,
                iou_threshold=self.iou_threshold,
            )

            per_class_iou_dist = calculate_per_class_iou_distribution(
                predictions=predictions,
                ground_truths=ground_truths,
                iou_threshold=self.iou_threshold,
            )

        # Log summary
        logger.info(f"Evaluation complete: mAP={overall_metrics.get('map', 0.0):.4f}, "
                   f"mAP@50={overall_metrics.get('map50', 0.0):.4f}, "
                   f"Precision={overall_metrics.get('precision', 0.0):.4f}, "
                   f"Recall={overall_metrics.get('recall', 0.0):.4f}")

        return {
            "overall": overall_metrics,
            "per_class": per_class_metrics,
            "iou_distribution": overall_iou_dist,
            "per_class_iou_distribution": per_class_iou_dist,
        }

    def evaluate_from_inference_results(
        self,
        inference_results: List[Dict[str, Any]],
        ground_truths_by_image: Dict[str, List[Dict[str, Any]]],
        target_class: Optional[str] = None,
        include_iou_distribution: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate from inference results and ground truth dictionaries.

        Args:
            inference_results: List of inference result dicts with format:
                [{"image_id": str, "detections": [{"bbox": {...}, "class_name": str, "confidence": float}]}]
            ground_truths_by_image: Dict mapping image_id to list of GT dicts:
                {"image_id": [{"bbox": {...}, "class_name": str}]}
            target_class: If provided, only evaluate this specific class
            include_iou_distribution: Whether to include IoU distribution analysis

        Returns:
            Evaluation results dictionary
        """
        # Convert to BoundingBox objects
        all_predictions = []
        all_ground_truths = []

        for result in inference_results:
            image_id = str(result["image_id"])
            image_width = result.get("image_width", 640)
            image_height = result.get("image_height", 640)

            # Parse predictions
            for detection in result.get("detections", []):
                try:
                    bbox = parse_detection_to_bbox(
                        detection,
                        image_id=image_id,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    # Filter by target class if specified
                    if target_class is None or bbox.class_name == target_class:
                        all_predictions.append(bbox)
                except Exception as e:
                    logger.warning(f"Failed to parse prediction bbox: {e}")

            # Parse ground truths
            gts = ground_truths_by_image.get(image_id, [])
            for gt in gts:
                try:
                    gt_detection = {
                        "bbox": gt.get("bbox", {}),
                        "class_name": gt.get("class_name", "unknown"),
                        "confidence": 1.0,
                    }
                    bbox = parse_detection_to_bbox(
                        gt_detection,
                        image_id=image_id,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    # Filter by target class if specified
                    if target_class is None or bbox.class_name == target_class:
                        all_ground_truths.append(bbox)
                except Exception as e:
                    logger.warning(f"Failed to parse GT bbox: {e}")

        # Perform evaluation
        return self.evaluate(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            target_class=target_class,
            include_iou_distribution=include_iou_distribution,
        )

    def compare_robustness(
        self,
        clean_results: Dict[str, Any],
        adversarial_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare clean vs adversarial evaluation results to calculate robustness metrics.

        Args:
            clean_results: Evaluation results from clean/original dataset
            adversarial_results: Evaluation results from adversarial dataset

        Returns:
            Dictionary containing:
            - robustness: Overall robustness metrics
            - per_class_robustness: Robustness metrics per class
        """
        logger.info("Calculating robustness metrics...")

        # Calculate overall robustness
        overall_robustness = calculate_robustness_metrics(
            clean_metrics=clean_results["overall"],
            adv_metrics=adversarial_results["overall"],
        )

        # Calculate per-class robustness
        per_class_robustness = {}
        clean_per_class = clean_results.get("per_class", {})
        adv_per_class = adversarial_results.get("per_class", {})

        # Get all classes
        all_classes = set(clean_per_class.keys()) | set(adv_per_class.keys())

        for class_name in all_classes:
            clean_metrics = clean_per_class.get(class_name, {})
            adv_metrics = adv_per_class.get(class_name, {})

            if clean_metrics and adv_metrics:
                per_class_robustness[class_name] = calculate_robustness_metrics(
                    clean_metrics=clean_metrics,
                    adv_metrics=adv_metrics,
                )

        logger.info(f"Robustness analysis complete: "
                   f"ΔmAP={overall_robustness.get('delta_map', 0.0):.4f}, "
                   f"Drop%={overall_robustness.get('drop_percentage', 0.0):.2f}%")

        return {
            "robustness": overall_robustness,
            "per_class_robustness": per_class_robustness,
        }

    def get_metrics_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of evaluation results.

        Args:
            results: Evaluation results from evaluate()

        Returns:
            Formatted string summary
        """
        overall = results.get("overall", {})
        per_class = results.get("per_class", {})

        summary_lines = [
            "=" * 80,
            "EVALUATION SUMMARY",
            "=" * 80,
            "",
            "Overall Metrics:",
            f"  mAP@0.5:0.95: {overall.get('map', 0.0):.4f}",
            f"  mAP@0.5:      {overall.get('map50', 0.0):.4f}",
            f"  mAP@0.75:     {overall.get('map75', 0.0):.4f}",
            f"  Precision:    {overall.get('precision', 0.0):.4f}",
            f"  Recall:       {overall.get('recall', 0.0):.4f}",
            f"  F1 Score:     {overall.get('f1', 0.0):.4f}",
            "",
            f"Total Ground Truths: {overall.get('total_gt', 0)}",
            f"Total Predictions:   {overall.get('total_pred', 0)}",
            f"True Positives:      {overall.get('total_tp', 0)}",
            f"False Positives:     {overall.get('total_fp', 0)}",
            f"False Negatives:     {overall.get('total_fn', 0)}",
            "",
            "=" * 80,
            "Per-Class Metrics:",
            "=" * 80,
        ]

        # Sort classes by mAP for better readability
        sorted_classes = sorted(
            per_class.items(),
            key=lambda x: x[1].get("map50", 0.0),
            reverse=True
        )

        for class_name, metrics in sorted_classes:
            summary_lines.extend([
                "",
                f"Class: {class_name}",
                f"  mAP@0.5:0.95: {metrics.get('map', 0.0):.4f}",
                f"  mAP@0.5:      {metrics.get('map50', 0.0):.4f}",
                f"  mAP@0.75:     {metrics.get('map75', 0.0):.4f}",
                f"  Precision:    {metrics.get('precision', 0.0):.4f}",
                f"  Recall:       {metrics.get('recall', 0.0):.4f}",
                f"  F1:           {metrics.get('f1', 0.0):.4f}",
                f"  GT Count:     {metrics.get('gt_count', 0)}",
                f"  Pred Count:   {metrics.get('pred_count', 0)}",
            ])

        summary_lines.append("=" * 80)

        return "\n".join(summary_lines)
