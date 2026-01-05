"""
Detection Attack Loss for adversarial attacks on object detectors.
Ported from AEGIS framework's DetectionLoss.

This loss is designed to REDUCE detection performance by:
1. Targeted Loss: Maximizes confidence for a specific target class
2. Agnostic Loss: Maximizes overall detection confidence (class-agnostic)

Minimizing this loss via gradient descent causes the detector to FAIL.
"""

import torch
import torch.nn as nn
from typing import Tuple

from .box_utils import box_iou, xywh2xyxy


class DetectionAttackLoss(nn.Module):
    """
    A highly configurable loss function for adversarial attacks on object detectors.

    This loss allows fine-grained control over how boxes and classes are selected
    and weighted to maximize attack effectiveness.

    Args:
        iou_threshold: Minimum IoU for a box proposal to be considered (default: 0.1)
        class_conf_threshold: Minimum confidence for a class to be considered for
                             agnostic loss (default: 0.05)
        top_k_boxes: Number of boxes to use in the loss. -1 for all boxes passing
                    the IoU threshold (default: 5)
        top_n_classes: Number of classes per box to sum for agnostic loss. -1 for
                      all classes passing the confidence threshold (default: 3)
        lambda_targeted: Weight for targeted loss component (default: 1.0)
        lambda_agnostic: Weight for agnostic loss component (default: 1.5)

    Example:
        >>> loss_fn = DetectionAttackLoss(iou_threshold=0.1, top_k_boxes=5)
        >>> # inference_output: (batch_size, num_proposals, 85) for YOLO
        >>> # pseudo_gt_boxes: (batch_size, 4) in xywh format (YOLO standard)
        >>> total_loss, targeted, agnostic = loss_fn(
        ...     inference_output, pseudo_gt_boxes, target_class_idx=0
        ... )
        >>> total_loss.backward()  # Gradients will reduce detection!
    """

    def __init__(
        self,
        iou_threshold: float = 0.1,
        class_conf_threshold: float = 0.05,
        top_k_boxes: int = 5,
        top_n_classes: int = 3,
        lambda_targeted: float = 1.0,
        lambda_agnostic: float = 1.5
    ):
        super().__init__()

        # Validate hyperparameters
        if not 0.0 <= iou_threshold < 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if not 0.0 <= class_conf_threshold < 1.0:
            raise ValueError("class_conf_threshold must be between 0.0 and 1.0")

        self.iou_threshold = iou_threshold
        self.class_conf_threshold = class_conf_threshold
        self.top_k_boxes = top_k_boxes
        self.top_n_classes = top_n_classes
        self.lambda_targeted = lambda_targeted
        self.lambda_agnostic = lambda_agnostic

    def forward(
        self,
        inference_output: torch.Tensor,
        pseudo_gt_boxes: torch.Tensor,
        target_class_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the detection attack loss.

        Args:
            inference_output: Model predictions
                            - YOLO v5 format: (batch_size, num_proposals, 85)
                              [cx, cy, w, h, objectness, class_0, ..., class_79]
                            - YOLO v8 format: (batch_size, num_proposals, 84)
                              [cx, cy, w, h, class_0, ..., class_79]
                            Note: Coordinates should be in xywh format (YOLO standard), scores NOT sigmoidized
            pseudo_gt_boxes: Pseudo ground truth boxes of shape (batch_size, 4)
                           Format: [cx, cy, w, h] in xywh format (YOLO standard)
            target_class_idx: Index of the target class for targeted attack

        Returns:
            Tuple of (total_loss, targeted_loss, agnostic_loss)
            - total_loss: Weighted sum of targeted and agnostic losses
            - targeted_loss: Loss component for target class
            - agnostic_loss: Loss component for all classes
        """
        device = inference_output.device
        batch_size = inference_output.shape[0]
        num_features = inference_output.shape[2]  # 84 or 85

        # Auto-detect YOLO version based on feature count
        # YOLO v5/v7: 85 (4 bbox + 1 obj + 80 classes)
        # YOLO v8+:    84 (4 bbox + 80 classes, no objectness)
        has_objectness = (num_features == 85)

        # Initialize accumulators (avoid in-place ops on leaf tensors)
        batch_loss_targeted = torch.tensor(0.0, device=device)
        batch_loss_agnostic = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            # Get single pseudo-GT box for this batch item (xywh format)
            gt_box_single_xywh = pseudo_gt_boxes[b].unsqueeze(0)  # (1, 4)

            # Skip if no valid ground truth box
            if torch.sum(gt_box_single_xywh) == 0:
                continue

            # Convert pseudo-GT box from xywh to xyxy format for IoU calculation
            gt_box_single = xywh2xyxy(gt_box_single_xywh)  # (1, 4)

            # Get predictions for this batch item
            preds = inference_output[b]  # (num_proposals, 84 or 85)

            # Skip if no predictions
            if preds is None or len(preds) == 0:
                continue

            # Parse YOLO output format
            # YOLO outputs boxes in xywh format (center x, center y, width, height)
            pred_boxes_xywh = preds[:, :4]  # (num_proposals, 4)

            # Convert to xyxy format for IoU calculation
            # box_iou requires xyxy format (x1, y1, x2, y2)
            pred_boxes = xywh2xyxy(pred_boxes_xywh)  # (num_proposals, 4)

            # DEBUG: Check if predictions are already sigmoidized
            import logging
            logger = logging.getLogger(__name__)
            if b == 0:  # Only log first batch item
                if has_objectness:
                    logger.debug(f"[AttackLoss] Raw obj scores (sample): min={preds[:, 4].min():.4f}, max={preds[:, 4].max():.4f}, mean={preds[:, 4].mean():.4f}")
                    logger.debug(f"[AttackLoss] Raw class scores (sample): min={preds[:, 5:].min():.4f}, max={preds[:, 5:].max():.4f}, mean={preds[:, 5:].mean():.4f}")
                else:
                    logger.debug(f"[AttackLoss] Raw class scores (sample): min={preds[:, 4:].min():.4f}, max={preds[:, 4:].max():.4f}, mean={preds[:, 4:].mean():.4f}")

            if has_objectness:
                # YOLO v5 format: [cx, cy, w, h, obj, class_0, ..., class_79]
                obj_scores = torch.sigmoid(preds[:, 4])  # (num_proposals,)
                class_scores = torch.sigmoid(preds[:, 5:])  # (num_proposals, 80)
            else:
                # YOLO v8 format: [cx, cy, w, h, class_0, ..., class_79]
                class_scores = torch.sigmoid(preds[:, 4:])  # (num_proposals, 80)
                # Use max class confidence as objectness proxy
                obj_scores = class_scores.max(dim=1)[0]  # (num_proposals,)

            if b == 0:  # Only log first batch item
                logger.debug(f"[AttackLoss] After sigmoid - obj_scores: min={obj_scores.min():.4f}, max={obj_scores.max():.4f}, mean={obj_scores.mean():.4f}")
                logger.debug(f"[AttackLoss] After sigmoid - class_scores: min={class_scores.min():.4f}, max={class_scores.max():.4f}, mean={class_scores.mean():.4f}")

            # Step 1: Filter boxes by IoU with pseudo-GT
            ious = box_iou(pred_boxes, gt_box_single).squeeze(-1)  # (num_proposals,)
            keep_mask = ious >= self.iou_threshold

            # Skip if no boxes pass IoU threshold
            if not torch.any(keep_mask):
                continue

            # Filter predictions by IoU threshold
            filtered_obj_scores = obj_scores[keep_mask]  # (num_filtered,)
            filtered_class_scores = class_scores[keep_mask]  # (num_filtered, 80)
            filtered_ious = ious[keep_mask]  # (num_filtered,)

            # Step 2: Calculate Targeted Loss
            # Score = objectness × target_class_confidence × IoU
            target_class_conf = filtered_class_scores[:, target_class_idx]
            targeted_scores = filtered_obj_scores * target_class_conf * filtered_ious

            # Step 3: Calculate Agnostic Loss
            # Create mask for classes above confidence threshold
            class_keep_mask = filtered_class_scores >= self.class_conf_threshold
            thresholded_class_scores = filtered_class_scores * class_keep_mask.float()

            # Select classes (Top-N or All)
            if self.top_n_classes == -1:
                # Use ALL classes that passed the threshold
                sum_class_conf = torch.sum(thresholded_class_scores, dim=1)
            else:
                # Find the top_n_classes from the thresholded scores
                n_classes_available = thresholded_class_scores.shape[1]
                n = min(self.top_n_classes, n_classes_available)

                top_n_class_conf_vals, _ = torch.topk(thresholded_class_scores, n, dim=1)
                sum_class_conf = torch.sum(top_n_class_conf_vals, dim=1)

            # Score = objectness × sum(class_confidences) × IoU
            agnostic_scores = filtered_obj_scores * sum_class_conf * filtered_ious

            # Step 4: Select Top-K boxes
            num_filtered_boxes = len(targeted_scores)

            if self.top_k_boxes == -1:
                k = num_filtered_boxes
            else:
                k = min(self.top_k_boxes, num_filtered_boxes)

            if k > 0:
                if self.top_k_boxes == -1:
                    # Use all boxes
                    sum_targeted_scores = torch.sum(targeted_scores)
                    sum_agnostic_scores = torch.sum(agnostic_scores)
                else:
                    # Use top-k boxes
                    top_k_targeted_boxes, _ = torch.topk(targeted_scores, k)
                    sum_targeted_scores = torch.sum(top_k_targeted_boxes)

                    top_k_agnostic_boxes, _ = torch.topk(agnostic_scores, k)
                    sum_agnostic_scores = torch.sum(top_k_agnostic_boxes)

                batch_loss_targeted = batch_loss_targeted + sum_targeted_scores
                batch_loss_agnostic = batch_loss_agnostic + sum_agnostic_scores

        # Step 5: Calculate final losses (average over batch)
        # Use squeeze() to convert from shape (1,) to scalar for cleaner output
        final_targeted_loss = (
            (batch_loss_targeted / batch_size).squeeze() if batch_size > 0
            else torch.zeros(1, device=device, requires_grad=True).squeeze()
        )
        final_agnostic_loss = (
            (batch_loss_agnostic / batch_size).squeeze() if batch_size > 0
            else torch.zeros(1, device=device, requires_grad=True).squeeze()
        )

        # Total loss is weighted combination
        total_loss = (
            self.lambda_targeted * final_targeted_loss +
            self.lambda_agnostic * final_agnostic_loss
        )

        # CRITICAL: Negate the loss for gradient ascent optimization
        # We use GRADIENT ASCENT to MAXIMIZE this loss
        # Detection scores are positive (higher = stronger detection)
        # To REDUCE detection, we need to MINIMIZE scores
        # By returning NEGATIVE scores and using gradient ascent,
        # we effectively perform gradient descent on the actual scores
        # This makes the attack suppress detections correctly
        return -total_loss, -final_targeted_loss, -final_agnostic_loss

    def __repr__(self):
        return (
            f"DetectionAttackLoss("
            f"iou_threshold={self.iou_threshold}, "
            f"class_conf_threshold={self.class_conf_threshold}, "
            f"top_k_boxes={self.top_k_boxes}, "
            f"top_n_classes={self.top_n_classes}, "
            f"lambda_targeted={self.lambda_targeted}, "
            f"lambda_agnostic={self.lambda_agnostic})"
        )
