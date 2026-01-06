"""
Distortion-Aware Attack for Object Detection (PyTorch Implementation).

This module implements the Distortion-Aware attack from the attack_detector paper.
Performs image-specific adversarial attacks with NCC-based distortion control.

Key features:
- Image-specific perturbation optimization (not universal)
- YOLO training loss (L_box + L_cls + L_dfl) for gradient computation
- Per-object NCC checking with configurable threshold
- Iterative gradient ascent with object masking

Reference:
- Original: /attack_detector/yolov8_coco_attack.py
- Migrated from: universal_noise_pytorch.py (image_specific mode)

CRITICAL FIXES (2026-01-06):
- Use current_pseudo_gt for gradient computation (not initial)
- Add retain_grad() to prevent gradient loss
- Scale gradient back to [0, 1] range (divide by 255)
- Ensure model wrapper is in training mode

This implementation is mathematically equivalent to attack_detector.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm.auto import trange

from app.ai.attacks.attack import EvasionAttack
from app.ai.estimators.estimator import BaseEstimator
from app.ai.estimators.object_detection.object_detector import ObjectDetectorMixin

if TYPE_CHECKING:
    from app.ai.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_ncc_torch(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """
    Compute Normalized Cross Correlation between two images (PyTorch version).

    Mirrors attack_detector implementation but optimized for PyTorch.

    Args:
        image1: Tensor of shape (C, H, W) or (1, C, H, W)
        image2: Tensor of shape (C, H, W) or (1, C, H, W)

    Returns:
        NCC value (higher is more similar, 1.0 = identical)
    """
    # Remove batch dimension if present
    if image1.dim() == 4:
        image1 = image1.squeeze(0)
    if image2.dim() == 4:
        image2 = image2.squeeze(0)

    # Convert to grayscale (weighted average)
    # CRITICAL: Match OpenCV's BGR2GRAY conversion exactly
    # OpenCV uses BGR channel order, so we need to match that
    # Formula: 0.299*R + 0.587*G + 0.114*B
    # But with BGR order: 0.114*B + 0.587*G + 0.299*R
    if image1.shape[0] == 3:
        # BGR to grayscale (matching cv2.cvtColor(BGR2GRAY))
        # image[0] = B, image[1] = G, image[2] = R
        gray1 = 0.114 * image1[0] + 0.587 * image1[1] + 0.299 * image1[2]
        gray2 = 0.114 * image2[0] + 0.587 * image2[1] + 0.299 * image2[2]
    else:
        gray1 = image1[0]
        gray2 = image2[0]

    # Normalize (subtract mean)
    mean1 = gray1.mean()
    mean2 = gray2.mean()
    gray1_centered = gray1 - mean1
    gray2_centered = gray2 - mean2

    # Compute standard deviations
    std1 = gray1_centered.std()
    std2 = gray2_centered.std()

    # Compute NCC
    # NCC = Σ((x - μx)(y - μy)) / (σx * σy * N)
    numerator = (gray1_centered * gray2_centered).sum()
    denominator = std1 * std2 * gray1.numel()

    # Add small epsilon for numerical stability
    ncc = (numerator / (denominator + 1e-8)).item()

    # Subtract small value like original implementation
    ncc -= 1e-6

    return ncc


# ============================================================================
# Pseudo Ground Truth Generator
# ============================================================================

class _PseudoGTGenerator:
    """
    Generate pseudo ground truth labels using ART's predict API.

    This enables unsupervised adversarial attacks by using the model's own
    predictions as training targets. Now fully ART-compliant.
    """

    def __init__(
        self,
        estimator,
        target_class_id: int = 0,
        confidence_threshold: float = 0.0,
        top_k_detections: int = -1,
        device: str = 'cpu'
    ):
        """
        Args:
            estimator: ART estimator (for predict API)
            target_class_id: Class ID to target (e.g., 0 for 'person' in COCO)
            confidence_threshold: Minimum confidence for pseudo-GT
            top_k_detections: Number of detections to keep per image:
                             -1 = all detections above threshold (multi-target, default)
                              1 = single highest confidence detection (AEGIS-style)
                              N = top N detections
            device: Device to run on ('cpu' or 'cuda')
        """
        self.estimator = estimator
        self.target_class_id = target_class_id
        self.confidence_threshold = confidence_threshold
        self.top_k_detections = top_k_detections
        self.device = device

        logger.info(
            f"PseudoGTGenerator initialized: "
            f"target_class={target_class_id}, threshold={confidence_threshold}, "
            f"top_k={top_k_detections if top_k_detections > 0 else 'all'}"
        )

    def generate_from_estimator(
        self,
        images: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """
        Generate pseudo-GT using ART's predict API (fully ART-compliant).

        Args:
            images: Input images (B, C, H, W) - PyTorch tensor in [0, 1] range

        Returns:
            List of dicts with 'boxes' and 'labels' for each image
        """
        pseudo_gts = []

        # Convert to NumPy for ART interface
        x_numpy = images.detach().cpu().numpy()
        if not self.estimator.channels_first:
            x_numpy = np.transpose(x_numpy, (0, 2, 3, 1))

        # Denormalize if needed for ART
        if self.estimator.clip_values is not None:
            x_numpy = x_numpy * self.estimator.clip_values[1]

        # Use ART's predict API - handles all preprocessing automatically
        predictions = self.estimator.predict(x=x_numpy)

        # Process each prediction
        for idx, pred in enumerate(predictions):
            try:
                if 'boxes' not in pred or len(pred['boxes']) == 0:
                    logger.debug(f"No detections for image {idx}")
                    pseudo_gts.append(self._empty_pseudo_gt())
                    continue

                # predict() returns xyxy format, but this framework uses xywh
                boxes_xyxy = torch.from_numpy(pred['boxes']).float().to(self.device)
                scores = torch.from_numpy(pred['scores']).float().to(self.device)
                classes = torch.from_numpy(pred['labels']).long().to(self.device)

                # Convert xyxy to xywh (YOLO standard for this project)
                from app.ai.losses.box_utils import xyxy2xywh
                boxes_xywh = xyxy2xywh(boxes_xyxy)

                # CRITICAL DEBUG: Check coordinate scale
                if len(boxes_xywh) > 0:
                    max_coord = boxes_xywh.max().item()
                    logger.debug(f"Pseudo-GT boxes: max_coord={max_coord:.2f} (should be in pixel scale, NOT normalized)")

                # Filter by target class
                target_mask = (classes == self.target_class_id)
                target_boxes = boxes_xywh[target_mask]
                target_labels = classes[target_mask]
                target_scores = scores[target_mask]

                if len(target_boxes) > 0:
                    # Apply confidence threshold
                    conf_mask = target_scores >= self.confidence_threshold
                    filtered_boxes = target_boxes[conf_mask]
                    filtered_labels = target_labels[conf_mask]
                    filtered_scores = target_scores[conf_mask]

                    if len(filtered_boxes) > 0:
                        # Select top-k detections based on configuration
                        if self.top_k_detections == 1:
                            # AEGIS-style: single highest confidence detection
                            top_idx = torch.argmax(filtered_scores)
                            pseudo_gt = {
                                'boxes': filtered_boxes[top_idx:top_idx + 1],
                                'labels': filtered_labels[top_idx:top_idx + 1]
                            }
                            logger.debug(
                                f"Pseudo-GT for image {idx}: single target (AEGIS-style); "
                                f"score={filtered_scores[top_idx].item():.3f}"
                            )
                        elif self.top_k_detections > 1:
                            # Top-K detections
                            k = min(self.top_k_detections, len(filtered_boxes))
                            top_k_indices = torch.topk(filtered_scores, k).indices
                            pseudo_gt = {
                                'boxes': filtered_boxes[top_k_indices],
                                'labels': filtered_labels[top_k_indices]
                            }
                            logger.debug(
                                f"Pseudo-GT for image {idx}: top-{k} targets; "
                                f"scores={filtered_scores[top_k_indices].cpu().numpy()}"
                            )
                        else:
                            # All detections above threshold (multi-target, default)
                            pseudo_gt = {
                                'boxes': filtered_boxes,
                                'labels': filtered_labels
                            }
                            logger.debug(
                                f"Pseudo-GT for image {idx}: {len(filtered_boxes)} targets (multi-target); "
                                f"score_range=[{filtered_scores.min().item():.3f}, {filtered_scores.max().item():.3f}]"
                            )
                        pseudo_gts.append(pseudo_gt)
                    else:
                        logger.debug(
                            f"No detections for image {idx} with class {self.target_class_id} "
                            f"above threshold {self.confidence_threshold}"
                        )
                        pseudo_gts.append(self._empty_pseudo_gt())
                else:
                    logger.debug(f"No detections for image {idx} with class {self.target_class_id}")
                    pseudo_gts.append(self._empty_pseudo_gt())

            except Exception as e:
                logger.error(f"Error processing result {idx}: {e}")
                pseudo_gts.append(self._empty_pseudo_gt())

        return pseudo_gts

    def _empty_pseudo_gt(self) -> dict[str, torch.Tensor]:
        """Create empty pseudo-GT."""
        return {
            'boxes': torch.zeros(0, 4, dtype=torch.float32, device=self.device),
            'labels': torch.zeros(0, dtype=torch.int64, device=self.device)
        }

    def torch_to_numpy_labels(
        self,
        labels: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, np.ndarray]]:
        """Convert PyTorch labels to NumPy arrays."""
        numpy_labels = []
        for label in labels:
            numpy_label = {
                'boxes': label['boxes'].cpu().numpy().astype(np.float32),
                'labels': label['labels'].cpu().numpy().astype(np.int64)
            }
            numpy_labels.append(numpy_label)
        return numpy_labels


# ============================================================================
# Distortion-Aware Attack Class
# ============================================================================

class DistortionAwareAttackPyTorch(EvasionAttack):
    """
    Distortion-Aware Attack for Object Detection.
    
    Image-specific adversarial attack with NCC-based distortion control.
    Equivalent to attack_detector's yolov8_coco_attack_ncc implementation.
    
    Attack flow:
    1. For each image:
       2. Initialize perturbation to zero
       3. For max_iter iterations:
          a. Apply current perturbation
          b. Detect objects on adversarial image
          c. Compute gradient using YOLO training loss
          d. Create mask with NCC checking
          e. Update perturbation with gradient ascent
          f. Stop if all objects suppressed or NCC threshold exceeded
    """
    
    attack_params = EvasionAttack.attack_params + [
        "eps",
        "eps_step", 
        "max_iter",
        "ncc_threshold",
        "target_class_id",
        "verbose",
    ]
    
    _estimator_requirements = (BaseEstimator, ObjectDetectorMixin)
    
    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        eps: float = 1.0,
        eps_step: float = 0.392,
        max_iter: int = 3000,
        ncc_threshold: float = 0.6,
        target_class_id: int = 0,
        verbose: bool = True,
    ):
        """
        Create a DistortionAwareAttackPyTorch instance.
        
        Args:
            estimator: Trained object detector
            eps: Maximum perturbation epsilon (default 1.0 in [0,1] scale = 255 in [0,255])
            eps_step: Step size for gradient ascent (default 0.392 = 100/255)
            max_iter: Maximum iterations per image (default 3000)
            ncc_threshold: NCC threshold for stopping (default 0.6)
            target_class_id: Target class to attack (default 0 = person in COCO)
            verbose: Show progress bars
        """
        super().__init__(estimator=estimator)
        
        # Device
        self.device = self.estimator.device
        self._torch_model = self.estimator.model
        
        # Attack parameters
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.ncc_threshold = ncc_threshold
        self.target_class_id = target_class_id
        self.verbose = verbose

        # Distortion-Aware specific parameters
        self.use_model_loss = True  # Always use YOLO model loss
        self.apply_mask = True      # Always apply object masking
        
        # Clip bounds
        self.clip_min, self.clip_max = self._get_clip_bounds()
        
        # Input scale (255 if model expects [0,255], 1 if [0,1])
        self._input_scale = float(self.estimator.clip_values[1]) if self.estimator.clip_values is not None else 1.0
        
        # Normalize eps/eps_step if needed
        if self._input_scale > 1.0:
            if self.eps > 1.0:
                self.eps = self.eps / self._input_scale
            if self.eps_step > 1.0:
                self.eps_step = self.eps_step / self._input_scale
        
        # Pseudo-GT generator
        self._pseudo_gt_gen = _PseudoGTGenerator(
            estimator=self.estimator,
            target_class_id=target_class_id,
            confidence_threshold=0.0,
            top_k_detections=-1,  # All detections
            device=self.device
        )
        
        logger.info(
            f"DistortionAwareAttack initialized: "
            f"eps={eps}, step={eps_step}, max_iter={max_iter}, "
            f"ncc_threshold={ncc_threshold}, target_class={target_class_id}"
        )
    
    def _get_clip_bounds(self) -> tuple[float, float]:
        """Get clipping bounds in normalized [0,1] scale."""
        if self.estimator.clip_values is not None:
            clip_min = float(self.estimator.clip_values[0])
            clip_max = float(self.estimator.clip_values[1])
            if clip_max > 1.0:
                clip_min = clip_min / clip_max
                clip_max = 1.0
            return clip_min, clip_max
        return 0.0, 1.0
    
    def _check_params(self) -> None:
        """Check attack parameters are valid."""
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.eps_step <= 0:
            raise ValueError("eps_step must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not 0 <= self.ncc_threshold <= 1:
            raise ValueError("ncc_threshold must be in [0, 1]")
    
    def generate(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial examples.
        
        Args:
            x: Images in NumPy format (N, H, W, C) or (N, C, H, W)
            y: Optional labels (not used, will generate pseudo-GT)
            
        Returns:
            Adversarial images in same format as input
        """
        logger.info(f"Starting Distortion-Aware attack on {len(x)} images")
        
        # Preprocess
        x_torch, x_original = self._preprocess_and_convert(x)
        batch_size = x_torch.shape[0]
        
        # Attack each image independently
        x_adv_list = []
        
        pbar = trange(batch_size, desc="Attacking images", disable=not self.verbose)
        for img_idx in pbar:
            img_torch = x_torch[img_idx:img_idx + 1]  # (1, C, H, W)
            
            # Generate pseudo-GT
            img_pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(img_torch)
            img_pseudo_gt = img_pseudo_gts[0] if img_pseudo_gts else None
            
            # Attack this image
            img_adv, _ = self._attack_single_image(
                img_torch, img_pseudo_gt, img_idx, pbar
            )
            
            x_adv_list.append(img_adv)
        
        # Stack results
        x_adv_torch = torch.cat(x_adv_list, dim=0)
        
        # Convert back to NumPy
        x_adv = self._torch_to_numpy(x_adv_torch, x_original)
        
        logger.info("Distortion-Aware attack completed")
        return x_adv
    
    def _preprocess_and_convert(self, x: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Preprocess and convert NumPy to PyTorch."""
        x_original = x.copy()
        x_preprocessed = self.estimator._apply_preprocessing(x, y=None, fit=False)[0]
        
        # Convert to torch
        x_torch = torch.from_numpy(x_preprocessed).float().to(self.device)
        
        # Ensure channels first
        if not self.estimator.channels_first and x_torch.dim() == 4:
            x_torch = x_torch.permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] if needed
        if self._input_scale > 1.0:
            x_torch = x_torch / self._input_scale
        
        return x_torch, x_original
    
    def _torch_to_numpy(self, x_torch: torch.Tensor, _x_original: np.ndarray) -> np.ndarray:
        """Convert PyTorch tensor back to NumPy.

        Args:
            x_torch: PyTorch tensor to convert
            _x_original: Original NumPy array (for shape reference, currently unused)
        """
        # Denormalize if needed
        if self._input_scale > 1.0:
            x_torch = x_torch * self._input_scale
        
        # Convert to NumPy
        x_numpy = x_torch.detach().cpu().numpy()
        
        # Match original format (channels last if needed)
        if not self.estimator.channels_first and x_numpy.ndim == 4:
            x_numpy = np.transpose(x_numpy, (0, 2, 3, 1))
        
        return x_numpy

    # ========================================================================
    # Core Attack Methods
    # ========================================================================

    def _attack_single_image(
        self,
        img_torch: torch.Tensor,
        pseudo_gt: dict | None,
        img_idx: int,
        pbar: trange,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Attack a single image with iterative gradient ascent (attack_detector style).

        CRITICAL: This implementation exactly matches attack_detector's approach:
        1. Compute initial gradient from original image
        2. Loop:
           - Get current detections
           - Create mask with NCC checking
           - Update image directly (not perturbation)
           - Check stopping conditions
           - Recompute gradient from updated image

        Args:
            img_torch: Single image tensor (1, C, H, W)
            pseudo_gt: Pseudo ground truth for this image (or None)
            img_idx: Image index for logging
            pbar: Progress bar for updates

        Returns:
            Tuple of (adversarial image, perturbation)
        """
        # Check if image has detections
        if pseudo_gt is None or 'boxes' not in pseudo_gt or len(pseudo_gt['boxes']) == 0:
            logger.debug(f"Image {img_idx}: No detections, skipping")
            pbar.set_postfix({"img": img_idx, "status": "no_det"})
            perturbation = torch.zeros_like(img_torch)
            return img_torch, perturbation

        # Get initial detections count
        initial_det_count = len(pseudo_gt['boxes'])
        pbar.set_postfix({"img": img_idx, "dets": initial_det_count})

        # CRITICAL FIX: Work in [0, 255] scale like attack_detector
        # Original attack_detector uses [0, 255] throughout
        # Gradient magnitude is 255x larger in [0, 255] scale!
        img_torch_255 = img_torch * 255.0  # Convert to [0, 255]
        img_original_255 = img_torch_255.clone()

        # CRITICAL: Compute initial gradient from original image (before loop)
        # This matches attack_detector which computes gradient at the start
        img_to_attack = img_torch_255.clone().detach().requires_grad_(True)
        attack_grad = self._compute_gradient_with_model_loss(img_to_attack, pseudo_gt)

        if attack_grad is None:
            logger.warning(f"Image {img_idx}: Failed to compute initial gradient")
            perturbation = torch.zeros_like(img_torch)
            return img_torch, perturbation

        # Current attacking image (will be modified in-place style)
        img_current_255 = img_torch_255.clone()

        # Adjust step size for [0, 255] scale
        # Original: step=100 in [0, 255]
        # We're now in [0, 255], so use original step size
        step_size_255 = self.eps_step * 255.0  # 0.392 * 255 ≈ 100

        # Iterative attack loop (like attack_detector's 5000 iterations)
        for iter_num in range(self.max_iter):
            # 1. Get detections for current image (matches attack_detector's model.predict)
            # CRITICAL FIX: Convert [0, 255] to [0, 1] for pseudo-GT generation
            current_pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(img_current_255 / 255.0)
            current_pseudo_gt = current_pseudo_gts[0] if current_pseudo_gts else None

            if current_pseudo_gt is None or 'boxes' not in current_pseudo_gt or len(current_pseudo_gt['boxes']) == 0:
                # No more detections - attack successful!
                logger.debug(f"Image {img_idx}: All detections suppressed at iteration {iter_num}")
                pbar.set_postfix({"img": img_idx, "iter": iter_num, "status": "success"})
                break

            current_det_count = len(current_pseudo_gt['boxes'])

            # 2. Create mask with NCC checking (matches attack_detector's mask creation)
            if self.apply_mask and self.ncc_threshold > 0:
                # CRITICAL FIX: Convert [0, 255] to [0, 1] for NCC computation
                mask, objects_within_threshold = self._create_mask_with_ncc_check(
                    img_current_255 / 255.0, img_original_255 / 255.0, current_pseudo_gt
                )

                # If all objects exceed NCC threshold, stop (matches attack_detector's count == number_object)
                if objects_within_threshold == 0:
                    logger.debug(f"Image {img_idx}: All objects exceed NCC threshold at iteration {iter_num}")
                    pbar.set_postfix({"img": img_idx, "iter": iter_num, "status": "ncc_stop"})
                    break
            elif self.apply_mask:
                mask = self._create_simple_mask(current_pseudo_gt, img_torch.shape)
            else:
                mask = torch.ones_like(attack_grad)

            # 3. Update image directly with masked gradient (matches attack_detector's new_image = image + step*grad*mask)
            # CRITICAL FIX: Use img_current_255 (not img_current)
            img_new_255 = img_current_255 + step_size_255 * attack_grad * mask

            # CRITICAL FIX Bug #2: Clamp in [0, 255] scale (not [0, 1])
            # Option A: Use [0, 255] clipping for numerical stability
            img_current_255 = torch.clamp(img_new_255, 0.0, 255.0)

            # Option B (original attack_detector): No clipping at all
            # img_current_255 = img_new_255

            # Debug: Check actual image change
            if iter_num % 50 == 0:
                max_diff = (img_current_255 - img_original_255).abs().max().item()
                mean_diff = (img_current_255 - img_original_255).abs().mean().item()
                logger.debug(f"Iter {iter_num}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

            # 4. Recompute gradient from updated image (matches attack_detector's gradient recomputation)
            # CRITICAL: This is done AFTER updating the image, matching attack_detector exactly
            img_current_grad_255 = img_current_255.clone().detach().requires_grad_(True)

            # Get fresh pseudo-GT for gradient computation (current detections)
            # CRITICAL FIX: Convert [0, 255] to [0, 1] for pseudo-GT generation
            grad_pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(img_current_grad_255 / 255.0)
            grad_pseudo_gt = grad_pseudo_gts[0] if grad_pseudo_gts else None

            if grad_pseudo_gt is not None and len(grad_pseudo_gt.get('boxes', [])) > 0:
                # CRITICAL FIX: Pass [0, 255] scale image to gradient computation
                new_grad = self._compute_gradient_with_model_loss(img_current_grad_255, grad_pseudo_gt)
                if new_grad is not None:
                    attack_grad = new_grad

            # Update progress bar
            if iter_num % 10 == 0 or current_det_count < initial_det_count:
                pbar.set_postfix({
                    "img": img_idx,
                    "iter": iter_num,
                    "dets": f"{current_det_count}/{initial_det_count}"
                })

        # Final perturbation (in [0, 255] scale)
        perturbation_255 = img_current_255 - img_original_255

        # Convert back to [0, 1] scale for return
        img_current = img_current_255 / 255.0
        perturbation = perturbation_255 / 255.0

        return img_current, perturbation

    def _compute_gradient_with_model_loss(
        self,
        img_adv_255: torch.Tensor,
        pseudo_gt: dict,
    ) -> torch.Tensor | None:
        """
        Compute gradient using model's training loss (attack_detector style).

        This matches attack_detector's approach of using the YOLO training loss
        (L_box + L_cls + L_dfl) to compute gradients.

        CRITICAL: This function now works entirely in [0, 255] scale to match attack_detector!

        Args:
            img_adv_255: Adversarial image in [0, 255] scale with requires_grad=True (1, C, H, W)
            pseudo_gt: Pseudo ground truth dict with 'boxes' (xywh, pixel) and 'labels'

        Returns:
            Gradient tensor in [0, 255] scale (1, C, H, W) or None if computation fails
        """
        try:
            # Prepare targets in YOLO training format
            # Format: [batch_idx, class_id, x_center_norm, y_center_norm, width_norm, height_norm]
            boxes_pixel = pseudo_gt['boxes']  # (N, 4) in xywh pixel format
            labels = pseudo_gt['labels']  # (N,)

            if len(boxes_pixel) == 0:
                logger.debug("No boxes in pseudo_gt for gradient computation")
                return None

            # Normalize boxes to [0, 1] range (YOLO expects normalized coordinates)
            h, w = img_adv_255.shape[2:]

            # CRITICAL DEBUG: Check if boxes are already normalized
            max_box_val = boxes_pixel.max().item() if len(boxes_pixel) > 0 else 0
            logger.debug(f"Boxes BEFORE norm: max={max_box_val:.2f}, shape={boxes_pixel.shape}")
            if max_box_val <= 1.0:
                logger.warning(f"Boxes may already be normalized (max={max_box_val:.4f})! Double normalization will break gradient!")

            boxes_norm = boxes_pixel.clone()
            boxes_norm[:, 0] = boxes_pixel[:, 0] / w  # cx
            boxes_norm[:, 1] = boxes_pixel[:, 1] / h  # cy
            boxes_norm[:, 2] = boxes_pixel[:, 2] / w  # width
            boxes_norm[:, 3] = boxes_pixel[:, 3] / h  # height

            logger.debug(f"Boxes AFTER norm: max={boxes_norm.max().item():.4f}, first_box={boxes_norm[0] if len(boxes_norm) > 0 else 'none'}")

            # Create targets tensor: [batch_idx, class_id, cx, cy, w, h]
            batch_idx = torch.zeros(len(boxes_norm), 1, device=self.device)
            targets = torch.cat([
                batch_idx,
                labels.unsqueeze(1).float(),
                boxes_norm
            ], dim=1)  # (N, 6)

            # CRITICAL FIX Bug #3: img_adv_255 is already in [0, 255] scale!
            # DO NOT multiply by self._input_scale again!
            # The model wrapper expects [0, 255] input, and we're providing it.
            img_adv_255.retain_grad()  # Ensure gradient is retained
            img_adv_scaled = img_adv_255  # Already in correct scale!

            # Ensure model is in training mode (required for loss computation)
            was_training = self._torch_model.training
            self._torch_model.train()

            # Get model wrapper (PyTorchYoloLossWrapper)
            model_wrapper = self.estimator.model
            if not hasattr(model_wrapper, 'forward'):
                logger.error("Model wrapper doesn't have forward method")
                return None

            # CRITICAL: Ensure model wrapper is also in training mode
            model_wrapper.train()

            # Forward pass with loss computation
            # This uses YOLO's v8DetectionLoss (L_box + L_cls + L_dfl)
            # exactly like attack_detector's trainer.criterion()
            loss_dict = model_wrapper.forward(img_adv_scaled, targets)

            # Restore original model mode
            self._torch_model.train(mode=was_training)

            # Extract total loss
            if isinstance(loss_dict, dict):
                if 'loss_total' in loss_dict:
                    loss = loss_dict['loss_total']
                else:
                    # Fallback: sum all loss components
                    loss = sum(v for k, v in loss_dict.items() if 'loss' in k and isinstance(v, torch.Tensor))
                    logger.debug(f"Loss components: {list(loss_dict.keys())}")
            else:
                logger.warning(f"Unexpected loss format: {type(loss_dict)}")
                return None

            # Verify loss requires gradient
            if not loss.requires_grad:
                logger.warning("Loss doesn't require gradient - check model training mode")
                return None

            # CRITICAL DEBUG: Log loss value
            loss_value = loss.item()
            logger.debug(f"Loss value: {loss_value:.6f}")
            if loss_value < 1e-6:
                logger.error(f"Loss is almost zero ({loss_value:.10f})! Gradient will be useless!")

            # Compute gradients via backpropagation
            # This gives us ∂L/∂image (how to change image to maximize loss)
            loss.backward()

            # Extract gradient from adversarial image
            gradient = img_adv_255.grad
            if gradient is None:
                logger.warning("Gradient is None after backward pass")
                logger.warning("This may happen if retain_grad() was not called or computation graph was broken")
                return None

            # CRITICAL FIX Bug #3: Gradient is now in [0, 255] scale!
            # We work entirely in [0, 255] scale like attack_detector:
            # - img_adv_255: [0, 255] input
            # - img_adv_scaled: [0, 255] (no scaling, already correct)
            # - gradient: [0, 255] (∂L/∂img_adv_255)
            # - step_size_255: ~100 (in [0, 255] scale)
            # This matches attack_detector exactly!
            gradient_255 = gradient.detach()

            # Debug logging
            logger.debug(f"Gradient stats: norm={gradient_255.norm().item():.6f}, "
                        f"min={gradient_255.min().item():.6f}, "
                        f"max={gradient_255.max().item():.6f}")

            return gradient_255

        except Exception as e:
            logger.error(f"Error computing gradient with model loss: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return None

    def _create_mask_with_ncc_check(
        self,
        img_current: torch.Tensor,
        img_original: torch.Tensor,
        pseudo_gt: dict,
    ) -> tuple[torch.Tensor, int]:
        """
        Create mask with per-object NCC checking (attack_detector style).

        Args:
            img_current: Current adversarial image (1, C, H, W)
            img_original: Original clean image (1, C, H, W)
            pseudo_gt: Pseudo ground truth with boxes

        Returns:
            Tuple of (mask tensor, number of objects within NCC threshold)
        """
        from app.ai.losses.box_utils import xywh2xyxy

        mask = torch.zeros_like(img_current)

        boxes_xywh = pseudo_gt['boxes']
        if len(boxes_xywh) == 0:
            return mask, 0

        boxes_xyxy = xywh2xyxy(boxes_xywh)

        h, w = img_current.shape[2:]
        objects_within_threshold = 0

        for box in boxes_xyxy:
            # Get bbox coordinates
            x1 = max(0, min(int(box[0].item()), w))
            y1 = max(0, min(int(box[1].item()), h))
            x2 = max(0, min(int(box[2].item()), w))
            y2 = max(0, min(int(box[3].item()), h))

            if x1 >= x2 or y1 >= y2:
                continue

            # Extract bbox regions
            bbox_current = img_current[:, :, y1:y2, x1:x2]
            bbox_original = img_original[:, :, y1:y2, x1:x2]

            # Compute NCC for this bbox
            ncc = compute_ncc_torch(bbox_current, bbox_original)

            # Debug: Log NCC values for first few boxes
            if objects_within_threshold < 3:
                logger.debug(f"  Box {objects_within_threshold}: NCC={ncc:.4f} vs threshold={self.ncc_threshold:.4f}")

            # Only add to mask if NCC is above threshold
            # (higher NCC = more similar = less distorted)
            if ncc >= self.ncc_threshold:
                mask[:, :, y1:y2, x1:x2] = 1.0
                objects_within_threshold += 1

        logger.debug(f"Mask created: {objects_within_threshold}/{len(boxes_xyxy)} objects within NCC threshold")

        return mask, objects_within_threshold

    def _create_simple_mask(
        self,
        pseudo_gt: dict,
        img_shape: tuple,
    ) -> torch.Tensor:
        """
        Create simple binary mask from bounding boxes (without NCC checking).

        Args:
            pseudo_gt: Pseudo ground truth with boxes
            img_shape: Image shape (1, C, H, W)

        Returns:
            Mask tensor
        """
        from app.ai.losses.box_utils import xywh2xyxy

        mask = torch.zeros(img_shape, device=self.device)

        boxes_xywh = pseudo_gt['boxes']
        if len(boxes_xywh) == 0:
            return mask

        boxes_xyxy = xywh2xyxy(boxes_xywh)

        h, w = img_shape[2:]

        for box in boxes_xyxy:
            x1 = max(0, min(int(box[0].item()), w))
            y1 = max(0, min(int(box[1].item()), h))
            x2 = max(0, min(int(box[2].item()), w))
            y2 = max(0, min(int(box[3].item()), h))

            if x1 < x2 and y1 < y2:
                mask[:, :, y1:y2, x1:x2] = 1.0

        return mask
