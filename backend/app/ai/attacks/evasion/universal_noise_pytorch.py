"""
Universal Noise Attack with full PyTorch implementation.

This version restores all functionality from the original AEGIS framework:
- Pseudo ground truth generation
- Differentiable loss computation
- Object masking

CRITICAL FIXES (2026-01-02):
- Process entire dataset at once (not mini-batches) for consistency
- Gradient ascent to maximize detection loss (make detection fail)
- Sum gradients across all samples (like AEGIS ensemble)
- Direct SGD-style updates (not Adam optimizer)

External interface remains NumPy-based for ART compatibility.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from app.ai.attacks.attack import EvasionAttack
from app.ai.estimators.estimator import BaseEstimator, LossGradientsMixin
from app.ai.estimators.object_detection.object_detector import ObjectDetectorMixin
from app.ai.summary_writer import SummaryWriter
from app.ai import config
from app.ai.lr_scheduler import create_lr_scheduler

if TYPE_CHECKING:
    from app.ai.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


# ============================================================================
# Internal Classes (Private to this attack module)
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
        confidence_threshold: float = 0.3,
        device: str = 'cpu'
    ):
        """
        Args:
            estimator: ART estimator (for predict API)
            target_class_id: Class ID to target (e.g., 0 for 'person' in COCO)
            confidence_threshold: Minimum confidence for pseudo-GT
            device: Device to run on ('cpu' or 'cuda')
        """
        self.estimator = estimator
        self.target_class_id = target_class_id
        self.confidence_threshold = confidence_threshold
        self.device = device

        logger.info(
            f"PseudoGTGenerator initialized: "
            f"target_class={target_class_id}, threshold={confidence_threshold}"
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

                # Filter by target class and confidence
                mask = (classes == self.target_class_id) & (scores >= self.confidence_threshold)
                filtered_boxes = boxes_xywh[mask]
                filtered_labels = classes[mask]
                filtered_scores = scores[mask]

                # If no detections, use empty arrays
                if len(filtered_boxes) == 0:
                    logger.debug(f"No detections for image {idx} with class {self.target_class_id}")
                    pseudo_gts.append(self._empty_pseudo_gt())
                else:
                    # Use highest confidence detection
                    best_idx = filtered_scores.argmax()
                    pseudo_gt = {
                        'boxes': filtered_boxes[best_idx:best_idx+1],  # (1, 4) in xywh
                        'labels': filtered_labels[best_idx:best_idx+1]  # (1,)
                    }
                    pseudo_gts.append(pseudo_gt)
                    logger.debug(
                        f"Pseudo-GT for image {idx}: "
                        f"box={filtered_boxes[best_idx].cpu().numpy()}, "
                        f"score={filtered_scores[best_idx].item():.3f}"
                    )

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


class UniversalNoiseAttackPyTorch(EvasionAttack):
    """
    Universal Noise Attack with full functionality restored using PyTorch.

    External interface: NumPy (ART compatible)
    Internal implementation: PyTorch (for differentiability and performance)
    """

    attack_params = EvasionAttack.attack_params + [
        "eps",
        "eps_step",
        "max_iter",
        "batch_size",
        "apply_mask",
        "target_class_id",
        "summary_writer",
        "verbose",
        "return_perturbation",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        eps: float = 0.3,
        eps_step: float = 0.01,
        max_iter: int = 50,
        batch_size: int = 4,
        apply_mask: bool = True,
        target_class_id: int = 0,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
        scheduler_type: str = "constant",
        scheduler_params: dict | None = None,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ):
        """
        Create a UniversalNoiseAttackPyTorch attack instance.

        :param estimator: A trained object detector.
        :param eps: Maximum perturbation epsilon.
        :param eps_step: Learning rate for perturbation optimization.
        :param max_iter: Maximum number of iterations.
        :param batch_size: Batch size for training.
        :param apply_mask: Whether to apply masking to object regions only.
        :param target_class_id: Target class ID for generating pseudo ground truth.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator, summary_writer=summary_writer)

        # Device management
        self.device = self.estimator.device
        self._torch_model = self.estimator.model

        # Attack parameters
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.apply_mask = apply_mask
        self.target_class_id = target_class_id
        self.verbose = verbose
        self.progress_callback = progress_callback
        # Always clamp to detector scale; default to 0-255 if estimator does not define clip values
        self.clip_min, self.clip_max = self._get_clip_bounds()
        self._check_params()

        # Use provided scheduler configuration or defaults
        self.lr_scheduler_type = scheduler_type
        self.lr_scheduler_params = scheduler_params if scheduler_params is not None else {}

        # Universal perturbation (PyTorch parameter)
        self._perturbation_torch: nn.Parameter | None = None
        self._perturbation: np.ndarray | None = None

        # Configure attack-specific loss for the estimator
        self._configure_attack_loss()

        logger.info(f"UniversalNoiseAttackPyTorch initialized on device: {self.device}")

        # Initialize pseudo-GT generator (now ART-compliant)
        self._pseudo_gt_gen = _PseudoGTGenerator(
            estimator=self.estimator,  # Pass estimator for ART predict API
            target_class_id=target_class_id,
            confidence_threshold=0.3,
            device=self.device
        )


    def generate(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        *,
        return_perturbation: bool = False,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Generate universal adversarial perturbation.

        Args:
            x: Sample images (N, H, W, C) or (N, C, H, W) - NumPy
            y: Optional labels (will use pseudo-GT if None)
            return_perturbation: If True, also return the learned universal perturbation.

        Returns:
            Adversarial images. If `return_perturbation=True`, also returns the learned perturbation.
        """
        logger.info(f"Starting Universal Noise generation with {len(x)} images")

        # Use base class preprocessing
        x_torch, x_original = self._preprocess_and_convert(x)

        # Initialize perturbation using base class
        self._initialize_perturbation(x_torch)

        # Train perturbation using PyTorch
        self._train_universal_perturbation_pytorch(x_torch, y)

        # Apply perturbation
        if self.apply_mask:
            # Use provided labels or generate pseudo-GT for masking
            if y is not None:
                # Convert y to torch format for masking
                pseudo_gts = []
                for label in y:
                    from app.ai.losses.box_utils import xyxy2xywh

                    boxes_xyxy = torch.from_numpy(label['boxes']).float().to(self.device)
                    pseudo_gt = {
                        'boxes': xyxy2xywh(boxes_xyxy),
                        'labels': torch.from_numpy(label['labels']).long().to(self.device)
                    }
                    pseudo_gts.append(pseudo_gt)
                logger.info(f"Using provided labels for masking")
            else:
                # Generate pseudo-GT for masking using ART predict API
                pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(x_torch)
                logger.info(f"Generated pseudo-GT for masking")
            x_adv_torch = self._apply_perturbation_with_mask(x_torch, pseudo_gts)
        else:
            with torch.no_grad():
                x_adv_torch = torch.clamp(x_torch + self._perturbation_torch, self.clip_min, self.clip_max)

        # Convert back to NumPy (use ART's reverse conversion)
        x_adv = self._torch_to_numpy(x_adv_torch, x)
        self._perturbation = self._torch_to_numpy(self._perturbation_torch, x)

        logger.info("Universal Noise generation completed")

        # Return based on return_perturbation flag (follow EvasionAttack contract)
        if return_perturbation:
            return x_adv, self._perturbation
        else:
            return x_adv

    def _train_universal_perturbation_pytorch(
        self,
        x: torch.Tensor,
        y: list | None = None
    ):
        """
        Train universal perturbation using PyTorch with ART loss.

        FIXED: Process entire dataset at once (like AEGIS) instead of mini-batches
        to maintain consistency across all samples.

        Args:
            x: Input images (N, C, H, W)
            y: Optional labels. If provided, use as ground truth. If None, generate pseudo-GT.
        """
        logger.info(f"Training Universal Noise for {self.max_iter} iterations on {len(x)} images")

        # Use provided labels or generate pseudo ground truth
        if y is not None:
            logger.info(f"Using provided labels (y) for {len(y)} images")
            # Convert y to torch format if needed
            if isinstance(y, list) and len(y) > 0:
                if isinstance(y[0], dict):
                    # Already in dict format with 'boxes', 'labels'
                    pseudo_gts = []
                    for label in y:
                        from app.ai.losses.box_utils import xyxy2xywh

                        boxes_xyxy = torch.from_numpy(label['boxes']).float().to(self.device)
                        pseudo_gt = {
                            'boxes': xyxy2xywh(boxes_xyxy),
                            'labels': torch.from_numpy(label['labels']).long().to(self.device)
                        }
                        pseudo_gts.append(pseudo_gt)
                    pseudo_gts_numpy = self._pseudo_gt_gen.torch_to_numpy_labels(pseudo_gts)
                else:
                    raise ValueError(f"Unsupported y format: {type(y[0])}")
            else:
                # Generate pseudo-GT if y is empty
                logger.info("Provided y is empty, generating pseudo-GT")
                pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(x)
                pseudo_gts_numpy = self._pseudo_gt_gen.torch_to_numpy_labels(pseudo_gts)
        else:
            # Generate pseudo ground truth using ART predict API
            logger.info("No labels provided, generating pseudo-GT")
            pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(x)
            pseudo_gts_numpy = self._pseudo_gt_gen.torch_to_numpy_labels(pseudo_gts)
            logger.info(f"Generated pseudo-GT for {len(pseudo_gts)} images")

        # Filter out samples with empty boxes once at the beginning
        valid_indices = [i for i, label_dict in enumerate(pseudo_gts_numpy) if len(label_dict.get('boxes', [])) > 0]

        if len(valid_indices) == 0:
            logger.error("No valid detections found in any image. Cannot train universal perturbation.")
            return

        # Filter all data to only include valid samples
        x_valid = x[valid_indices]
        pseudo_gts_valid = [pseudo_gts[i] for i in valid_indices]
        pseudo_gts_numpy_valid = [pseudo_gts_numpy[i] for i in valid_indices]

        logger.info(f"Training on {len(valid_indices)}/{len(x)} images with detections")

        # Training loop - AEGIS style: process ALL data each iteration
        for i_iter in trange(self.max_iter, desc="Universal Noise Training", disable=not self.verbose):
            # Apply perturbation to ALL valid images at once
            if self.apply_mask:
                x_adv = self._apply_perturbation_with_mask(x_valid, pseudo_gts_valid)
            else:
                x_adv = torch.clamp(x_valid + self._perturbation_torch, self.clip_min, self.clip_max)

            # Compute gradient for ALL samples (like AEGIS ensemble loss)
            # Denormalize: internal [0, 1] -> estimator expects [0, clip_max]
            if self.estimator.clip_values is not None:
                x_for_loss = x_adv * self.estimator.clip_values[1]
            else:
                x_for_loss = x_adv

            # Get gradients from ART loss_gradient
            # This computes gradients for detection loss on all samples

            # Prepare kwargs for attack-specific loss
            # Extract pseudo-GT boxes as (batch_size, 4) tensor for attack loss
            pseudo_gt_boxes_batch = torch.stack([
                pseudo_gt['boxes'][0] if len(pseudo_gt['boxes']) > 0
                else torch.zeros(4, device=self.device)
                for pseudo_gt in pseudo_gts_valid
            ])  # (batch_size, 4)

            gradients = self.estimator.loss_gradient(
                x=x_for_loss,
                y=pseudo_gts_numpy_valid,
                pseudo_gt_boxes=pseudo_gt_boxes_batch,  # For attack loss
                target_class_idx=self.target_class_id   # For targeted attack
            )

            # Ensure gradients are torch tensors on the correct device
            if isinstance(gradients, np.ndarray):
                grad_torch = torch.from_numpy(gradients).to(self.device)
            else:
                grad_torch = gradients.to(self.device)

            # Ensure channel order matches internal representation
            if not self.estimator.channels_first and grad_torch.ndim == 4:
                grad_torch = grad_torch.permute(0, 3, 1, 2)

            # CRITICAL FIX: Sum gradients across ALL samples (like AEGIS)
            # Instead of averaging, sum to accumulate attack effect
            grad_sum = grad_torch.sum(dim=0, keepdim=True)

            # Normalize by number of samples for stability
            grad_avg = grad_sum / len(valid_indices)

            # CRITICAL FIX: Gradient DESCENT (not ascent) to minimize detection
            # ART's loss_gradient gives positive gradient direction
            # We want to INCREASE the loss (make detection fail)
            # So we use gradient ASCENT: perturbation -= learning_rate * (-gradient)
            # Which is: perturbation += learning_rate * gradient
            with torch.no_grad():
                # Direct SGD-style update (like AEGIS)
                # Negate gradient for gradient ascent (maximize detection loss)
                self._perturbation_torch.data += self.eps_step * grad_avg.squeeze(0)

                # Enforce epsilon constraint
                self._enforce_epsilon(self.eps)

            # Logging
            if self.verbose and (i_iter % 10 == 0 or i_iter == self.max_iter - 1):
                with torch.no_grad():
                    pert_norm = self._perturbation_torch.norm().item()
                    grad_norm = grad_avg.norm().item()
                    logger.info(
                        f"Iter {i_iter}/{self.max_iter}: "
                        f"Perturbation norm = {pert_norm:.6f}, "
                        f"Gradient norm = {grad_norm:.6f}, "
                        f"LR = {self.eps_step:.6f}"
                    )

            # Progress callback for SSE updates
            if self.progress_callback and (i_iter % 10 == 0 or i_iter == self.max_iter - 1):
                try:
                    perturbation_norm = self._perturbation_torch.norm().item()
                    self.progress_callback(i_iter, self.max_iter, perturbation_norm, self.eps_step)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

    def _apply_perturbation_with_mask(
        self,
        x: torch.Tensor,
        pseudo_gts: list
    ) -> torch.Tensor:
        """
        Apply perturbation only to detected object regions.

        Args:
            x: Input images (B, C, H, W)
            pseudo_gts: List of pseudo ground truth dicts

        Returns:
            Perturbed images (B, C, H, W)
        """
        x_adv = x.clone()

        for i in range(x.shape[0]):
            if 'boxes' not in pseudo_gts[i] or len(pseudo_gts[i]['boxes']) == 0:
                continue

            boxes = pseudo_gts[i]['boxes']
            from app.ai.losses.box_utils import xywh2xyxy
            boxes_xyxy = xywh2xyxy(boxes)

            # Create mask for this image
            mask = torch.zeros(1, x.shape[2], x.shape[3], device=self.device)

            for box in boxes_xyxy:
                x1, y1, x2, y2 = box[:4].int()

                # Ensure coordinates are within bounds
                h, w = x.shape[2], x.shape[3]
                x1 = max(0, min(x1.item(), w))
                y1 = max(0, min(y1.item(), h))
                x2 = max(0, min(x2.item(), w))
                y2 = max(0, min(y2.item(), h))

                if x1 < x2 and y1 < y2:
                    mask[:, y1:y2, x1:x2] = 1.0

            # Apply masked perturbation
            perturbation_masked = self._perturbation_torch[0] * mask
            x_adv[i] = torch.clamp(x[i] + perturbation_masked, self.clip_min, self.clip_max)

        return x_adv

    def _configure_attack_loss(self) -> None:
        """
        Configure the estimator to use attack-specific loss (DetectionAttackLoss).

        This method encapsulates the attack loss configuration within the attack class,
        following proper OOP principles instead of having the service layer mutate
        private fields.
        """
        from app.ai.losses import AttackLossRegistry

        # Configure estimator for universal noise attack
        self.estimator._use_attack_loss = True
        self.estimator._attack_type = 'universal_noise'
        self.estimator._attack_loss_config = {
            'iou_threshold': 0.1,           # Min IoU to consider a box
            'class_conf_threshold': 0.05,   # Min confidence for agnostic loss
            'top_k_boxes': 5,               # Use top-5 boxes
            'top_n_classes': 3,             # Sum top-3 class scores
            'lambda_targeted': 1.0,         # Weight for targeted loss
            'lambda_agnostic': 1.5          # Weight for agnostic loss
        }

        # Initialize the custom attack loss
        self.estimator._custom_attack_loss = AttackLossRegistry.get(
            self.estimator._attack_type,
            self.estimator._attack_loss_config
        )

        logger.info(
            f"Attack loss configured: {self.estimator._attack_type} "
            f"with config {self.estimator._attack_loss_config}"
        )

    def _check_params(self):
        """Check validity of parameters."""
        if not isinstance(self.eps, (int, float)) or self.eps <= 0:
            raise ValueError("eps must be positive")
        if not isinstance(self.eps_step, (int, float)) or self.eps_step <= 0:
            raise ValueError("eps_step must be positive")
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("max_iter must be non-negative integer")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be positive integer")

    def _initialize_perturbation(self, x: torch.Tensor) -> None:
        """
        Initialize universal perturbation tensor.

        Args:
            x: Input tensor to determine shape (B, C, H, W)
        """
        if self._perturbation_torch is None:
            self._perturbation_torch = nn.Parameter(
                torch.zeros(
                    1, x.shape[1], x.shape[2], x.shape[3],
                    dtype=torch.float32,
                    device=self.device
                )
            )
            logger.info(f"Initialized perturbation: {self._perturbation_torch.shape}")

    def _preprocess_and_convert(self, x: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess inputs using ART's standard preprocessing.

        Args:
            x: Input images (N, H, W, C) or (N, C, H, W) - NumPy

        Returns:
            Tuple of (preprocessed torch tensor, original numpy array)
        """
        x_torch, _ = self.estimator._preprocess_and_convert_inputs(
            x=x, y=None, fit=False, no_grad=True
        )
        # Move to the estimator device explicitly to avoid mixed CPU/GPU tensors later
        x_torch = x_torch.to(self.device)
        logger.debug(f"Input converted to torch: {x_torch.shape}, device: {x_torch.device}")
        return x_torch, x

    def _torch_to_numpy(self, x_torch: torch.Tensor, x_original: np.ndarray) -> np.ndarray:
        """
        Convert PyTorch tensor back to NumPy array matching original format.

        Args:
            x_torch: Torch tensor to convert
            x_original: Original NumPy array (for shape reference)

        Returns:
            NumPy array in the same format as x_original
        """
        x_np = x_torch.detach().cpu().numpy()

        # Denormalize if needed
        if self.estimator.clip_values is not None:
            min_val, max_val = self.estimator.clip_values
            if max_val > 1 and x_np.max() <= 1.0:
                x_np = x_np * max_val
        elif self.clip_max > 1 and x_np.max() <= 1.0:
            # Only rescale if tensors are still normalized
            x_np = x_np * self.clip_max

        # Handle channels_first/last to match original format
        if not self.estimator.channels_first and x_np.ndim == 4:
            # (N, C, H, W) -> (N, H, W, C)
            x_np = np.transpose(x_np, (0, 2, 3, 1))

        return x_np.astype(np.float32)

    def _get_clip_bounds(self) -> tuple[float, float]:
        """
        Determine clipping bounds for INTERNAL normalized [0, 1] scale.

        IMPORTANT: Internally, we work in [0, 1] normalized scale.
        We only scale to estimator.clip_values (e.g., [0, 255]) when
        passing to estimator.loss_gradient().
        """
        # Always use [0, 1] for internal operations
        # This avoids scale mixing issues
        return 0.0, 1.0

    @property
    def perturbation(self) -> np.ndarray | None:
        """Get the current universal perturbation (NumPy)."""
        return self._perturbation

    def set_perturbation(self, perturbation: np.ndarray) -> None:
        """
        Set a pre-trained universal perturbation.

        Args:
            perturbation: Perturbation to set (NumPy array)
        """
        self._perturbation = perturbation
        self._perturbation_torch = nn.Parameter(
            torch.from_numpy(perturbation).float().to(self.device)
        )

    def _enforce_epsilon(self, eps: float) -> None:
        """
        Project perturbation into L_inf ball of radius `eps` (same scale as internal tensors).
        """
        if self._perturbation_torch is None:
            return
        # Clamp around 0 since perturbation is additive
        self._perturbation_torch.data = torch.clamp(self._perturbation_torch.data, -eps, eps)

    def apply_perturbation(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        apply_mask: bool = True
    ) -> np.ndarray:
        """
        Apply pre-trained universal perturbation to new images with proper masking.

        This method is for inference after training. It applies the learned perturbation
        with the same masking logic used during training.

        Args:
            x: Input images (N, H, W, C) or (N, C, H, W) - NumPy
            y: Optional labels for masking. If None, generate pseudo-GT.
            apply_mask: Whether to apply masking. Default True.

        Returns:
            Adversarial images (NumPy array)
        """
        if self._perturbation_torch is None:
            raise ValueError("Perturbation not trained yet. Call generate() first or set_perturbation().")

        logger.info(f"Applying universal perturbation to {len(x)} images (apply_mask={apply_mask})")

        # Convert to torch
        x_torch, x_original = self._preprocess_and_convert(x)

        # Apply perturbation with or without masking
        if apply_mask:
            # Use provided labels or generate pseudo-GT for masking
            if y is not None:
                # Convert y to torch format
                pseudo_gts = []
                for label in y:
                    if isinstance(label, dict) and 'boxes' in label:
                        from app.ai.losses.box_utils import xyxy2xywh

                        boxes_xyxy = torch.from_numpy(label['boxes']).float().to(self.device)
                        pseudo_gt = {
                            'boxes': xyxy2xywh(boxes_xyxy),
                            'labels': torch.from_numpy(label['labels']).long().to(self.device)
                        }
                        pseudo_gts.append(pseudo_gt)
                    else:
                        # Empty pseudo-GT if no boxes
                        pseudo_gts.append({
                            'boxes': torch.zeros(0, 4, dtype=torch.float32, device=self.device),
                            'labels': torch.zeros(0, dtype=torch.int64, device=self.device)
                        })
                logger.info(f"Using provided labels for masking")
            else:
                # Generate pseudo-GT using ART predict API
                pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(x_torch)
                logger.info(f"Generated pseudo-GT for masking")

            x_adv_torch = self._apply_perturbation_with_mask(x_torch, pseudo_gts)
        else:
            with torch.no_grad():
                x_adv_torch = torch.clamp(x_torch + self._perturbation_torch, self.clip_min, self.clip_max)

        # Convert back to NumPy
        x_adv = self._torch_to_numpy(x_adv_torch, x_original)

        logger.info(f"Applied perturbation to {len(x)} images")
        return x_adv
