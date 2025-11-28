"""
Universal Noise Attack with full PyTorch implementation.

This version restores all functionality from the original AEGIS framework:
- Pseudo ground truth generation
- Differentiable loss computation
- Object masking

External interface remains NumPy-based for ART compatibility.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from app.ai.attacks.attack import EvasionAttack
from app.ai.estimators.estimator import BaseEstimator, LossGradientsMixin
from app.ai.estimators.object_detection.object_detector import ObjectDetectorMixin
from app.ai.summary_writer import SummaryWriter
from app.ai import config

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

                boxes = torch.from_numpy(pred['boxes']).float().to(self.device)
                scores = torch.from_numpy(pred['scores']).float().to(self.device)
                classes = torch.from_numpy(pred['labels']).long().to(self.device)

                # Filter by target class and confidence
                mask = (classes == self.target_class_id) & (scores >= self.confidence_threshold)
                filtered_boxes = boxes[mask]
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
                        'boxes': filtered_boxes[best_idx:best_idx+1],  # (1, 4)
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
        # Always clamp to detector scale; default to 0-255 if estimator does not define clip values
        self.clip_min, self.clip_max = self._get_clip_bounds()
        self._check_params()

        # Universal perturbation (PyTorch parameter)
        self._perturbation_torch: nn.Parameter | None = None
        self._perturbation: np.ndarray | None = None

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
                    pseudo_gt = {
                        'boxes': torch.from_numpy(label['boxes']).float().to(self.device),
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
        # Always return perturbation for service compatibility
        # (service code expects tuple to extract perturbation)
        return x_adv, self._perturbation

    def _train_universal_perturbation_pytorch(
        self,
        x: torch.Tensor,
        y: list | None = None
    ):
        """
        Train universal perturbation using PyTorch with ART loss.

        Uses PyTorch DataLoader for efficient batch processing (Type 1 pattern).

        Args:
            x: Input images (N, C, H, W)
            y: Optional labels. If provided, use as ground truth. If None, generate pseudo-GT.
        """
        logger.info(f"Training Universal Noise for {self.max_iter} iterations")

        # Use provided labels or generate pseudo ground truth
        if y is not None:
            logger.info(f"Using provided labels (y) for {len(y)} images")
            # Convert y to torch format if needed
            if isinstance(y, list) and len(y) > 0:
                if isinstance(y[0], dict):
                    # Already in dict format with 'boxes', 'labels'
                    pseudo_gts = []
                    for label in y:
                        pseudo_gt = {
                            'boxes': torch.from_numpy(label['boxes']).float().to(self.device),
                            'labels': torch.from_numpy(label['labels']).long().to(self.device)
                        }
                        pseudo_gts.append(pseudo_gt)
                    pseudo_gts_numpy = y  # Already in numpy format
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

        # Create custom dataset for labels (cannot use TensorDataset with list of dicts)
        class LabeledImageDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels_torch, labels_numpy):
                self.images = images
                self.labels_torch = labels_torch
                self.labels_numpy = labels_numpy

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return self.images[idx], self.labels_torch[idx], self.labels_numpy[idx]

        # Custom collate function for object detection that keeps labels as list of dicts
        def collate_fn_od(batch):
            images = torch.stack([item[0] for item in batch])
            labels_torch = [item[1] for item in batch]  # Keep as list of dicts
            labels_numpy = [item[2] for item in batch]  # Keep as list of dicts
            return images, labels_torch, labels_numpy

        # Create dataset and dataloader (following PGD PyTorch pattern)
        dataset = LabeledImageDataset(x, pseudo_gts, pseudo_gts_numpy)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle to maintain order
            drop_last=False,
            collate_fn=collate_fn_od  # Use custom collate for variable-size boxes
        )

        logger.info(f"Created DataLoader: {len(dataset)} samples, {len(data_loader)} batches")

        # Training loop with DataLoader (Type 1: PyTorch DataLoader pattern)
        for i_iter in trange(self.max_iter, desc="Universal Noise Training", disable=not self.verbose):
            for batch_id, (x_batch, y_batch_torch, y_batch_numpy) in enumerate(data_loader):
                # Move batch to device
                x_batch = x_batch.to(self.device)

                # Apply perturbation
                if self.apply_mask:
                    # For masking, need y_batch_torch with tensors on correct device
                    y_batch_torch_device = [{k: v.to(self.device) for k, v in y_i.items()} for y_i in y_batch_torch]
                    x_adv_batch = self._apply_perturbation_with_mask(x_batch, y_batch_torch_device)
                else:
                    x_adv_batch = torch.clamp(x_batch + self._perturbation_torch, self.clip_min, self.clip_max)

                # Filter out samples with empty boxes before calling loss_gradient
                valid_indices = [i for i, label_dict in enumerate(y_batch_numpy) if len(label_dict.get('boxes', [])) > 0]

                if len(valid_indices) == 0:
                    # All samples have no detections, skip this batch
                    logger.warning("All samples in batch have no detected objects, skipping gradient update")
                    continue

                # Filter x and y to only include valid samples
                x_adv_batch_valid = x_adv_batch[valid_indices]
                y_batch_numpy_valid = [y_batch_numpy[i] for i in valid_indices]

                # Keep everything on a single device (match other attacks like DPatch)
                x_adv_batch_valid = x_adv_batch_valid.detach()

                # Denormalize: internal [0, 1] -> estimator expects [0, clip_max]
                if self.estimator.clip_values is not None:
                    x_for_loss = x_adv_batch_valid * self.estimator.clip_values[1]
                else:
                    x_for_loss = x_adv_batch_valid

                gradients_valid = self.estimator.loss_gradient(x=x_for_loss, y=y_batch_numpy_valid)

                # Ensure gradients are torch tensors on the correct device
                if isinstance(gradients_valid, np.ndarray):
                    grad_torch_valid = torch.from_numpy(gradients_valid).to(self.device)
                else:
                    grad_torch_valid = gradients_valid.to(self.device)

                # Ensure channel order matches internal representation
                if not self.estimator.channels_first and grad_torch_valid.ndim == 4:
                    grad_torch_valid = grad_torch_valid.permute(0, 3, 1, 2)

                # Normalize gradients back to [0, 1] space (our internal optimization space)
                if self.estimator.clip_values is not None:
                    grad_torch_valid = grad_torch_valid / self.estimator.clip_values[1]

                # Gradient ascent: update perturbation to maximize loss
                # Average gradients across valid samples only
                grad_avg = grad_torch_valid.mean(dim=0, keepdim=True)

                with torch.no_grad():
                    # Sign gradient for robustness (like FGSM)
                    self._perturbation_torch.data += self.eps_step * torch.sign(grad_avg)

                    # Enforce epsilon constraint using base class
                    self._enforce_epsilon(self.eps)

                if i_iter == 0 and batch_id == 0:
                    logger.info(f"Using ART loss_gradient: grad shape={grad_torch_valid.shape}, norm={grad_torch_valid.norm():.4f}")

            if self.verbose and (i_iter % 10 == 0 or i_iter == self.max_iter - 1):
                logger.info(f"Iter {i_iter}/{self.max_iter}: Perturbation norm = {self._perturbation_torch.norm().item():.6f}")

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

            # Create mask for this image
            mask = torch.zeros(1, x.shape[2], x.shape[3], device=self.device)

            for box in boxes:
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
        Determine clipping bounds; fall back to detector-friendly [0, 255].
        """
        if self.estimator.clip_values is not None:
            return float(self.estimator.clip_values[0]), float(self.estimator.clip_values[1])
        return 0.0, 255.0

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
