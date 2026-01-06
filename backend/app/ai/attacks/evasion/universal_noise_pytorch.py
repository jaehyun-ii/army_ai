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
import torch.nn.functional as F
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
    if image1.shape[0] == 3:
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        gray1 = 0.299 * image1[0] + 0.587 * image1[1] + 0.114 * image1[2]
        gray2 = 0.299 * image2[0] + 0.587 * image2[1] + 0.114 * image2[2]
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
        "top_k_detections",
        "confidence_threshold",
        "use_model_loss",
        "ncc_threshold",
        "attack_mode",
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
        top_k_detections: int = -1,
        confidence_threshold: float = 0.0,
        use_model_loss: bool = False,
        ncc_threshold: float = 0.0,
        attack_mode: str = "universal",
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
        :param top_k_detections: Number of detections to keep per image when generating pseudo-GT:
                                -1 = all detections above threshold (multi-target, default)
                                 1 = single highest confidence detection (AEGIS-style)
                                 N = top N detections
        :param confidence_threshold: Minimum confidence for pseudo-GT detections.
        :param use_model_loss: If True, use model's training loss directly (like attack_detector).
                              If False, use custom DetectionAttackLoss (default).
        :param ncc_threshold: NCC threshold for distortion-aware stopping (0.0 = disabled).
                             If > 0, stop attacking objects when NCC < threshold.
                             Typical values: 0.6-0.9 (higher = more similar).
        :param attack_mode: Attack mode - "universal" (default) or "image_specific".
                           "universal": Learn single perturbation for all images.
                           "image_specific": Optimize perturbation per image (like attack_detector).
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
        self.top_k_detections = top_k_detections
        self.confidence_threshold = confidence_threshold
        self.use_model_loss = use_model_loss
        self.ncc_threshold = ncc_threshold
        self.attack_mode = attack_mode
        self.verbose = verbose
        self.progress_callback = progress_callback
        # Always clamp to normalized detector scale internally
        self.clip_min, self.clip_max = self._get_clip_bounds()

        # Validate attack mode
        if attack_mode not in ["universal", "image_specific"]:
            raise ValueError(f"attack_mode must be 'universal' or 'image_specific', got {attack_mode}")

        # Normalize eps/eps_step to [0, 1] internal scale if estimator expects [0, 255]
        self._input_scale = float(self.estimator.clip_values[1]) if self.estimator.clip_values is not None else 1.0
        if self._input_scale > 1.0:
            if self.eps > 1.0:
                self.eps = self.eps / self._input_scale
            if self.eps_step > 1.0:
                self.eps_step = self.eps_step / self._input_scale

        self._check_params()

        # Use provided scheduler configuration or defaults
        self.lr_scheduler_type = scheduler_type
        self.lr_scheduler_params = scheduler_params if scheduler_params is not None else {}

        # Universal perturbation (PyTorch parameter)
        self._perturbation_torch: nn.Parameter | None = None
        self._perturbation: np.ndarray | None = None

        # Optimizer and scheduler (like original AEGIS)
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

        # Configure attack-specific loss for the estimator (only if not using model loss)
        if not use_model_loss:
            self._configure_attack_loss()
            logger.info(f"Using custom DetectionAttackLoss")
        else:
            logger.info(f"Using model's training loss directly (attack_detector style)")

        logger.info(f"UniversalNoiseAttackPyTorch initialized on device: {self.device}")
        logger.info(f"Attack mode: {attack_mode}")
        if ncc_threshold > 0:
            logger.info(f"NCC threshold enabled: {ncc_threshold}")

        # Initialize pseudo-GT generator (now ART-compliant with multi-target support)
        self._pseudo_gt_gen = _PseudoGTGenerator(
            estimator=self.estimator,  # Pass estimator for ART predict API
            target_class_id=target_class_id,
            confidence_threshold=confidence_threshold,
            top_k_detections=top_k_detections,
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
        Generate adversarial perturbation.

        Behavior depends on attack_mode:
        - "universal": Learn single perturbation for all images (default)
        - "image_specific": Optimize perturbation per image (attack_detector style)

        Args:
            x: Sample images (N, H, W, C) or (N, C, H, W) - NumPy
            y: Optional labels (will use pseudo-GT if None)
            return_perturbation: If True, also return the learned perturbation.

        Returns:
            Adversarial images. If `return_perturbation=True`, also returns the learned perturbation.
        """
        if self.attack_mode == "universal":
            return self._generate_universal(x, y, return_perturbation=return_perturbation, **kwargs)
        else:  # image_specific
            return self._generate_image_specific(x, y, return_perturbation=return_perturbation, **kwargs)

    def _generate_universal(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        *,
        return_perturbation: bool = False,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Generate universal adversarial perturbation (original implementation).
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
                    boxes_xyxy = torch.from_numpy(label.get('boxes', np.zeros((0, 4), dtype=np.float32))).float().to(self.device)
                    boxes_xywh = self._boxes_to_xywh_pixels(
                        boxes_xyxy,
                        (x_torch.shape[2], x_torch.shape[3])
                    )
                    pseudo_gt = {
                        'boxes': boxes_xywh,
                        'labels': torch.from_numpy(label.get('labels', np.array([], dtype=np.int64))).long().to(self.device)
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
        Train universal perturbation using PyTorch optimizer (like original AEGIS).

        Uses mini-batch processing with AdamW optimizer and CosineAnnealingLR scheduler.

        Args:
            x: Input images (N, C, H, W)
            y: Optional labels. If provided, use as ground truth. If None, generate pseudo-GT.
        """
        logger.info(f"Training Universal Noise for {self.max_iter} iterations on {len(x)} images (batch_size={self.batch_size})")

        # Use provided labels or generate pseudo ground truth
        if y is not None:
            logger.info(f"Using provided labels (y) for {len(y)} images")
            # Convert y to torch format for mask/loss (xywh center, pixel)
            if isinstance(y, list) and len(y) > 0:
                if isinstance(y[0], dict):
                    pseudo_gts = []
                    img_h, img_w = x.shape[2], x.shape[3]
                    for label in y:
                        boxes_np = label.get('boxes', np.zeros((0, 4), dtype=np.float32))
                        boxes_t = torch.from_numpy(boxes_np).float().to(self.device)
                        boxes_xywh = self._boxes_to_xywh_pixels(boxes_t, (img_h, img_w))
                        pseudo_gt = {
                            'boxes': boxes_xywh,
                            'labels': torch.from_numpy(label.get('labels', np.array([], dtype=np.int64))).long().to(self.device)
                        }
                        pseudo_gts.append(pseudo_gt)
                    y_for_loss = self._pseudo_gts_to_numpy_xyxy(pseudo_gts)
                else:
                    raise ValueError(f"Unsupported y format: {type(y[0])}")
            else:
                # Generate pseudo-GT if y is empty
                logger.info("Provided y is empty, generating pseudo-GT")
                pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(x)
                y_for_loss = self._pseudo_gts_to_numpy_xyxy(pseudo_gts)
        else:
            # Generate pseudo ground truth using ART predict API
            logger.info("No labels provided, generating pseudo-GT")
            pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(x)
            y_for_loss = self._pseudo_gts_to_numpy_xyxy(pseudo_gts)
            logger.info(f"Generated pseudo-GT for {len(pseudo_gts)} images")

        # Filter out samples with empty boxes once at the beginning
        valid_indices = [i for i, gt in enumerate(pseudo_gts) if len(gt.get('boxes', [])) > 0]

        if len(valid_indices) == 0:
            logger.error("No valid detections found in any image. Cannot train universal perturbation.")
            return

        # Filter all data to only include valid samples
        x_valid = x[valid_indices]
        pseudo_gts_valid = [pseudo_gts[i] for i in valid_indices]
        y_for_loss_valid = [y_for_loss[i] for i in valid_indices]

        logger.info(f"Training on {len(valid_indices)}/{len(x)} images with detections")

        # DEBUG: Verify pseudo_gts have valid boxes
        if len(pseudo_gts_valid) > 0:
            first_gt = pseudo_gts_valid[0]
            logger.debug(f"First pseudo_gt: boxes shape={first_gt['boxes'].shape if 'boxes' in first_gt else 'N/A'}, boxes={first_gt.get('boxes', 'N/A')}")
            logger.debug(f"First pseudo_gt: labels={first_gt.get('labels', 'N/A')}")

        # Create DataLoader for mini-batch processing (like original AEGIS)
        from torch.utils.data import Dataset, DataLoader

        # Custom dataset that includes images and their indices
        class IndexedDataset(Dataset):
            def __init__(self, images):
                self.images = images

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return self.images[idx], idx

        dataset = IndexedDataset(x_valid)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle to maintain pseudo-GT alignment
            drop_last=False
        )

        # Training loop - AEGIS style: epoch-based with mini-batches
        global_step = 0
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            num_batches = 0

            # Progress bar for this epoch
            pbar = trange(len(dataloader), desc=f"Epoch {epoch+1}/{self.max_iter}", disable=not self.verbose)

            for batch_idx, (x_batch, indices) in enumerate(dataloader):
                # Get corresponding pseudo-GT for this batch using actual indices
                indices_list = indices.tolist()
                pseudo_gts_batch = [pseudo_gts_valid[i] for i in indices_list]
                y_for_loss_batch = [y_for_loss_valid[i] for i in indices_list]

                # Zero gradients (like AEGIS)
                self._optimizer.zero_grad()

                # Apply perturbation to batch
                if self.apply_mask:
                    x_adv = self._apply_perturbation_with_mask(x_batch, pseudo_gts_batch)
                else:
                    x_adv = torch.clamp(x_batch + self._perturbation_torch, self.clip_min, self.clip_max)

                # Denormalize for loss computation
                if self.estimator.clip_values is not None:
                    x_for_loss = x_adv * self.estimator.clip_values[1]
                else:
                    x_for_loss = x_adv

                # Prepare pseudo-GT boxes for attack loss (keep all boxes per image)
                max_boxes = max(len(pseudo_gt['boxes']) if 'boxes' in pseudo_gt else 0 for pseudo_gt in pseudo_gts_batch)
                if max_boxes == 0:
                    pseudo_gt_boxes_batch = torch.zeros(len(pseudo_gts_batch), 1, 4, device=self.device)
                else:
                    pseudo_gt_boxes_batch = torch.zeros(len(pseudo_gts_batch), max_boxes, 4, device=self.device)
                    for idx_img, pseudo_gt in enumerate(pseudo_gts_batch):
                        if 'boxes' in pseudo_gt and len(pseudo_gt['boxes']) > 0:
                            num = len(pseudo_gt['boxes'])
                            pseudo_gt_boxes_batch[idx_img, :num, :] = pseudo_gt['boxes'][:num]

                # Compute loss (now positive, like AEGIS)
                with torch.set_grad_enabled(True):
                    loss = self.estimator.compute_loss(
                        x=x_for_loss,
                        y=y_for_loss_batch,
                        pseudo_gt_boxes=pseudo_gt_boxes_batch,
                        target_class_idx=self.target_class_id
                    )

                # Get gradients using ART API
                # CRITICAL: PyTorchObjectDetector.loss_gradient() returns gradients in [0, 1] scale!
                # (See pytorch_object_detector.py:511-512: grads = grads / clip_values[1])
                gradients = self.estimator.loss_gradient(
                    x=x_for_loss,  # [0, 255] NumPy input
                    y=y_for_loss_batch,
                    pseudo_gt_boxes=pseudo_gt_boxes_batch,
                    target_class_idx=self.target_class_id
                )
                # gradients: [0, 1] scale (already normalized by estimator)

                # Convert to torch and assign to parameter
                if isinstance(gradients, np.ndarray):
                    grad_torch = torch.from_numpy(gradients).to(self.device)
                else:
                    grad_torch = gradients.to(self.device)

                if not self.estimator.channels_first and grad_torch.ndim == 4:
                    grad_torch = grad_torch.permute(0, 3, 1, 2)

                # CRITICAL: Gradient scale verification
                # - estimator.loss_gradient() returns [0, 1] scale gradients
                # - perturbation is in [0, 1] scale
                # - optimizer LR = eps_step * 255 (e.g., 0.001 * 255 = 0.255)
                # This combination is CORRECT:
                #   perturbation -= LR * grad = 0.255 * grad_[0,1]
                # NO additional normalization needed!

                # CRITICAL FIX: Sum gradients across batch (NOT average!)
                # Original AEGIS sums loss across batch, so gradients should also be summed
                # Averaging would make gradients 1/batch_size smaller, killing learning
                grad_sum = grad_torch.sum(dim=0, keepdim=True)

                # Assign gradient to perturbation parameter
                # grad_sum shape: (1, C, H, W) matches parameter shape
                # grad_sum scale: [0, 1] (matches perturbation scale)
                self._perturbation_torch.grad = grad_sum

                # Optimizer step (like AEGIS using AdamW)
                self._optimizer.step()

                # Post-step update: enforce epsilon constraint
                with torch.no_grad():
                    self._enforce_epsilon(self.eps)

                # Accumulate loss
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Update progress bar
                pbar.update(1)
                if batch_idx % 10 == 0:
                    current_lr = self._scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.6f}'
                    })

            pbar.close()

            # Learning rate scheduler step (like AEGIS)
            self._scheduler.step()

            # Epoch logging
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            current_lr = self._scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1}/{self.max_iter}: "
                f"Avg Loss = {avg_epoch_loss:.6f}, "
                f"LR = {current_lr:.6f}, "
                f"Pert norm = {self._perturbation_torch.norm().item():.6f}"
            )

            # Progress callback for SSE updates
            if self.progress_callback:
                try:
                    perturbation_norm = self._perturbation_torch.norm().item()
                    self.progress_callback(epoch, self.max_iter, avg_epoch_loss, current_lr)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

    def _boxes_to_xywh_pixels(
        self,
        boxes: torch.Tensor,
        image_shape: tuple[int, int]
    ) -> torch.Tensor:
        """
        Normalize input boxes to xywh (center) in pixel coordinates.

        Assumes normalized YOLO labels are in cxcywh format; otherwise treats boxes as xyxy pixels.
        """
        if boxes.numel() == 0:
            return boxes

        img_h, img_w = image_shape
        max_val = boxes.abs().max().item()
        if max_val <= 1.5:
            # Normalized YOLO cxcywh -> scale to pixels
            cx = boxes[:, 0] * img_w
            cy = boxes[:, 1] * img_h
            w = boxes[:, 2] * img_w
            h = boxes[:, 3] * img_h
            return torch.stack([cx, cy, w, h], dim=1)

        from app.ai.losses.box_utils import xyxy2xywh
        return xyxy2xywh(boxes)

    def _pseudo_gts_to_numpy_xyxy(
        self,
        pseudo_gts: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, np.ndarray]]:
        """Convert pseudo-GT (xywh, torch) to numpy labels in xyxy format for ART."""
        from app.ai.losses.box_utils import xywh2xyxy

        labels_numpy = []
        for gt in pseudo_gts:
            boxes = gt.get('boxes', torch.zeros(0, 4, device=self.device))
            labels = gt.get('labels', torch.zeros(0, dtype=torch.int64, device=self.device))
            if boxes.numel() > 0:
                boxes_xyxy = xywh2xyxy(boxes)
            else:
                boxes_xyxy = boxes
            labels_numpy.append({
                "boxes": boxes_xyxy.detach().cpu().numpy().astype(np.float32),
                "labels": labels.detach().cpu().numpy().astype(np.int64),
            })
        return labels_numpy

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
        total_mask_coverage = 0.0  # Track how much of the image is masked

        for i in range(x.shape[0]):
            if 'boxes' not in pseudo_gts[i] or len(pseudo_gts[i]['boxes']) == 0:
                logger.debug(f"Image {i}: No boxes in pseudo_gt")
                continue

            boxes = pseudo_gts[i]['boxes']
            from app.ai.losses.box_utils import xywh2xyxy
            boxes_xyxy = xywh2xyxy(boxes)

            # DEBUG: Log box values to understand coordinate system (only once)
            if i == 0 and not hasattr(self, '_logged_boxes'):  # Log first image only ONCE
                logger.debug(f"Image {i}: boxes_xywh = {boxes}")
                logger.debug(f"Image {i}: boxes_xyxy = {boxes_xyxy}")
                logger.debug(f"Image {i}: Image shape (H, W) = ({x.shape[2]}, {x.shape[3]})")
                self._logged_boxes = True

            # Create mask for this image
            mask = torch.zeros(1, x.shape[2], x.shape[3], device=self.device)

            for box_idx, box in enumerate(boxes_xyxy):
                # Boxes from estimator.predict() are in absolute pixel units (ART returns xyxy in pixels)
                # Only scale if they look normalized (<=1); otherwise, use as-is.
                h, w = x.shape[2], x.shape[3]
                box_vals = box[:4]
                # Heuristic: treat as normalized if coords are very small (<=4)
                if box_vals.abs().max() <= 4.0:
                    x1 = int(box[0].item() * w)
                    y1 = int(box[1].item() * h)
                    x2 = int(box[2].item() * w)
                    y2 = int(box[3].item() * h)
                else:
                    x1 = int(box[0].item())
                    y1 = int(box[1].item())
                    x2 = int(box[2].item())
                    y2 = int(box[3].item())

                # DEBUG: Log first box coordinates (only once)
                if i == 0 and box_idx == 0 and not hasattr(self, '_logged_coords'):
                    logger.debug(f"Image {i}, Box {box_idx}: Normalized - x1={box[0].item():.4f}, y1={box[1].item():.4f}, x2={box[2].item():.4f}, y2={box[3].item():.4f}")
                    logger.debug(f"Image {i}, Box {box_idx}: Before clamping - x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    self._logged_coords_before = True

                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if i == 0 and box_idx == 0 and not hasattr(self, '_logged_coords'):
                    logger.debug(f"Image {i}, Box {box_idx}: After clamping - x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    logger.debug(f"Image {i}, Box {box_idx}: Valid box? {x1 < x2 and y1 < y2}")
                    self._logged_coords = True

                if x1 < x2 and y1 < y2:
                    mask[:, y1:y2, x1:x2] = 1.0

            # Apply masked perturbation
            perturbation_masked = self._perturbation_torch[0] * mask
            x_adv[i] = torch.clamp(x[i] + perturbation_masked, self.clip_min, self.clip_max)

            # Track mask coverage
            mask_ratio = mask.sum().item() / mask.numel()
            total_mask_coverage += mask_ratio

        # Log mask statistics once per batch (reduce spam - only log occasionally)
        avg_mask_coverage = total_mask_coverage / x.shape[0] if x.shape[0] > 0 else 0.0
        if not hasattr(self, '_mask_log_count'):
            self._mask_log_count = 0
        if self._mask_log_count < 3:  # Only log first 3 times
            logger.debug(f"Mask coverage: {avg_mask_coverage:.2%} of image area on average")
            self._mask_log_count += 1

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
        num_classes = None
        has_objectness = None
        model_wrapper = getattr(self.estimator, "model", None)
        if model_wrapper is not None and hasattr(model_wrapper, "num_classes"):
            num_classes = int(model_wrapper.num_classes)
        elif model_wrapper is not None and hasattr(model_wrapper, "model") and hasattr(model_wrapper.model, "nc"):
            num_classes = int(model_wrapper.model.nc)
        if model_wrapper is not None and hasattr(model_wrapper, "has_objectness"):
            has_objectness = bool(model_wrapper.has_objectness)

        self.estimator._attack_loss_config = {
            'iou_threshold': 0.1,           # Min IoU to consider a box
            'class_conf_threshold': 0.05,   # Min confidence for agnostic loss
            'top_k_boxes': -1,              # Use all boxes passing IoU threshold
            'top_n_classes': 3,             # Sum top-3 class scores
            'lambda_targeted': 1.0,         # Weight for targeted loss
            'lambda_agnostic': 2.0,         # Weight for agnostic loss (AEGIS default)
            'num_classes': num_classes,
        }
        if has_objectness is not None:
            self.estimator._attack_loss_config['has_objectness'] = has_objectness

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
        Initialize universal perturbation tensor and optimizer.

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

            # Initialize AdamW optimizer (like original AEGIS)
            # CRITICAL: loss_gradient returns grads w.r.t. 0-255 inputs, while the perturbation lives in 0-1 space.
            # Scale LR by clip_values[1] to apply updates in the normalized scale.
            effective_lr = self.eps_step
            if self.estimator.clip_values is not None:
                effective_lr = self.eps_step * self.estimator.clip_values[1]  # 0.01 * 255 = 2.55

            self._optimizer = torch.optim.AdamW(
                [self._perturbation_torch],
                lr=effective_lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            scale_factor = self.estimator.clip_values[1] if self.estimator.clip_values is not None else 1.0
            logger.info(f"Initialized AdamW optimizer with LR: {effective_lr} (eps_step={self.eps_step} * scale={scale_factor})")

            # Initialize CosineAnnealingLR scheduler (like original AEGIS)
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self.max_iter,
                eta_min=1e-5
            )
            logger.info(f"Initialized CosineAnnealingLR scheduler with T_max: {self.max_iter}")

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
            perturbation: Perturbation to set (NumPy array in any format)
        """
        # Rescale if perturbation stored in pixel scale
        pert = perturbation
        abs_max = float(np.nanmax(np.abs(pert))) if pert.size else 0.0
        if self._input_scale > 1.0 and abs_max > 1.0:
            pert = pert / self._input_scale
        if np.isnan(pert).any():
            pert = np.nan_to_num(pert, nan=0.0, posinf=0.0, neginf=0.0)
        self._perturbation = pert

        # Convert to torch tensor
        pert_torch = torch.from_numpy(pert).float()

        # CRITICAL FIX: If perturbation is in channels-last format (H, W, C),
        # convert to channels-first (C, H, W) for internal use
        if pert_torch.ndim == 4 and not self.estimator.channels_first:
            # Input is (1, H, W, C), need (1, C, H, W)
            pert_torch = pert_torch.permute(0, 3, 1, 2)
            logger.debug(f"Converted perturbation from channels-last to channels-first: {pert_torch.shape}")

        self._perturbation_torch = nn.Parameter(pert_torch.to(self.device))
        logger.info(f"Set perturbation with shape: {self._perturbation_torch.shape}")

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
        logger.debug(f"Perturbation shape: {self._perturbation_torch.shape}, Input shape: {x.shape}")

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
                        boxes_np = label.get('boxes', np.zeros((0, 4), dtype=np.float32))
                        boxes_t = torch.from_numpy(boxes_np).float().to(self.device)
                        boxes_xywh = self._boxes_to_xywh_pixels(
                            boxes_t,
                            (x_torch.shape[2], x_torch.shape[3])
                        )

                        pseudo_gt = {
                            'boxes': boxes_xywh,
                            'labels': torch.from_numpy(label.get('labels', np.array([], dtype=np.int64))).long().to(self.device)
                        }
                        pseudo_gts.append(pseudo_gt)
                    else:
                        # Empty pseudo-GT if no boxes
                        pseudo_gts.append({
                            'boxes': torch.zeros(0, 4, dtype=torch.float32, device=self.device),
                            'labels': torch.zeros(0, dtype=torch.int64, device=self.device)
                        })
                logger.info(f"Using provided labels for masking")
                x_adv_torch = self._apply_perturbation_with_mask(x_torch, pseudo_gts)
            else:
                # Generate pseudo-GT using ART predict API
                pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(x_torch)

                # Count how many images have valid detections
                valid_count = sum(1 for gt in pseudo_gts if 'boxes' in gt and len(gt['boxes']) > 0)
                logger.info(f"Generated pseudo-GT for masking: {valid_count}/{len(pseudo_gts)} images have detections")

                if valid_count == 0:
                    logger.warning("No detections found in any image - perturbations will not be applied!")
                    logger.warning("Falling back to unmasked perturbation application")
                    # Apply perturbation without masking as fallback
                    with torch.no_grad():
                        x_adv_torch = torch.clamp(x_torch + self._perturbation_torch, self.clip_min, self.clip_max)
                else:
                    x_adv_torch = self._apply_perturbation_with_mask(x_torch, pseudo_gts)
        else:
            with torch.no_grad():
                x_adv_torch = torch.clamp(x_torch + self._perturbation_torch, self.clip_min, self.clip_max)

        # Convert back to NumPy
        x_adv = self._torch_to_numpy(x_adv_torch, x_original)

        # Verify perturbation was actually applied
        diff = np.abs(x_adv - x).mean()
        logger.info(f"Applied perturbation to {len(x)} images (mean diff: {diff:.6f})")

        if diff < 0.01:
            logger.warning(f"Perturbation appears to have minimal effect (mean diff={diff:.6f})")

        return x_adv

    def _generate_image_specific(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        *,
        return_perturbation: bool = False,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Generate image-specific adversarial perturbations (attack_detector style).

        Each image gets its own optimized perturbation.

        Args:
            x: Sample images (N, H, W, C) or (N, C, H, W) - NumPy
            y: Optional labels (will use pseudo-GT if None)
            return_perturbation: If True, also return the learned perturbations.

        Returns:
            Adversarial images. If `return_perturbation=True`, also returns perturbations.
        """
        logger.info(f"Starting Image-Specific attack on {len(x)} images (attack_detector style)")

        # Preprocess
        x_torch, x_original = self._preprocess_and_convert(x)
        batch_size = x_torch.shape[0]

        # Prepare outputs
        x_adv_list = []
        perturbations_list = []

        # Convert labels if provided
        if y is not None:
            pseudo_gts = []
            for label in y:
                boxes_xyxy = torch.from_numpy(label.get('boxes', np.zeros((0, 4), dtype=np.float32))).float().to(self.device)
                boxes_xywh = self._boxes_to_xywh_pixels(
                    boxes_xyxy,
                    (x_torch.shape[2], x_torch.shape[3])
                )
                pseudo_gt = {
                    'boxes': boxes_xywh,
                    'labels': torch.from_numpy(label.get('labels', np.array([], dtype=np.int64))).long().to(self.device)
                }
                pseudo_gts.append(pseudo_gt)
            logger.info(f"Using provided labels for {len(pseudo_gts)} images")
        else:
            pseudo_gts = None

        # Attack each image independently
        pbar = trange(batch_size, desc="Attacking images", disable=not self.verbose)
        for img_idx in pbar:
            img_torch = x_torch[img_idx:img_idx + 1]  # (1, C, H, W)

            # Get pseudo-GT for this image
            if pseudo_gts is not None:
                img_pseudo_gt = pseudo_gts[img_idx]
            else:
                # Generate pseudo-GT using estimator
                img_pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(img_torch)
                img_pseudo_gt = img_pseudo_gts[0] if img_pseudo_gts else None

            # Attack this image
            img_adv, img_pert = self._attack_single_image(
                img_torch, img_pseudo_gt, img_idx, pbar
            )

            x_adv_list.append(img_adv)
            perturbations_list.append(img_pert)

        # Stack results
        x_adv_torch = torch.cat(x_adv_list, dim=0)
        perturbations_torch = torch.cat(perturbations_list, dim=0)

        # Convert back to NumPy
        x_adv = self._torch_to_numpy(x_adv_torch, x_original)
        perturbations = self._torch_to_numpy(perturbations_torch, x_original)

        logger.info("Image-Specific attack completed")

        if return_perturbation:
            return x_adv, perturbations
        else:
            return x_adv

    def _attack_single_image(
        self,
        img_torch: torch.Tensor,
        pseudo_gt: dict | None,
        img_idx: int,
        pbar: trange,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Attack a single image with iterative gradient ascent (attack_detector style).

        Args:
            img_torch: Single image tensor (1, C, H, W)
            pseudo_gt: Pseudo ground truth for this image (or None)
            img_idx: Image index for logging
            pbar: Progress bar for updates

        Returns:
            Tuple of (adversarial image, perturbation)
        """
        # Initialize perturbation for this image
        perturbation = torch.zeros_like(img_torch, requires_grad=False)
        img_original = img_torch.clone()

        # Check if image has detections
        if pseudo_gt is None or 'boxes' not in pseudo_gt or len(pseudo_gt['boxes']) == 0:
            logger.debug(f"Image {img_idx}: No detections, skipping")
            pbar.set_postfix({"img": img_idx, "status": "no_det"})
            return img_torch, perturbation

        # Get initial detections count
        initial_det_count = len(pseudo_gt['boxes'])
        pbar.set_postfix({"img": img_idx, "dets": initial_det_count})

        # Step size (convert from normalized to pixel scale like attack_detector)
        # attack_detector uses step=100 in pixel scale [0, 255]
        # We use eps_step in normalized [0, 1] scale
        step_size = self.eps_step

        # Iterative attack loop (like attack_detector's 5000 iterations)
        for iter_num in range(self.max_iter):
            # Create image with current perturbation
            img_adv = img_torch + perturbation
            img_adv_clamped = torch.clamp(img_adv, self.clip_min, self.clip_max)

            # Get detections for current adversarial image
            current_pseudo_gts = self._pseudo_gt_gen.generate_from_estimator(img_adv_clamped)
            current_pseudo_gt = current_pseudo_gts[0] if current_pseudo_gts else None

            if current_pseudo_gt is None or 'boxes' not in current_pseudo_gt or len(current_pseudo_gt['boxes']) == 0:
                # No more detections - attack successful!
                logger.debug(f"Image {img_idx}: All detections suppressed at iteration {iter_num}")
                pbar.set_postfix({"img": img_idx, "iter": iter_num, "status": "success"})
                break

            current_det_count = len(current_pseudo_gt['boxes'])

            # Compute gradient using model loss
            # CRITICAL: Create new tensor for gradient computation (breaks previous computation graph)
            img_adv_grad = img_adv_clamped.clone().detach().requires_grad_(True)

            if self.use_model_loss:
                # Use model's training loss directly (attack_detector style)
                # CRITICAL FIX: Use current_pseudo_gt instead of initial pseudo_gt!
                # The detection results change as the image is perturbed, so we must
                # compute gradients based on current detections, not initial ones.
                gradient = self._compute_gradient_with_model_loss(img_adv_grad, current_pseudo_gt)
            else:
                # Use custom attack loss
                gradient = self._compute_gradient_with_attack_loss(img_adv_grad, [current_pseudo_gt])

            if gradient is None:
                logger.debug(f"Image {img_idx}: Gradient is None at iteration {iter_num}")
                break

            # Create mask with NCC checking (if enabled)
            if self.apply_mask and self.ncc_threshold > 0:
                mask, objects_within_threshold = self._create_mask_with_ncc_check(
                    img_adv_clamped, img_original, current_pseudo_gt
                )

                if objects_within_threshold == 0:
                    # All objects exceed distortion threshold - stop attacking
                    logger.debug(f"Image {img_idx}: All objects exceed NCC threshold at iteration {iter_num}")
                    pbar.set_postfix({"img": img_idx, "iter": iter_num, "status": "ncc_stop"})
                    break
            elif self.apply_mask:
                # Mask without NCC checking
                mask = self._create_simple_mask(current_pseudo_gt, img_torch.shape)
            else:
                # No masking
                mask = torch.ones_like(gradient)

            # Update perturbation (gradient ascent with mask)
            perturbation = perturbation + step_size * gradient * mask

            # Enforce epsilon constraint
            perturbation = torch.clamp(perturbation, -self.eps, self.eps)

            # Update progress bar
            if iter_num % 10 == 0 or current_det_count < initial_det_count:
                pbar.set_postfix({
                    "img": img_idx,
                    "iter": iter_num,
                    "dets": f"{current_det_count}/{initial_det_count}"
                })

        # Final adversarial image
        img_adv_final = torch.clamp(img_torch + perturbation, self.clip_min, self.clip_max)

        return img_adv_final, perturbation

    def _compute_gradient_with_model_loss(
        self,
        img_adv: torch.Tensor,
        pseudo_gt: dict,
    ) -> torch.Tensor | None:
        """
        Compute gradient using model's training loss (attack_detector style).

        This matches attack_detector's approach of using the YOLO training loss
        (L_box + L_cls + L_dfl) to compute gradients.

        Args:
            img_adv: Adversarial image with requires_grad=True (1, C, H, W)
            pseudo_gt: Pseudo ground truth dict with 'boxes' (xywh, pixel) and 'labels'

        Returns:
            Gradient tensor (1, C, H, W) or None if computation fails
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
            h, w = img_adv.shape[2:]
            boxes_norm = boxes_pixel.clone()
            boxes_norm[:, 0] = boxes_pixel[:, 0] / w  # cx
            boxes_norm[:, 1] = boxes_pixel[:, 1] / h  # cy
            boxes_norm[:, 2] = boxes_pixel[:, 2] / w  # width
            boxes_norm[:, 3] = boxes_pixel[:, 3] / h  # height

            # Create targets tensor: [batch_idx, class_id, cx, cy, w, h]
            batch_idx = torch.zeros(len(boxes_norm), 1, device=self.device)
            targets = torch.cat([
                batch_idx,
                labels.unsqueeze(1).float(),
                boxes_norm
            ], dim=1)  # (N, 6)

            # CRITICAL: Scale image to [0, 255] range if model expects it
            # The model wrapper expects denormalized input
            # IMPORTANT: We need to retain gradient on the input tensor
            # since we'll be computing gradient w.r.t. the [0,1] scale input
            img_adv.retain_grad()  # Ensure gradient is retained on intermediate tensor
            img_adv_scaled = img_adv * self._input_scale

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

            # Compute gradients via backpropagation
            # This gives us ∂L/∂image (how to change image to maximize loss)
            loss.backward()

            # Extract gradient from adversarial image
            gradient = img_adv.grad
            if gradient is None:
                logger.warning("Gradient is None after backward pass")
                logger.warning("This may happen if retain_grad() was not called or computation graph was broken")
                return None

            # CRITICAL FIX: Scale gradient back to [0, 1] range
            # The gradient was computed w.r.t. img_adv_scaled (which is in [0, 255] scale)
            # Using chain rule: ∂L/∂img_adv = ∂L/∂img_adv_scaled * ∂img_adv_scaled/∂img_adv
            #                                = ∂L/∂img_adv_scaled * self._input_scale
            # But we want gradient in [0, 1] scale to match perturbation scale, so divide by input_scale
            # This matches attack_detector where step=100 is in [0,255] scale, but we use step=0.392 in [0,1] scale
            gradient_normalized = gradient.detach() / self._input_scale

            return gradient_normalized

        except Exception as e:
            logger.error(f"Error computing gradient with model loss: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return None

    def _compute_gradient_with_attack_loss(
        self,
        img_adv: torch.Tensor,
        pseudo_gts: list[dict],
    ) -> torch.Tensor | None:
        """
        Compute gradient using custom attack loss.

        Args:
            img_adv: Adversarial image with requires_grad=True (1, C, H, W)
            pseudo_gts: List of pseudo ground truths

        Returns:
            Gradient tensor or None
        """
        try:
            # Prepare for loss computation
            pseudo_gt_boxes_list = []
            for gt in pseudo_gts:
                if 'boxes' in gt and len(gt['boxes']) > 0:
                    pseudo_gt_boxes_list.append(gt['boxes'])
                else:
                    pseudo_gt_boxes_list.append(torch.zeros(0, 4, device=self.device))

            if len(pseudo_gt_boxes_list) == 0 or all(len(b) == 0 for b in pseudo_gt_boxes_list):
                return None

            # Use estimator's loss_gradient method
            # Scale to estimator's expected range
            img_adv_scaled = img_adv * self._input_scale

            # Prepare pseudo_gt_boxes as batch
            max_boxes = max(len(b) for b in pseudo_gt_boxes_list)
            pseudo_gt_boxes_batch = torch.zeros(
                len(pseudo_gt_boxes_list), max_boxes, 4,
                device=self.device
            )
            for i, boxes in enumerate(pseudo_gt_boxes_list):
                if len(boxes) > 0:
                    pseudo_gt_boxes_batch[i, :len(boxes)] = boxes

            gradients = self.estimator.loss_gradient(
                x=img_adv_scaled,
                y=None,
                pseudo_gt_boxes=pseudo_gt_boxes_batch,
                target_class_idx=self.target_class_id
            )

            if gradients is None:
                return None

            # Convert to torch and normalize
            gradient = torch.from_numpy(gradients).float().to(self.device)
            gradient = gradient / self._input_scale

            return gradient

        except Exception as e:
            logger.warning(f"Error computing gradient with attack loss: {e}")
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

            # Only add to mask if NCC is above threshold
            # (higher NCC = more similar = less distorted)
            if ncc >= self.ncc_threshold:
                mask[:, :, y1:y2, x1:x2] = 1.0
                objects_within_threshold += 1

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
