"""
Noise OSFD Attack with full PyTorch implementation.

This version restores all functionality from the original AEGIS framework:
- Differentiable feature extraction
- RRB augmentation
- End-to-end PyTorch optimization

External interface remains NumPy-based for ART compatibility.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
from torchvision.transforms import functional as TF
import random

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

class _FeatureExtractor(nn.Module):
    """
    Differentiable feature extractor using PyTorch hooks.
    Based on AEGIS framework implementation.

    This allows extracting intermediate features while maintaining gradient flow,
    which is essential for OSFD attack.

    NOTE: This custom implementation is necessary because:
    1. OSFD requires gradients to flow through features for backpropagation
    2. ART's get_activations() is designed for inference (no_grad context)
    3. The attack needs to optimize perturbations based on feature-space MSE loss
    4. Standard ART APIs don't support differentiable feature extraction with gradients

    This is one of the few justified cases for bypassing ART, as the attack's
    core mechanism (feature distortion) requires gradient-enabled feature extraction
    that ART's current API doesn't provide.
    """

    def __init__(self, model: nn.Module, layers_to_hook: list[int]):
        """
        Args:
            model: PyTorch model (e.g., YOLO model.model)
            layers_to_hook: List of layer indices to extract features from
        """
        super().__init__()

        # Get the sequential model layers
        # Handle PyTorchYoloLossWrapper -> YOLO model -> model.model (Sequential)
        unwrapped_model = model

        # Unwrap wrappers until we find the actual model
        while hasattr(unwrapped_model, 'model') and not isinstance(unwrapped_model.model, nn.Sequential):
            unwrapped_model = unwrapped_model.model
            logger.debug(f"Unwrapped to: {type(unwrapped_model)}")

        # Store the unwrapped model for forward pass
        self._unwrapped_model = unwrapped_model

        # Detect model type and get layers
        if hasattr(unwrapped_model, 'model') and isinstance(unwrapped_model.model, nn.Sequential):
            # YOLO-style model
            self.model_layers = unwrapped_model.model
            self._model_type = 'sequential'
            logger.debug(f"Using model.model (YOLO-style): {len(self.model_layers)} layers")
        elif isinstance(unwrapped_model, nn.Sequential):
            # Direct Sequential model
            self.model_layers = unwrapped_model
            self._model_type = 'sequential'
            logger.debug(f"Using model directly (Sequential): {len(self.model_layers)} layers")
        elif hasattr(unwrapped_model, 'backbone') and hasattr(unwrapped_model, 'neck'):
            # MMDetection-style model (EfficientDet, Faster R-CNN, etc.)
            # Use named_modules() to get all modules in a list
            self.model_layers = list(unwrapped_model.named_modules())
            self._model_type = 'mmdet'
            logger.debug(f"Using MMDetection-style model: {len(self.model_layers)} named modules")
        else:
            raise TypeError(
                "Model must be nn.Sequential, have .model attribute, or be MMDetection-style (with backbone/neck). "
                f"Got {type(model)} -> {type(unwrapped_model)}"
            )

        self.layers_to_hook = sorted(layers_to_hook)
        self.features = {}
        self.hooks = []

        # Register hooks for specified layers
        for layer_idx in self.layers_to_hook:
            try:
                if self._model_type == 'sequential':
                    # For Sequential models, use direct indexing
                    layer_module = self.model_layers[layer_idx]
                    hook = layer_module.register_forward_hook(
                        self.save_features_hook(layer_idx)
                    )
                    self.hooks.append(hook)
                    logger.debug(f"Registered hook for layer {layer_idx}: {type(layer_module).__name__}")
                else:
                    # For MMDetection models, use named_modules
                    if layer_idx < len(self.model_layers):
                        layer_name, layer_module = self.model_layers[layer_idx]
                        # Skip empty modules
                        if len(list(layer_module.children())) == 0:
                            hook = layer_module.register_forward_hook(
                                self.save_features_hook(layer_idx)
                            )
                            self.hooks.append(hook)
                            logger.debug(f"Registered hook for layer {layer_idx} ({layer_name}): {type(layer_module).__name__}")
                        else:
                            logger.debug(f"Skipping container module at {layer_idx} ({layer_name})")
                    else:
                        logger.warning(f"Layer index {layer_idx} out of range (model has {len(self.model_layers)} modules)")
            except IndexError:
                raise ValueError(
                    f"Layer index {layer_idx} is out of bounds for model "
                    f"with {len(self.model_layers)} layers."
                )

    def save_features_hook(self, layer_id: int):
        """Create hook function for saving features."""
        def fn(module, input, output):
            # Handle tuple outputs (e.g., from C2f layers in YOLO)
            if isinstance(output, tuple):
                self.features[layer_id] = output[0]
            else:
                self.features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Differentiable forward pass through model.
        Uses hooks to capture features without manually iterating layers.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dict mapping layer indices to feature tensors (with gradients)
        """
        self.features = {}

        # YOLO models have complex interconnections between layers
        # We cannot simply iterate through layers one by one
        # Instead, we run the full model and rely on hooks to capture features

        # Get the parent model that contains model_layers
        if hasattr(self, '_parent_model'):
            parent_model = self._parent_model
        else:
            # Try to find parent model
            parent_model = None
            for module in [m for m in dir(self) if not m.startswith('_')]:
                attr = getattr(self, module, None)
                if attr is not None and hasattr(attr, 'model') and attr.model is self.model_layers:
                    parent_model = attr
                    break

            # If still not found, create a simple wrapper
            if parent_model is None:
                # Just run through model_layers but handle errors gracefully
                try:
                    # For YOLO, we need to call the parent model, not iterate layers
                    # Try to get from the unwrapped_model we stored
                    if hasattr(self, '_unwrapped_model'):
                        parent_model = self._unwrapped_model
                    else:
                        # Fallback: just use model_layers directly and hope for the best
                        parent_model = None
                except:
                    parent_model = None

        # Run the model to trigger hooks
        try:
            if parent_model is not None:
                # Check if it's an MMDetection model
                if self._model_type == 'mmdet':
                    # MMDetection models: use extract_feat for feature extraction
                    # This runs through backbone and neck without needing data_samples
                    parent_model.eval()  # Set to eval mode to disable dropout/batchnorm updates

                    # For feature extraction, we just need a forward pass through the backbone
                    # Run through backbone and neck to get features
                    if hasattr(parent_model, 'extract_feat'):
                        # Use extract_feat method if available (MMDetection models)
                        # Don't use no_grad - we need gradients for the attack!
                        _ = parent_model.extract_feat(x)
                    else:
                        # Fallback: run through backbone manually
                        if hasattr(parent_model, 'backbone'):
                            _ = parent_model.backbone(x)
                        else:
                            raise RuntimeError("MMDetection model has no extract_feat or backbone method")
                else:
                    # YOLO-style model
                    _ = parent_model(x)
            else:
                # Fallback: try to run model_layers as a sequential
                # This may fail for YOLO but it's our last resort
                if self._model_type == 'sequential':
                    _ = nn.Sequential(*self.model_layers)(x)
                else:
                    raise RuntimeError("Cannot run MMDetection model without parent_model")
        except Exception as e:
            logger.warning(f"Error during forward pass: {e}")
            # Features may still have been captured by hooks before the error
            if not self.features:
                raise

        return self.features

    def remove_hooks(self):
        """Remove all registered hooks."""
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            logger.debug("All hooks removed")

    def __del__(self):
        """Cleanup hooks on deletion."""
        try:
            self.remove_hooks()
        except:
            pass  # Ignore errors during cleanup


class _RRBAugmentation(nn.Module):
    """
    Random Rotation, Resize, and Blur augmentation.

    This augmentation ensures robustness and transferability of universal perturbations
    by training them to be effective under various transformations.

    Based on AEGIS framework implementation.
    """

    def __init__(
        self,
        rotation_degrees: float = 7.0,
        resize_scale: tuple = (0.9, 1.1),
        blur_kernel: int = 3,
        blur_sigma: tuple = (0.1, 2.0),
        stride: int = 32,
        apply_rotation: bool = True,
        apply_resize: bool = True,
        apply_blur: bool = True,
    ):
        """
        Args:
            rotation_degrees: Max rotation angle in degrees (±)
            resize_scale: (min_scale, max_scale) for random resize
            blur_kernel: Gaussian blur kernel size (must be odd)
            blur_sigma: (min_sigma, max_sigma) for Gaussian blur
            stride: Architectural constraint (e.g., 32 for YOLO)
            apply_rotation: Enable rotation augmentation
            apply_resize: Enable resize augmentation
            apply_blur: Enable blur augmentation
        """
        super().__init__()
        self.rotation_degrees = rotation_degrees
        self.resize_scale = resize_scale
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1  # Ensure odd
        self.blur_sigma = blur_sigma
        self.stride = stride
        self.apply_rotation = apply_rotation
        self.apply_resize = apply_resize
        self.apply_blur = apply_blur

        logger.info(
            f"RRB Augmentation initialized: "
            f"rotation={apply_rotation}, resize={apply_resize}, blur={apply_blur}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RRB augmentation to batch.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Augmented tensor (B, C, H, W)
        """
        augmented_batch = []
        original_size = x.shape[-2:]  # (H, W)

        for img in x:
            # Add batch dimension for processing
            img = img.unsqueeze(0)  # (1, C, H, W)

            # 1. Random Rotation
            if self.apply_rotation:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                img = TF.rotate(
                    img,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    expand=False
                )

            # 2. Random Resize
            if self.apply_resize:
                scale = random.uniform(*self.resize_scale)
                new_h = int(original_size[0] * scale)
                new_w = int(original_size[1] * scale)

                # Ensure divisibility by stride (architectural constraint)
                new_h = ((new_h + self.stride - 1) // self.stride) * self.stride
                new_w = ((new_w + self.stride - 1) // self.stride) * self.stride

                # Resize to new size
                img = F.interpolate(
                    img,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )

                # Resize back to original size
                img = F.interpolate(
                    img,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )

            # 3. Random Gaussian Blur
            if self.apply_blur and random.random() > 0.5:
                sigma = random.uniform(*self.blur_sigma)
                # Ensure kernel size is list for TF.gaussian_blur
                kernel_size = [self.blur_kernel, self.blur_kernel]
                img = TF.gaussian_blur(img, kernel_size=kernel_size, sigma=[sigma, sigma])

            augmented_batch.append(img.squeeze(0))

        return torch.stack(augmented_batch)


class NoiseOSFDPyTorch(EvasionAttack):
    """
    Noise OSFD Attack with full functionality restored using PyTorch.

    External interface: NumPy (ART compatible)
    Internal implementation: PyTorch (for differentiability and performance)
    """

    attack_params = EvasionAttack.attack_params + [
        "eps",
        "eps_step",
        "max_iter",
        "batch_size",
        "feature_layer_indices",
        "amplification_factor",
        "apply_augmentation",
        "summary_writer",
        "verbose",
        "return_perturbation",
        "lr_scheduler_type",
        "lr_scheduler_params",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        eps: float = 8.0 / 255.0,  # AEGIS default: 8 in [0,255] scale = 0.031 in [0,1] scale
        eps_step: float = 0.001,  # AEGIS default: 0.001 (learning rate)
        max_iter: int = 100,  # AEGIS default: 100 epochs
        batch_size: int = 4,
        feature_layer_indices: list[int] | None = None,
        amplification_factor: float = 3.0,  # AEGIS default: 3.0 (not 10.0!)
        apply_augmentation: bool = True,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
        lr_scheduler_type: str = "constant",
        lr_scheduler_params: dict | None = None,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ):
        """
        Create a NoiseOSFDPyTorch attack instance.

        :param estimator: A trained object detector.
        :param eps: Maximum perturbation epsilon (L-inf norm). Default: 8/255 ≈ 0.031 (AEGIS paper).
        :param eps_step: Learning rate for perturbation optimization. Default: 0.001 (AEGIS paper).
        :param max_iter: Maximum number of iterations. Default: 100 (AEGIS paper).
        :param batch_size: Batch size for training.
        :param feature_layer_indices: Indices of layers to extract features from. Default: [10, 15, 20].
        :param amplification_factor: Factor K for amplifying benign features. Default: 3.0 (AEGIS paper).
                                      Setting this too high (e.g., 10.0) causes extreme perturbations
                                      that result in black/white images.
        :param apply_augmentation: Whether to apply RRB augmentation.
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
        self.feature_layer_indices = feature_layer_indices or [10, 15, 20]
        self.amplification_factor = amplification_factor
        self.apply_augmentation = apply_augmentation
        self.verbose = verbose
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_params = lr_scheduler_params if lr_scheduler_params is not None else {}
        self.progress_callback = progress_callback
        self.clip_min, self.clip_max = self._get_clip_bounds()
        self._input_scale = float(self.estimator.clip_values[1]) if self.estimator.clip_values is not None else 1.0

        # Auto-scale epsilon if provided in [0, 255] scale instead of [0, 1]
        original_eps = self.eps
        original_eps_step = self.eps_step
        if self._input_scale > 1.0:
            if self.eps > 1.0:
                self.eps = self.eps / self._input_scale
                logger.info(f"Auto-scaled eps from {original_eps} to {self.eps} (input_scale={self._input_scale})")
            if self.eps_step > 1.0:
                self.eps_step = self.eps_step / self._input_scale
                logger.info(f"Auto-scaled eps_step from {original_eps_step} to {self.eps_step} (input_scale={self._input_scale})")

        self._check_params()

        # Universal perturbation (PyTorch parameter)
        self._perturbation_torch: nn.Parameter | None = None
        self._perturbation: np.ndarray | None = None

        logger.info(
            f"NoiseOSFDPyTorch initialized on device: {self.device}\n"
            f"  - eps: {self.eps:.4f} (L-inf norm bound)\n"
            f"  - eps_step: {self.eps_step:.4f} (learning rate)\n"
            f"  - max_iter: {self.max_iter}\n"
            f"  - amplification_factor: {self.amplification_factor}\n"
            f"  - feature_layers: {self.feature_layer_indices}\n"
            f"  - apply_augmentation: {self.apply_augmentation}"
        )

        # Initialize internal PyTorch components
        self._feature_extractor = None
        self._augmentation = None
        if self.apply_augmentation:
            self._augmentation = _RRBAugmentation(
                rotation_degrees=7.0,
                resize_scale=(0.9, 1.1),
                blur_kernel=3,
                blur_sigma=(0.1, 2.0),
                stride=32,
                apply_rotation=True,
                apply_resize=True,
                apply_blur=True
            ).to(self.device)


    def generate(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        *,
        return_perturbation: bool = False,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Generate universal adversarial perturbation using OSFD.

        Args:
            x: Sample images (N, H, W, C) or (N, C, H, W) - NumPy
            y: Optional target labels with bounding boxes. If provided, apply perturbation only to those regions.
            return_perturbation: If True, also return the learned universal perturbation.

        Returns:
            Adversarial images. If `return_perturbation=True`, also returns the learned perturbation.
        """
        logger.info(f"Starting OSFD generation with {len(x)} images")

        # Use base class preprocessing
        x_torch, x_original = self._preprocess_and_convert(x)

        # Initialize perturbation using base class
        self._initialize_perturbation(x_torch)

        # Train perturbation using PyTorch
        self._train_osfd_perturbation_pytorch(x_torch)

        # Apply perturbation (with optional bbox masking)
        num_samples = x_torch.shape[0]
        batch_size = min(self.batch_size, num_samples)
        x_adv_batches = []

        if y is not None:
            logger.info("Applying perturbation with bbox masking from provided labels")
            # Convert y to torch format
            pseudo_gts = []
            for label in y:
                from app.ai.losses.box_utils import xyxy2xywh

                boxes_xyxy = torch.from_numpy(label['boxes']).float().to(self.device)
                pseudo_gt = {
                    'boxes': xyxy2xywh(boxes_xyxy),
                    'labels': torch.from_numpy(label['labels']).long().to(self.device)
                }
                pseudo_gts.append(pseudo_gt)

            for start in range(0, num_samples, batch_size):
                batch_x = x_torch[start:start + batch_size].to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_gts = pseudo_gts[start:start + batch_size]
                with torch.no_grad():
                    x_adv_batch = self._apply_perturbation_with_mask(batch_x, batch_gts)
                x_adv_batches.append(self._torch_to_numpy(x_adv_batch, x[start:start + batch_size]))
        else:
            # Apply perturbation to entire image
            logger.info("Applying perturbation to entire images (no masking)")
            for start in range(0, num_samples, batch_size):
                batch_x = x_torch[start:start + batch_size].to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                with torch.no_grad():
                    x_adv_batch = torch.clamp(
                        batch_x + self._perturbation_torch,
                        self.clip_min,
                        self.clip_max
                    )
                x_adv_batches.append(self._torch_to_numpy(x_adv_batch, x[start:start + batch_size]))

        x_adv = np.concatenate(x_adv_batches, axis=0)
        self._perturbation = self._torch_to_numpy(self._perturbation_torch, x[:1])

        # Cleanup
        if self._feature_extractor is not None:
            self._feature_extractor.remove_hooks()
            self._feature_extractor = None

        logger.info("OSFD generation completed")
        # Always return perturbation for service compatibility
        # (service code expects tuple to extract perturbation)
        return x_adv, self._perturbation

    def _train_osfd_perturbation_pytorch(self, x: torch.Tensor):
        """
        Train universal perturbation using PyTorch (fully differentiable).

        This is the core OSFD algorithm:
        1. Extract benign features
        2. Apply perturbation + augmentation
        3. Extract adversarial features
        4. Maximize MSE between adversarial and amplified benign features

        Uses mini-batch processing for memory-efficient training.

        Args:
            x: Input images (N, C, H, W) - PyTorch tensor
        """
        logger.info(f"Training OSFD perturbation for {self.max_iter} iterations")

        # Setup feature extractor with hooks
        self._feature_extractor = _FeatureExtractor(
            self._torch_model,
            self.feature_layer_indices
        )
        logger.info(f"Feature extractor ready for layers: {self.feature_layer_indices}")

        # Setup optimizer
        optimizer = torch.optim.Adam([self._perturbation_torch], lr=self.eps_step)

        # Create learning rate scheduler
        scheduler = create_lr_scheduler(
            optimizer=optimizer,
            scheduler_type=self.lr_scheduler_type,
            scheduler_params=self.lr_scheduler_params,
            max_iter=self.max_iter,
        )

        if scheduler is not None:
            logger.info(f"Using {self.lr_scheduler_type} LR scheduler")
        else:
            logger.info("Using constant learning rate (no scheduler)")

        # MSE loss function
        mse_loss = nn.MSELoss()

        num_samples = x.shape[0]
        batch_size = min(self.batch_size, num_samples)
        full_batch = batch_size >= num_samples
        logger.info(
            f"Training on {num_samples} images with batch_size={batch_size} "
            f"({'full-batch' if full_batch else 'mini-batch'})"
        )

        # Training loop - process in batches (full-batch when batch_size >= num_samples)
        for i_iter in trange(self.max_iter, desc="OSFD Training", disable=not self.verbose):
            optimizer.zero_grad()

            total_loss_value = 0.0
            layer_losses = {layer_idx: 0.0 for layer_idx in self.feature_layer_indices}

            for start in range(0, num_samples, batch_size):
                batch_x = x[start:start + batch_size].to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_weight = batch_x.shape[0] / float(num_samples)

                # Extract benign features for this batch (no gradients needed)
                with torch.no_grad():
                    benign_features = self._feature_extractor(batch_x)

                # Apply perturbation to batch
                x_adv = torch.clamp(
                    batch_x + self._perturbation_torch,
                    self.clip_min,
                    self.clip_max
                )

                # Apply augmentation for robustness
                if self.apply_augmentation and self._augmentation is not None:
                    x_adv_aug = self._augmentation(x_adv)
                else:
                    x_adv_aug = x_adv

                # Extract adversarial features (with gradients)
                adv_features = self._feature_extractor(x_adv_aug)

                # Compute OSFD loss: maximize MSE between adversarial and amplified benign features
                batch_loss = torch.tensor(0.0, device=self.device)

                for layer_idx in self.feature_layer_indices:
                    if layer_idx not in benign_features or layer_idx not in adv_features:
                        logger.warning(f"Layer {layer_idx} not found in features, skipping")
                        continue

                    # Target: amplified benign features
                    target_features = self.amplification_factor * benign_features[layer_idx].detach()
                    adv_feat = adv_features[layer_idx]

                    # Handle shape mismatch with interpolation
                    if adv_feat.shape != target_features.shape:
                        target_size = target_features.shape[2:]
                        adv_feat = F.interpolate(
                            adv_feat,
                            size=target_size,
                            mode='bilinear',
                            align_corners=False
                        )
                        if i_iter == 0:
                            logger.debug(
                                f"Resized layer {layer_idx} features from {adv_features[layer_idx].shape} "
                                f"to {adv_feat.shape}"
                            )

                    # MSE loss for this layer
                    layer_loss = mse_loss(adv_feat, target_features)
                    batch_loss += layer_loss
                    layer_losses[layer_idx] += layer_loss.item() * batch_weight

                # OSFD maximizes distortion -> minimize negative loss
                loss_to_backward = -batch_loss * batch_weight
                loss_to_backward.backward()
                total_loss_value += batch_loss.item() * batch_weight

            optimizer.step()

            # Enforce epsilon constraint using base class
            self._enforce_epsilon(self.eps)

            avg_epoch_loss = total_loss_value
            avg_layer_losses = layer_losses

            # Step the learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Logging
            if self.verbose and (i_iter % 10 == 0 or i_iter == self.max_iter - 1):
                logger.info(
                    f"Iter {i_iter}/{self.max_iter}: Avg OSFD Loss = {avg_epoch_loss:.6f}, "
                    f"LR = {current_lr:.6f}, Layers: {avg_layer_losses}"
                )

            # Progress callback for SSE updates
            if self.progress_callback and (i_iter % 10 == 0 or i_iter == self.max_iter - 1):
                try:
                    self.progress_callback(i_iter, self.max_iter, avg_epoch_loss, current_lr)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

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
        if not isinstance(self.amplification_factor, (int, float)) or self.amplification_factor <= 0:
            raise ValueError("amplification_factor must be positive")

    def _apply_perturbation_with_mask(
        self,
        x: torch.Tensor,
        pseudo_gts: list
    ) -> torch.Tensor:
        """
        Apply perturbation only to detected object regions (bbox masking).

        Args:
            x: Input images (B, C, H, W)
            pseudo_gts: List of pseudo ground truth dicts with 'boxes' and 'labels'

        Returns:
            Perturbed images (B, C, H, W) with perturbation applied only to target regions
        """
        x_adv = x.clone()

        for i in range(x.shape[0]):
            if 'boxes' not in pseudo_gts[i] or len(pseudo_gts[i]['boxes']) == 0:
                # No boxes for this image, keep original
                continue

            boxes = pseudo_gts[i]['boxes'].to(self.device)
            from app.ai.losses.box_utils import xywh2xyxy
            boxes_xyxy = xywh2xyxy(boxes)

            # Create mask for this image (C, H, W) to match perturbation shape
            mask = torch.zeros(x.shape[1], x.shape[2], x.shape[3], device=self.device, dtype=torch.float32)

            for box in boxes_xyxy:
                # Convert box coordinates to integers on CPU (for indexing)
                box_cpu = box[:4].cpu()
                x1 = int(box_cpu[0].item())
                y1 = int(box_cpu[1].item())
                x2 = int(box_cpu[2].item())
                y2 = int(box_cpu[3].item())

                # Ensure coordinates are within bounds
                h, w = x.shape[2], x.shape[3]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if x1 < x2 and y1 < y2:
                    mask[:, y1:y2, x1:x2] = 1.0

            # Apply masked perturbation - ensure all tensors are on the same device
            perturbation_masked = self._perturbation_torch[0].to(self.device) * mask
            x_i = x[i].to(self.device)
            x_adv[i] = torch.clamp(x_i + perturbation_masked, self.clip_min, self.clip_max)

        return x_adv

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
        x_torch = x_torch.to(dtype=torch.float32)
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
            x_np = x_np * self.clip_max

        # Handle channels_first/last to match original format
        if not self.estimator.channels_first and x_np.ndim == 4:
            # (N, C, H, W) -> (N, H, W, C)
            x_np = np.transpose(x_np, (0, 2, 3, 1))

        return x_np.astype(np.float32)

    def _get_clip_bounds(self) -> tuple[float, float]:
        """
        Determine clipping bounds for INTERNAL normalized [0, 1] scale.
        """
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
        pert = perturbation
        abs_max = float(np.nanmax(np.abs(pert))) if pert.size else 0.0
        if self._input_scale > 1.0 and abs_max > 1.0:
            pert = pert / self._input_scale
        if np.isnan(pert).any():
            pert = np.nan_to_num(pert, nan=0.0, posinf=0.0, neginf=0.0)
        self._perturbation = pert
        self._perturbation_torch = nn.Parameter(
            torch.from_numpy(pert).float().to(self.device)
        )

    def _enforce_epsilon(self, eps: float) -> None:
        """
        Project perturbation into L_inf ball of radius `eps` (same scale as internal tensors).
        """
        if self._perturbation_torch is None:
            return
        # Clamp around 0 since perturbation is additive
        self._perturbation_torch.data = torch.clamp(self._perturbation_torch.data, -eps, eps)

    def _generate_pseudo_gts_for_mask(self, x: np.ndarray) -> list[dict[str, torch.Tensor]]:
        """
        Generate pseudo-GT boxes for masking using the estimator predict API.
        """
        from app.ai.losses.box_utils import xyxy2xywh

        predictions = self.estimator.predict(x=x)
        pseudo_gts = []

        for pred in predictions:
            if 'boxes' not in pred or len(pred['boxes']) == 0:
                pseudo_gts.append({
                    'boxes': torch.zeros(0, 4, dtype=torch.float32, device=self.device),
                    'labels': torch.zeros(0, dtype=torch.int64, device=self.device),
                })
                continue

            boxes_xyxy = torch.from_numpy(pred['boxes']).float().to(self.device)
            boxes_xywh = xyxy2xywh(boxes_xyxy)

            labels = pred.get('labels')
            if labels is None:
                labels_t = torch.zeros(len(boxes_xywh), dtype=torch.int64, device=self.device)
            else:
                labels_t = torch.from_numpy(labels).long().to(self.device)

            pseudo_gts.append({
                'boxes': boxes_xywh,
                'labels': labels_t,
            })

        return pseudo_gts

    def apply_perturbation(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        apply_mask: bool = True
    ) -> np.ndarray:
        """
        Apply pre-trained OSFD perturbation to new images.

        Args:
            x: Input images (N, H, W, C) or (N, C, H, W) - NumPy
            y: Optional labels for masking. If None, use pseudo-GT from predictor.
            apply_mask: Whether to apply masking. Default True.

        Returns:
            Adversarial images (NumPy array)
        """
        if self._perturbation_torch is None:
            raise ValueError("Perturbation not trained yet. Call generate() first or set_perturbation().")

        x_original = x
        x_is_hwc = x.ndim == 4 and x.shape[-1] in [1, 3]
        x_input = x
        if self.estimator.channels_first and x_is_hwc:
            x_input = np.transpose(x, (0, 3, 1, 2))

        x_torch, _ = self.estimator._preprocess_and_convert_inputs(
            x=x_input, y=None, fit=False, no_grad=True
        )
        x_torch = x_torch.to(self.device, dtype=torch.float32)

        if apply_mask:
            if y is not None:
                pseudo_gts = []
                for label in y:
                    from app.ai.losses.box_utils import xyxy2xywh

                    boxes_xyxy = torch.from_numpy(label['boxes']).float().to(self.device)
                    pseudo_gt = {
                        'boxes': xyxy2xywh(boxes_xyxy),
                        'labels': torch.from_numpy(label['labels']).long().to(self.device)
                    }
                    pseudo_gts.append(pseudo_gt)
            else:
                pseudo_gts = self._generate_pseudo_gts_for_mask(x_input)

            x_adv_torch = self._apply_perturbation_with_mask(x_torch, pseudo_gts)
        else:
            with torch.no_grad():
                x_adv_torch = torch.clamp(x_torch + self._perturbation_torch, self.clip_min, self.clip_max)

        x_adv = self._torch_to_numpy(x_adv_torch, x_input)
        if self.estimator.channels_first and x_is_hwc and x_adv.ndim == 4:
            x_adv = np.transpose(x_adv, (0, 2, 3, 1))

        # Ensure output format matches input
        if x_original.ndim == 4 and x_original.shape[-1] in [1, 3]:
            return x_adv.astype(np.float32)
        return x_adv.astype(np.float32)
