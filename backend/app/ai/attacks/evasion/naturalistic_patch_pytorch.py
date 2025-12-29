"""
Naturalistic Adversarial Patch Attack using BigGAN.

This module implements GAN-based adversarial patch generation for object detectors,
producing naturalistic-looking patches through latent space optimization.

| Paper link: Naturalistic Physical Adversarial Patch (Int'l Journal of Information Security 2025)
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
import math
import numpy as np
from typing import Optional, TYPE_CHECKING, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

from app.ai.attacks.attack import EvasionAttack
from app.ai.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from app.ai.summary_writer import SummaryWriter
from app.ai.gan.biggan.loader import load_biggan, generate_from_latent
from app.services.gan_weights_service import gan_weights_service

if TYPE_CHECKING:
    from app.ai.utils import PYTORCH_OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class TotalVariation(nn.Module):
    """
    Total Variation loss for patch smoothness.

    Calculates the total variation (TV) of a patch to encourage
    smooth transitions and reduce high-frequency artifacts.
    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.

        Args:
            adv_patch: Patch tensor (shape: [C, H, W])

        Returns:
            TV loss value
        """
        # Horizontal differences
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 1e-6))
        # Vertical differences
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 1e-6))
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class NaturalisticPatchPyTorch(EvasionAttack):
    """
    Naturalistic adversarial patch attack using BigGAN latent space optimization.

    Generates adversarial patches by optimizing latent vectors in BigGAN's latent space,
    producing naturalistic-looking patches that fool object detectors while maintaining
    visual realism through GAN priors.

    | Paper link: https://link.springer.com/article/10.1007/s10207-024-00947-0
    """

    attack_params = EvasionAttack.attack_params + [
        "gan_class_id",
        "tv_weight",
        "overlap_weight",
        "alpha_latent",
        "patch_scale",
        "learning_rate",
        "max_iter",
        "batch_size",
        "max_latent_value",
        "enable_deformator",
        "deformator_path",
        "enable_shift_deformator",
        "enable_latent_rounding",
        "latent_rounding_precision",
        "enable_random_location",
        "enable_rotation",
        "rotation_range",
        "enable_appearance_aug",
        "min_contrast",
        "max_contrast",
        "min_brightness",
        "max_brightness",
        "noise_factor",
        "noise_distribution",
        "vertical_offset_ratio",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(
        self,
        estimator: "PYTORCH_OBJECT_DETECTOR_TYPE",
        gan_class_id: int = 259,  # Pomeranian
        tv_weight: float = 0.0,
        overlap_weight: float = 0.0,
        alpha_latent: float = 1.0,
        patch_scale: float = 0.2,
        learning_rate: float = 0.02,
        max_iter: int = 1000,
        batch_size: int = 8,
        max_latent_value: float = 8.0,
        enable_deformator: bool = False,
        deformator_path: Optional[str] = None,
        enable_shift_deformator: bool = True,
        enable_latent_rounding: bool = True,
        latent_rounding_precision: int = 4,
        enable_random_location: bool = False,
        enable_rotation: bool = False,
        rotation_range: float = 20.0,
        enable_appearance_aug: bool = True,
        min_contrast: float = 0.8,
        max_contrast: float = 1.2,
        min_brightness: float = -0.1,
        max_brightness: float = 0.1,
        noise_factor: float = 0.10,
        noise_distribution: str = 'gaussian',
        vertical_offset_ratio: float = -0.05,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
    ):
        """
        Create an instance of NaturalisticPatchPyTorch.

        Args:
            estimator: A trained PyTorch object detector (YOLO, RT-DETR, etc.)
            gan_class_id: ImageNet class ID for BigGAN generation (default: 259 = Pomeranian)
            tv_weight: Total variation loss weight (0.0-0.1, default: 0.0)
                Higher values produce smoother patches but may reduce attack effectiveness.
            overlap_weight: Overlap loss weight (0.0-0.1, default: 0.0)
                Measures detection probability change (with/without patch).
                Higher values maximize attack impact. Set to 0.0 to disable (recommended).
            alpha_latent: Latent space mixing factor (0.0-1.0, default: 1.0)
                - With deformator+shift: z = (1-α)*rand + α*fixed, shift is optimized.
                - Without deformator: z = (1-α)*rand + α*latent_shift.
            patch_scale: Patch scale relative to detected person bbox (default: 0.2)
                0.2 means patch size = 20% of sqrt(bbox_width² + bbox_height²)
            learning_rate: Learning rate for Adam optimizer (default: 0.02)
            max_iter: Maximum optimization iterations (default: 1000)
            batch_size: Batch size for patch generation (default: 8)
            max_latent_value: Maximum absolute value for latent clamping (default: 8.0)
            enable_deformator: Use deformator for latent transformation (default: False)
            deformator_path: Path to deformator weights (optional)
            enable_shift_deformator: Use deformator with shift (default: True)
            enable_latent_rounding: Round latent to fixed precision (default: True)
                Helps stabilize training by preventing micro-oscillations.
            latent_rounding_precision: Decimal places for latent rounding (default: 4)
            enable_random_location: Add random jitter to patch location (default: False)
            enable_rotation: Apply random rotation to patch (default: False)
            rotation_range: Rotation angle range in degrees (default: 20.0)
            enable_appearance_aug: Apply appearance augmentations (default: True)
                Includes contrast, brightness, and noise variations.
            min_contrast: Minimum contrast multiplier (default: 0.8)
            max_contrast: Maximum contrast multiplier (default: 1.2)
            min_brightness: Minimum brightness offset (default: -0.1)
            max_brightness: Maximum brightness offset (default: 0.1)
            noise_factor: Noise magnitude multiplier (default: 0.10)
            noise_distribution: Noise distribution type (default: 'gaussian')
                - 'gaussian': Normal(0, noise_factor) - more realistic, matches camera sensor noise
                - 'uniform': Uniform(-noise_factor, noise_factor) - original NAP implementation
            vertical_offset_ratio: Vertical offset as ratio of image height (default: -0.05)
                Negative values move patch upward (e.g., -0.05 = 5% upward)
            summary_writer: SummaryWriter for logging or SSE updates
            verbose: Show progress bar
        """
        super().__init__(estimator=estimator, summary_writer=summary_writer)

        self.gan_class_id = gan_class_id
        self.tv_weight = tv_weight
        self.overlap_weight = overlap_weight
        self.alpha_latent = alpha_latent
        self.patch_scale = patch_scale
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.max_latent_value = max_latent_value
        self.enable_deformator = enable_deformator
        self.deformator_path = deformator_path
        self.enable_shift_deformator = enable_shift_deformator
        self.enable_latent_rounding = enable_latent_rounding
        self.latent_rounding_precision = latent_rounding_precision
        self.enable_random_location = enable_random_location
        self.enable_rotation = enable_rotation
        self.rotation_range = rotation_range
        self.enable_appearance_aug = enable_appearance_aug
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.noise_factor = noise_factor
        self.vertical_offset_ratio = vertical_offset_ratio
        self.verbose = verbose

        # Validate noise distribution
        if noise_distribution not in ('gaussian', 'uniform'):
            raise ValueError(
                f"noise_distribution must be 'gaussian' or 'uniform', got '{noise_distribution}'"
            )
        self.noise_distribution = noise_distribution

        # Ensure GAN weights are downloaded
        logger.info("Ensuring BigGAN weights are available...")
        weights_path = gan_weights_service.ensure_biggan_weights()
        logger.info(f"BigGAN weights ready at {weights_path}")

        # Load BigGAN generator
        logger.info(f"Loading BigGAN for class {gan_class_id}...")

        # Validate deformator configuration
        if enable_deformator and not deformator_path:
            raise ValueError(
                "enable_deformator=True but deformator_path is not provided. "
                "Please provide a valid deformator_path or set enable_deformator=False."
            )

        # Only load deformator if explicitly enabled
        deformator_path_to_use = deformator_path if enable_deformator else None
        self.generator, self.deformator = load_biggan(
            weights_path=str(weights_path),
            target_class=gan_class_id,
            device=self.estimator.device.type,
            deformator_path=deformator_path_to_use,
        )

        # Initialize latent vectors
        self.dim_z = self.generator.dim_z
        # Fixed latent: base point in latent space
        # Use uniform distribution to match original NAP setup
        self.fixed_latent = torch.rand(self.dim_z, device=self.estimator.device)
        logger.info("Initialized fixed_latent with Uniform(0, 1) distribution")
        self.latent_shift = nn.Parameter(
            torch.randn(self.dim_z).to(self.estimator.device),
            requires_grad=True
        )

        # Optimizer (match original NAP defaults)
        self.optimizer = torch.optim.Adam(
            [self.latent_shift],
            lr=self.learning_rate,
            betas=(0.5, 0.999),
            amsgrad=True,
        )

        # Total variation loss
        self.tv_loss_fn = TotalVariation()

        # Current patch
        self.current_patch = None

        logger.info(
            f"NaturalisticPatchPyTorch initialized (class={gan_class_id}, "
            f"tv_weight={tv_weight}, overlap_weight={overlap_weight}, "
            f"alpha={alpha_latent}, noise={noise_distribution}, deformator={bool(self.deformator)})"
        )

    def _generate_patch_from_latent(self) -> torch.Tensor:
        """
        Generate patch image from current latent vectors.

        Returns:
            Generated patch tensor (shape: [3, H, W]) in range [0, 1]
        """
        # Clamp latent shift to prevent extreme values
        # FIXED: Use in-place clamp without breaking gradient flow
        with torch.no_grad():
            self.latent_shift.clamp_(-self.max_latent_value, self.max_latent_value)
            if self.enable_latent_rounding:
                factor = 10 ** self.latent_rounding_precision
                self.latent_shift.copy_(torch.round(self.latent_shift * factor) / factor)

        use_deformator = self.enable_deformator and self.deformator is not None

        if use_deformator:
            # Original NAP: z = (1 - alpha) * rand + alpha * fixed
            z = torch.rand(self.dim_z, device=self.estimator.device)
            if self.alpha_latent > 0:
                z = (1 - self.alpha_latent) * z + self.alpha_latent * self.fixed_latent

            if self.enable_shift_deformator:
                shift = torch.clamp(
                    self.latent_shift * self.max_latent_value,
                    -self.max_latent_value,
                    self.max_latent_value,
                )
                patch = generate_from_latent(
                    self.generator,
                    z.unsqueeze(0),
                    shift=shift.unsqueeze(0),
                    deformator=self.deformator,
                )
            else:
                z = (1 - self.alpha_latent) * z + self.alpha_latent * self.latent_shift
                patch = self.generator(z.unsqueeze(0))
        else:
            # No deformator: latent_merged = (1 - alpha) * rand + alpha * latent_shift
            latent = torch.rand(self.dim_z, device=self.estimator.device)
            if self.alpha_latent > 0:
                latent = (1 - self.alpha_latent) * latent + self.alpha_latent * self.latent_shift
            patch = self.generator(latent.unsqueeze(0))

        # Convert from [-1, 1] to [0, 1] range
        patch = (patch + 1) * 0.5
        patch = torch.clamp(patch, 0, 1)

        return patch[0]  # Remove batch dimension

    def _compute_detection_loss(
        self,
        patched_images: torch.Tensor,
        labels: Optional[list[dict[str, torch.Tensor]]] = None,
        target_class: Optional[int] = None,
        clean_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute detection loss with gradient flow and optional overlap score.

        Prefer ART estimator's compute_loss() (training loss) to preserve gradients.
        Fallback to a surrogate loss when labels are unavailable.

        Args:
            patched_images: Patched image batch (shape: [B, C, H, W])
            labels: ART-style labels (list of dicts with "boxes" and "labels")
            target_class: Optional target class ID for fallback logging
            clean_images: Optional clean images for overlap score calculation

        Returns:
            Tuple of (detection_loss, overlap_score)
            - detection_loss: Detection loss on patched images (requires_grad=True)
            - overlap_score: |loss(patched) - loss(clean)| if clean_images provided, else 0
        """
        # Detection loss on patched images
        if labels is not None:
            loss_det_patched = self.estimator.compute_loss(x=patched_images, y=labels)
        else:
            logger.warning(
                "Detection labels missing; using surrogate loss to keep gradients. "
                f"target_class={target_class}"
            )
            loss_det_patched = patched_images.mean()

        # Compute overlap score if clean images provided
        overlap_score = torch.tensor(0.0, device=patched_images.device)
        if clean_images is not None and labels is not None:
            # Detach clean images to prevent gradient flow (we only want gradient through patched)
            with torch.no_grad():
                loss_det_clean = self.estimator.compute_loss(x=clean_images, y=labels)

            # Overlap score: absolute difference in detection loss
            overlap_score = torch.abs(loss_det_patched - loss_det_clean)

        return loss_det_patched, overlap_score

    def _apply_patch_to_images(
        self,
        images: torch.Tensor,
        patch: torch.Tensor,
        labels: Optional[list[dict[str, torch.Tensor]]] = None,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply patch to images at detected object locations.

        Args:
            images: Image batch (shape: [B, C, H, W])
            patch: Patch to apply (shape: [C, H_patch, W_patch])
            labels: Optional ART-style labels (list of dicts with "boxes" and "labels")
                    Boxes are expected in pixel coordinates (x1, y1, x2, y2).
            target_class: Optional target class ID to filter boxes.

        Returns:
            Patched images
        """
        batch_size, _C, H, W = images.shape

        # Scale patch to match image range if needed
        # Patch is [0, 1], but images might be [0, 255]
        patch_to_apply = patch
        clip_max = 1.0
        if hasattr(self.estimator, 'clip_values') and self.estimator.clip_values[1] > 1.0:
            clip_max = float(self.estimator.clip_values[1])
            patch_to_apply = patch * clip_max

        if labels and len(labels) == batch_size:
            patched_batch = []
            any_box = False
            for b in range(batch_size):
                patched = images[b]
                boxes = labels[b].get("boxes")
                label_ids = labels[b].get("labels")

                if boxes is None or boxes.numel() == 0:
                    patched_batch.append(patched)
                    continue

                if label_ids is not None and not isinstance(label_ids, torch.Tensor):
                    label_ids = torch.as_tensor(label_ids, dtype=torch.int64, device=images.device)

                applied = False
                for i in range(boxes.shape[0]):
                    if target_class is not None and label_ids is not None:
                        if int(label_ids[i].item()) != int(target_class):
                            continue

                    x1, y1, x2, y2 = boxes[i]
                    x1_i = int(torch.clamp(x1.round(), 0, W - 1).item())
                    y1_i = int(torch.clamp(y1.round(), 0, H - 1).item())
                    x2_i = int(torch.clamp(x2.round(), 0, W).item())
                    y2_i = int(torch.clamp(y2.round(), 0, H).item())

                    if x2_i <= x1_i or y2_i <= y1_i:
                        continue

                    bbox_w = x2_i - x1_i
                    bbox_h = y2_i - y1_i
                    patch_size = int(round(math.sqrt(bbox_w ** 2 + bbox_h ** 2) * self.patch_scale))
                    patch_size = max(1, min(patch_size, H, W))
                    if patch_size < 2:
                        continue

                    patch_resized = torch.nn.functional.interpolate(
                        patch_to_apply.unsqueeze(0),
                        size=(patch_size, patch_size),
                        mode='bilinear',
                        align_corners=False
                    )[0]

                    center_x = (x1_i + x2_i) // 2
                    center_y = (y1_i + y2_i) // 2
                    if self.enable_random_location:
                        jitter_x = torch.empty(1, device=patch_resized.device).uniform_(-0.4, 0.4)
                        jitter_y = torch.empty(1, device=patch_resized.device).uniform_(-0.4, 0.4)
                        center_x += int(round(bbox_w * float(jitter_x.item())))
                        center_y += int(round(bbox_h * float(jitter_y.item())))
                    if self.vertical_offset_ratio != 0.0:
                        center_y += int(round(self.vertical_offset_ratio * H))

                    center_x = max(0, min(center_x, W - 1))
                    center_y = max(0, min(center_y, H - 1))

                    if self.enable_appearance_aug:
                        patch_norm = patch_resized / clip_max
                        contrast = torch.empty(1, device=patch_resized.device).uniform_(
                            self.min_contrast, self.max_contrast
                        )
                        brightness = torch.empty(1, device=patch_resized.device).uniform_(
                            self.min_brightness, self.max_brightness
                        )
                        # Generate noise based on distribution type
                        if self.noise_distribution == 'gaussian':
                            noise = torch.randn_like(patch_norm) * self.noise_factor
                        else:  # 'uniform'
                            noise = (torch.rand_like(patch_norm) * 2 - 1) * self.noise_factor
                        patch_norm = torch.clamp(patch_norm * contrast + brightness + noise, 0.0, 1.0)
                        patch_resized = patch_norm * clip_max

                    if self.enable_rotation and self.rotation_range > 0:
                        angle_deg = torch.empty(1, device=patch_resized.device).uniform_(
                            -self.rotation_range, self.rotation_range
                        )
                        angle = angle_deg * (math.pi / 180.0)
                        cos_a = torch.cos(angle)
                        sin_a = torch.sin(angle)
                        theta = torch.stack(
                            [cos_a, -sin_a, torch.zeros_like(cos_a),
                             sin_a, cos_a, torch.zeros_like(cos_a)],
                            dim=1
                        ).view(1, 2, 3)
                        grid = F.affine_grid(
                            theta,
                            size=(1, patch_resized.shape[0], patch_size, patch_size),
                            align_corners=False,
                        )
                        patch_resized = F.grid_sample(
                            patch_resized.unsqueeze(0),
                            grid,
                            align_corners=False,
                            padding_mode="zeros",
                        )[0]

                    patch_x1 = max(0, min(center_x - patch_size // 2, W - patch_size))
                    patch_y1 = max(0, min(center_y - patch_size // 2, H - patch_size))
                    patch_x2 = patch_x1 + patch_size
                    patch_y2 = patch_y1 + patch_size

                    pad_left = patch_x1
                    pad_right = W - patch_x2
                    pad_top = patch_y1
                    pad_bottom = H - patch_y2

                    patch_canvas = torch.nn.functional.pad(
                        patch_resized,
                        (pad_left, pad_right, pad_top, pad_bottom),
                    )
                    mask = torch.nn.functional.pad(
                        torch.ones_like(patch_resized),
                        (pad_left, pad_right, pad_top, pad_bottom),
                    )

                    patched = patched * (1 - mask) + patch_canvas
                    applied = True

                patched_batch.append(patched)
                any_box = any_box or applied

            if any_box:
                return torch.stack(patched_batch, dim=0)

        # Fallback: apply patch to center of image
        patch_h, patch_w = patch_to_apply.shape[1:]
        target_size = max(1, int(min(H, W) * self.patch_scale))
        if target_size != patch_h or target_size != patch_w:
            patch_resized = torch.nn.functional.interpolate(
                patch_to_apply.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )[0]
        else:
            patch_resized = patch_to_apply

        if self.enable_appearance_aug:
            patch_norm = patch_resized / clip_max
            contrast = torch.empty(1, device=patch_resized.device).uniform_(
                self.min_contrast, self.max_contrast
            )
            brightness = torch.empty(1, device=patch_resized.device).uniform_(
                self.min_brightness, self.max_brightness
            )
            # Generate noise based on distribution type
            if self.noise_distribution == 'gaussian':
                noise = torch.randn_like(patch_norm) * self.noise_factor
            else:  # 'uniform'
                noise = (torch.rand_like(patch_norm) * 2 - 1) * self.noise_factor
            patch_norm = torch.clamp(patch_norm * contrast + brightness + noise, 0.0, 1.0)
            patch_resized = patch_norm * clip_max

        if self.enable_rotation and self.rotation_range > 0:
            angle_deg = torch.empty(1, device=patch_resized.device).uniform_(
                -self.rotation_range, self.rotation_range
            )
            angle = angle_deg * (math.pi / 180.0)
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            theta = torch.stack(
                [cos_a, -sin_a, torch.zeros_like(cos_a),
                 sin_a, cos_a, torch.zeros_like(cos_a)],
                dim=1
            ).view(1, 2, 3)
            grid = F.affine_grid(
                theta,
                size=(1, patch_resized.shape[0], target_size, target_size),
                align_corners=False,
            )
            patch_resized = F.grid_sample(
                patch_resized.unsqueeze(0),
                grid,
                align_corners=False,
                padding_mode="zeros",
            )[0]

        y_start = (H - target_size) // 2
        x_start = (W - target_size) // 2
        y_end = y_start + target_size
        x_end = x_start + target_size

        pad_left = x_start
        pad_right = W - x_end
        pad_top = y_start
        pad_bottom = H - y_end

        patch_canvas = torch.nn.functional.pad(
            patch_resized,
            (pad_left, pad_right, pad_top, pad_bottom),
        )
        mask = torch.nn.functional.pad(
            torch.ones_like(patch_resized),
            (pad_left, pad_right, pad_top, pad_bottom),
        )

        patch_canvas = patch_canvas.unsqueeze(0).expand(batch_size, -1, -1, -1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return images * (1 - mask) + patch_canvas

    def generate(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray | list[dict[str, Any]]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate naturalistic adversarial patch.

        Args:
            x: Sample images (shape: [B, C, H, W] or [B, H, W, C])
               Range can be [0, 1] or [0, 255] (auto-normalized).
            y: Optional labels. Supported formats:
               - ART format: list[dict] with keys "boxes", "labels" (and optional "scores")
               - NAP format: np.ndarray [B, max_objects, 5] with [x_center, y_center, width, height, class]
            **kwargs: Additional arguments

        Returns:
            Tuple of (patch, mask)
            - patch: Generated adversarial patch (shape: [C, H, W])
            - mask: Patch mask (shape: [C, H, W])
        """
        logger.info(
            f"Generating naturalistic patch: {self.max_iter} iterations, "
            f"class_id={self.gan_class_id}, tv_weight={self.tv_weight}"
        )

        # Convert images to torch tensors (support NCHW or NHWC)
        images_np = x.astype(np.float32)
        if images_np.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape={images_np.shape}")

        if images_np.shape[1] in (1, 3):
            images_tensor = torch.from_numpy(images_np)
        elif images_np.shape[-1] in (1, 3):
            images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unrecognized image format: shape={images_np.shape}")

        # Normalize to estimator clip range if needed
        clip_max = 1.0
        if hasattr(self.estimator, "clip_values") and self.estimator.clip_values is not None:
            clip_max = float(self.estimator.clip_values[1])
        max_val = float(images_np.max()) if images_np.size else 0.0
        if max_val <= 1.5 and clip_max > 1.0:
            images_tensor = images_tensor * clip_max
        elif max_val > 1.5 and clip_max <= 1.0:
            images_tensor = images_tensor / max_val if max_val > 0 else images_tensor

        images_tensor = images_tensor.to(self.estimator.device)

        # Prepare labels for compute_loss
        labels: Optional[list[dict[str, torch.Tensor]]] = None
        target_class: Optional[int] = None
        if y is not None:
            if isinstance(y, list):
                labels = []
                for label_dict in y:
                    boxes = label_dict.get("boxes", np.zeros((0, 4), dtype=np.float32))
                    label_ids = label_dict.get("labels", np.zeros((0,), dtype=np.int64))
                    scores = label_dict.get("scores")

                    boxes_t = torch.as_tensor(boxes, dtype=torch.float32, device=self.estimator.device)
                    labels_t = torch.as_tensor(label_ids, dtype=torch.int64, device=self.estimator.device)
                    label_out: dict[str, torch.Tensor] = {"boxes": boxes_t, "labels": labels_t}
                    if scores is not None:
                        label_out["scores"] = torch.as_tensor(scores, dtype=torch.float32, device=self.estimator.device)
                    labels.append(label_out)

            elif isinstance(y, np.ndarray):
                labels = []
                _, _C, H, W = images_tensor.shape
                for b in range(y.shape[0]):
                    boxes_b = []
                    labels_b = []
                    for ann in y[b]:
                        if ann is None or len(ann) < 5:
                            continue
                        x_center, y_center, width, height, cls_id = ann[:5]
                        if np.isnan(cls_id):
                            continue
                        # Detect normalized coords
                        if max(x_center, y_center, width, height) <= 1.5:
                            x_center *= W
                            y_center *= H
                            width *= W
                            height *= H
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        boxes_b.append([x1, y1, x2, y2])
                        labels_b.append(int(cls_id))

                    if boxes_b:
                        boxes_t = torch.as_tensor(boxes_b, dtype=torch.float32, device=self.estimator.device)
                        labels_t = torch.as_tensor(labels_b, dtype=torch.int64, device=self.estimator.device)
                    else:
                        boxes_t = torch.zeros((0, 4), dtype=torch.float32, device=self.estimator.device)
                        labels_t = torch.zeros((0,), dtype=torch.int64, device=self.estimator.device)
                    labels.append({"boxes": boxes_t, "labels": labels_t})

        if target_class is None and labels:
            for label_entry in labels:
                if label_entry["labels"].numel() > 0:
                    target_class = int(label_entry["labels"][0].item())
                    break

        # Save clean images for overlap loss computation
        clean_images_tensor = None
        if self.overlap_weight > 0.0 and labels is not None:
            clean_images_tensor = images_tensor.clone().detach()
            logger.info(f"Overlap loss enabled (weight={self.overlap_weight})")

        # Training loop
        iterator = trange(self.max_iter, desc="NAP Training", disable=not self.verbose)

        for iteration in iterator:
            self.optimizer.zero_grad()

            # Generate patch from latent
            patch = self._generate_patch_from_latent()
            self.current_patch = patch

            # Apply patch to images
            patched_images = self._apply_patch_to_images(
                images_tensor,
                patch,
                labels=labels,
                target_class=target_class,
            )

            # Compute detection loss and overlap score
            loss_det, overlap_score = self._compute_detection_loss(
                patched_images,
                labels=labels,
                target_class=target_class,
                clean_images=clean_images_tensor,
            )

            # Compute total variation loss
            loss_tv = self.tv_loss_fn(patch)

            # Untargeted attack: maximize detector loss, minimize TV
            loss_attack = -loss_det

            # Overlap loss: maximize detection change (with/without patch)
            loss_overlap = -overlap_score

            # Total loss
            loss = loss_attack + self.tv_weight * loss_tv + self.overlap_weight * loss_overlap

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Log progress
            if iteration % 10 == 0:
                postfix = {
                    'det_loss': f'{loss_det.item():.4f}',
                    'tv_loss': f'{loss_tv.item():.4f}',
                    'attack_loss': f'{loss_attack.item():.4f}',
                }
                if self.overlap_weight > 0.0:
                    postfix['overlap'] = f'{overlap_score.item():.4f}'
                postfix['total'] = f'{loss.item():.4f}'
                iterator.set_postfix(postfix)

                # SSE update if available
                if isinstance(self.summary_writer, SummaryWriter):
                    self.summary_writer.add_scalar('loss/detection', loss_det.item(), iteration)
                    self.summary_writer.add_scalar('loss/attack', loss_attack.item(), iteration)
                    self.summary_writer.add_scalar('loss/tv', loss_tv.item(), iteration)
                    if self.overlap_weight > 0.0:
                        self.summary_writer.add_scalar('loss/overlap', overlap_score.item(), iteration)
                    self.summary_writer.add_scalar('loss/total', loss.item(), iteration)

        logger.info("Patch generation complete")

        # Convert final patch to numpy (keep CHW for patch_service)
        patch_np = self.current_patch.detach().cpu().numpy()
        clip_max = 1.0
        if hasattr(self.estimator, "clip_values") and self.estimator.clip_values is not None:
            clip_max = float(self.estimator.clip_values[1])
        if clip_max > 1.0:
            patch_np = np.clip(patch_np * clip_max, 0, clip_max).astype(np.uint8)
        else:
            patch_np = np.clip(patch_np, 0.0, 1.0).astype(np.float32)

        # Create mask (all ones for full patch), keep CHW
        mask_np = np.ones_like(patch_np)

        return patch_np, mask_np
