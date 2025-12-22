"""
Naturalistic Adversarial Patch Attack using BigGAN.

This module implements GAN-based adversarial patch generation for object detectors,
producing naturalistic-looking patches through latent space optimization.

| Paper link: Naturalistic Physical Adversarial Patch (Int'l Journal of Information Security 2025)
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
import numpy as np
from typing import Optional, TYPE_CHECKING, Tuple
import torch
import torch.nn as nn
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
        "alpha_latent",
        "patch_scale",
        "learning_rate",
        "max_iter",
        "batch_size",
        "max_latent_value",
        "enable_deformator",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(
        self,
        estimator: "PYTORCH_OBJECT_DETECTOR_TYPE",
        gan_class_id: int = 259,  # Pomeranian
        tv_weight: float = 0.0,
        alpha_latent: float = 1.0,
        patch_scale: float = 0.2,
        learning_rate: float = 0.02,
        max_iter: int = 1000,
        batch_size: int = 8,
        max_latent_value: float = 8.0,
        enable_deformator: bool = False,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
    ):
        """
        Create an instance of NaturalisticPatchPyTorch.

        Args:
            estimator: A trained PyTorch object detector (YOLO, RT-DETR, etc.)
            gan_class_id: ImageNet class ID for BigGAN generation (default: 259 = Pomeranian)
            tv_weight: Total variation loss weight (0.0-0.1, default: 0.0)
            alpha_latent: Latent space mixing factor (0.0-1.0, default: 1.0)
                         1.0 = optimize shift only, 0.0 = optimize from random latent
            patch_scale: Patch scale relative to detected person bbox (default: 0.2)
            learning_rate: Learning rate for Adam optimizer (default: 0.02)
            max_iter: Maximum optimization iterations (default: 1000)
            batch_size: Batch size for patch generation (default: 8)
            max_latent_value: Maximum absolute value for latent clamping (default: 8.0)
            enable_deformator: Use deformator for latent transformation (default: False)
            summary_writer: SummaryWriter for logging or SSE updates
            verbose: Show progress bar
        """
        super().__init__(estimator=estimator, summary_writer=summary_writer)

        self.gan_class_id = gan_class_id
        self.tv_weight = tv_weight
        self.alpha_latent = alpha_latent
        self.patch_scale = patch_scale
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.max_latent_value = max_latent_value
        self.enable_deformator = enable_deformator
        self.verbose = verbose

        # Ensure GAN weights are downloaded
        logger.info("Ensuring BigGAN weights are available...")
        weights_path = gan_weights_service.ensure_biggan_weights()
        logger.info(f"BigGAN weights ready at {weights_path}")

        # Load BigGAN generator
        logger.info(f"Loading BigGAN for class {gan_class_id}...")
        self.generator, self.deformator = load_biggan(
            weights_path=str(weights_path),
            target_class=gan_class_id,
            device=self.estimator.device.type,
            deformator_path=None,  # TODO: Add deformator support
        )

        # Initialize latent vectors
        self.dim_z = self.generator.dim_z
        self.fixed_latent = torch.randn(self.dim_z).to(self.estimator.device)
        self.latent_shift = nn.Parameter(
            torch.randn(self.dim_z).to(self.estimator.device),
            requires_grad=True
        )

        # Optimizer
        self.optimizer = torch.optim.Adam([self.latent_shift], lr=self.learning_rate)

        # Total variation loss
        self.tv_loss_fn = TotalVariation()

        # Current patch
        self.current_patch = None

        logger.info(
            f"NaturalisticPatchPyTorch initialized (class={gan_class_id}, "
            f"tv_weight={tv_weight}, alpha={alpha_latent})"
        )

    def _generate_patch_from_latent(self) -> torch.Tensor:
        """
        Generate patch image from current latent vectors.

        Returns:
            Generated patch tensor (shape: [3, H, W]) in range [0, 1]
        """
        # Clamp latent shift to prevent extreme values
        self.latent_shift.data = torch.clamp(
            self.latent_shift.data,
            -self.max_latent_value,
            self.max_latent_value
        )

        # Mix fixed latent with optimized shift
        if self.alpha_latent > 0:
            # Start from fixed latent and add shift
            z = (1 - self.alpha_latent) * torch.randn(
                self.dim_z, device=self.estimator.device
            ) + self.alpha_latent * self.fixed_latent
        else:
            # Start from random latent
            z = torch.randn(self.dim_z, device=self.estimator.device)

        # Generate image using BigGAN
        with torch.no_grad():
            # For deformator path (NAP's approach)
            if self.enable_deformator and self.deformator is not None:
                # Apply deformator to shift
                shift = self.latent_shift.unsqueeze(0)
                patch = generate_from_latent(
                    self.generator,
                    z.unsqueeze(0),
                    shift=shift,
                    deformator=self.deformator
                )
            else:
                # Direct shift application
                z_shifted = z + self.latent_shift
                patch = self.generator(z_shifted.unsqueeze(0))

        # Convert from [-1, 1] to [0, 1] range
        patch = (patch + 1) * 0.5
        patch = torch.clamp(patch, 0, 1)

        return patch[0]  # Remove batch dimension

    def _compute_detection_loss(
        self,
        patched_images: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """
        Compute detection loss from patched images.

        For naturalistic patches, we want to MAXIMIZE detection confidence
        (higher confidence = better patch).

        Args:
            patched_images: Patched image batch (shape: [B, C, H, W])
            target_class: Target class ID (e.g., 0 for person in COCO)

        Returns:
            Detection loss (mean confidence)
        """
        # Run detector
        predictions = self.estimator.model(patched_images, verbose=False)

        # Extract confidence scores for target class
        confidences = []
        for pred in predictions:
            boxes = pred.boxes
            for box in boxes:
                if box.cls.cpu().item() == target_class:
                    confidences.append(box.conf.cpu())

        if len(confidences) > 0:
            loss_det = torch.mean(torch.stack(confidences)).to(self.estimator.device)
        else:
            # No detections - worst case for attack
            loss_det = torch.tensor(1.0).to(self.estimator.device)

        return loss_det

    def _apply_patch_to_images(
        self,
        images: torch.Tensor,
        patch: torch.Tensor,
        labels: Optional[np.ndarray] = None  # noqa: ARG002 - Reserved for future bbox-based placement
    ) -> torch.Tensor:
        """
        Apply patch to images at detected object locations.

        Args:
            images: Image batch (shape: [B, C, H, W])
            patch: Patch to apply (shape: [C, H_patch, W_patch])
            labels: Optional bbox labels (shape: [B, max_objects, 5])
                   Format: [x_center, y_center, width, height, class]

        Returns:
            Patched images
        """
        batch_size, _C, H, W = images.shape
        patched_images = images.clone()

        # For simplicity, apply patch to center of image
        # TODO: Implement bbox-based placement using labels
        patch_h, patch_w = patch.shape[1:]

        # Resize patch to scale relative to image
        target_size = int(min(H, W) * self.patch_scale)
        if target_size != patch_h or target_size != patch_w:
            patch_resized = torch.nn.functional.interpolate(
                patch.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )[0]
        else:
            patch_resized = patch

        # Apply to center of each image
        y_start = (H - target_size) // 2
        x_start = (W - target_size) // 2
        y_end = y_start + target_size
        x_end = x_start + target_size

        for b in range(batch_size):
            patched_images[b, :, y_start:y_end, x_start:x_end] = patch_resized

        return patched_images

    def generate(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate naturalistic adversarial patch.

        Args:
            x: Sample images (shape: [B, H, W, C]) in range [0, 255]
            y: Optional labels (shape: [B, max_objects, 5])
               Format: [x_center, y_center, width, height, class]
            **kwargs: Additional arguments

        Returns:
            Tuple of (patch, mask)
            - patch: Generated adversarial patch (shape: [H, W, C])
            - mask: Patch mask (shape: [H, W, C])
        """
        logger.info(
            f"Generating naturalistic patch: {self.max_iter} iterations, "
            f"class_id={self.gan_class_id}, tv_weight={self.tv_weight}"
        )

        # Convert images to torch tensors
        images_np = x.astype(np.float32) / 255.0
        images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).to(self.estimator.device)

        # Extract target class from labels (default to 0 = person)
        target_class = 0  # COCO person class
        if y is not None and len(y) > 0:
            target_class = int(y[0, 0, 4]) if y.shape[1] > 0 else 0

        # Training loop
        iterator = trange(self.max_iter, desc="NAP Training", disable=not self.verbose)

        for iteration in iterator:
            self.optimizer.zero_grad()

            # Generate patch from latent
            patch = self._generate_patch_from_latent()
            self.current_patch = patch

            # Apply patch to images
            patched_images = self._apply_patch_to_images(images_tensor, patch, y)

            # Compute detection loss
            loss_det = self._compute_detection_loss(patched_images, target_class)

            # Compute total variation loss
            loss_tv = self.tv_loss_fn(patch)

            # Total loss
            loss = loss_det + self.tv_weight * loss_tv

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Log progress
            if iteration % 10 == 0:
                iterator.set_postfix({
                    'det_loss': f'{loss_det.item():.4f}',
                    'tv_loss': f'{loss_tv.item():.4f}',
                    'total': f'{loss.item():.4f}'
                })

                # SSE update if available
                if isinstance(self.summary_writer, SummaryWriter):
                    self.summary_writer.add_scalar('loss/detection', loss_det.item(), iteration)
                    self.summary_writer.add_scalar('loss/tv', loss_tv.item(), iteration)
                    self.summary_writer.add_scalar('loss/total', loss.item(), iteration)

        logger.info("Patch generation complete")

        # Convert final patch to numpy
        patch_np = self.current_patch.detach().cpu().permute(1, 2, 0).numpy()
        patch_np = (patch_np * 255).astype(np.uint8)

        # Create mask (all ones for full patch)
        mask_np = np.ones_like(patch_np)

        return patch_np, mask_np
