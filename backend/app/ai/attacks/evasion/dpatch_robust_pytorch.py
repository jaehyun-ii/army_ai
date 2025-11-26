# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
PyTorch implementation of a variation of the adversarial patch attack `DPatch` for object detectors.

This version follows Lee & Kolter (2019) in using sign gradients with expectations over transformations.
The particular transformations supported in this implementation are cropping, rotations by multiples of 90 degrees,
and changes in the brightness of the image.

Uses PyTorch DataLoader for efficient batch processing (Type 1 pattern).

| Paper link (original DPatch): https://arxiv.org/abs/1806.02299v4
| Paper link (physical-world patch from Lee & Kolter): https://arxiv.org/abs/1906.11897
"""
from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

from app.ai.attacks.attack import EvasionAttack
from app.ai.estimators.estimator import BaseEstimator, LossGradientsMixin
from app.ai.estimators.object_detection.object_detector import ObjectDetectorMixin
from app.ai import config
from app.ai.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from app.ai.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class RobustDPatchPyTorch(EvasionAttack):
    """
    PyTorch implementation of RobustDPatch with Expectation over Transformations (EoT).

    External interface: NumPy (ART compatible)
    Internal implementation: PyTorch (for performance and Type 1 DataLoader pattern)

    | Paper link (original DPatch): https://arxiv.org/abs/1806.02299v4
    | Paper link (physical-world patch): https://arxiv.org/abs/1906.11897
    """

    attack_params = EvasionAttack.attack_params + [
        "patch_shape",
        "learning_rate",
        "max_iter",
        "batch_size",
        "patch_location",
        "crop_range",
        "brightness_range",
        "rotation_weights",
        "sample_size",
        "targeted",
        "summary_writer",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        patch_shape: tuple[int, int, int] = (40, 40, 3),
        patch_location: tuple[int, int] = (0, 0),
        crop_range: tuple[int, int] = (0, 0),
        brightness_range: tuple[float, float] = (1.0, 1.0),
        rotation_weights: tuple[float, float, float, float] | tuple[int, int, int, int] = (1, 0, 0, 0),
        sample_size: int = 1,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        targeted: bool = False,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.RobustDPatchPyTorch`.

        :param estimator: A trained object detector.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape (height, width, nb_channels).
        :param patch_location: The location of the adversarial patch as a tuple of shape (upper left x, upper left y).
        :param crop_range: By how much the images may be cropped as a tuple of shape (height, width).
        :param brightness_range: Range for randomly adjusting the brightness of the image.
        :param rotation_weights: Sampling weights for random image rotations by (0, 90, 180, 270) degrees
                                 counter-clockwise.
        :param sample_size: Number of samples to be used in expectations over transformation (EoT).
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
        :param verbose: Show progress bars.
        """

        super().__init__(estimator=estimator, summary_writer=summary_writer)

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.patch_location = patch_location
        self.crop_range = crop_range
        self.brightness_range = brightness_range
        self.rotation_weights = rotation_weights
        self.sample_size = sample_size
        self._targeted = targeted
        self._check_params()

        # Device management
        self.device = self.estimator.device
        self._torch_model = self.estimator.model

        # Initialize patch as PyTorch parameter
        if self.estimator.clip_values is None:
            patch_init = torch.zeros(patch_shape, dtype=torch.float32, device=self.device)
        else:
            patch_init = (
                torch.randint(0, 256, size=patch_shape, device=self.device).float()
                / 255
                * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                + self.estimator.clip_values[0]
            )

        self._patch = nn.Parameter(patch_init)

        logger.info(f"RobustDPatchPyTorch initialized on device: {self.device}")

    def generate(
        self,
        x: np.ndarray,
        y: list[dict[str, np.ndarray]] | None = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate RobustDPatch using PyTorch with DataLoader (Type 1 pattern).

        :param x: Sample images (N, H, W, C) or (N, C, H, W) - NumPy.
        :param y: Target labels for object detector.
        :return: Adversarial patch (NumPy array).
        """
        # Validate inputs
        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError("The color channel index of the images and the patch have to be identical.")
        if y is None and self.targeted:
            raise ValueError("The targeted version of RobustDPatch attack requires target labels provided to `y`.")
        if y is not None and not self.targeted:
            raise ValueError("The untargeted version of RobustDPatch attack does not use True labels provided to 'y'.")
        if x.ndim != 4:
            raise ValueError("The adversarial patch can only be applied to images.")

        # Check whether patch fits into the cropped images
        if self.estimator.channels_first:
            image_height, image_width = x.shape[2:4]
        else:
            image_height, image_width = x.shape[1:3]

        # Convert labels to PyTorch format if needed
        if not self.estimator.native_label_is_pytorch_format and y is not None:
            from app.ai.estimators.object_detection.utils import convert_tf_to_pt
            y = convert_tf_to_pt(y=y, height=x.shape[1], width=x.shape[2])

        # Validate patch location
        if y is not None:
            for i_image in range(x.shape[0]):
                y_i = y[i_image]["boxes"]
                for i_box in range(y_i.shape[0]):
                    x_1, y_1, x_2, y_2 = y_i[i_box]
                    if (
                        x_1 < self.crop_range[1]
                        or y_1 < self.crop_range[0]
                        or x_2 > image_width - self.crop_range[1] + 1
                        or y_2 > image_height - self.crop_range[0] + 1
                    ):
                        raise ValueError("Cropping is intersecting with at least one box, reduce `crop_range`.")

        if (
            self.patch_location[0] + self.patch_shape[0] > image_height - self.crop_range[0]
            or self.patch_location[1] + self.patch_shape[1] > image_width - self.crop_range[1]
        ):
            raise ValueError("The patch (partially) lies outside the cropped image.")

        logger.info(f"Starting RobustDPatch generation with {len(x)} images, EoT sample_size={self.sample_size}")

        # Convert to PyTorch tensors
        x_torch, _ = self._preprocess_and_convert(x)

        # Convert labels to PyTorch if provided
        if y is not None:
            y_torch = self._labels_to_torch(y)
        else:
            y_torch = None

        # Train patch using PyTorch DataLoader with EoT
        self._train_patch_pytorch(x_torch, y_torch)

        # Convert patch back to NumPy
        patch_numpy = self._patch_to_numpy()

        logger.info("RobustDPatch generation completed")
        return patch_numpy

    def _preprocess_and_convert(self, x: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Preprocess inputs using ART's standard preprocessing."""
        x_torch, _ = self.estimator._preprocess_and_convert_inputs(
            x=x, y=None, fit=False, no_grad=True
        )
        logger.debug(f"Input converted to torch: {x_torch.shape}, device: {x_torch.device}")
        return x_torch, x

    def _labels_to_torch(self, y: list[dict[str, np.ndarray]]) -> list[dict[str, torch.Tensor]]:
        """Convert NumPy labels to PyTorch tensors."""
        y_torch = []
        for label in y:
            label_torch = {
                'boxes': torch.from_numpy(label['boxes']).float().to(self.device),
                'labels': torch.from_numpy(label['labels']).long().to(self.device),
                'scores': torch.from_numpy(label['scores']).float().to(self.device)
            }
            y_torch.append(label_torch)
        return y_torch

    def _train_patch_pytorch(
        self,
        x: torch.Tensor,
        y: list[dict[str, torch.Tensor]] | None
    ):
        """
        Train patch using PyTorch with DataLoader and EoT (Type 1 pattern).

        Args:
            x: Input images (B, C, H, W)
            y: Target labels (for targeted attack) or None (for untargeted)
        """
        logger.info(f"Training RobustDPatch for {self.max_iter} iterations")

        # Create custom dataset
        class RobustPatchDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                if self.labels is not None:
                    return self.images[idx], self.labels[idx]
                else:
                    return self.images[idx], None

        # Create DataLoader (Type 1 pattern)
        dataset = RobustPatchDataset(x, y)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        logger.info(f"Created DataLoader: {len(dataset)} samples, {len(data_loader)} batches")

        # Training loop with EoT
        for i_step in trange(self.max_iter, desc="RobustDPatch Training", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info(f"Training Step: {i_step + 1}")

            patch_gradients_old = torch.zeros_like(self._patch)

            # EoT: Multiple transformation samples
            for e_step in range(self.sample_size):
                if e_step == 0 or (e_step + 1) % 100 == 0:
                    logger.debug(f"EoT Step: {e_step + 1}")

                # Process batches using DataLoader (Type 1)
                for batch_images, batch_labels in data_loader:
                    batch_images = batch_images.to(self.device)

                    # Sample and apply random transformations
                    patched_images, patch_targets, transforms = self._augment_images_with_patch(
                        batch_images, batch_labels, self._patch
                    )

                    # Compute loss and gradients
                    loss = self._compute_loss(patched_images, patch_targets)

                    # Backward pass
                    loss.backward()

                    # Reverse transformations on gradients
                    if self._patch.grad is not None:
                        # The gradients are already computed, we just need to untransform them
                        gradients_untransformed = self._untransform_gradients(
                            self._patch.grad.data, transforms
                        )

                        # Accumulate gradients
                        patch_gradients = patch_gradients_old + gradients_untransformed
                        logger.debug(
                            f"Gradient percentage diff: {torch.mean((torch.sign(patch_gradients) != torch.sign(patch_gradients_old)).float()).item():.4f}"
                        )
                        patch_gradients_old = patch_gradients

                        # Zero gradients for next iteration
                        self._patch.grad.zero_()

            # Write summary
            if self.summary_writer is not None and i_step % 10 == 0:
                # Create sample patched images for visualization
                with torch.no_grad():
                    x_patched, y_patched, _ = self._augment_images_with_patch(
                        x, y, self._patch
                    )
                self.summary_writer.update(
                    batch_id=0,
                    global_step=i_step,
                    grad=patch_gradients_old.cpu().numpy(),
                    patch=self._patch.detach().cpu().numpy(),
                    estimator=self.estimator,
                    x=x_patched.cpu().numpy(),
                    y=y_patched,
                    targeted=self.targeted,
                )

            # Update patch with sign gradient
            with torch.no_grad():
                self._patch.data += torch.sign(patch_gradients_old) * (1 - 2 * int(self.targeted)) * self.learning_rate

                # Clip patch values
                if self.estimator.clip_values is not None:
                    self._patch.data.clamp_(
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1]
                    )

        if self.summary_writer is not None:
            self.summary_writer.reset()

    def _augment_images_with_patch(
        self,
        x: torch.Tensor,
        y: list[dict[str, torch.Tensor]] | torch.Tensor | None,
        patch: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], dict[str, float | int]]:
        """
        Apply patch to images with random transformations (crop, rotate, brightness).

        Args:
            x: Input images (B, C, H, W)
            y: Target labels (list of dicts or None)
            patch: Patch to apply

        Returns:
            Tuple of (patched images, patch targets, transformations)
        """
        transformations: dict[str, float | int] = {}
        x_patch = x.clone()

        # Handle channels_first/last
        if self.estimator.channels_first:
            # (B, C, H, W) -> (B, H, W, C) for easier processing
            x_patch = x_patch.permute(0, 2, 3, 1)
            patch_copy = patch.permute(1, 2, 0) if patch.dim() == 3 else patch
        else:
            patch_copy = patch

        # Apply patch at specified location
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_copy.shape[0], y_1 + patch_copy.shape[1]
        x_patch[:, x_1:x_2, y_1:y_2, :] = patch_copy

        # 1) Crop images
        crop_x = random.randint(0, self.crop_range[0])
        crop_y = random.randint(0, self.crop_range[1])
        x_1_crop, y_1_crop = crop_x, crop_y
        x_2_crop = x_patch.shape[1] - crop_x
        y_2_crop = x_patch.shape[2] - crop_y
        x_patch = x_patch[:, x_1_crop:x_2_crop, y_1_crop:y_2_crop, :]

        transformations.update({"crop_x": crop_x, "crop_y": crop_y})

        # 2) Rotate images
        rot90 = random.choices([0, 1, 2, 3], weights=self.rotation_weights)[0]
        x_patch = torch.rot90(x_patch, k=rot90, dims=(1, 2))

        transformations.update({"rot90": rot90})

        # Transform labels if provided (for targeted attack)
        if y is not None and self.targeted:
            y_copy = []
            image_width = x.shape[2] if self.estimator.channels_first else x.shape[2]
            image_height = x.shape[1] if self.estimator.channels_first else x.shape[1]

            for i_image in range(x_patch.shape[0]):
                if isinstance(y, list):
                    y_b = y[i_image]["boxes"].clone()
                    labels = y[i_image]["labels"]
                    scores = y[i_image]["scores"]
                else:
                    # Handle batch tensor case
                    continue

                # Apply rotation transformation to boxes
                x_1_arr = y_b[:, 0]
                y_1_arr = y_b[:, 1]
                x_2_arr = y_b[:, 2]
                y_2_arr = y_b[:, 3]
                box_width = x_2_arr - x_1_arr
                box_height = y_2_arr - y_1_arr

                if rot90 == 0:
                    x_1_new, y_1_new = x_1_arr, y_1_arr
                    x_2_new, y_2_new = x_2_arr, y_2_arr
                elif rot90 == 1:
                    x_1_new = y_1_arr
                    y_1_new = image_width - x_1_arr - box_width
                    x_2_new = y_1_arr + box_height
                    y_2_new = image_width - x_1_arr
                elif rot90 == 2:
                    x_1_new = image_width - x_2_arr
                    y_1_new = image_height - y_2_arr
                    x_2_new = x_1_new + box_width
                    y_2_new = y_1_new + box_height
                else:  # rot90 == 3
                    x_1_new = image_height - y_1_arr - box_height
                    y_1_new = x_1_arr
                    x_2_new = image_height - y_1_arr
                    y_2_new = x_1_arr + box_width

                y_i = {
                    "boxes": torch.stack([x_1_new, y_1_new, x_2_new, y_2_new], dim=1),
                    "labels": labels,
                    "scores": scores
                }
                y_copy.append(y_i)
        else:
            y_copy = None

        # 3) Adjust brightness
        brightness = random.uniform(*self.brightness_range)
        x_patch = torch.round(brightness * x_patch / self.learning_rate) * self.learning_rate

        transformations.update({"brightness": brightness})

        logger.debug(f"Transformations: {transformations}")

        # Create patch targets
        if self.targeted:
            patch_targets = y_copy
        else:
            # For untargeted attack, get predictions
            # Convert back to channels_first if needed
            if self.estimator.channels_first:
                x_patch_pred = x_patch.permute(0, 3, 1, 2)
            else:
                x_patch_pred = x_patch

            # Get predictions
            x_patch_numpy = x_patch_pred.detach().cpu().numpy()
            if not self.estimator.channels_first:
                x_patch_numpy = np.transpose(x_patch_numpy, (0, 2, 3, 1))
            if self.estimator.clip_values is not None:
                x_patch_numpy = x_patch_numpy * self.estimator.clip_values[1]

            predictions = self.estimator.predict(x=x_patch_numpy, standardise_output=True)

            patch_targets = []
            for i_image in range(x_patch.shape[0]):
                target_dict = {
                    "boxes": torch.from_numpy(predictions[i_image]["boxes"]).float().to(self.device),
                    "labels": torch.from_numpy(predictions[i_image]["labels"]).long().to(self.device),
                    "scores": torch.from_numpy(predictions[i_image]["scores"]).float().to(self.device)
                }
                patch_targets.append(target_dict)

        # Convert back to channels_first if needed
        if self.estimator.channels_first:
            x_patch = x_patch.permute(0, 3, 1, 2)

        return x_patch, patch_targets, transformations

    def _untransform_gradients(
        self,
        gradients: torch.Tensor,
        transforms: dict[str, int | float]
    ) -> torch.Tensor:
        """
        Revert transformation on gradients.

        Args:
            gradients: The gradients to be reverse transformed (same shape as patch)
            transforms: The transformations in forward direction

        Returns:
            Reverse-transformed gradients
        """
        # This is a simplified version - in practice, gradients flow through
        # the patch directly, so we mainly need to account for brightness scaling
        gradients_transformed = gradients.clone()

        # Account for brightness adjustment
        gradients_transformed = transforms["brightness"] * gradients_transformed

        return gradients_transformed

    def _compute_loss(
        self,
        x: torch.Tensor,
        targets: list[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute detection loss for patch optimization.

        Args:
            x: Patched images (B, C, H, W)
            targets: Target labels

        Returns:
            Loss value
        """
        # Use estimator's compute_loss method
        loss = self.estimator.compute_loss(x=x, y=targets)
        return loss

    def _patch_to_numpy(self) -> np.ndarray:
        """Convert patch from PyTorch to NumPy."""
        patch_np = self._patch.detach().cpu().numpy()
        return patch_np.astype(np.float32)

    def apply_patch(
        self,
        x: np.ndarray,
        patch_external: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched (NumPy).
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :return: The patched images (NumPy).
        """
        # Convert to PyTorch
        x_torch, _ = self._preprocess_and_convert(x)

        # Use external patch if provided
        if patch_external is not None:
            patch_torch = torch.from_numpy(patch_external).float().to(self.device)
        else:
            patch_torch = self._patch

        # Apply patch (without transformations)
        x_patch = x_torch.clone()

        # Handle channels_first/last
        if self.estimator.channels_first:
            x_patch = x_patch.permute(0, 2, 3, 1)
            if patch_torch.dim() == 3:
                patch_local = patch_torch.permute(1, 2, 0)
            else:
                patch_local = patch_torch
        else:
            patch_local = patch_torch

        # Apply patch at specified location
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_local.shape[0], y_1 + patch_local.shape[1]

        if x_2 > x_patch.shape[1] or y_2 > x_patch.shape[2]:
            raise ValueError("The patch (partially) lies outside the image.")

        x_patch[:, x_1:x_2, y_1:y_2, :] = patch_local

        # Convert back to channels_first if needed
        if self.estimator.channels_first:
            x_patch = x_patch.permute(0, 3, 1, 2)

        # Convert back to NumPy
        patched_numpy = x_patch.detach().cpu().numpy()
        if not self.estimator.channels_first:
            patched_numpy = np.transpose(patched_numpy, (0, 2, 3, 1))
        if self.estimator.clip_values is not None:
            patched_numpy = patched_numpy * self.estimator.clip_values[1]

        return patched_numpy.astype(np.float32)

    @property
    def targeted(self) -> bool:
        """Return whether the attack is targeted."""
        return self._targeted

    def _check_params(self) -> None:
        """Validate parameters."""
        if not isinstance(self.patch_shape, (tuple, list)) or not all(isinstance(s, int) for s in self.patch_shape):
            raise ValueError("The patch shape must be either a tuple or list of integers.")
        if len(self.patch_shape) != 3:
            raise ValueError("The length of patch shape must be 3.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if self.learning_rate <= 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The number of optimization steps must be of type int.")
        if self.max_iter <= 0:
            raise ValueError("The number of optimization steps must be greater than 0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if self.batch_size <= 0:
            raise ValueError("The batch size must be greater than 0.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

        if not isinstance(self.patch_location, (tuple, list)) or not all(
            isinstance(s, int) for s in self.patch_location
        ):
            raise ValueError("The patch location must be either a tuple or list of integers.")
        if len(self.patch_location) != 2:
            raise ValueError("The length of patch location must be 2.")

        if not isinstance(self.crop_range, (tuple, list)) or not all(isinstance(s, int) for s in self.crop_range):
            raise ValueError("The crop range must be either a tuple or list of integers.")
        if len(self.crop_range) != 2:
            raise ValueError("The length of crop range must be 2.")

        if self.crop_range[0] > self.crop_range[1]:
            raise ValueError("The first element of the crop range must be less or equal to the second one.")

        if self.patch_location[0] < self.crop_range[0] or self.patch_location[1] < self.crop_range[1]:
            raise ValueError("The patch location must be outside the crop range.")

        if not isinstance(self.brightness_range, (tuple, list)) or not all(
            isinstance(s, float) for s in self.brightness_range
        ):
            raise ValueError("The brightness range must be either a tuple or list of floats.")
        if len(self.brightness_range) != 2:
            raise ValueError("The length of brightness range must be 2.")

        if self.brightness_range[0] < 0.0:
            raise ValueError("The brightness range must be >= 0.0.")

        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError("The first element of the brightness range must be less or equal to the second one.")

        if not isinstance(self.rotation_weights, (tuple, list)) or not all(
            isinstance(s, (float, int)) for s in self.rotation_weights
        ):
            raise ValueError("The rotation sampling weights must be provided as tuple or list of float or int values.")
        if len(self.rotation_weights) != 4:
            raise ValueError("The number of rotation sampling weights must be 4.")

        if not all(s >= 0.0 for s in self.rotation_weights):
            raise ValueError("The rotation sampling weights must be non-negative.")

        if all(s == 0.0 for s in self.rotation_weights):
            raise ValueError("At least one of the rotation sampling weights must be strictly greater than zero.")

        if not isinstance(self.sample_size, int):
            raise ValueError("The EOT sample size must be of type int.")
        if self.sample_size <= 0:
            raise ValueError("The EOT sample size must be greater than 0.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The argument `targeted` has to be of type bool.")
