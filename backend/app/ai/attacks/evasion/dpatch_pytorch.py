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
PyTorch implementation of the adversarial patch attack `DPatch` for object detectors.

This is a PyTorch-native version that uses DataLoader for efficient batch processing,
while maintaining the same algorithm as the original DPatch.

| Paper link: https://arxiv.org/abs/1806.02299v4
"""
from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from app.ai.attacks.attack import EvasionAttack
from app.ai.estimators.estimator import BaseEstimator, LossGradientsMixin
from app.ai.estimators.object_detection.object_detector import ObjectDetectorMixin
from app.ai import config
from app.ai.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from app.ai.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class DPatchPyTorch(EvasionAttack):
    """
    PyTorch implementation of the DPatch attack.

    External interface: NumPy (ART compatible)
    Internal implementation: PyTorch (for performance and Type 1 DataLoader pattern)

    | Paper link: https://arxiv.org/abs/1806.02299v4
    """

    attack_params = EvasionAttack.attack_params + [
        "patch_shape",
        "learning_rate",
        "max_iter",
        "batch_size",
        "summary_writer",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        patch_shape: tuple[int, int, int] = (40, 40, 3),
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.DPatchPyTorch`.

        :param estimator: A trained object detector.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape (height, width, nb_channels).
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
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

        # Patch is a learnable parameter (C, H, W) or (H, W, C) based on channels_first
        self._patch = nn.Parameter(patch_init)

        self.target_label: int | np.ndarray | list[int] | None = []

        logger.info(f"DPatchPyTorch initialized on device: {self.device}")

    def generate(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        target_label: int | np.ndarray | list[int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate DPatch using PyTorch with DataLoader (Type 1 pattern).

        :param x: Sample images (N, H, W, C) or (N, C, H, W) - NumPy.
        :param y: True labels of type `list[dict[np.ndarray]]` for untargeted attack, one dictionary per input image.
                  The keys and values of the dictionary are:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image
                  - scores [N]: the scores or each prediction.
        :param target_label: The target label of the DPatch attack.
        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :return: Adversarial patch (NumPy array).
        """
        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.copy()

        # Validate mask
        mask = self._check_mask(mask, x)

        # Validate inputs
        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError("The color channel index of the images and the patch have to be identical.")
        if x.ndim != 4:
            raise ValueError("The adversarial patch can only be applied to images.")

        # Process target labels
        if target_label is not None:
            if isinstance(target_label, int):
                self.target_label = [target_label] * x.shape[0]
            elif isinstance(target_label, np.ndarray):
                if target_label.shape not in ((x.shape[0], 1), (x.shape[0],)):
                    raise ValueError("The target_label has to be a 1-dimensional array.")
                self.target_label = target_label.tolist()
            else:
                if not len(target_label) == x.shape[0] or not isinstance(target_label, list):
                    raise ValueError("The target_label as list of integers needs to of length number of images in `x`.")
                self.target_label = target_label

        logger.info(f"Starting DPatch generation with {len(x)} images")

        # Convert to PyTorch tensors
        x_torch, x_original = self._preprocess_and_convert(x)

        # Generate initial patch locations and create dataset
        patch_transforms = self._generate_patch_locations(x_torch, mask)

        # Create pseudo ground truth for training
        patch_target_list = self._create_patch_targets(
            x_torch, patch_transforms, self.target_label, y
        )

        # Train patch using PyTorch DataLoader
        self._train_patch_pytorch(x_torch, patch_transforms, patch_target_list)

        # Convert patch back to NumPy
        patch_numpy = self._patch_to_numpy()

        logger.info("DPatch generation completed")
        return patch_numpy

    def _preprocess_and_convert(self, x: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Preprocess inputs using ART's standard preprocessing."""
        x_torch, _ = self.estimator._preprocess_and_convert_inputs(
            x=x, y=None, fit=False, no_grad=True
        )
        logger.debug(f"Input converted to torch: {x_torch.shape}, device: {x_torch.device}")
        return x_torch, x

    def _generate_patch_locations(
        self,
        x: torch.Tensor,
        mask: np.ndarray | None
    ) -> list[dict[str, int]]:
        """
        Generate random patch locations for each image.

        Args:
            x: Input images (B, C, H, W)
            mask: Optional mask for valid patch locations

        Returns:
            List of dicts with patch coordinates for each image
        """
        transforms = []

        # Get image dimensions
        if self.estimator.channels_first:
            img_h, img_w = x.shape[2], x.shape[3]
            patch_h, patch_w = self.patch_shape[1], self.patch_shape[2]
        else:
            img_h, img_w = x.shape[1], x.shape[2]
            patch_h, patch_w = self.patch_shape[0], self.patch_shape[1]

        for i in range(x.shape[0]):
            if mask is None:
                # Random location
                i_x_1 = random.randint(0, img_h - 1 - patch_h)
                i_y_1 = random.randint(0, img_w - 1 - patch_w)
            else:
                # Use mask to select valid locations
                if mask.shape[0] == 1:
                    mask_2d = mask[0, :, :]
                else:
                    mask_2d = mask[i, :, :]

                edge_x_0 = patch_h // 2
                edge_x_1 = patch_h - edge_x_0
                edge_y_0 = patch_w // 2
                edge_y_1 = patch_w - edge_y_0

                mask_2d = mask_2d.copy()
                mask_2d[0:edge_x_0, :] = False
                mask_2d[-edge_x_1:, :] = False
                mask_2d[:, 0:edge_y_0] = False
                mask_2d[:, -edge_y_1:] = False

                num_pos = np.argwhere(mask_2d).shape[0]
                if num_pos == 0:
                    raise ValueError(f"No valid positions in mask for image {i}")
                pos_id = np.random.choice(num_pos, size=1)
                pos = np.argwhere(mask_2d > 0)[pos_id[0]]
                i_x_1 = pos[0] - edge_x_0
                i_y_1 = pos[1] - edge_y_0

            i_x_2 = i_x_1 + patch_h
            i_y_2 = i_y_1 + patch_w

            transforms.append({
                "i_x_1": i_x_1,
                "i_y_1": i_y_1,
                "i_x_2": i_x_2,
                "i_y_2": i_y_2
            })

        return transforms

    def _create_patch_targets(
        self,
        x: torch.Tensor,
        transforms: list[dict[str, int]],
        target_label: list | None,
        y: list | None
    ) -> list[dict[str, torch.Tensor]]:
        """
        Create target labels for patch training.

        Args:
            x: Input images (B, C, H, W)
            transforms: Patch location transforms
            target_label: Target labels (for targeted attack)
            y: Ground truth labels (for untargeted attack)

        Returns:
            List of target dicts for each image
        """
        patch_target = []

        if target_label and y is None:
            # Targeted attack: use patch location as target box
            for i_image in range(x.shape[0]):
                if isinstance(target_label, int):
                    t_l = target_label
                else:
                    t_l = target_label[i_image]

                i_x_1 = transforms[i_image]["i_x_1"]
                i_x_2 = transforms[i_image]["i_x_2"]
                i_y_1 = transforms[i_image]["i_y_1"]
                i_y_2 = transforms[i_image]["i_y_2"]

                target_dict = {
                    "boxes": torch.tensor([[i_x_1, i_y_1, i_x_2, i_y_2]], dtype=torch.float32, device=self.device),
                    "labels": torch.tensor([t_l], dtype=torch.int64, device=self.device),
                    "scores": torch.tensor([1.0], dtype=torch.float32, device=self.device)
                }
                patch_target.append(target_dict)
        else:
            # Untargeted attack: use predictions or ground truth
            if y is not None:
                predictions = y
            else:
                # Get predictions from model
                x_numpy = x.detach().cpu().numpy()
                if not self.estimator.channels_first:
                    x_numpy = np.transpose(x_numpy, (0, 2, 3, 1))
                if self.estimator.clip_values is not None:
                    x_numpy = x_numpy * self.estimator.clip_values[1]
                predictions = self.estimator.predict(x=x_numpy, standardise_output=True)

            for i_image in range(x.shape[0]):
                target_dict = {
                    "boxes": torch.from_numpy(predictions[i_image]["boxes"]).float().to(self.device),
                    "labels": torch.from_numpy(predictions[i_image]["labels"]).long().to(self.device),
                    "scores": torch.from_numpy(predictions[i_image]["scores"]).float().to(self.device)
                }
                patch_target.append(target_dict)

        return patch_target

    def _train_patch_pytorch(
        self,
        x: torch.Tensor,
        transforms: list[dict[str, int]],
        patch_targets: list[dict[str, torch.Tensor]]
    ):
        """
        Train patch using PyTorch with DataLoader (Type 1 pattern).

        Args:
            x: Input images (B, C, H, W)
            transforms: Patch location transforms
            patch_targets: Target labels for each image
        """
        logger.info(f"Training DPatch for {self.max_iter} iterations")

        # Create custom dataset
        class PatchDataset(torch.utils.data.Dataset):
            def __init__(self, images, transforms_list, targets):
                self.images = images
                self.transforms_list = transforms_list
                self.targets = targets

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return self.images[idx], self.transforms_list[idx], self.targets[idx]

        # Create DataLoader (Type 1 pattern)
        dataset = PatchDataset(x, transforms, patch_targets)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        logger.info(f"Created DataLoader: {len(dataset)} samples, {len(data_loader)} batches")

        # Training loop
        for i_step in trange(self.max_iter, desc="DPatch Training", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info(f"Training Step: {i_step + 1}")

            patch_gradients = torch.zeros_like(self._patch)

            # Process batches using DataLoader (Type 1)
            for batch_images, batch_transforms, batch_targets in data_loader:
                batch_images = batch_images.to(self.device)

                # Apply patch to images
                patched_images = self._apply_patch_to_batch(
                    batch_images, batch_transforms
                )

                # Compute loss and gradients
                loss = self._compute_loss(patched_images, batch_targets)

                # Backward pass
                loss.backward()

                # Extract patch gradients from image gradients
                # Accumulate gradients for patch update
                if self._patch.grad is not None:
                    patch_gradients += self._patch.grad.data
                    self._patch.grad.zero_()

            # Update patch with sign gradient
            with torch.no_grad():
                if self.target_label:
                    self._patch.data -= torch.sign(patch_gradients) * self.learning_rate
                else:
                    self._patch.data += torch.sign(patch_gradients) * self.learning_rate

                # Clip patch values
                if self.estimator.clip_values is not None:
                    self._patch.data.clamp_(
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1]
                    )

            # Write summary
            if self.summary_writer is not None and i_step % 10 == 0:
                self.summary_writer.update(
                    batch_id=0,
                    global_step=i_step,
                    grad=patch_gradients.cpu().numpy(),
                    patch=self._patch.detach().cpu().numpy(),
                    estimator=self.estimator,
                    x=None,
                    y=None,
                    targeted=self.target_label is not None,
                )

        if self.summary_writer is not None:
            self.summary_writer.reset()

    def _apply_patch_to_batch(
        self,
        images: torch.Tensor,
        transforms: list[dict[str, int]]
    ) -> torch.Tensor:
        """
        Apply patch to a batch of images.

        Args:
            images: Batch of images (B, C, H, W) or (B, H, W, C)
            transforms: Patch locations for each image

        Returns:
            Patched images
        """
        patched = images.clone()

        for i, trans in enumerate(transforms):
            i_x_1 = trans["i_x_1"]
            i_x_2 = trans["i_x_2"]
            i_y_1 = trans["i_y_1"]
            i_y_2 = trans["i_y_2"]

            if self.estimator.channels_first:
                patched[i, :, i_x_1:i_x_2, i_y_1:i_y_2] = self._patch
            else:
                patched[i, i_x_1:i_x_2, i_y_1:i_y_2, :] = self._patch

        return patched

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

        # Handle channels_first/last
        if self.estimator.channels_first:
            # (C, H, W) -> (H, W, C)
            patch_np = np.transpose(patch_np, (1, 2, 0))

        return patch_np.astype(np.float32)

    def _check_mask(self, mask: np.ndarray | None, x: np.ndarray) -> np.ndarray | None:
        """Validate mask shape."""
        if mask is not None:
            i_h = 1 if self.estimator.channels_first else 0
            i_w = 2 if self.estimator.channels_first else 1

            if (
                mask.dtype != bool
                or not (mask.shape[0] == 1 or mask.shape[0] == x.shape[0])
                or not (mask.shape[1] == x.shape[i_h + 1] and mask.shape[2] == x.shape[i_w + 1])
            ):
                raise ValueError(
                    "The shape of `mask` has to be equal to the shape of a single samples (1, H, W) or the"
                    "shape of `x` (N, H, W) without their channel dimensions."
                )

            if mask.shape[0] == 1:
                mask = np.repeat(mask, repeats=x.shape[0], axis=0)

        return mask

    def apply_patch(
        self,
        x: np.ndarray,
        patch_external: np.ndarray | None = None,
        random_location: bool = False,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched (NumPy).
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :param random_location: True if patch location should be random.
        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched images (NumPy).
        """
        # Convert to PyTorch
        x_torch, _ = self._preprocess_and_convert(x)

        # Use external patch if provided
        if patch_external is not None:
            patch_torch = torch.from_numpy(patch_external).float().to(self.device)
            if not self.estimator.channels_first:
                # (H, W, C) -> (C, H, W)
                patch_torch = patch_torch.permute(2, 0, 1)
        else:
            patch_torch = self._patch

        # Generate patch locations
        transforms = self._generate_patch_locations(x_torch, mask)

        # Temporarily replace patch
        original_patch = self._patch
        self._patch = nn.Parameter(patch_torch)

        # Apply patch
        patched_torch = self._apply_patch_to_batch(x_torch, transforms)

        # Restore original patch
        self._patch = original_patch

        # Convert back to NumPy
        patched_numpy = patched_torch.detach().cpu().numpy()
        if not self.estimator.channels_first:
            patched_numpy = np.transpose(patched_numpy, (0, 2, 3, 1))
        if self.estimator.clip_values is not None:
            patched_numpy = patched_numpy * self.estimator.clip_values[1]

        return patched_numpy.astype(np.float32)

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
