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
        :param verbose: Show progress bars.
        """

        super().__init__(estimator=estimator, summary_writer=summary_writer)

        # Store patch_shape as (H, W, C) for API compatibility with original RobustDPatch
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

        # Device management (consistent with other PyTorch attacks)
        self.device = self.estimator.device
        self._torch_model = self.estimator.model

        # Initialize patch as PyTorch parameter in (C, H, W) format (PyTorch standard)
        # Convert from (H, W, C) API to (C, H, W) internal representation
        patch_shape_internal = (patch_shape[2], patch_shape[0], patch_shape[1])  # (C, H, W)

        if self.estimator.clip_values is None:
            patch_init = torch.zeros(patch_shape_internal, dtype=torch.float32, device=self.device)
        else:
            patch_init = (
                torch.randint(0, 256, size=patch_shape_internal, device=self.device).float()
                / 255
                * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                + self.estimator.clip_values[0]
            )

        # Patch is a learnable parameter in (C, H, W) format (PyTorch standard)
        self._patch = nn.Parameter(patch_init)

        logger.info(f"RobustDPatchPyTorch initialized on device: {self.device}, patch shape (C,H,W): {self._patch.shape}")

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
        # patch_shape is always (H, W, C) format, so channel is at index 2
        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[2]:
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
        # Only validate crop_range if it's not (0, 0) - no cropping means no intersection issues
        if y is not None and (self.crop_range[0] > 0 or self.crop_range[1] > 0):
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
        Train patch using PyTorch with DataLoader and EoT in a fully-differentiable manner.
        This version avoids the costly NumPy-PyTorch conversions of the original implementation.

        Args:
            x: Input images (B, C, H, W)
            y: Target labels (for targeted attack) or None (for untargeted)
        """
        logger.info(f"Training RobustDPatch for {self.max_iter} iterations using pure PyTorch backend.")

        # Custom collate fn
        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            targets = [item[1] for item in batch] # List of dicts
            return images, targets

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

        # Create DataLoader
        dataset = RobustPatchDataset(x, y)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn
        )

        logger.info(f"Created DataLoader: {len(dataset)} samples, {len(data_loader)} batches")
        
        # Optimizer for the patch (used mainly for zero_grad)
        optimizer = torch.optim.Adam([self._patch], lr=self.learning_rate)

        # Training loop with EoT
        for i_step in trange(self.max_iter, desc="RobustDPatch Training", disable=not self.verbose):
            
            # Zero the gradients for the patch at the beginning of each full data pass
            optimizer.zero_grad()

            # EoT: Multiple transformation samples
            for e_step in range(self.sample_size):
                
                # Process batches using DataLoader
                for batch_images, batch_labels in data_loader:
                    batch_images = batch_images.to(self.device)

                    # Sample and apply random transformations
                    patched_images, patch_targets, transforms = self._augment_images_with_patch(
                        batch_images, batch_labels, self._patch
                    )
                    
                    # Ensure patched images are within valid range
                    if self.estimator.clip_values is not None:
                        patched_images.clamp_(self.estimator.clip_values[0], self.estimator.clip_values[1])

                    # Compute loss using the estimator's native PyTorch loss function
                    # The loss is automatically averaged over the batch by the estimator's loss function.
                    # We accumulate gradients by calling .backward() repeatedly.
                    loss = self.estimator.compute_loss(x=patched_images, y=patch_targets, standardise_output=True)

                    # For untargeted attacks, we want to maximize the loss
                    if not self.targeted:
                        loss = -loss

                    # Backward pass to accumulate gradients for the patch.
                    # PyTorch automatically sums gradients from multiple backward calls
                    # into the .grad attribute of leaf tensors (our patch).
                    loss.backward()

            # After iterating through all EoT samples and all batches, update the patch
            with torch.no_grad():
                grad = self._patch.grad
                if grad is not None:
                    # Update patch with the sign of the accumulated gradient
                    self._patch.data += torch.sign(grad) * (1 - 2 * int(self.targeted)) * self.learning_rate

                # Clip patch values
                if self.estimator.clip_values is not None:
                    self._patch.data.clamp_(
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1]
                    )
            
            # Summary writer
            if self.summary_writer is not None and i_step % 10 == 0:
                 self.summary_writer.update(
                    batch_id=0,
                    global_step=i_step,
                    grad=self._patch.grad.cpu().numpy() if self._patch.grad is not None else None,
                    patch=self._patch.detach().cpu().numpy(),
                    estimator=self.estimator,
                    x=None,
                    y=None,
                    targeted=self.targeted,
                )

        if self.summary_writer is not None:
            self.summary_writer.reset()


    def _augment_images_with_patch(
        self,
        x: torch.Tensor,
        y: list[dict[str, torch.Tensor]] | None,
        patch: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], dict[str, float | int]]:
        """
        Apply patch to images with random transformations (crop, rotate, brightness).

        Args:
            x: Input images (B, C, H, W) - always channels_first after preprocessing
            y: Target labels (for targeted attack) or None
            patch: Patch tensor (C, H, W)

        Returns:
            Tuple of (augmented images, targets, transformations)
        """
        transformations: dict[str, float | int] = {}
        x_patch = x.clone()

        # Apply patch at specified location
        # x is (B, C, H, W), patch is (C, H, W)
        x_1, y_1 = self.patch_location
        # self._patch is (C, H, W), so height and width are:
        x_2 = x_1 + self._patch.shape[1]  # H
        y_2 = y_1 + self._patch.shape[2]  # W

        # Apply patch: x_patch[B, C, H, W] = patch[C, H, W]
        x_patch[:, :, x_1:x_2, y_1:y_2] = self._patch

        # 1) Crop images
        crop_x = random.randint(0, self.crop_range[0])
        crop_y = random.randint(0, self.crop_range[1])
        
        # x_patch is (B, C, H, W)
        # Crop is applied to H and W dimensions
        h_start = crop_x
        w_start = crop_y
        h_end = x_patch.shape[2] - crop_x
        w_end = x_patch.shape[3] - crop_y
        
        x_patch = x_patch[:, :, h_start:h_end, w_start:w_end]

        transformations.update({"crop_x": crop_x, "crop_y": crop_y})

        # 2) Rotate images
        rot90 = random.choices([0, 1, 2, 3], weights=self.rotation_weights)[0]
        x_patch = torch.rot90(x_patch, k=rot90, dims=(2, 3))

        transformations.update({"rot90": rot90})

        # Transform labels if provided (for targeted attack)
        if y is not None and self.targeted:
             # ... (Similar logic to original, adapted for Tensor)
             # For brevity, implementing just the image part first as per request focus
             # But need to implement label transform for targeted to work.
             # (Keeping implementation from previous draft which looked correct for logic)
             y_copy = []
             image_width = x.shape[3] # W
             image_height = x.shape[2] # H

             for i_image in range(x_patch.shape[0]):
                 if y[i_image] is None: continue 
                 
                 y_b = y[i_image]["boxes"].clone()
                 labels = y[i_image]["labels"]
                 scores = y[i_image]["scores"]
                 
                 # Boxes are [x1, y1, x2, y2]
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
                 else: # 3
                     x_1_new = image_height - y_1_arr - box_height
                     y_1_new = x_1_arr
                     x_2_new = image_height - y_1_arr
                     y_2_new = x_1_arr + box_width
                 
                 # Adjust for crop?
                 # Original NumPy implementation did NOT adjust boxes for crop in `_augment_images_with_patch`?
                 # Let's check `dpatch_robust.py`.
                 # It does crop `x_copy`. 
                 # Then it does rotation.
                 # Then it updates labels for rotation.
                 # It does NOT update labels for crop. This seems to be a bug or assumption in original code?
                 # Wait, `x_1, y_1 = crop_x, crop_y`. `x_copy = x_copy[:, x_1:x_2...]`.
                 # If you crop the image, the object coordinates shift.
                 # If original code didn't shift boxes, I should replicate that behavior if "meaning must be same".
                 # Or maybe `crop` is just reducing the canvas but keeping coordinates relative to original?
                 # No, CNNs see the new canvas.
                 # If I check `dpatch_robust.py` again... `transformations.update({"crop_x": crop_x...})`.
                 # It seems it ignores crop for labels. This is strange for a robust attack if it targets specific boxes.
                 # BUT, `RobustDPatch` often uses `targeted` where we want to hallucinate an object.
                 # The target box is usually the patch itself?
                 # In `dpatch_robust.py`: `y_i["boxes"][:, 0] = x_1_new`.
                 # It seems to only handle rotation.
                 # I will stick to the original logic: only rotate labels.
                 
                 y_new = {
                     "boxes": torch.stack([x_1_new, y_1_new, x_2_new, y_2_new], dim=1),
                     "labels": labels,
                     "scores": scores
                 }
                 y_copy.append(y_new)
        else:
            y_copy = None

        # 3) Adjust brightness
        brightness = random.uniform(*self.brightness_range)
        x_patch = torch.round(brightness * x_patch / self.learning_rate) * self.learning_rate

        transformations.update({"brightness": brightness})

        # Create patch targets
        if self.targeted:
            patch_targets = y_copy
        else:
            # Untargeted: predict on augmented image
            # We need to run prediction on the augmented image x_patch
            # Convert to NumPy for ART
            x_patch_np = x_patch.detach().cpu().numpy()
            if not self.estimator.channels_first:
                 x_patch_np = np.transpose(x_patch_np, (0, 2, 3, 1))
            if self.estimator.clip_values is not None:
                 x_patch_np = x_patch_np * self.estimator.clip_values[1]
            
            predictions = self.estimator.predict(x=x_patch_np, standardise_output=True)
            
            patch_targets = []
            for i in range(len(predictions)):
                t = {
                    "boxes": torch.from_numpy(predictions[i]["boxes"]).float().to(self.device),
                    "labels": torch.from_numpy(predictions[i]["labels"]).long().to(self.device),
                    "scores": torch.from_numpy(predictions[i]["scores"]).float().to(self.device)
                }
                patch_targets.append(t)

        return x_patch, patch_targets, transformations

    def _untransform_gradients(
        self,
        gradients: torch.Tensor,
        transforms: dict[str, int | float]
    ) -> torch.Tensor:
        """
        Revert transformation on gradients.

        Args:
            gradients: The gradients (B, C, H_crop, W_crop) after transformations
            transforms: The transformations in forward direction

        Returns:
            Reverse-transformed gradients mapped back to patch size (B, C, H_patch, W_patch)
        """
        # 1. Brightness: scale gradients by brightness factor
        gradients = transforms["brightness"] * gradients

        # 2. Undo Rotation
        rot90 = transforms["rot90"]
        k = (4 - rot90) % 4  # Reverse rotation
        gradients = torch.rot90(gradients, k=k, dims=(2, 3))

        # 3. Undo Cropping (Extract patch region from gradient)
        # The gradients are now aligned with the UN-ROTATED, but CROPPED image.
        # Dimensions: (B, C, H_crop, W_crop).
        # We want to extract the gradients for the patch.
        # The patch is at `self.patch_location` in the ORIGINAL image.
        # The cropped image starts at `(crop_x, crop_y)` of the ORIGINAL image.
        # So the patch in the cropped image is at:
        # p_x = patch_x - crop_x
        # p_y = patch_y - crop_y

        crop_x = int(transforms["crop_x"])
        crop_y = int(transforms["crop_y"])

        # Patch coordinates in original image
        px, py = self.patch_location
        # self._patch is (C, H, W), so patch height and width are:
        ph, pw = self._patch.shape[1], self._patch.shape[2]

        # Patch coordinates in cropped gradient tensor
        gx = px - crop_x
        gy = py - crop_y

        # Extract patch gradient region
        grad_patch = gradients[:, :, gx:gx+ph, gy:gy+pw]

        return grad_patch

    def _patch_to_numpy(self) -> np.ndarray:
        """
        Convert patch from PyTorch to NumPy.

        Internal patch is (C, H, W), return as (C, H, W) for consistency with other PyTorch attacks.
        """
        patch_np = self._patch.detach().cpu().numpy()

        # Return in (C, H, W) format for consistency with AdversarialPatchPyTorch
        # patch_service.py will handle conversion to (H, W, C) for saving

        # Denormalize if needed (consistent with other PyTorch attacks)
        if self.estimator.clip_values is not None:
            min_val, max_val = self.estimator.clip_values
            if max_val > 1:
                patch_np = patch_np * max_val

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
            # External patch is (H, W, C), convert to (C, H, W)
            patch_torch = torch.from_numpy(patch_external).float().to(self.device)
            patch_torch = patch_torch.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        else:
            patch_torch = self._patch  # Already (C, H, W)

        # Apply patch (without transformations) at specified location
        x_patch = x_torch.clone()

        # x_patch is (B, C, H, W), patch_torch is (C, H, W)
        x_1, y_1 = self.patch_location
        x_2 = x_1 + patch_torch.shape[1]  # H
        y_2 = y_1 + patch_torch.shape[2]  # W

        x_patch[:, :, x_1:x_2, y_1:y_2] = patch_torch

        # Convert back to NumPy (consistent with other attacks)
        patched_numpy = x_patch.detach().cpu().numpy()
        # Always (B, C, H, W) -> (B, H, W, C) for API
        patched_numpy = np.transpose(patched_numpy, (0, 2, 3, 1))

        # Denormalize if needed
        if self.estimator.clip_values is not None:
            min_val, max_val = self.estimator.clip_values
            if max_val > 1:
                patched_numpy = patched_numpy * max_val

        return patched_numpy.astype(np.float32)

    @property
    def targeted(self) -> bool:
        """Return whether the attack is targeted."""
        return self._targeted

    def _check_params(self) -> None:
        """Validate parameters."""
        # (Copied from original class to ensure completeness)
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

        # Note: crop_range is (height_crop, width_crop), not comparable to each other
        # Original implementation has this check but it's semantically incorrect
        # Keeping for compatibility but should be reconsidered

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