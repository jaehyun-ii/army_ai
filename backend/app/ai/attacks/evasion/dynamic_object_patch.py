# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# This file is an extension of adversarial_patch_pytorch.py, modified to
# dynamically place and scale the patch based on object detection bounding boxes.
#
"""
This module implements the DynamicObjectPatchPyTorch attack.
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
import random
from typing import TYPE_CHECKING, List, Dict, Union

import torch
import torchvision
import numpy as np
from tqdm.auto import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau

from app.ai.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch

if TYPE_CHECKING:
    from app.ai.utils import PYTORCH_OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class DynamicObjectPatchPyTorch(AdversarialPatchPyTorch):
    """
    An adversarial patch attack for object detectors that dynamically places and
    scales the patch on a specific target class object's bounding box.

    The patch is placed at the center of a randomly chosen bounding box of the target class,
    with a size relative to the bounding box's width.
    This version incorporates an adaptive learning rate scheduler and a 'reset to best' mechanism.
    """

    def __init__(
        self,
        estimator: "PYTORCH_OBJECT_DETECTOR_TYPE",
        target_class_id: int,
        scale_factor: float = 0.3,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        batch_size: int = 16,
        patch_shape: tuple[int, int, int] = (3, 100, 100),
        optimizer: str = "Adam",
        targeted: bool = False,
        verbose: bool = True,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ):
        super().__init__(
            estimator=estimator,
            rotation_max=0.0,
            scale_min=scale_factor,
            scale_max=scale_factor,
            distortion_scale_max=0.0,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            patch_shape=patch_shape,
            optimizer=optimizer,
            targeted=targeted,
            verbose=verbose,
        )
        self.target_class_id = target_class_id
        self.scale_factor = scale_factor
        
        if optimizer == "Adam":
            self._optimizer = torch.optim.Adam([self._patch], lr=learning_rate)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer}' not supported.")

        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode='max' if not self.targeted else 'min',
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

        self.best_patch_loss = -float('inf') if not self.targeted else float('inf')
        self.best_patch = self._patch.clone().detach()

        logger.info(f"DynamicObjectPatchPyTorch initialized to target class ID: {self.target_class_id}")
        logger.info(f"Using ReduceLROnPlateau scheduler with patience={scheduler_patience} and factor={scheduler_factor}")

    def generate(self, x: np.ndarray, y: Union[np.ndarray, List[Dict[str, np.ndarray]]], **kwargs) -> tuple[np.ndarray, np.ndarray]:
        import torch.utils.data

        class ObjectDetectionDataset(torch.utils.data.Dataset):
            def __init__(self, x_data: np.ndarray, y_data: List[Dict[str, np.ndarray]]):
                self.x = x_data
                self.y = y_data

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx: int):
                img = torch.from_numpy(self.x[idx])
                target = {k: torch.from_numpy(v) for k, v in self.y[idx].items()}
                return img, target
        
        def collate_fn(batch):
            return torch.stack([item[0] for item in batch], 0), [item[1] for item in batch]

        if isinstance(y, np.ndarray):
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        else:
            dataset = ObjectDetectionDataset(x, y)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn
        )

        for i_iter in trange(self.max_iter, desc="Dynamic Patch PyTorch", disable=not self.verbose):
            epoch_loss = 0.0
            num_batches = 0

            for images, targets in data_loader:
                images = images.to(self.estimator.device)
                
                target_list = []
                for t in targets:
                    target_list.append({k: v.to(self.estimator.device) for k, v in t.items()})

                batch_loss = self._train_step(images=images, target=target_list, mask=None)
                
                if isinstance(batch_loss, torch.Tensor):
                    epoch_loss += batch_loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            is_best = (avg_epoch_loss > self.best_patch_loss) if not self.targeted else (avg_epoch_loss < self.best_patch_loss)
            if is_best:
                self.best_patch_loss = avg_epoch_loss
                self.best_patch = self._patch.clone().detach()

            lr_before = self._optimizer.param_groups[0]['lr']
            self._scheduler.step(avg_epoch_loss)
            lr_after = self._optimizer.param_groups[0]['lr']

            if lr_after < lr_before:
                logger.info(f"Learning rate reduced to {lr_after:.6f}. Resetting patch to best found so far.")
                with torch.no_grad():
                    self._patch.copy_(self.best_patch)
            
            if self.verbose:
                logger.info(
                    f"Epoch {i_iter + 1}/{self.max_iter}, Avg. Loss: {avg_epoch_loss:.4f}, "
                    f"Best Loss: {self.best_patch_loss:.4f}, Current LR: {lr_after:.6f}"
                )

        return (
            self.best_patch.cpu().numpy(),
            self._get_circular_patch_mask(nb_samples=1).cpu().numpy()[0],
        )

    def _overlay_on_bboxes(self, images: torch.Tensor, patch: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if not self.estimator.channels_first:
            images = torch.permute(images, (0, 3, 1, 2))

        nb_samples = images.shape[0]
        image_height, image_width = images.shape[2], images.shape[3]

        small_mask = self._get_circular_patch_mask(nb_samples=1)[0].float().to(self.estimator.device)
        patch_mask_canvas = torch.zeros(self.patch_shape[0], image_height, image_width).to(self.estimator.device)
        patch_with_grad_canvas = torch.zeros(self.patch_shape[0], image_height, image_width).to(self.estimator.device)

        patch_center_h, patch_center_w = image_height // 2, image_width // 2
        patch_h, patch_w = self.patch_shape[1], self.patch_shape[2]
        h_start, w_start = patch_center_h - patch_h // 2, patch_center_w - patch_w // 2
        h_end, w_end = h_start + patch_h, w_start + patch_w
        
        patch_mask_canvas[:, h_start:h_end, w_start:w_end] = small_mask
        patch_with_grad_canvas[:, h_start:h_end, w_start:w_end] = patch.to(self.estimator.device)

        final_patched_images = []

        for i_sample in range(nb_samples):
            image = images[i_sample]
            target = targets[i_sample]

            target_indices = (target["labels"] == self.target_class_id).nonzero().squeeze(dim=1)

            if target_indices.nelement() == 0:
                final_patched_images.append(image)
                continue

            random_idx_pos = random.randint(0, len(target_indices) - 1)
            random_idx = target_indices[random_idx_pos]
            box = target["boxes"][random_idx]
            
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            
            target_patch_width = box_width * self.scale_factor
            if target_patch_width <= 0:
                final_patched_images.append(image)
                continue
            
            im_scale = float(target_patch_width / patch_w)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x_shift = center_x - patch_center_w
            y_shift = center_y - patch_center_h
            
            transformed_mask = torchvision.transforms.functional.affine(
                img=patch_mask_canvas,
                angle=0.0,
                translate=[x_shift.item(), y_shift.item()],
                scale=im_scale,
                shear=[0, 0],
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                fill=0.0,
            )

            transformed_patch = torchvision.transforms.functional.affine(
                img=patch_with_grad_canvas,
                angle=0.0,
                translate=[x_shift.item(), y_shift.item()],
                scale=im_scale,
                shear=[0, 0],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                fill=0.0,
            )

            inverted_mask = 1.0 - transformed_mask
            patched_image = image * inverted_mask + transformed_patch
            final_patched_images.append(patched_image)

        final_batch = torch.stack(final_patched_images, dim=0)

        if not self.estimator.channels_first:
            final_batch = torch.permute(final_batch, (0, 2, 3, 1))

        return final_batch

    def _train_step(self, images: torch.Tensor, target: List[Dict[str, torch.Tensor]], mask: None) -> torch.Tensor:
        self._optimizer.zero_grad()
        loss = self._loss(images, target, mask)
        loss.backward()
        self._optimizer.step()
        with torch.no_grad():
            self._patch[:] = torch.clamp(
                self._patch, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
            )
        return loss

    def _loss(self, images: torch.Tensor, target: List[Dict[str, torch.Tensor]], mask: None) -> torch.Tensor:
        """
        Compute loss using estimator's compute_loss method.
        This follows the original ART implementation pattern.
        """
        patched_input = self._overlay_on_bboxes(images, self._patch, target)
        patched_input = torch.clamp(
            patched_input,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        # Use estimator.compute_loss() like original ART implementation
        # This handles YOLO, Faster R-CNN, and other models correctly
        loss = self.estimator.compute_loss(x=patched_input, y=target)

        if not self.targeted:
            loss = -loss
        return loss