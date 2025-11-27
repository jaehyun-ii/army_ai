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
This module implements the Projected Gradient Descent attack `ProjectedGradientDescent` as an iterative method in which,
after each iteration, the perturbation is projected on a lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from app.ai.config import ART_NUMPY_DTYPE
from app.ai.summary_writer import SummaryWriter
from app.ai.estimators.estimator import BaseEstimator, LossGradientsMixin
from app.ai.estimators.classification.classifier import ClassifierMixin
from app.ai.estimators.object_detection.object_detector import ObjectDetectorMixin
from app.ai.attacks.attack import EvasionAttack
from app.ai.utils import compute_success, random_sphere, compute_success_array, get_labels_np_array, check_and_transform_label_format

if TYPE_CHECKING:

    import torch
    from app.ai.estimators.classification.pytorch import PyTorchClassifier
    from app.ai.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class _PseudoGTGenerator:
    """
    Generate pseudo ground truth labels using ART's predict API.

    This enables unsupervised adversarial attacks by using the model's own
    predictions as training targets for object detection models.
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

    def generate_from_numpy(
        self,
        x: np.ndarray
    ) -> list[dict[str, np.ndarray]]:
        """
        Generate pseudo-GT using ART's predict API.

        Args:
            x: Input images in NumPy format

        Returns:
            List of dicts with 'boxes' and 'labels' for each image (NumPy arrays)
        """
        import torch

        pseudo_gts = []

        # Use ART's predict API - handles all preprocessing automatically
        predictions = self.estimator.predict(x=x)

        # Process each prediction
        for idx, pred in enumerate(predictions):
            try:
                if 'boxes' not in pred or len(pred['boxes']) == 0:
                    pseudo_gts.append(self._empty_pseudo_gt_numpy())
                    continue

                boxes = pred['boxes']
                scores = pred['scores']
                classes = pred['labels']

                # Filter by target class and confidence
                mask = (classes == self.target_class_id) & (scores >= self.confidence_threshold)
                filtered_boxes = boxes[mask]
                filtered_labels = classes[mask]
                filtered_scores = scores[mask]

                # If no detections, use empty arrays
                if len(filtered_boxes) == 0:
                    pseudo_gts.append(self._empty_pseudo_gt_numpy())
                else:
                    # Use highest confidence detection
                    best_idx = filtered_scores.argmax()
                    pseudo_gt = {
                        'boxes': filtered_boxes[best_idx:best_idx+1],  # (1, 4)
                        'labels': filtered_labels[best_idx:best_idx+1]  # (1,)
                    }
                    pseudo_gts.append(pseudo_gt)

            except Exception as e:
                logger.error(f"Error processing result {idx}: {e}")
                pseudo_gts.append(self._empty_pseudo_gt_numpy())

        return pseudo_gts

    def _empty_pseudo_gt_numpy(self) -> dict[str, np.ndarray]:
        """Create empty pseudo-GT (NumPy)."""
        return {
            'boxes': np.zeros((0, 4), dtype=np.float32),
            'labels': np.zeros(0, dtype=np.int64)
        }


class ProjectedGradientDescentPyTorch(EvasionAttack):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on a lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "decay",
        "max_iter",
        "targeted",
        "num_random_init",
        "batch_size",
        "random_eps",
        "summary_writer",
        "verbose",
        "target_class_id",
    ]
    # Support both classification (ClassifierMixin) and object detection (ObjectDetectorMixin)
    # The tuple (ClassifierMixin, ObjectDetectorMixin) means estimator must inherit from at least one of them
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, (ClassifierMixin, ObjectDetectorMixin))  # type: ignore

    def __init__(
        self,
        estimator: "PyTorchClassifier | OBJECT_DETECTOR_TYPE",
        norm: int | float | str = np.inf,
        eps: int | float | np.ndarray = 0.3,
        eps_step: int | float | np.ndarray = 0.1,
        decay: float | None = None,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
        target_class_id: int = 0,
    ):
        """
        Create a :class:`.ProjectedGradientDescentPyTorch` instance.

        :param estimator: A trained estimator (classifier or object detector).
        :param norm: The norm of the adversarial perturbation, supporting  "inf", `np.inf` or a real `p >= 1`.
                     Currently, when `p` is not infinity, the projection step only rescales the noise, which may be
                     suboptimal for `p != 2`.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step is
                           modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
                           is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               'runs/exp1', 'runs/exp2', etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        :param target_class_id: Target class ID for object detection models (used for pseudo-GT generation). Ignored for classification models.
        """
        if not estimator.all_framework_preprocessing:
            raise NotImplementedError(
                "The framework-specific implementation only supports framework-specific preprocessing."
            )

        if summary_writer and num_random_init > 1:
            raise ValueError("TensorBoard is not yet supported for more than 1 random restart (num_random_init>1).")

        super().__init__(estimator=estimator, summary_writer=summary_writer)

        # Set attack parameters
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.decay = decay
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.random_eps = random_eps
        self.verbose = verbose
        self.target_class_id = target_class_id

        # Check if this is an object detection model
        self._is_object_detector = isinstance(estimator, ObjectDetectorMixin)

        # Initialize pseudo-GT generator for object detection models
        if self._is_object_detector:
            import torch
            device = estimator.device if hasattr(estimator, 'device') else 'cpu'
            self._pseudo_gt_gen = _PseudoGTGenerator(
                estimator=estimator,
                target_class_id=target_class_id,
                confidence_threshold=0.3,
                device=device
            )
            logger.info(f"Initialized PGD for object detection with target_class_id={target_class_id}")
        else:
            self._pseudo_gt_gen = None
            logger.info("Initialized PGD for classification")

        # Validate parameters
        self._check_params()

        self._batch_id = 0
        self._i_max_iter = 0

    def _get_mask(self, x: np.ndarray, **kwargs) -> np.ndarray | None:
        """
        Get mask from kwargs.

        :param x: Input samples.
        :param kwargs: Additional keyword arguments.
        :return: Mask if provided in kwargs, otherwise None.
        """
        return kwargs.get("mask", None)

    def _check_compatibility_input_and_eps(self, x: np.ndarray) -> None:
        """
        Check the compatibility of input samples and epsilon values.

        :param x: Input samples.
        """
        if isinstance(self.eps, np.ndarray):
            # Check if eps shape is compatible with x
            if self.eps.shape[0] != x.shape[0]:
                raise ValueError(
                    f"The first dimension of `eps` must match the batch size of `x`. "
                    f"Got eps.shape[0]={self.eps.shape[0]}, x.shape[0]={x.shape[0]}"
                )

        if isinstance(self.eps_step, np.ndarray):
            # Check if eps_step shape is compatible with x
            if self.eps_step.shape[0] != x.shape[0]:
                raise ValueError(
                    f"The first dimension of `eps_step` must match the batch size of `x`. "
                    f"Got eps_step.shape[0]={self.eps_step.shape[0]}, x.shape[0]={x.shape[0]}"
                )

    def _random_eps(self) -> None:
        """
        Apply random epsilon sampling from truncated normal distribution.
        Only applies when self.random_eps is True.
        """
        if self.random_eps:
            # Sample epsilon from truncated normal distribution
            if isinstance(self.eps, (int, float)):
                # Sample a random value from normal distribution truncated at [0, 2*eps]
                self.eps = np.abs(np.random.normal(loc=self.eps, scale=self.eps / 2))
                self.eps = np.clip(self.eps, 0, 2 * self.eps)

                # Adjust eps_step to maintain the ratio
                if isinstance(self.eps_step, (int, float)):
                    ratio = self.eps_step / self.eps if self.eps > 0 else 0.1
                    self.eps_step = ratio * self.eps

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import torch

        mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        # Set up targets
        targets = self._set_targets(x, y)

        # Create dataset - handle object detection (list) vs classification (array) differently
        if self._is_object_detector:
            # For object detection, targets is a list of dicts, use custom dataset
            class ObjectDetectionDataset(torch.utils.data.Dataset):
                def __init__(self, images, labels, mask=None):
                    self.images = torch.from_numpy(images.astype(ART_NUMPY_DTYPE))
                    self.labels = labels  # list of dicts
                    self.mask = mask
                    if mask is not None:
                        if len(mask.shape) == len(images.shape):
                            self.mask = torch.from_numpy(mask.astype(ART_NUMPY_DTYPE))
                        else:
                            self.mask = torch.from_numpy(np.array([mask.astype(ART_NUMPY_DTYPE)] * images.shape[0]))

                def __len__(self):
                    return len(self.images)

                def __getitem__(self, idx):
                    if self.mask is not None:
                        return self.images[idx], self.labels[idx], self.mask[idx]
                    return self.images[idx], self.labels[idx], None

            # Custom collate function for object detection that keeps labels as list of dicts
            def collate_fn_od(batch):
                images = torch.stack([item[0] for item in batch])
                labels = [item[1] for item in batch]  # Keep as list of dicts
                masks = None
                if batch[0][2] is not None:
                    masks = torch.stack([item[2] for item in batch])
                return images, labels, masks

            dataset = ObjectDetectionDataset(x, targets, mask)
            collate_fn = collate_fn_od
        else:
            # For classification, targets is a numpy array, use TensorDataset
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise, (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                        torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                        torch.from_numpy(mask.astype(ART_NUMPY_DTYPE)),
                    )

                else:
                    dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                        torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                        torch.from_numpy(np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])),
                    )

            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                )
            collate_fn = None

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn
        )

        # Start to compute adversarial examples
        adv_x = x.astype(ART_NUMPY_DTYPE)

        # Compute perturbation with batching
        for batch_id, batch_all in enumerate(
            tqdm(data_loader, desc="PGD - Batches", leave=False, disable=not self.verbose)
        ):

            self._batch_id = batch_id

            if mask is not None:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
            else:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            batch_eps: int | float | np.ndarray
            batch_eps_step: int | float | np.ndarray

            # Compute batch_eps and batch_eps_step
            if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
                if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                    batch_eps_step = self.eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = self.eps
                    batch_eps_step = self.eps_step

            else:
                batch_eps = self.eps
                batch_eps_step = self.eps_step

            for rand_init_num in range(max(1, self.num_random_init)):
                if rand_init_num == 0:
                    # first iteration: use the adversarial examples as they are the only ones we have now
                    adv_x[batch_index_1:batch_index_2] = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )
                else:
                    adversarial_batch = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )

                    # return the successful adversarial examples (only for classification)
                    if not self._is_object_detector:
                        attack_success = compute_success_array(
                            self.estimator,
                            batch,
                            batch_labels,
                            adversarial_batch,
                            self.targeted,
                            batch_size=self.batch_size,
                        )
                        adv_x[batch_index_1:batch_index_2][attack_success] = adversarial_batch[attack_success]
                    else:
                        # For object detection, always use the new adversarial examples
                        adv_x[batch_index_1:batch_index_2] = adversarial_batch

        # Compute success rate only for classification models
        if not self._is_object_detector:
            logger.info(
                "Success rate of attack: %.2f%%",
                100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size),
            )
        else:
            logger.info("PGD attack completed for object detection model")

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return adv_x

    def _generate_batch(
        self,
        x: "torch.Tensor",
        targets: "torch.Tensor | list",
        mask: "torch.Tensor",
        eps: int | float | np.ndarray,
        eps_step: int | float | np.ndarray,
    ) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param targets: Target values. For classification: tensor of shape `(nb_samples, nb_classes)`.
                       For object detection: list of dicts with 'boxes' and 'labels'.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch

        inputs = x.to(self.estimator.device)

        # Handle targets - for object detection it's a list, for classification it's a tensor
        if not self._is_object_detector:
            targets = targets.to(self.estimator.device)
        # For object detection, targets is already a list of dicts, keep as is

        adv_x = torch.clone(inputs)
        momentum = torch.zeros(inputs.shape).to(self.estimator.device)

        if mask is not None:
            mask = mask.to(self.estimator.device)

        for i_max_iter in range(self.max_iter):
            self._i_max_iter = i_max_iter
            adv_x = self._compute_pytorch(
                adv_x, inputs, targets, mask, eps, eps_step, self.num_random_init > 0 and i_max_iter == 0, momentum
            )

        return adv_x.cpu().detach().numpy()

    def _compute_perturbation_pytorch(
        self, x: "torch.Tensor", y: "torch.Tensor | list", mask: "torch.Tensor" | None, momentum: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values. For classification: tensor of shape `(nb_samples, nb_classes)`.
                  For object detection: list of dicts with 'boxes' and 'labels'.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :return: Perturbations.
        """
        import torch

        # For object detection, filter out samples with empty boxes
        if self._is_object_detector and isinstance(y, list):
            # Find indices of samples with valid detections
            valid_indices = [i for i, label_dict in enumerate(y) if len(label_dict.get('boxes', [])) > 0]

            if len(valid_indices) == 0:
                # All samples have no detections, return zero gradient
                logger.warning("All samples have no detected objects, returning zero gradients")
                return torch.zeros_like(x)

            # Filter x and y to only include valid samples
            x_valid = x[valid_indices]
            y_valid = [y[i] for i in valid_indices]

            # Get gradient only for valid samples
            grad_valid = self.estimator.loss_gradient(x=x_valid, y=y_valid) * (-1 if self.targeted else 1)

            # Create full gradient tensor with zeros for invalid samples
            grad = torch.zeros_like(x)
            grad[valid_indices] = grad_valid
        else:
            # For classification, compute gradient normally
            grad = self.estimator.loss_gradient(x=x, y=y) * (-1 if self.targeted else 1)

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            y_numpy = y if self._is_object_detector else y.cpu().detach().numpy()
            self.summary_writer.update(
                batch_id=self._batch_id,
                global_step=self._i_max_iter,
                grad=grad.cpu().detach().numpy(),
                patch=None,
                estimator=self.estimator,
                x=x.cpu().detach().numpy(),
                y=y_numpy,
                targeted=self.targeted,
            )

        # Check for nan before normalisation and replace with 0
        if torch.any(grad.isnan()):  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad[grad.isnan()] = 0.0

        # Apply mask
        if mask is not None:
            grad = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), grad)

        # Compute gradient momentum
        if self.decay is not None:
            # Update momentum in-place (important).
            # The L1 normalization for accumulation is an arbitrary choice of the paper.
            grad_2d = grad.reshape(len(grad), -1)
            norm1 = torch.linalg.norm(grad_2d, ord=1, dim=1, keepdim=True)
            normalized_grad = (grad_2d * norm1.where(norm1 == 0, 1 / norm1)).reshape(grad.shape)
            momentum *= self.decay
            momentum += normalized_grad
            # Use the momentum to compute the perturbation, instead of the gradient
            grad = momentum

        # Apply norm bound
        norm: float = np.inf if self.norm == "inf" else float(self.norm)
        grad_2d = grad.reshape(len(grad), -1)
        if norm == np.inf:
            grad_2d = torch.ones_like(grad_2d)
        elif norm == 1:
            i_max = torch.argmax(grad_2d.abs(), dim=1)
            grad_2d = torch.zeros_like(grad_2d)
            grad_2d[range(len(grad_2d)), i_max] = 1
        elif norm > 1:
            conjugate = norm / (norm - 1)
            q_norm = torch.linalg.norm(grad_2d, ord=conjugate, dim=1, keepdim=True)
            grad_2d = (grad_2d.abs() * q_norm.where(q_norm == 0, 1 / q_norm)) ** (conjugate - 1)

        grad = grad_2d.reshape(grad.shape) * grad.sign()

        assert x.shape == grad.shape

        return grad

    def _apply_perturbation_pytorch(
        self, x: "torch.Tensor", perturbation: "torch.Tensor", eps_step: int | float | np.ndarray
    ) -> "torch.Tensor":
        """
        Apply perturbation on examples.

        :param x: Current adversarial examples.
        :param perturbation: Current perturbations.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch

        eps_step = np.array(eps_step, dtype=ART_NUMPY_DTYPE)
        perturbation_step = torch.tensor(eps_step).to(self.estimator.device) * perturbation
        perturbation_step[torch.isnan(perturbation_step)] = 0
        x = x + perturbation_step
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x = torch.max(
                torch.min(x, torch.tensor(clip_max).to(self.estimator.device)),
                torch.tensor(clip_min).to(self.estimator.device),
            )

        return x

    def _compute_pytorch(
        self,
        x: "torch.Tensor",
        x_init: "torch.Tensor",
        y: "torch.Tensor | list",
        mask: "torch.Tensor",
        eps: int | float | np.ndarray,
        eps_step: int | float | np.ndarray,
        random_init: bool,
        momentum: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values. For classification: tensor of shape `(nb_samples, nb_classes)`.
                  For object detection: list of dicts with 'boxes' and 'labels'.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :return: Adversarial examples.
        """
        import torch

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()

            random_perturbation_array = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            random_perturbation = torch.from_numpy(random_perturbation_array).to(self.estimator.device)

            if mask is not None:
                random_perturbation = random_perturbation * mask

            x_adv = x + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = torch.max(
                    torch.min(x_adv, torch.tensor(clip_max).to(self.estimator.device)),
                    torch.tensor(clip_min).to(self.estimator.device),
                )

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation_pytorch(x_adv, y, mask, momentum)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation_pytorch(x_adv, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)

        # Recompute x_adv
        x_adv = perturbation + x_init

        return x_adv

    @staticmethod
    def _projection(
        values: "torch.Tensor",
        eps: int | float | np.ndarray,
        norm_p: int | float | str,
        *,
        suboptimal: bool = True,
    ) -> "torch.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Values to clip.
        :param eps: If a scalar, the norm of the L_p ball onto which samples are projected. Equivalently in general,
                    can be any array of non-negatives broadcastable with `values`, and the projection occurs onto the
                    unit ball for the weighted L_{p, w} norm with `w = 1 / eps`. Currently, for any given sample,
                    non-uniform weights are only supported with infinity norm. Example: To specify sample-wise scalar,
                    you can provide `eps.shape = (n_samples,) + (1,) * values[0].ndim`.
        :param norm_p: Lp norm to use for clipping, with `norm_p > 0`. Only 2, `np.inf` and "inf" are supported
                       with `suboptimal=False` for now.
        :param suboptimal: If `True` simply projects by rescaling to Lp ball. Fast but may be suboptimal for
                           `norm_p != 2`.
                       Ignored when `norm_p in [np.inf, "inf"]` because optimal solution is fast. Defaults to `True`.
        :return: Values of `values` after projection.
        """
        import torch

        norm = np.inf if norm_p == "inf" else float(norm_p)
        assert norm > 0

        values_tmp = values.reshape(len(values), -1)  # (n_samples, d)

        eps = np.broadcast_to(eps, values.shape)
        eps = eps.reshape(len(eps), -1)  # (n_samples, d)
        assert np.all(eps >= 0)
        if norm != np.inf and not np.all(eps == eps[:, [0]]):
            raise NotImplementedError(
                "Projection onto the weighted L_p ball is currently not supported with finite `norm_p`."
            )

        if (suboptimal or norm == 2) and norm != np.inf:  # Simple rescaling
            values_norm = torch.linalg.norm(values_tmp, ord=norm, dim=1, keepdim=True)  # (n_samples, 1)
            values_tmp = values_tmp * values_norm.where(
                values_norm == 0,
                torch.minimum(
                    torch.ones(1).to(values_tmp.device), torch.tensor(eps).to(values_tmp.device) / values_norm
                ),
            )
        else:  # Optimal
            if norm == np.inf:  # Easy exact case
                values_tmp = values_tmp.sign() * torch.minimum(
                    values_tmp.abs(), torch.tensor(eps).to(values_tmp.device)
                )
            elif norm >= 1:  # Convex optim
                raise NotImplementedError(
                    "Finite values of `norm_p >= 1` are currently not supported with `suboptimal=False`."
                )
            else:  # Non-convex optim
                raise NotImplementedError("Values of `norm_p < 1` are currently not supported with `suboptimal=False`")

        values = values_tmp.reshape(values.shape).to(values.dtype)

        return values

    def _set_targets(self, x: np.ndarray, y: np.ndarray | list | None) -> np.ndarray | list:
        """
        Check and set up targets.

        :param x: An array with the original inputs.
        :param y: Target values. For classification: (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). For object detection: list of dicts with 'boxes' and 'labels'.
                  Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :return: The targets.
        """
        # Handle object detection models
        if self._is_object_detector:
            if y is None:
                # Generate pseudo ground truth for object detection
                logger.info("Generating pseudo-GT for object detection model")
                targets = self._pseudo_gt_gen.generate_from_numpy(x)
            else:
                # Use provided labels
                targets = y
            return targets

        # Handle classification models (original logic)
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            targets = y

        return targets

    def _check_params(self) -> None:
        """Validate attack parameters."""
        norm: float = np.inf if self.norm == "inf" else float(self.norm)
        if norm < 1:
            raise ValueError('Norm order must be either "inf", `np.inf` or a real `p >= 1`.')

        if not (
            isinstance(self.eps, (int, float))
            and isinstance(self.eps_step, (int, float))
            or isinstance(self.eps, np.ndarray)
            and isinstance(self.eps_step, np.ndarray)
        ):
            raise TypeError(
                "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same type of `int`"
                ", `float`, or `np.ndarray`."
            )

        if isinstance(self.eps, (int, float)):
            if self.eps < 0:
                raise ValueError("The perturbation size `eps` has to be non-negative.")
        else:
            if (self.eps < 0).any():
                raise ValueError("The perturbation size `eps` has to be non-negative.")

        if isinstance(self.eps_step, (int, float)):
            if self.eps_step <= 0:
                raise ValueError("The perturbation step-size `eps_step` has to be positive.")
        else:
            if (self.eps_step <= 0).any():
                raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
            if self.eps.shape != self.eps_step.shape:
                raise ValueError(
                    "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same shape."
                )

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, int):
            raise TypeError("The number of random initialisations has to be of type integer.")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if self.max_iter < 0:
            raise ValueError("The number of iterations `max_iter` has to be a non-negative integer.")

        if self.decay is not None and self.decay < 0.0:
            raise ValueError("The decay factor `decay` has to be a nonnegative float.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The verbose has to be a Boolean.")
