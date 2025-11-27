# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements the Fast Gradient Method attack in PyTorch. This implementation includes the original Fast
Gradient Sign Method attack and extends it to other norms, therefore it is called the Fast Gradient Method.

| Paper link: https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from app.ai.summary_writer import SummaryWriter
from app.ai.attacks.evasion.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch

if TYPE_CHECKING:
    from app.ai.estimators.classification.pytorch import PyTorchClassifier
    from app.ai.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class FastGradientMethodPyTorch(ProjectedGradientDescentPyTorch):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity norm (and is known as the "Fast
    Gradient Sign Method"). This implementation extends the attack to other norms, and is therefore called the Fast
    Gradient Method. This is the PyTorch implementation which is faster and supports GPU acceleration.

    | Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(
        self,
        estimator: "PyTorchClassifier | OBJECT_DETECTOR_TYPE",
        norm: int | float | str = np.inf,
        eps: int | float | np.ndarray = 0.3,
        eps_step: int | float | np.ndarray = 0.1,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        summary_writer: str | bool | SummaryWriter = False,
        target_class_id: int = 0,
    ):
        """
        Create a :class:`.FastGradientMethodPyTorch` instance.

        :param estimator: A trained PyTorch classifier or object detector.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", `np.inf` or a real `p >= 1`.
        :param eps: Attack step size (input variation).
        :param eps_step: Step size of input variation for minimal perturbation computation.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :param num_random_init: Number of random initialisations within the epsilon ball. For random_init=0 starting at
            the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               'runs/exp1', 'runs/exp2', etc. for each new experiment to compare across them.
        :param target_class_id: Target class ID for object detection models (used for pseudo-GT generation). Ignored for classification models.
        """
        # FGSM is PGD with max_iter=1
        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=None,
            max_iter=1,  # FGSM is a single-step attack
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=False,
            summary_writer=summary_writer,
            verbose=False,  # No need for progress bar in single iteration
            target_class_id=target_class_id,
        )

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        logger.info("Creating FGSM adversarial samples with PyTorch.")
        return super().generate(x=x, y=y, **kwargs)
