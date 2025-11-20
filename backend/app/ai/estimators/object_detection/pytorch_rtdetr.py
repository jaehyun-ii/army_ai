# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2025
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
This module implements the task specific estimator for PyTorch RT-DETR object detectors.

| Paper link: https://arxiv.org/abs/2304.08069
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from app.ai.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

if TYPE_CHECKING:

    import torch

    from app.ai.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from app.ai.defences.preprocessor.preprocessor import Preprocessor
    from app.ai.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchRTDETR(PyTorchObjectDetector):
    """
    This module implements the model- and task specific estimator for RT-DETR object detector models in PyTorch.

    | Paper link: https://arxiv.org/abs/2304.08069
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        input_shape: tuple[int, ...] = (3, 640, 640),
        optimizer: "torch.optim.Optimizer" | None = None,
        clip_values: "CLIP_VALUES_TYPE" | None = None,
        channels_first: bool = True,
        preprocessing_defences: "Preprocessor" | list["Preprocessor"] | None = None,
        postprocessing_defences: "Postprocessor" | list["Postprocessor"] | None = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: tuple[str, ...] = (
            "loss_giou",
            "loss_cls",
            "loss_bbox",
        ),
        device_type: str = "gpu",
        is_ultralytics: bool = True,
    ):
        """
        Initialization.

        :param model: RT-DETR model wrapped as demonstrated in examples.
                      The output of the model is `list[dict[str, torch.Tensor]]`, one for each input image.
                      The fields of the dict are as follows:

                      - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                        0 <= y1 < y2 <= H.
                      - labels [N]: the labels for each image.
                      - scores [N]: the scores of each prediction.
        :param input_shape: The shape of one input sample.
        :param optimizer: The optimizer for training the classifier.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_giou', 'loss_cls',
                              'loss_bbox'.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        :param is_ultralytics: Whether the model is from Ultralytics RT-DETR implementation.
        """
        # Store device_type before calling super().__init__
        self._temp_device_type = device_type

        if is_ultralytics:
            from app.ai.estimators.object_detection.pytorch_rtdetr_loss_wrapper import PyTorchRTDETRLossWrapper

            # Determine the actual device
            import torch
            if device_type == "cpu":
                device = "cpu"
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = PyTorchRTDETRLossWrapper(model, device=str(device))

        super().__init__(
            model=model,
            input_shape=input_shape,
            optimizer=optimizer,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            attack_losses=attack_losses,
            device_type=device_type,
        )

    def _translate_labels(self, labels: list[dict[str, "torch.Tensor"]]) -> "torch.Tensor":
        """
        Translate object detection labels from ART format (torchvision) to the model format (RT-DETR) and
        move tensors to GPU, if applicable.

        :param labels: Object detection labels in format x1y1x2y2 (torchvision).
        :return: Object detection labels in format xcycwh (RT-DETR).
        """
        import torch

        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        labels_xcycwh_list = []

        for i, label_dict in enumerate(labels):
            # create 2D tensor to encode labels and bounding boxes
            label_xcycwh = torch.zeros(len(label_dict["boxes"]), 6, device=self.device)
            label_xcycwh[:, 0] = i
            label_xcycwh[:, 1] = label_dict["labels"]
            label_xcycwh[:, 2:6] = label_dict["boxes"]

            # normalize bounding boxes to [0, 1]
            label_xcycwh[:, 2:6:2] /= width
            label_xcycwh[:, 3:6:2] /= height

            # convert from x1y1x2y2 to xcycwh
            label_xcycwh[:, 4] -= label_xcycwh[:, 2]
            label_xcycwh[:, 5] -= label_xcycwh[:, 3]
            label_xcycwh[:, 2] += label_xcycwh[:, 4] / 2
            label_xcycwh[:, 3] += label_xcycwh[:, 5] / 2

            labels_xcycwh_list.append(label_xcycwh)

        if labels_xcycwh_list:
            labels_xcycwh_tensor = torch.cat(labels_xcycwh_list, dim=0)
        else:
            labels_xcycwh_tensor = torch.zeros((0, 6), device=self.device)

        return labels_xcycwh_tensor
