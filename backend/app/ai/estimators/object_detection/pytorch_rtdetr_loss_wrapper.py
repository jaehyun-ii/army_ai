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
PyTorch-specific RT-DETR loss wrapper for app.ai.
"""

import torch


class PyTorchRTDETRLossWrapper(torch.nn.Module):
    """Wrapper for RT-DETR models to handle loss dict format."""

    def __init__(self, model, device="cpu"):
        super().__init__()
        self.model = model
        self.device = device

        try:
            from ultralytics.models.rtdetr import RTDETRPredictor
            from ultralytics.models.utils.loss import RTDETRDetectionLoss

            # Initialize predictor to get args
            self.predictor = RTDETRPredictor()
            self.model.args = self.predictor.args

            # Initialize criterion for RT-DETR
            # RT-DETR uses different parameters than YOLO
            nc = getattr(model, 'nc', 80)  # number of classes
            self.model.criterion = RTDETRDetectionLoss(nc=nc)

            # Move criterion's internal tensors to the specified device
            device_obj = torch.device(device)
            if hasattr(self.model.criterion, 'device'):
                self.model.criterion.device = device_obj

        except ImportError as e:
            raise ImportError(
                "The 'ultralytics' package is required for RT-DETR models but not installed."
            ) from e

    def forward(self, images, targets=None):
        """
        Forward pass with loss computation.

        :param images: Input images tensor
        :param targets: List of target dicts with 'boxes', 'labels', 'scores'
        :return: Tuple of (total_loss, loss_components_dict)
        """
        items = {"img": images}

        if targets is not None:
            device_obj = torch.device(self.device)

            # Create batch_idx tensor
            batch_idx = torch.cat([
                torch.full_like(t['labels'], i, dtype=torch.float).to(device_obj)
                for i, t in enumerate(targets)
            ])
            items["batch_idx"] = batch_idx

            # Concatenate boxes and labels (ensure all on same device)
            items["cls"] = torch.cat([t['labels'].to(device_obj) for t in targets])
            items["bboxes"] = torch.cat([t['boxes'].to(device_obj) for t in targets])

        # Call model's loss method
        loss_tuple = self.model.loss(items)

        # RT-DETR returns similar format to YOLO
        # Convert to dict format for ART
        if isinstance(loss_tuple, tuple) and len(loss_tuple) == 2:
            loss_tensor = loss_tuple[0]

            # Create dict with RT-DETR loss component names
            # RT-DETR typically has: [giou_loss, cls_loss, l1_loss]
            loss_components = {
                "loss_giou": loss_tensor[0] if loss_tensor.numel() > 0 else torch.tensor(0.0, device=loss_tensor.device),
                "loss_cls": loss_tensor[1] if loss_tensor.numel() > 1 else torch.tensor(0.0, device=loss_tensor.device),
                "loss_l1": loss_tensor[2] if loss_tensor.numel() > 2 else torch.tensor(0.0, device=loss_tensor.device),
                "loss_total": loss_tensor.sum()
            }
            return loss_tensor.sum(), loss_components
        else:
            raise ValueError(f"Unexpected loss format from RT-DETR model: {type(loss_tuple)}")
