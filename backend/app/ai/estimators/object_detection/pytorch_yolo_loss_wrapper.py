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
PyTorch-specific YOLO loss wrapper for ART for yolo versions 8 and above.
"""

import torch


class PyTorchYoloLossWrapper(torch.nn.Module):
    """Wrapper for YOLO v8+ models to handle loss dict format."""

    def __init__(self, model, name):
        super().__init__()
        self.model = model
        self.return_raw_predictions = False  # Flag for attack loss mode
        # Determine number of classes for dynamic feature-size checks
        if hasattr(model, "nc"):
            self.num_classes = int(model.nc)
        elif hasattr(model, "names"):
            self.num_classes = len(model.names)
        else:
            # Fallback to COCO classes if not available
            self.num_classes = 80
        # YOLOv8/v10 heads do not expose a separate objectness logit.
        self.has_objectness = False
        try:
            from ultralytics.models.yolo.detect import DetectionPredictor
            from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss

            self.detection_predictor = DetectionPredictor()
            self.model.args = self.detection_predictor.args
            if "v10" in name:
                self.model.criterion = E2EDetectLoss(model)
            else:
                self.model.criterion = v8DetectionLoss(model)
        except ImportError as e:
            raise ImportError("The 'ultralytics' package is required for YOLO v8+ models but not installed.") from e

    def forward(self, x, targets=None):
        """Transforms the target to dict expected by model.loss"""
        # When return_raw_predictions=True, skip training loss and return raw predictions
        # This allows attack loss computation while keeping wrapper in training mode
        if self.training and not self.return_raw_predictions:
            if targets is None:
                raise ValueError("Targets should not be None when training.")

            # Align model/loss buffers with incoming tensor device to avoid CPU/GPU mix
            device = x.device
            self.model.to(device)
            if hasattr(self.model, "criterion"):
                criterion = self.model.criterion
                # Some Ultralytics loss classes are not nn.Modules, so update their device/tensors manually
                if hasattr(criterion, "device"):
                    criterion.device = device
                if hasattr(criterion, "stride"):
                    if isinstance(criterion.stride, torch.Tensor):
                        criterion.stride = criterion.stride.to(device)
                    else:
                        criterion.stride = torch.as_tensor(criterion.stride, device=device, dtype=x.dtype)
                if hasattr(criterion, "bbox_loss") and hasattr(criterion.bbox_loss, "to"):
                    criterion.bbox_loss = criterion.bbox_loss.to(device)
                if hasattr(criterion, "bce") and hasattr(criterion.bce, "to"):
                    criterion.bce = criterion.bce.to(device)
                # v8DetectionLoss may not implement .to(), so guard it
                if hasattr(criterion, "to"):
                    criterion.to(device)
                # ultralytics loss keeps `proj` as a plain tensor (not registered buffer)
                if hasattr(criterion, "proj"):
                    criterion.proj = criterion.proj.to(device)
                # Move assigner to device (contains anchor points and other tensors)
                if hasattr(criterion, "assigner"):
                    if hasattr(criterion.assigner, "to"):
                        criterion.assigner.to(device)
                    # Also set device attribute for ultralytics compatibility
                    if hasattr(criterion.assigner, "device"):
                        criterion.assigner.device = device

            # Ensure targets tensor is on the same device
            targets = targets.to(device)

            # Create items dict and ensure all tensors are on the same device
            items = {}
            items["batch_idx"] = targets[:, 0].to(device)
            items["bboxes"] = targets[:, 2:6].to(device)
            items["cls"] = targets[:, 1].to(device)
            items["img"] = x.to(device)
            loss, loss_components = self.model.loss(items)
            loss_components_dict = {"loss_total": loss.sum()}
            loss_components_dict["loss_box"] = loss_components[0].sum()
            loss_components_dict["loss_cls"] = loss_components[1].sum()
            loss_components_dict["loss_dfl"] = loss_components[2].sum()
            return loss_components_dict
        else:
            # If attack loss mode, return decoded raw predictions in xywh (center) + class logits
            if self.return_raw_predictions:
                return self._decode_ultralytics_yolo(x)

            # Get raw predictions from model (before NMS/postprocessing)
            # CRITICAL: Must bypass Ultralytics' __call__ which uses @smart_inference_mode()
            # This decorator blocks gradients even with torch.set_grad_enabled(True)
            if hasattr(self.model, '_predict_once'):
                preds = self.model._predict_once(x)
            else:
                preds = self.model(x)

            if isinstance(preds, tuple):
                preds = preds[0]

            # Normal inference mode: Apply NMS and postprocessing
            self.detection_predictor.model = self.model
            self.detection_predictor.batch = [x]
            preds = self.detection_predictor.postprocess(preds, x, x)
            items = []
            for pred in preds:
                items.append(
                    {"boxes": pred.boxes.xyxy, "scores": pred.boxes.conf, "labels": pred.boxes.cls.type(torch.int)}
                )
            return items

    def _decode_ultralytics_yolo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode YOLO v8+ raw head outputs into xywh (center, pixel) + class logits.
        Mirrors AEGIS differentiable_api.py behavior to keep gradients.
        """
        if not hasattr(self.model, "model"):
            raise ValueError("Ultralytics YOLO decode requires model.model with a Detect head.")

        head = self.model.model[-1]
        if not hasattr(head, "reg_max") or not hasattr(head, "nc") or not hasattr(head, "stride"):
            raise ValueError("Detect head missing reg_max/nc/stride needed for raw decode.")

        reg_max = int(head.reg_max)
        num_classes = int(head.nc)
        box_split_size = reg_max * 4
        cls_split_size = num_classes

        # Force train mode to get raw feature maps, then restore previous state
        was_training = self.model.training
        self.model.train()
        raw_feature_maps = self.model(x)
        self.model.train(mode=was_training)

        if isinstance(raw_feature_maps, tuple):
            raw_feature_maps = raw_feature_maps[0]

        if torch.is_tensor(raw_feature_maps):
            raw_feature_maps = [raw_feature_maps]

        if not isinstance(raw_feature_maps, (list, tuple)) or not raw_feature_maps:
            raise ValueError("Expected list of raw feature maps from YOLO head for decoding.")

        feature_maps = [fm for fm in raw_feature_maps if torch.is_tensor(fm)]
        if not feature_maps:
            raise ValueError("No tensor feature maps found for YOLO decoding.")

        for fm in feature_maps:
            if fm.dim() != 4:
                raise ValueError(f"Expected 4D feature maps for YOLO decode, got shape {tuple(fm.shape)}.")

        batch_size = x.shape[0]

        # Flatten feature maps into (B, N, F)
        output_components = [
            fm.view(batch_size, fm.shape[1], -1).permute(0, 2, 1)
            for fm in feature_maps
        ]
        inference_output = torch.cat(output_components, dim=1)

        expected_features = box_split_size + cls_split_size
        if inference_output.size(-1) < expected_features:
            raise ValueError(
                f"YOLO head output has {inference_output.size(-1)} features, "
                f"expected at least {expected_features} (reg_max={reg_max}, nc={num_classes})."
            )
        if inference_output.size(-1) != expected_features:
            inference_output = inference_output[..., :expected_features]

        box_raw, cls_raw = inference_output.split((box_split_size, cls_split_size), dim=-1)

        # Build anchor points and stride tensors
        strides = head.stride
        if not torch.is_tensor(strides):
            strides = torch.tensor(strides, device=x.device, dtype=box_raw.dtype)
        else:
            strides = strides.to(device=x.device, dtype=box_raw.dtype)

        if len(strides) != len(feature_maps):
            raise ValueError(
                f"Stride count ({len(strides)}) does not match feature maps ({len(feature_maps)})."
            )

        anchor_points_list = []
        strides_tensor_list = []
        for fm, stride in zip(feature_maps, strides):
            h, w = fm.shape[2:]
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing='ij'
            )
            anchor_points_list.append(torch.stack((grid_x, grid_y), 2).view(1, -1, 2))
            strides_tensor_list.append(torch.full((1, h * w, 1), stride, device=x.device, dtype=box_raw.dtype))

        anchor_points = torch.cat(anchor_points_list, dim=1).expand(batch_size, -1, -1)
        strides_tensor = torch.cat(strides_tensor_list, dim=1).expand(batch_size, -1, -1)

        # Decode distances (DFL / reg_max)
        box_dist_softmax = box_raw.view(batch_size, -1, 4, reg_max).softmax(3)
        reg_max_range = torch.arange(reg_max, device=x.device, dtype=box_raw.dtype)
        box_offsets = (box_dist_softmax * reg_max_range).sum(3)

        lt, rb = box_offsets.chunk(2, -1)
        box_xyxy = torch.cat((anchor_points - lt, anchor_points + rb), -1) * strides_tensor

        # Convert to xywh (center) to match DetectionAttackLoss expectation
        from app.ai.losses.box_utils import xyxy2xywh

        box_xywh = xyxy2xywh(box_xyxy)
        return torch.cat([box_xywh, cls_raw], dim=-1)
