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
            # Get raw predictions from model (before NMS/postprocessing)
            # For attack loss mode, we need raw predictions with gradients
            # CRITICAL: Set internal YOLO model to eval mode for correct prediction format
            # But keep gradients enabled with torch.set_grad_enabled(True)

            if self.return_raw_predictions:
                # Attack loss mode: need raw predictions with gradients
                import logging
                logger = logging.getLogger(__name__)

                was_training = self.model.training
                self.model.eval()

                logger.debug(f"[Wrapper] Before model call: x.requires_grad={x.requires_grad}, torch.is_grad_enabled()={torch.is_grad_enabled()}")

                # Use a no_grad=False context to ensure gradients flow
                # Even in eval mode, we can compute gradients by enabling grad explicitly
                with torch.set_grad_enabled(True):
                    # Prefer internal _predict_once to get raw head outputs without NMS/inference_mode
                    if hasattr(self.model, "_predict_once"):
                        logger.debug("[Wrapper] Using self.model._predict_once(x)")
                        preds = self.model._predict_once(x)
                    elif hasattr(self.model, "forward"):
                        logger.debug("[Wrapper] Using self.model.forward(x)")
                        preds = self.model.forward(x)
                    else:
                        logger.debug("[Wrapper] Using self.model(x)")
                        preds = self.model(x)

                logger.debug(f"[Wrapper] After model call: preds type={type(preds)}, requires_grad={preds.requires_grad if isinstance(preds, torch.Tensor) else 'N/A'}")

                # Restore training mode
                if was_training:
                    self.model.train()
            else:
                # Normal inference: just call forward
                if hasattr(self.model, 'forward'):
                    preds = self.model.forward(x)
                else:
                    preds = self.model(x)

            # If attack loss mode, return raw predictions in YOLO format (xywh)
            # Raw predictions need to be in format: (batch_size, num_proposals, features)
            # IMPORTANT: Keep boxes in xywh format (YOLO standard format for this project)
            if self.return_raw_predictions:
                # YOLO v8+ outputs: (batch_size, 84, 8400) for 80-class models
                #   - 84 = 4 bbox (xywh) + 80 classes (no objectness)
                #   - 8400 = total anchors from all feature levels
                # YOLO v5 outputs: (batch_size, 85, num_anchors)
                #   - 85 = 4 bbox (xywh) + 1 obj + 80 classes

                # Accept dynamic class counts instead of hard-coding 84/85
                expected_no_obj = 4 + self.num_classes          # v8 style (no objectness)
                expected_with_obj = 5 + self.num_classes        # v5 style (with objectness)
                valid_feature_dims = {expected_no_obj, expected_with_obj}
                min_reasonable_feat = 5       # at least 4 coords + 1 score/class
                max_reasonable_feat = 512     # anchors dimension is much larger (thousands)
                seen_shapes: list[str] = []

                def tensor_to_preds(t: torch.Tensor) -> list[torch.Tensor]:
                    """Normalize various tensor shapes to (B, N, F)."""
                    out: list[torch.Tensor] = []
                    seen_shapes.append(str(tuple(t.shape)))

                    if t.dim() == 3:
                        # (B, F, N) or (B, N, F)
                        if min_reasonable_feat <= t.size(1) <= max_reasonable_feat:
                            out.append(t.permute(0, 2, 1))
                        elif min_reasonable_feat <= t.size(2) <= max_reasonable_feat:
                            out.append(t)
                    elif t.dim() == 4:
                        # Try each axis (except batch) as feature dim
                        b, d1, d2, d3 = t.shape
                        candidates = [
                            (1, d1, (0, 2, 3, 1)),
                            (2, d2, (0, 1, 3, 2)),
                            (3, d3, (0, 1, 2, 3)),  # already last
                        ]
                        for axis_idx, feat_dim, perm in candidates:
                            if min_reasonable_feat <= feat_dim <= max_reasonable_feat:
                                permuted = t.permute(*perm) if perm != (0, 1, 2, 3) else t
                                out.append(permuted.reshape(b, -1, feat_dim))
                                break
                    return out

                def collect_tensors(obj) -> list[torch.Tensor]:
                    out: list[torch.Tensor] = []
                    if torch.is_tensor(obj):
                        out.extend(tensor_to_preds(obj))
                        return out
                    if isinstance(obj, (list, tuple)):
                        for item in obj:
                            out.extend(collect_tensors(item))
                        return out
                    if isinstance(obj, dict):
                        for key in ("preds", "pred", "output", "outputs", "raw"):
                            if key in obj:
                                out.extend(collect_tensors(obj[key]))
                        return out
                    return out

                pred_tensors: list[torch.Tensor] = collect_tensors(preds)

                # Fallback: try calling underlying .model if nothing found
                if not pred_tensors and hasattr(self.model, "model"):
                    try:
                        raw_preds = self.model.model(x)
                        pred_tensors = collect_tensors(raw_preds)
                    except Exception:
                        pass

                if not pred_tensors:
                    # Multiple prediction heads - concatenate along spatial dimension
                    # Handle nested lists/tuples from some Ultralytics variants.
                        raise ValueError(
                            "No valid prediction tensors found for attack loss. "
                            f"Expected YOLO raw preds with ~{valid_feature_dims} features "
                            f"(found none with feature dim in [{min_reasonable_feat}, {max_reasonable_feat}]). "
                            f"Seen shapes: {seen_shapes}"
                        )

                # Group tensors by feature dimension to avoid mismatched cat
                by_feat: dict[int, list[torch.Tensor]] = {}
                for pred in pred_tensors:
                    feat_dim = pred.size(2)
                    if feat_dim not in valid_feature_dims and not (min_reasonable_feat <= feat_dim <= max_reasonable_feat):
                        continue
                    by_feat.setdefault(int(feat_dim), []).append(pred)

                if not by_feat:
                    raise ValueError(
                        f"No compatible YOLO prediction tensors found for attack loss. "
                        f"Expected feature dimension in {valid_feature_dims} or "
                        f"any value within [{min_reasonable_feat}, {max_reasonable_feat}]. "
                        f"Seen shapes: {seen_shapes}"
                    )

                # Prefer the expected feature dims; otherwise take the largest group
                chosen_feat = None
                for f in valid_feature_dims:
                    if f in by_feat:
                        chosen_feat = f
                        break
                if chosen_feat is None:
                    # Choose the feature size with the most tensors
                    chosen_feat = max(by_feat.items(), key=lambda kv: len(kv[1]))[0]

                tensors_to_cat = by_feat[chosen_feat]
                concatenated_preds = torch.cat(tensors_to_cat, dim=1)  # (B, total_anchors, F_chosen)
                return concatenated_preds
            else:
                # Single prediction tensor: (batch, features, num_anchors)
                # Transpose to (batch, num_anchors, features)
                if preds.dim() == 3:
                    if min_reasonable_feat <= preds.size(1) <= max_reasonable_feat:
                        # YOLO v8/v5 format: (batch, features, num_anchors)
                        return preds.permute(0, 2, 1)
                    if min_reasonable_feat <= preds.size(2) <= max_reasonable_feat:
                        # Already (batch, anchors, features)
                        return preds
                else:
                    # Already in correct format or unknown format
                    return preds

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
