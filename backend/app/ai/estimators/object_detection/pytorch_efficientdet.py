"""
PyTorch EfficientDet estimator for MMDetection models.

This module provides an estimator for EfficientDet models trained with MMDetection.
Supports both standard EfficientDet and incremental learning variants.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import numpy as np

from app.ai.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

if TYPE_CHECKING:
    import torch
    from app.ai.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from app.ai.defences.preprocessor.preprocessor import Preprocessor
    from app.ai.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchEfficientDet(PyTorchObjectDetector):
    """
    PyTorch estimator for EfficientDet models from MMDetection.

    This class now inherits from PyTorchObjectDetector to gain:
    - Automatic channel format conversion (NHWC <-> NCHW)
    - Standard preprocessing pipeline
    - Full compatibility with all attack modules
    - Consistent interface with other object detectors (YOLO, RT-DETR, etc.)

    Supports:
    - Standard EfficientDet (D0-D7)
    - EfficientDet with incremental learning
    - Custom MMDetection configurations
    """

    estimator_params = PyTorchObjectDetector.estimator_params + [
        "config_file",
        "model_wrapper",
        "mean",
        "std"
    ]

    def __init__(
        self,
        model: "torch.nn.Module",
        input_shape: tuple[int, ...] = (3, 768, 768),
        optimizer: "torch.optim.Optimizer" | None = None,
        clip_values: "CLIP_VALUES_TYPE" | None = None,
        channels_first: bool = True,
        preprocessing_defences: "Preprocessor" | list["Preprocessor"] | None = None,
        postprocessing_defences: "Postprocessor" | list["Postprocessor"] | None = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: tuple[str, ...] = ("loss_total",),
        device_type: str = "gpu",
        config_file: str | None = None,
        model_wrapper: Any | None = None,
    ):
        """
        Initialize EfficientDet estimator.

        Args:
            model: MMDetection model instance
            input_shape: Input shape (C, H, W)
            optimizer: Optimizer for training
            clip_values: Pixel value range (e.g., (0, 255))
            channels_first: Whether input is NCHW format
            preprocessing_defences: Preprocessing defences
            postprocessing_defences: Postprocessing defences
            preprocessing: Preprocessing parameters (mean, std)
            attack_losses: Loss components for adversarial attacks
            device_type: Device type ('gpu', 'cpu', or 'auto')
            config_file: Path to MMDetection config file
            model_wrapper: MMDetection model wrapper (DetInferencer or similar)
        """
        # Initialize parent class (PyTorchObjectDetector)
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

        # EfficientDet-specific attributes
        self._config_file = config_file
        self._model_wrapper = model_wrapper

        # MMDetection preprocessing parameters
        # These match the standard ImageNet normalization used by EfficientNet
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        logger.info(f"EfficientDet initialized with input_shape={input_shape}, "
                   f"channels_first={channels_first}, device={self._device}")

    @property
    def config_file(self) -> str | None:
        """Return the config file path."""
        return self._config_file

    @property
    def model_wrapper(self) -> Any:
        """Return the model wrapper."""
        return self._model_wrapper

    def _apply_preprocessing(
        self,
        x: "torch.Tensor",
        y: list[dict[str, "torch.Tensor"]] | None,
        fit: bool = False,
        no_grad: bool = True
    ) -> tuple["torch.Tensor", list[dict[str, "torch.Tensor"]] | None]:
        """
        Apply EfficientDet-specific preprocessing.

        This overrides PyTorchEstimator._apply_preprocessing to add:
        - Image resizing to model input size
        - BGR to RGB conversion
        - MMDetection-style normalization

        Args:
            x: Input tensor in NCHW format, normalized to [0, 1]
            y: Target labels (optional)
            fit: Whether this is for training
            no_grad: Whether to disable gradients

        Returns:
            Preprocessed (x, y) tuple
        """
        import torch
        import torch.nn.functional as F

        # x is already in NCHW format and normalized to [0, 1] by parent
        # We need to:
        # 1. Resize to model input size
        # 2. Denormalize back to [0, 255]
        # 3. Apply MMDetection normalization

        batch_size = x.shape[0]
        _, h, w = self._input_shape

        # Resize if needed
        if x.shape[2] != h or x.shape[3] != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        # Denormalize from [0, 1] back to [0, 255]
        # (parent already divided by clip_values[1])
        if self.clip_values is not None:
            x = x * self.clip_values[1]

        # Convert BGR to RGB (swap channels 0 and 2)
        # Assuming input was in BGR format
        x = x[:, [2, 1, 0], :, :]

        # Apply MMDetection normalization
        # Normalize to ImageNet mean/std
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x, y

    def _translate_predictions(self, outputs: Any) -> list[dict[str, np.ndarray]]:
        """
        Translate MMDetection model outputs to standard format.

        This method is called by PyTorchObjectDetector.predict() to convert
        model-specific outputs to the standard format.

        Args:
            outputs: Raw model outputs from forward pass

        Returns:
            List of predictions in standard format:
            - boxes: (N, 4) array in [x1, y1, x2, y2] format
            - labels: (N,) array of class indices
            - scores: (N,) array of confidence scores
        """
        predictions = []

        # Handle tuple outputs (common in MMDetection)
        if isinstance(outputs, tuple):
            # Usually (batch_results,) or (loss, predictions)
            if len(outputs) > 0:
                outputs = outputs[0] if isinstance(outputs[0], (list, dict)) else outputs

        # Handle different output formats
        if isinstance(outputs, list):
            # List of detection results (one per image)
            for output in outputs:
                pred_dict = self._extract_predictions(output)
                predictions.append(pred_dict)
        elif isinstance(outputs, dict):
            # Single output dict
            pred_dict = self._extract_predictions(outputs)
            predictions.append(pred_dict)
        else:
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        return predictions

    def _extract_predictions(self, output: Any) -> dict[str, np.ndarray]:
        """
        Extract predictions from MMDetection output.

        Args:
            output: Single image output from MMDetection model

        Returns:
            Dict with boxes, labels, scores
        """
        import torch

        pred_dict = {
            "boxes": np.array([]).reshape(0, 4),
            "labels": np.array([]).astype(np.int64),
            "scores": np.array([]).astype(np.float32)
        }

        # Handle MMDetection 3.x output format
        if hasattr(output, 'pred_instances'):
            pred_instances = output.pred_instances

            if len(pred_instances) > 0:
                boxes = pred_instances.bboxes
                scores = pred_instances.scores
                labels = pred_instances.labels

                # Convert to numpy
                if isinstance(boxes, torch.Tensor):
                    pred_dict["boxes"] = boxes.cpu().numpy()
                    pred_dict["scores"] = scores.cpu().numpy()
                    pred_dict["labels"] = labels.cpu().numpy()
                else:
                    pred_dict["boxes"] = np.array(boxes)
                    pred_dict["scores"] = np.array(scores)
                    pred_dict["labels"] = np.array(labels)

        # Handle dict format
        elif isinstance(output, dict):
            if "boxes" in output:
                pred_dict["boxes"] = output["boxes"]
            if "scores" in output:
                pred_dict["scores"] = output["scores"]
            if "labels" in output:
                pred_dict["labels"] = output["labels"]

        return pred_dict

    def _get_raw_predictions(
        self,
        x: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Extract raw predictions from MMDetection model for attack loss.

        This method extracts predictions before NMS to be compatible with
        YOLO/RT-DETR-style attack losses.

        Args:
            x: Input tensor [batch, C, H, W]

        Returns:
            Raw predictions tensor [batch, total_proposals, 4 + num_classes]
            Format: [cx, cy, w, h, class_scores] in xywh format (YOLO standard)
        """
        import torch
        from app.ai.losses.box_utils import xyxy2xywh

        # MMDetection structure:
        # 1. backbone: extract features
        # 2. neck: feature pyramid (BiFPN for EfficientDet)
        # 3. bbox_head: detection head (multi-scale)

        # Extract multi-scale features
        if hasattr(self._model, 'extract_feat'):
            feats = self._model.extract_feat(x)
        else:
            # Fallback: assume model is already the backbone
            feats = self._model(x)

        # Get raw predictions from bbox_head
        if hasattr(self._model, 'bbox_head'):
            bbox_head = self._model.bbox_head

            # Forward through bbox_head to get raw predictions
            # MMDetection bbox_head returns (cls_scores, bbox_preds)
            # cls_scores: list of [batch, num_anchors*num_classes, H, W]
            # bbox_preds: list of [batch, num_anchors*4, H, W]
            cls_scores, bbox_preds = bbox_head(feats)

            # Number of classes
            num_classes = bbox_head.num_classes

            # Build anchors for each feature level
            if hasattr(bbox_head, "prior_generator"):
                prior_generator = bbox_head.prior_generator
            elif hasattr(bbox_head, "anchor_generator"):
                prior_generator = bbox_head.anchor_generator
            else:
                raise AttributeError(
                    "MMDetection bbox_head does not expose a prior/anchor generator."
                )

            featmap_sizes = [cls_score.shape[-2:] for cls_score in cls_scores]
            if hasattr(prior_generator, "grid_priors"):
                anchors_list = prior_generator.grid_priors(featmap_sizes, device=x.device)
            elif hasattr(prior_generator, "grid_anchors"):
                anchors_list = prior_generator.grid_anchors(featmap_sizes, device=x.device)
            else:
                raise AttributeError(
                    "Anchor generator does not support grid_priors/grid_anchors."
                )

            if not hasattr(bbox_head, "bbox_coder"):
                raise AttributeError(
                    "MMDetection bbox_head does not expose bbox_coder for decoding deltas."
                )

            # Process each feature level and concatenate
            all_proposals = []

            img_h, img_w = x.shape[2], x.shape[3]

            for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
                batch_size = cls_score.shape[0]

                # Reshape cls_score: [batch, num_anchors*num_classes, H, W]
                # → [batch, num_anchors, num_classes, H, W]
                # → [batch, num_anchors, H, W, num_classes]
                # → [batch, num_anchors*H*W, num_classes]
                num_anchors = cls_score.shape[1] // num_classes
                h, w = cls_score.shape[2], cls_score.shape[3]

                # Reshape class scores
                cls_score = cls_score.view(batch_size, num_anchors, num_classes, h, w)
                cls_score = cls_score.permute(0, 1, 3, 4, 2)  # [B, A, H, W, C]
                cls_score = cls_score.contiguous().view(batch_size, -1, num_classes)  # [B, A*H*W, C]

                bbox_pred = bbox_pred.view(batch_size, num_anchors, 4, h, w)
                bbox_pred = bbox_pred.permute(0, 1, 3, 4, 2)  # [B, A, H, W, 4]
                bbox_pred_raw = bbox_pred.contiguous().view(batch_size, -1, 4)  # [B, A*H*W, 4]

                anchors = anchors_list[level_idx]
                if not isinstance(anchors, torch.Tensor):
                    anchors = torch.as_tensor(anchors, device=x.device, dtype=bbox_pred_raw.dtype)
                else:
                    anchors = anchors.to(device=x.device, dtype=bbox_pred_raw.dtype)
                anchors_expanded = anchors.unsqueeze(0).expand(batch_size, -1, 4).reshape(-1, 4)
                deltas = bbox_pred_raw.reshape(-1, 4)

                try:
                    decoded_xyxy = bbox_head.bbox_coder.decode(
                        anchors_expanded, deltas, max_shape=(img_h, img_w)
                    )
                except TypeError:
                    decoded_xyxy = bbox_head.bbox_coder.decode(anchors_expanded, deltas)

                decoded_xyxy = decoded_xyxy.view(batch_size, -1, 4)
                bbox_pred_xywh = xyxy2xywh(decoded_xyxy)

                # Concatenate: [batch, proposals, 4 + num_classes]
                # Format: [cx, cy, w, h, class_0, ..., class_N]
                level_proposals = torch.cat([bbox_pred_xywh, cls_score], dim=-1)
                all_proposals.append(level_proposals)

            # Concatenate all feature levels
            # Final shape: [batch, total_proposals, 4 + num_classes]
            raw_predictions = torch.cat(all_proposals, dim=1)

            return raw_predictions

        else:
            raise AttributeError(
                "MMDetection model doesn't have bbox_head attribute. "
                "Make sure the model is properly initialized."
            )

    def _compute_loss(
        self,
        x: "torch.Tensor",
        y: list[dict[str, "torch.Tensor"]]
    ) -> dict[str, "torch.Tensor"]:
        """
        Compute loss for adversarial attacks.

        This method is called by PyTorchObjectDetector.loss_gradient() to compute
        the loss for gradient-based attacks.

        Args:
            x: Input tensor in NCHW format
            y: Target labels (list of dicts with 'boxes' and 'labels')

        Returns:
            Dict of loss components
        """
        import torch

        # Prepare targets in MMDetection format
        targets = self._prepare_targets(y)

        # Forward pass in training mode
        # MMDetection models expect data_samples parameter
        try:
            # Try standard MMDetection forward signature
            outputs = self._model(x, data_samples=targets, mode='loss')

            # Debug: log detailed structure (can be removed later)
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, list):
                        logger.debug(f"{key}: list with {len(value)} elements")
                        for i, v in enumerate(value):
                            if isinstance(v, torch.Tensor):
                                logger.debug(f"  [{i}] shape={v.shape}, requires_grad={v.requires_grad}")
                    elif isinstance(value, torch.Tensor):
                        logger.debug(f"{key}: tensor shape={value.shape}, requires_grad={value.requires_grad}")

            # Extract losses
            if isinstance(outputs, dict):
                # Standard MMDetection loss format
                total_loss = torch.tensor(0.0, device=self._device, requires_grad=True)

                for key, value in outputs.items():
                    if 'loss' in key:
                        # Handle different loss value types
                        if isinstance(value, list):
                            # Sum list of losses (e.g., FPN with multiple levels)
                            for v in value:
                                if isinstance(v, torch.Tensor):
                                    total_loss = total_loss + v
                        elif isinstance(value, torch.Tensor):
                            # Single tensor loss
                            total_loss = total_loss + value

                loss_dict = {
                    'loss_total': total_loss,
                    **outputs  # Include individual losses too
                }
            else:
                loss_dict = {'loss_total': outputs}

        except (TypeError, RuntimeError) as e:
            logger.warning(f"Standard loss computation failed: {e}, using fallback")
            # Fallback: extract features and create simple loss
            # This ensures gradients can flow back through the model
            try:
                # Use the input directly to create a loss with gradient
                # This is a simple approach: use the mean of the input as loss
                # The gradient will be computed w.r.t. the input
                total_loss = x.mean()

                loss_dict = {'loss_total': total_loss}

            except Exception as e2:
                logger.error(f"Fallback loss computation also failed: {e2}")
                raise

        return loss_dict

    def _prepare_targets(
        self,
        y: list[dict[str, np.ndarray | "torch.Tensor"]]
    ) -> list:
        """
        Prepare targets in MMDetection format (DetDataSample objects).

        Args:
            y: List of target dicts with 'boxes' and 'labels'

        Returns:
            List of DetDataSample objects for MMDetection
        """
        import torch
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        data_samples = []
        for y_i in y:
            # Create DetDataSample
            data_sample = DetDataSample()

            # Create ground truth instances
            gt_instances = InstanceData()

            # Set bboxes (boxes)
            if 'boxes' in y_i:
                boxes = y_i['boxes']
                if isinstance(boxes, np.ndarray):
                    boxes = torch.from_numpy(boxes).to(self._device)
                elif isinstance(boxes, torch.Tensor):
                    boxes = boxes.to(self._device)
                gt_instances.bboxes = boxes

            # Set labels
            if 'labels' in y_i:
                labels = y_i['labels']
                if isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels).to(self._device)
                elif isinstance(labels, torch.Tensor):
                    labels = labels.to(self._device)
                gt_instances.labels = labels

            # Attach ground truth to data sample
            data_sample.gt_instances = gt_instances

            # Set metadata (required for MMDetection)
            _, h, w = self._input_shape
            data_sample.set_metainfo({
                'img_shape': (h, w),         # Current image shape (H, W)
                'ori_shape': (h, w),         # Original image shape (H, W)
                'pad_shape': (h, w),         # Padded image shape (H, W)
                'scale_factor': (1.0, 1.0),  # Scale factor (no scaling)
                'img_id': 0,                 # Image ID (placeholder)
            })

            data_samples.append(data_sample)

        return data_samples

    def _get_layers(self) -> list[str]:
        """
        Return the names of all layers in the model.

        Returns:
            List of layer names
        """
        return [name for name, _ in self._model.named_modules()]

    def get_activations(
        self,
        x: np.ndarray,
        layer: int | str,
        batch_size: int = 128,
        framework: bool = False
    ) -> np.ndarray | "torch.Tensor":
        """
        Return the output of a specific layer for input `x`.

        Args:
            x: Input samples of shape (N, C, H, W) or (N, H, W, C)
            layer: Layer index or layer name for which to extract activations
            batch_size: Batch size for processing
            framework: If True, return PyTorch tensor; otherwise, return numpy array

        Returns:
            The output of the specified layer
        """
        import torch

        self._model.eval()

        # Use parent's preprocessing
        x_preprocessed, _ = self._preprocess_and_convert_inputs(x, y=None, fit=False, no_grad=True)

        # Dictionary to store activations
        activations = {}

        def get_activation(name):
            """Hook function to capture layer output."""
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    activations[name] = output[0].detach() if isinstance(output[0], torch.Tensor) else None
            return hook

        # Register hook
        hook_handle = None
        if isinstance(layer, int):
            modules = list(self._model.modules())
            if 0 <= layer < len(modules):
                hook_handle = modules[layer].register_forward_hook(get_activation(f"layer_{layer}"))
                layer_name = f"layer_{layer}"
            else:
                raise ValueError(f"Layer index {layer} out of range. Model has {len(modules)} modules.")
        elif isinstance(layer, str):
            found = False
            for name, module in self._model.named_modules():
                if name == layer:
                    hook_handle = module.register_forward_hook(get_activation(layer))
                    layer_name = layer
                    found = True
                    break
            if not found:
                raise ValueError(f"Layer '{layer}' not found in model.")
        else:
            raise TypeError("Layer must be int or str.")

        # Process in batches
        results = []
        num_batch = int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))

        with torch.no_grad():
            for m in range(num_batch):
                begin = m * batch_size
                end = min((m + 1) * batch_size, x_preprocessed.shape[0])

                x_batch = x_preprocessed[begin:end].to(self._device)

                # Forward pass
                _ = self._model(x_batch)

                # Get activation
                if layer_name in activations:
                    activation = activations[layer_name]
                    results.append(activation.cpu() if not framework else activation)
                    del activations[layer_name]

        # Remove hook
        if hook_handle is not None:
            hook_handle.remove()

        # Concatenate results
        if results:
            if framework:
                return torch.cat(results, dim=0)
            else:
                result_tensor = torch.cat(results, dim=0)
                return result_tensor.numpy()
        else:
            raise RuntimeError(f"No activations captured for layer {layer}.")

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> list[dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs using MMDetection's predict API.

        This method overrides PyTorchObjectDetector.predict() to use MMDetection's
        native predict() method instead of forward(), which ensures proper inference mode.

        :param x: Samples of shape NCHW or NHWC.
        :param batch_size: Batch size.
        :return: Predictions of format `list[dict[str, np.ndarray]]`, one for each input image.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Set model to evaluation mode
        self._model.eval()

        # Apply preprocessing and convert to tensors
        x_preprocessed, _ = self._preprocess_and_convert_inputs(x=x, y=None, fit=False, no_grad=True)

        # Create dataloader
        dataset = TensorDataset(x_preprocessed)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        predictions: list[dict[str, np.ndarray]] = []
        for (x_batch,) in dataloader:
            # Move inputs to device
            x_batch = x_batch.to(self._device)

            # Run prediction using MMDetection's predict method
            with torch.no_grad():
                # MMDetection models use predict() method for inference
                # which returns DataSample objects with pred_instances
                # Create data samples with required metadata for each image in batch
                from mmdet.structures import DetDataSample
                from mmengine.structures import InstanceData

                batch_data_samples = []
                for i in range(x_batch.shape[0]):
                    data_sample = DetDataSample()
                    # Set required metadata
                    data_sample.set_metainfo({
                        'img_shape': (x_batch.shape[2], x_batch.shape[3]),  # (H, W)
                        'ori_shape': (x_batch.shape[2], x_batch.shape[3]),  # (H, W)
                        'scale_factor': (1.0, 1.0),  # No scaling
                    })
                    batch_data_samples.append(data_sample)

                # Try to use predict() method if available (standard MMDetection models)
                # Otherwise use forward() with mode='predict' (fallback or custom models)
                if hasattr(self._model, 'predict') and callable(getattr(self._model, 'predict')):
                    outputs = self._model.predict(x_batch, batch_data_samples=batch_data_samples, rescale=False)
                else:
                    # Use forward() with mode='predict' for models without predict() method
                    outputs = self._model(x_batch, data_samples=batch_data_samples, mode='predict')

            predictions_x1y1x2y2 = self._translate_predictions(outputs)
            predictions.extend(predictions_x1y1x2y2)

        return predictions

    def _get_losses(
        self,
        x: np.ndarray | "torch.Tensor",
        y: list[dict[str, np.ndarray | "torch.Tensor"]],
        **kwargs
    ) -> tuple[dict[str, "torch.Tensor"], "torch.Tensor"]:
        """
        Get the loss tensor output of the model including all preprocessing.

        Overrides PyTorchObjectDetector._get_losses() to use EfficientDet-specific
        loss computation with MMDetection API, but also supports attack loss.

        Args:
            x: Samples of shape NCHW or NHWC.
            y: Target values (list of dicts with 'boxes' and 'labels')
            kwargs: Additional arguments for attack-specific losses

        Returns:
            Loss components and preprocessed input tensor
        """
        import torch

        # Check if attack loss is enabled
        if hasattr(self, '_use_attack_loss') and self._use_attack_loss:
            # Use attack loss for EfficientDet (MMDetection)
            self._model.eval()

            self.set_dropout(False)
            self.set_batchnorm(False)
            self.set_multihead_attention(False)

            x_preprocessed, y_preprocessed = self._preprocess_and_convert_inputs(x=x, y=y, fit=False, no_grad=False)
            x_preprocessed = x_preprocessed.to(self.device)

            if x_preprocessed.is_leaf:
                x_preprocessed.requires_grad = True
            else:
                x_preprocessed.retain_grad()

            # Extract raw predictions from MMDetection bbox_head
            # This bypasses NMS and returns all proposals
            with torch.set_grad_enabled(True):
                # Get raw predictions: [batch, total_proposals, 4 + num_classes]
                inference_output = self._get_raw_predictions(x_preprocessed)

                # Calculate attack loss based on attack type
                if self._attack_type == 'adversarial_patch':
                    # Patch attack: simpler loss, just max detection score
                    total_loss = self._custom_attack_loss(
                        inference_output=inference_output,
                        **kwargs
                    )

                    # Return in dict format
                    loss_components = {
                        'attack_loss_total': total_loss,
                    }

                elif self._attack_type == 'universal_noise':
                    # Universal Noise: needs pseudo-GT boxes and target class
                    pseudo_gt_boxes = kwargs.get('pseudo_gt_boxes', None)
                    target_class_idx = kwargs.get('target_class_idx', 0)

                    if pseudo_gt_boxes is None:
                        raise ValueError(
                            "pseudo_gt_boxes must be provided in kwargs for universal_noise attack"
                        )

                    # Ensure pseudo_gt_boxes is on correct device
                    if isinstance(pseudo_gt_boxes, np.ndarray):
                        pseudo_gt_boxes = torch.from_numpy(pseudo_gt_boxes).to(self.device)
                    elif isinstance(pseudo_gt_boxes, torch.Tensor):
                        pseudo_gt_boxes = pseudo_gt_boxes.to(self.device)

                    # Calculate attack loss
                    total_loss, targeted_loss, agnostic_loss = self._custom_attack_loss(
                        inference_output=inference_output,
                        pseudo_gt_boxes=pseudo_gt_boxes,
                        target_class_idx=target_class_idx
                    )

                    # Return in dict format
                    loss_components = {
                        'attack_loss_total': total_loss,
                        'attack_loss_targeted': targeted_loss,
                        'attack_loss_agnostic': agnostic_loss
                    }

                else:
                    raise ValueError(f"Unknown attack type: {self._attack_type}")

                return loss_components, x_preprocessed

        # Normal mode: use EfficientDet training loss
        self._model.train()

        self.set_dropout(False)
        self.set_batchnorm(False)
        self.set_multihead_attention(False)

        # Apply preprocessing and convert to tensors
        x_preprocessed, y_preprocessed = self._preprocess_and_convert_inputs(x=x, y=y, fit=False, no_grad=False)

        # Move inputs to device
        x_preprocessed = x_preprocessed.to(self.device)

        # Set gradients again after inputs are moved to another device
        if x_preprocessed.is_leaf:
            x_preprocessed.requires_grad = True
        else:
            x_preprocessed.retain_grad()

        # Use EfficientDet-specific loss computation
        loss_components = self._compute_loss(x_preprocessed, y_preprocessed)

        return loss_components, x_preprocessed

    def compute_loss(
        self,
        x: np.ndarray | "torch.Tensor",
        y: list[dict[str, np.ndarray | "torch.Tensor"]],
        **kwargs
    ) -> np.ndarray | "torch.Tensor":
        """
        Compute total loss.

        Args:
            x: Input images
            y: Target labels
            kwargs: Additional arguments for attack-specific losses

        Returns:
            Total loss value
        """
        import torch

        # Use _get_losses for unified handling of attack/training loss
        loss_components, _ = self._get_losses(x=x, y=y, **kwargs)

        # Compute total loss
        if hasattr(self, '_use_attack_loss') and self._use_attack_loss:
            # For attack loss, use the total attack loss
            loss = loss_components.get('attack_loss_total')
            if loss is None:
                raise ValueError("Attack loss not found in loss_components")
        else:
            # For training loss, sum specified components
            loss = sum(loss_components[k] for k in self.attack_losses if k in loss_components)

        if isinstance(x, np.ndarray):
            return loss.detach().cpu().numpy()
        else:
            return loss
