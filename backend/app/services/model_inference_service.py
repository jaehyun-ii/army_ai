"""
Model inference service using ART estimators.

This service provides inference capabilities using the ART (Adversarial Robustness Toolbox)
estimator framework for object detection models.
"""
import base64
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np

from app.ai.estimators.object_detection import model_factory

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box in normalized (0-1) coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width (same units as coordinates)."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height (same units as coordinates)."""
        return max(0.0, self.y2 - self.y1)

    @property
    def x_center(self) -> float:
        """Box center x coordinate."""
        return self.x1 + self.width / 2

    @property
    def y_center(self) -> float:
        """Box center y coordinate."""
        return self.y1 + self.height / 2


@dataclass
class Detection:
    """Single detection result."""
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float


@dataclass
class DetectionResult:
    """Detection results from inference."""
    detections: list[Detection]
    inference_time_ms: float = 0.0


class ModelInferenceService:
    """
    Inference service using ART estimators.

    This service manages estimator instances and provides inference capabilities.
    """

    def __init__(self):
        """Initialize inference service."""
        # In-memory cache of loaded estimators
        # Format: {version_id: {"estimator": estimator, "class_names": [...], ...}}
        self._loaded_estimators: Dict[str, Dict[str, Any]] = {}

    async def run_inference(
        self,
        version_id: str,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        """
        Run object detection inference on an image.

        Args:
            version_id: Model version ID
            image: Input image as numpy array (H, W, C) in BGR format
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS (not used in ART estimators)

        Returns:
            Detection results
        """
        import time

        # Check if estimator is loaded
        if version_id not in self._loaded_estimators:
            raise ValueError(
                f"Model {version_id} not loaded. Load it first using model management service."
            )

        estimator_data = self._loaded_estimators[version_id]
        estimator = estimator_data["estimator"]
        class_names = estimator_data.get("class_names", [])

        # Convert BGR to RGB (ART expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to model's expected input size
        # Get input size from estimator's input_shape: (C, H, W)
        if hasattr(estimator, 'input_shape') and estimator.input_shape is not None:
            # input_shape is (C, H, W), we need (H, W)
            expected_h, expected_w = estimator.input_shape[1], estimator.input_shape[2]
            current_h, current_w = image_rgb.shape[:2]

            if current_h != expected_h or current_w != expected_w:
                image_rgb = cv2.resize(image_rgb, (expected_w, expected_h))
                logger.info(f"Resized image from ({current_h}, {current_w}) to ({expected_h}, {expected_w})")

        # Add batch dimension and convert to NCHW format (standard for PyTorch/ART)
        # [H, W, C] -> [1, H, W, C] -> [1, C, H, W]
        # IMPORTANT: Keep in [0, 255] range for YOLO models (clip_values=(0, 255))
        # DO NOT normalize to [0, 1] - YOLO models handle normalization internally
        image_batch = np.expand_dims(image_rgb, axis=0).astype(np.float32)  # [1, H, W, C], keep [0, 255]
        image_batch = np.transpose(image_batch, (0, 3, 1, 2))  # [1, C, H, W]

        logger.info(f"Image batch shape: {image_batch.shape}, dtype: {image_batch.dtype}, range: [{image_batch.min():.3f}, {image_batch.max():.3f}]")

        # Run inference
        start_time = time.time()
        predictions = estimator.predict(image_batch)
        inference_time_ms = (time.time() - start_time) * 1000

        # Debug: Log predictions structure
        logger.info(f"Predictions type: {type(predictions)}")
        logger.info(f"Predictions length: {len(predictions) if predictions else 0}")
        if predictions and len(predictions) > 0:
            logger.info(f"First prediction type: {type(predictions[0])}")
            logger.info(f"First prediction keys: {predictions[0].keys() if isinstance(predictions[0], dict) else 'Not a dict'}")
            if isinstance(predictions[0], dict):
                for key, value in predictions[0].items():
                    logger.info(f"  {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")

        # Convert ART predictions to DetectionResult format
        # predictions is a list of dicts with 'boxes', 'labels', 'scores'
        detections = []
        if predictions and len(predictions) > 0:
            pred = predictions[0]  # First image in batch

            # Handle both tensor and numpy array formats
            import torch
            boxes = pred.get('boxes', np.array([]))
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()

            labels = pred.get('labels', np.array([]))
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()

            scores = pred.get('scores', np.array([]))
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()

            logger.info(f"After conversion - boxes: {len(boxes)}, labels: {len(labels)}, scores: {len(scores)}")

            original_h, original_w = image.shape[:2]
            inference_h, inference_w = image_rgb.shape[:2]
            # Prevent divide-by-zero
            inference_w = max(1, inference_w)
            inference_h = max(1, inference_h)

            logger.info(f"Normalizing prediction boxes using inference size {inference_w}x{inference_h}")

            for box, label, score in zip(boxes, labels, scores):
                # Filter by confidence threshold
                if score < conf_threshold:
                    continue

                # box format from ART: [x1, y1, x2, y2] relative to inference size
                x1, y1, x2, y2 = box

                # Normalize to 0-1 range using inference dimensions (equivalent to original dimensions)
                normalized_x1 = min(1.0, max(0.0, x1 / inference_w))
                normalized_y1 = min(1.0, max(0.0, y1 / inference_h))
                normalized_x2 = min(1.0, max(0.0, x2 / inference_w))
                normalized_y2 = min(1.0, max(0.0, y2 / inference_h))

                # Get class name
                class_id_int = int(label)
                class_name = class_names[class_id_int] if class_id_int < len(class_names) else f"class_{class_id_int}"

                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x1=float(normalized_x1),
                            y1=float(normalized_y1),
                            x2=float(normalized_x2),
                            y2=float(normalized_y2)
                        ),
                        class_id=class_id_int,
                        class_name=class_name,
                        confidence=float(score)
                    )
                )

        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time_ms
        )

    async def decode_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        return image

    def register_estimator(
        self,
        version_id: str,
        estimator: Any,
        class_names: list[str],
        **metadata
    ) -> None:
        """
        Register a loaded estimator for inference.

        Args:
            version_id: Model version ID
            estimator: ART estimator instance
            class_names: List of class names
            **metadata: Additional metadata to store
        """
        self._loaded_estimators[version_id] = {
            "estimator": estimator,
            "class_names": class_names,
            **metadata
        }

    def unregister_estimator(self, version_id: str) -> None:
        """
        Unregister and remove an estimator.

        Args:
            version_id: Model version ID
        """
        logger.debug(f"Unregistering estimator '{version_id}'")
        if version_id in self._loaded_estimators:
            del self._loaded_estimators[version_id]
            logger.info(f"Successfully unregistered estimator '{version_id}'")

    def is_loaded(self, version_id: str) -> bool:
        """Check if estimator is loaded."""
        return version_id in self._loaded_estimators

    def get_estimator(self, version_id: str) -> dict | None:
        """Get loaded estimator info by ID."""
        return self._loaded_estimators.get(version_id)

    def get_loaded_estimator_ids(self) -> list[str]:
        """Get list of loaded estimator IDs."""
        return list(self._loaded_estimators.keys())


model_inference_service = ModelInferenceService()
