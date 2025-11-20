"""
Base detector interface - Legacy support for old custom model system.

NOTE: This module is deprecated. Use ART estimators instead:
- app.ai.estimators.object_detection.PyTorchYolo
- app.ai.estimators.object_detection.PyTorchRTDETR
- app.ai.estimators.object_detection.PyTorchFasterRCNN

This file exists only for backward compatibility with old code that
imports DetectionResult and BaseObjectDetector.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class BoundingBox:
    """Bounding box for object detection."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) format."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return (x, y, width, height) format."""
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return (self.x1, self.y1, width, height)


@dataclass
class Detection:
    """Single object detection result."""
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    """Object detection result containing multiple detections."""
    detections: List[Detection]
    image_shape: Tuple[int, int, int]  # (height, width, channels)
    inference_time_ms: Optional[float] = None

    @property
    def num_detections(self) -> int:
        """Number of detected objects."""
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Filter detections by confidence threshold."""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionResult(
            detections=filtered,
            image_shape=self.image_shape,
            inference_time_ms=self.inference_time_ms,
        )

    def filter_by_class(self, class_ids: List[int]) -> "DetectionResult":
        """Filter detections by class IDs."""
        filtered = [d for d in self.detections if d.class_id in class_ids]
        return DetectionResult(
            detections=filtered,
            image_shape=self.image_shape,
            inference_time_ms=self.inference_time_ms,
        )


class BaseObjectDetector(ABC):
    """
    Base class for object detection models.

    DEPRECATED: Use ART estimators instead.
    This class exists only for backward compatibility.
    """

    def __init__(self, config: dict):
        """
        Initialize detector with config.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_loaded = False
        self.class_names = config.get('class_names', [])

    @abstractmethod
    def load_model(self, weights_path: str, **kwargs) -> None:
        """
        Load model from weights file.

        Args:
            weights_path: Path to model weights
            **kwargs: Additional arguments
        """
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Preprocessed image
        """
        pass

    @abstractmethod
    def predict(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Run model inference.

        Args:
            preprocessed_image: Preprocessed image

        Returns:
            Raw model predictions
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        predictions: np.ndarray,
        original_shape: Tuple[int, int, int],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        """
        Postprocess model predictions.

        Args:
            predictions: Raw model predictions
            original_shape: Original image shape (H, W, C)
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS

        Returns:
            DetectionResult with filtered detections
        """
        pass

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        """
        End-to-end detection pipeline.

        Args:
            image: Input image as numpy array (H, W, C)
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS

        Returns:
            DetectionResult with detections
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        original_shape = image.shape
        preprocessed = self.preprocess(image)
        predictions = self.predict(preprocessed)
        result = self.postprocess(
            predictions,
            original_shape,
            conf_threshold,
            iou_threshold,
        )
        return result
