"""
Module containing estimators for object detection.
Supports YOLO, RT-DETR, DETR, Detection Transformer, Faster R-CNN, and EfficientDet.
"""

from app.ai.estimators.object_detection.object_detector import ObjectDetectorMixin
from app.ai.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

# YOLO
from app.ai.estimators.object_detection.pytorch_yolo import PyTorchYolo

# RT-DETR
from app.ai.estimators.object_detection.pytorch_rtdetr import PyTorchRTDETR

# DETR and Detection Transformer
from app.ai.estimators.object_detection.pytorch_detection_transformer import PyTorchDetectionTransformer

# Faster R-CNN
from app.ai.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from app.ai.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN

# EfficientDet
from app.ai.estimators.object_detection.pytorch_efficientdet import PyTorchEfficientDet

# Model Factory - for loading .pt files automatically
from app.ai.estimators.object_detection.model_factory import ModelFactory, model_factory
