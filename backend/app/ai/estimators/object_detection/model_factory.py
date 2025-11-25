"""
Model factory for loading object detection models from .pt files.

Automatically detects model type and wraps with appropriate estimator.
Supports custom class mappings for models trained with different class orders.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

from app.ai.estimators.object_detection import (
    PyTorchYolo,
    PyTorchRTDETR,
    PyTorchFasterRCNN,
    PyTorchEfficientDet,
)
from app.ai.estimators.object_detection.class_mapper import (
    ClassMapper,
    detect_class_format,
    COCO_CLASSES,
)

logger = logging.getLogger(__name__)


def _register_custom_mmdet_modules():
    """Register custom MMDetection modules (EfficientDetIncre, etc.)."""
    try:
        # Import MMDetection registry first
        from mmdet.registry import MODELS

        # Check if already registered
        if 'EfficientDetIncre' in MODELS.module_dict:
            logger.info("EfficientDetIncre already registered")
            return True

        # First, try to import standard EfficientDet project modules from mmdetection
        try:
            import sys
            import os
            # Add mmdetection projects to path (Docker container path)
            mmdet_projects_path = "/app/mmdetection/projects"
            if os.path.exists(mmdet_projects_path) and mmdet_projects_path not in sys.path:
                sys.path.insert(0, mmdet_projects_path)
                logger.info(f"Added {mmdet_projects_path} to sys.path")

            # Import EfficientDet project modules
            from EfficientDet.efficientdet import (
                BiFPN,
                EfficientDet,
                EfficientDetSepBNHead,
                HuberLoss
            )
            logger.info("✓ Standard EfficientDet project modules imported successfully")
        except Exception as e:
            logger.debug(f"Could not import standard EfficientDet project (will use custom implementation): {e}")

        # Import custom incremental learning modules to trigger registration
        from app.ai.models.efficientdet_incre import (
            EfficientDetIncre,
            EfficientDetSepBNHeadIncre
        )

        # Verify registration succeeded
        if 'EfficientDetIncre' in MODELS.module_dict:
            logger.info(f"✓ EfficientDetIncre registered successfully in MODELS registry")
            return True
        else:
            logger.error(f"✗ EfficientDetIncre import succeeded but not found in registry")
            logger.error(f"Available custom models: {[k for k in MODELS.module_dict.keys() if 'Efficient' in k]}")
            return False

    except Exception as e:
        logger.error(f"Failed to register custom modules: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


class ModelTypeDetector:
    """Detect model type from .pt file or metadata."""

    @staticmethod
    def detect_from_filename(filename: str) -> str:
        """
        Detect model type from filename.

        Returns:
            'yolo', 'rtdetr', 'faster_rcnn', 'efficientdet', or 'unknown'
        """
        filename_lower = filename.lower()

        if any(x in filename_lower for x in ['yolo', 'yolov', 'yolo11', 'yolo10', 'yolo8']):
            return 'yolo'
        elif 'rtdetr' in filename_lower or 'rt-detr' in filename_lower:
            return 'rtdetr'
        elif 'faster' in filename_lower and 'rcnn' in filename_lower:
            return 'faster_rcnn'
        elif 'efficient' in filename_lower and 'det' in filename_lower:
            return 'efficientdet'

        return 'unknown'

    @staticmethod
    def detect_from_state_dict(model_path: str) -> str:
        """
        Detect model type by inspecting state dict keys.

        Returns:
            Model type string
        """
        try:
            state_dict = torch.load(model_path, map_location='cpu')

            # Get model keys
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    keys = state_dict['model'].state_dict().keys() if hasattr(state_dict['model'], 'state_dict') else []
                else:
                    keys = list(state_dict.keys())
            else:
                keys = []

            keys_str = ' '.join(str(k) for k in keys)

            # YOLO detection
            if any(x in keys_str for x in ['Detect', 'C2f', 'SPPF', 'Conv']):
                return 'yolo'

            # RT-DETR detection
            if any(x in keys_str for x in ['transformer', 'encoder', 'decoder']):
                return 'rtdetr'

            # Faster R-CNN detection
            if any(x in keys_str for x in ['rpn', 'roi_heads', 'backbone']):
                return 'faster_rcnn'

        except Exception as e:
            logger.warning(f"Failed to detect model type from state dict: {e}")

        return 'unknown'


class ModelFactory:
    """Factory for creating estimators from .pt files."""

    def __init__(self):
        self.detector = ModelTypeDetector()

    def load_model(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        input_size: Optional[List[int]] = None,
        device_type: str = "auto",
        clip_values: tuple = (0, 255),
    ):
        """
        Load model from .pt file and wrap with appropriate estimator.

        Args:
            model_path: Path to .pt file
            model_type: Model type ('yolo', 'rtdetr', 'faster_rcnn', 'efficientdet')
                       If None, auto-detect from filename
            class_names: List of class names (optional)
            input_size: [height, width] (optional, default [640, 640])
            device_type: 'gpu', 'cpu', or 'auto'
            clip_values: Image value range (default (0, 255))

        Returns:
            Configured estimator instance
        """
        model_path_obj = Path(model_path)

        # Auto-detect model type if not provided
        if model_type is None:
            model_type = self.detector.detect_from_filename(model_path_obj.name)
            if model_type == 'unknown':
                model_type = self.detector.detect_from_state_dict(model_path)
            logger.info(f"Auto-detected model type: {model_type}")

        # Set defaults
        input_size = input_size or [640, 640]
        class_names = class_names or ["object"]  # Default single class

        # Build config
        config = {
            "class_names": class_names,
            "input_size": input_size,
        }

        # Load based on model type
        if model_type == 'yolo':
            return self._load_yolo(model_path, config, device_type, clip_values)
        elif model_type == 'rtdetr':
            return self._load_rtdetr(model_path, config, device_type, clip_values)
        elif model_type == 'faster_rcnn':
            return self._load_faster_rcnn(model_path, config, device_type, clip_values)
        elif model_type == 'efficientdet':
            return self._load_efficientdet(model_path, config, device_type, clip_values)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _load_yolo(self, model_path: str, config: dict, device_type: str, clip_values: tuple):
        """Load YOLO model using ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is required for YOLO models. Install with: pip install ultralytics")

        # Load ultralytics model
        yolo_model = YOLO(model_path)

        # Determine model name from path
        filename = Path(model_path).stem.lower()
        if 'yolo11' in filename or 'yolov11' in filename:
            model_name = 'yolov11'
        elif 'yolo10' in filename or 'yolov10' in filename:
            model_name = 'yolov10'
        elif 'yolo9' in filename or 'yolov9' in filename:
            model_name = 'yolov9'
        elif 'yolo8' in filename or 'yolov8' in filename:
            model_name = 'yolov8'
        else:
            model_name = 'yolov8'  # Default

        logger.info(f"Loading YOLO model: {model_name}")

        # Create estimator
        # channels_first=True (default) because ART expects NCHW input format
        # is_ultralytics=True will automatically wrap the model with PyTorchYoloLossWrapper
        # NOTE: Do NOT use preprocessing parameter - YOLO expects [0, 255] range
        # attack_losses must match what PyTorchYoloLossWrapper.forward() returns
        estimator = PyTorchYolo(
            model=yolo_model.model,
            input_shape=(3, *config['input_size']),
            channels_first=True,  # Input is NCHW (standard for PyTorch)
            device_type=device_type,
            clip_values=clip_values,
            attack_losses=("loss_total",),  # PyTorchYoloLossWrapper returns loss_total
            is_ultralytics=True,
            model_name=model_name,
        )

        logger.info(f"YOLO model loaded successfully: {model_name}")
        return estimator

    def _load_rtdetr(self, model_path: str, config: dict, device_type: str, clip_values: tuple):
        """Load RT-DETR model using ultralytics."""
        try:
            from ultralytics import RTDETR
        except ImportError:
            raise ImportError("ultralytics is required for RT-DETR models. Install with: pip install ultralytics")

        # Load RT-DETR model
        rtdetr_model = RTDETR(model_path)

        logger.info("Loading RT-DETR model")

        # Create estimator
        estimator = PyTorchRTDETR(
            model=rtdetr_model.model,
            input_shape=(3, *config['input_size']),
            device_type=device_type,
            clip_values=clip_values,
        )

        logger.info("RT-DETR model loaded successfully")
        return estimator

    def _load_faster_rcnn(self, model_path: str, config: dict, device_type: str, clip_values: tuple):
        """Load Faster R-CNN model."""
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')

        # Create model architecture (this needs to be implemented based on your specific Faster R-CNN variant)
        # For now, we'll use torchvision's pre-trained model as base
        import torchvision

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            num_classes=len(config['class_names'])
        )

        # Load weights if compatible
        if isinstance(state_dict, dict) and 'model' in state_dict:
            model.load_state_dict(state_dict['model'])
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)

        logger.info("Loading Faster R-CNN model")

        # Create estimator
        estimator = PyTorchFasterRCNN(
            model=model,
            input_shape=(3, *config['input_size']),
            device_type=device_type,
            clip_values=clip_values,
        )

        logger.info("Faster R-CNN model loaded successfully")
        return estimator

    def _load_efficientdet(self, model_path: str, config: dict, device_type: str, clip_values: tuple):
        """Load EfficientDet model from MMDetection checkpoint."""
        try:
            # Import required modules
            import torch
            from mmengine.config import Config
            from mmengine.runner import load_checkpoint

            # Try to import mmdet - will fail gracefully if mmcv ops not available
            try:
                import mmdet
                import mmcv
                # Check if mmcv ops are available
                try:
                    from mmcv.ops import roi_align
                    from mmdet.registry import MODELS
                    mmdet_available = True
                    logger.info("MMDetection with CUDA ops available")
                except ImportError as e:
                    logger.warning(f"MMDetection CUDA ops not available: {e}")
                    # Try to use MMDet without ops
                    try:
                        from mmdet.registry import MODELS
                        mmdet_available = True
                        logger.info("Using MMDetection without CUDA ops (limited functionality)")
                    except ImportError:
                        mmdet_available = False
            except ImportError as e:
                logger.warning(f"MMDetection import failed: {e}")
                mmdet_available = False

            model_path_obj = Path(model_path)

            # Look for config file
            config_file = None
            possible_config_names = [
                'config.py',
                f'{model_path_obj.stem}.py',
                'efficientdet_config.py'
            ]

            for config_name in possible_config_names:
                config_path = model_path_obj.parent / config_name
                if config_path.exists():
                    config_file = str(config_path)
                    break

            if config_file is None:
                # Try to find any .py file in the directory
                py_files = list(model_path_obj.parent.glob('*.py'))
                if py_files:
                    config_file = str(py_files[0])
                    logger.warning(f"Using config file: {config_file}")

            if config_file is None:
                raise FileNotFoundError(
                    f"No config file found in {model_path_obj.parent}. "
                    "EfficientDet requires an MMDetection config file."
                )

            logger.info(f"Loading EfficientDet from checkpoint: {model_path}")
            logger.info(f"Using config: {config_file}")

            # Try to load config from checkpoint first (contains trained model config)
            checkpoint_loaded = False
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if 'meta' in checkpoint and 'cfg' in checkpoint['meta']:
                    cfg_str = checkpoint['meta']['cfg']
                    # cfg is stored as string, need to parse it
                    if isinstance(cfg_str, str):
                        # Write to temp file and load
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                            f.write(cfg_str)
                            temp_config_path = f.name
                        try:
                            cfg = Config.fromfile(temp_config_path)
                            logger.info("Loaded config from checkpoint metadata")
                            checkpoint_loaded = True
                        finally:
                            import os
                            os.unlink(temp_config_path)
                    else:
                        cfg = cfg_str
                        logger.info("Loaded config from checkpoint metadata")
                        checkpoint_loaded = True
                else:
                    # Load config from file
                    cfg = Config.fromfile(config_file)
                    logger.info("Loaded config from config file")
            except Exception as e:
                logger.warning(f"Could not load checkpoint metadata: {e}")
                cfg = Config.fromfile(config_file)

            # Build model from config
            if mmdet_available:
                try:
                    # Register all MMDet modules first
                    from mmdet.utils import register_all_modules
                    register_all_modules()

                    # Register custom modules (EfficientDetIncre, etc.)
                    custom_registered = _register_custom_mmdet_modules()
                    logger.info(f"Custom modules registration: {custom_registered}")

                    # Check if model type is in registry
                    model_type = cfg.model.get('type', '')
                    logger.info(f"Model type from config: {model_type}")

                    # Check if model type exists in registry
                    if model_type not in MODELS.module_dict:
                        # Log all available model types in registry
                        available_models = list(MODELS.module_dict.keys())
                        efficient_models = [k for k in available_models if 'Efficient' in k]
                        logger.error(f"✗ Model type '{model_type}' NOT found in MMDetection registry")
                        logger.error(f"Available EfficientDet models: {efficient_models}")
                        logger.error(f"Total registered models: {len(available_models)}")

                        # If custom module registration failed, provide clear error
                        if not custom_registered:
                            raise ImportError(
                                f"Model type '{model_type}' requires custom module registration, "
                                f"but registration failed. Check logs for details."
                            )
                        else:
                            raise ValueError(
                                f"Model type '{model_type}' not found in registry even after "
                                f"successful custom module import. This indicates a registration issue."
                            )

                    logger.info(f"✓ Model type '{model_type}' found in registry")

                    model = MODELS.build(cfg.model)
                    logger.info("Model built successfully from MMDetection config")

                    # Load checkpoint weights
                    # Note: Use weights_only=False for MMEngine checkpoints (PyTorch 2.6+)
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    logger.info("Checkpoint weights loaded successfully")
                    # Weights are already loaded, no need for additional load_checkpoint call

                except (ImportError, ValueError) as e:
                    # These are registration/import errors - re-raise with clear message
                    logger.error(f"Fatal: Model registration or import failed: {e}")
                    raise ImportError(
                        f"Failed to load EfficientDet model '{model_type}': {str(e)}\n\n"
                        "This is likely due to:\n"
                        "  1. Custom model class not properly registered with MMDetection\n"
                        "  2. Missing dependencies or import errors\n"
                        "  3. Incompatible MMDetection version\n\n"
                        "Check the logs above for detailed error information."
                    ) from e
                except Exception as e:
                    # Other errors during model building
                    logger.error(f"Error building model from config: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise RuntimeError(
                        f"Failed to build EfficientDet model from config: {str(e)}\n\n"
                        "The model config was loaded successfully, but building the model failed.\n"
                        "Check the logs above for detailed error information."
                    ) from e

            # Extract model configuration
            input_size = config.get('input_size', [768, 768])

            # Create estimator
            estimator = PyTorchEfficientDet(
                model=model,
                input_shape=(3, *input_size),
                device_type=device_type,
                clip_values=clip_values,
                config_file=config_file,
                model_wrapper=None,  # Will use model directly
            )

            logger.info("EfficientDet model loaded successfully")
            return estimator

        except ImportError as e:
            logger.error(f"Required modules not installed: {e}")
            raise ImportError(
                "MMEngine is required for EfficientDet models. "
                "Install with: pip install mmengine\n"
                "For full functionality, also install: pip install mmdet mmcv"
            )
        except Exception as e:
            logger.error(f"Failed to load EfficientDet model: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"EfficientDet model loading failed: {e}")


# Global factory instance
model_factory = ModelFactory()
