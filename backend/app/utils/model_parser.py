"""
Model file parser utilities for extracting metadata from .pt and .yaml files.
"""
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelParser:
    """Parse model files to extract metadata."""

    @staticmethod
    def parse_pt_file(file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a PyTorch .pt file.

        Args:
            file_path: Path to the .pt file

        Returns:
            Dictionary containing extracted metadata:
            - model_type: str (yolo, rtdetr, faster_rcnn, etc.)
            - class_names: List[str] (if available)
            - num_classes: int (if available)
            - input_size: List[int] (if available)
            - framework: str
            - additional_info: Dict[str, Any]
        """
        metadata = {
            "framework": "pytorch",
            "model_type": None,  # Will be set if detectable, otherwise config file will provide it
            "class_names": None,
            "num_classes": None,
            "input_size": None,
            "additional_info": {}
        }

        try:
            # Try to load as YOLO model first (most common case)
            try:
                model = YOLO(str(file_path))
                metadata["model_type"] = "yolo"

                # Extract class names
                if hasattr(model, 'names') and model.names:
                    if isinstance(model.names, dict):
                        # Convert dict to list: {0: 'person', 1: 'car'} -> ['person', 'car']
                        metadata["class_names"] = [model.names[i] for i in sorted(model.names.keys())]
                    elif isinstance(model.names, list):
                        metadata["class_names"] = model.names
                    metadata["num_classes"] = len(metadata["class_names"])

                # Extract input size (imgsz)
                if hasattr(model, 'overrides') and 'imgsz' in model.overrides:
                    imgsz = model.overrides['imgsz']
                    if isinstance(imgsz, int):
                        metadata["input_size"] = [imgsz, imgsz]
                    elif isinstance(imgsz, (list, tuple)):
                        metadata["input_size"] = list(imgsz)

                # Additional YOLO-specific info
                if hasattr(model, 'task'):
                    metadata["additional_info"]["task"] = model.task
                if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
                    metadata["additional_info"]["yaml_config"] = model.model.yaml

                logger.info(f"Successfully parsed YOLO model from {file_path}")
                return metadata

            except Exception as yolo_error:
                logger.debug(f"Not a YOLO model: {yolo_error}")

            # Try to load as raw PyTorch checkpoint
            # Use weights_only=False for MMEngine checkpoints (PyTorch 2.6+)
            checkpoint = torch.load(str(file_path), map_location='cpu', weights_only=False)

            # Check if it's a state dict or full model
            if isinstance(checkpoint, dict):
                # Extract metadata from checkpoint
                if 'model' in checkpoint:
                    model_state = checkpoint['model']

                    # Try to detect model type from state dict keys
                    state_keys = list(model_state.state_dict().keys() if hasattr(model_state, 'state_dict') else [])

                    if any('yolo' in key.lower() for key in state_keys):
                        metadata["model_type"] = "yolo"
                    elif any('detr' in key.lower() for key in state_keys):
                        metadata["model_type"] = "detr"
                    elif any('rcnn' in key.lower() for key in state_keys):
                        metadata["model_type"] = "faster_rcnn"

                # Extract class names if available
                if 'names' in checkpoint:
                    names = checkpoint['names']
                    if isinstance(names, dict):
                        metadata["class_names"] = [names[i] for i in sorted(names.keys())]
                    elif isinstance(names, list):
                        metadata["class_names"] = names
                    metadata["num_classes"] = len(metadata["class_names"])

                # Extract other metadata
                for key in ['nc', 'num_classes']:
                    if key in checkpoint:
                        metadata["num_classes"] = checkpoint[key]

                if 'imgsz' in checkpoint:
                    imgsz = checkpoint['imgsz']
                    if isinstance(imgsz, int):
                        metadata["input_size"] = [imgsz, imgsz]
                    elif isinstance(imgsz, (list, tuple)):
                        metadata["input_size"] = list(imgsz)

                # Store additional checkpoint info
                metadata["additional_info"]["checkpoint_keys"] = list(checkpoint.keys())

                logger.info(f"Successfully parsed PyTorch checkpoint from {file_path}")

        except Exception as e:
            logger.error(f"Error parsing .pt file {file_path}: {e}", exc_info=True)
            metadata["additional_info"]["parse_error"] = str(e)

        return metadata

    @staticmethod
    def parse_config_file(file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a configuration file (YAML or Python).

        Args:
            file_path: Path to the config file (.yaml/.yml or .py)

        Returns:
            Dictionary containing extracted metadata:
            - class_names: List[str]
            - num_classes: int
            - input_size: List[int] (if available)
            - model_type: str (if detectable)
            - raw_yaml: Dict (original yaml content, empty for .py files)
        """
        metadata = {
            "class_names": None,
            "num_classes": None,
            "input_size": None,
            "model_type": "unknown",
            "raw_yaml": {}
        }

        # Check file extension
        file_ext = file_path.suffix.lower()

        # For .py config files (MMDetection), extract basic info
        if file_ext == '.py':
            logger.info(f"Parsing Python config file {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract classes from metainfo dict
                # Pattern: metainfo = dict(classes=(...))
                import re

                # Look for metainfo dict with classes tuple
                metainfo_pattern = r'metainfo\s*=\s*dict\s*\(\s*classes\s*=\s*\((.*?)\)\s*[,\)]'
                match = re.search(metainfo_pattern, content, re.DOTALL)

                if match:
                    classes_str = match.group(1)
                    # Extract quoted strings
                    class_names = re.findall(r"['\"]([^'\"]+)['\"]", classes_str)

                    if class_names:
                        metadata["class_names"] = class_names
                        metadata["num_classes"] = len(class_names)
                        logger.info(f"Extracted {len(class_names)} classes from Python config")

                # Try to extract input size from test_pipeline's Resize scale
                # Pattern matches: scale=(768, 768) or scale=(\n    768,\n    768,\n)
                # First, try to find scale in test_pipeline section
                test_pipeline_match = re.search(r'test_pipeline\s*=\s*\[(.*?)(?=\n\w+\s*=|\Z)', content, re.DOTALL)
                if test_pipeline_match:
                    test_pipeline_section = test_pipeline_match.group(1)
                    scale_pattern = r'scale\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*[,\)]'
                    scale_match = re.search(scale_pattern, test_pipeline_section, re.MULTILINE)
                    if scale_match:
                        h, w = int(scale_match.group(1)), int(scale_match.group(2))
                        metadata["input_size"] = [h, w]
                        logger.info(f"Extracted input size from test_pipeline: {metadata['input_size']}")

                # If not found in test_pipeline, use smallest scale (likely inference size)
                if not metadata["input_size"]:
                    scale_pattern = r'scale\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*[,\)]'
                    scales = re.findall(scale_pattern, content, re.MULTILINE | re.DOTALL)
                    if scales:
                        # Convert to integers and find minimum
                        scales_int = [(int(h), int(w)) for h, w in scales]
                        min_scale = min(scales_int, key=lambda x: x[0] * x[1])  # Smallest area
                        metadata["input_size"] = list(min_scale)
                        logger.info(f"Extracted input size (smallest): {metadata['input_size']}")

                metadata["model_type"] = "efficientdet"

            except Exception as e:
                logger.error(f"Error parsing Python config file {file_path}: {e}")
                metadata["model_type"] = "efficientdet"

            return metadata

        # Parse YAML files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)

            if not isinstance(yaml_data, dict):
                logger.warning(f"Config file {file_path} does not contain a dictionary")
                return metadata

            metadata["raw_yaml"] = yaml_data

            # Extract number of classes
            if 'nc' in yaml_data:
                metadata["num_classes"] = int(yaml_data['nc'])

            # Extract class names
            if 'names' in yaml_data:
                names = yaml_data['names']
                if isinstance(names, dict):
                    # Convert dict to list: {0: 'person', 1: 'car'} -> ['person', 'car']
                    metadata["class_names"] = [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    metadata["class_names"] = names

                if metadata["class_names"]:
                    metadata["num_classes"] = len(metadata["class_names"])

            # Extract input size
            if 'imgsz' in yaml_data:
                imgsz = yaml_data['imgsz']
                if isinstance(imgsz, int):
                    metadata["input_size"] = [imgsz, imgsz]
                elif isinstance(imgsz, (list, tuple)):
                    metadata["input_size"] = list(imgsz)

            # Try to detect model type from YAML content
            yaml_str = str(yaml_data).lower()
            if 'yolo' in yaml_str:
                metadata["model_type"] = "yolo"
            elif 'detr' in yaml_str or 'rtdetr' in yaml_str:
                metadata["model_type"] = "detr"
            elif 'faster' in yaml_str and 'rcnn' in yaml_str:
                metadata["model_type"] = "faster_rcnn"

            logger.info(f"Successfully parsed config file from {file_path}")

        except Exception as e:
            logger.error(f"Error parsing config file {file_path}: {e}", exc_info=True)
            metadata["raw_yaml"]["parse_error"] = str(e)

        return metadata

    @staticmethod
    def parse_yaml_file(file_path: Path) -> Dict[str, Any]:
        """
        Backward compatibility alias for parse_config_file.
        Deprecated: Use parse_config_file instead.
        """
        return ModelParser.parse_config_file(file_path)

    @staticmethod
    def merge_metadata(pt_metadata: Dict[str, Any], yaml_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge metadata from .pt and .yaml files, with .pt taking precedence.

        Args:
            pt_metadata: Metadata extracted from .pt file
            yaml_metadata: Metadata extracted from .yaml file (optional)

        Returns:
            Merged metadata dictionary
        """
        merged = pt_metadata.copy()

        if yaml_metadata is None:
            return merged

        # YAML provides fallback values when PT doesn't have them
        for key in ['class_names', 'num_classes', 'input_size', 'model_type']:
            if merged.get(key) is None and yaml_metadata.get(key) is not None:
                merged[key] = yaml_metadata[key]

        # Add raw YAML data to additional info
        if "raw_yaml" in yaml_metadata:
            merged["additional_info"]["yaml_config"] = yaml_metadata["raw_yaml"]

        return merged

    @staticmethod
    def extract_model_info(
        weights_path: Optional[Path] = None,
        yaml_path: Optional[Path] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extract comprehensive model information from .pt and/or config files.

        Args:
            weights_path: Path to .pt file (optional)
            yaml_path: Path to config file - .yaml/.yml or .py (optional)

        Returns:
            Tuple of (metadata, validation_errors)
            - metadata: Extracted and merged metadata
            - validation_errors: List of validation error messages
        """
        errors = []

        if not weights_path and not yaml_path:
            errors.append("At least one of weights_path or yaml_path must be provided")
            return {}, errors

        pt_metadata = {}
        yaml_metadata = {}

        # Parse .pt file
        if weights_path:
            if not weights_path.exists():
                errors.append(f"Weights file not found: {weights_path}")
            else:
                pt_metadata = ModelParser.parse_pt_file(weights_path)

        # Parse config file (.yaml/.yml or .py)
        if yaml_path:
            if not yaml_path.exists():
                errors.append(f"Config file not found: {yaml_path}")
            else:
                yaml_metadata = ModelParser.parse_config_file(yaml_path)

        # Merge metadata
        if pt_metadata or yaml_metadata:
            merged = ModelParser.merge_metadata(pt_metadata, yaml_metadata if yaml_metadata else None)
        else:
            merged = {}
            errors.append("Failed to extract metadata from any file")

        # Validate required fields
        if not merged.get("class_names"):
            errors.append("Could not extract class names from model or YAML file")

        if merged.get("model_type") == "unknown":
            errors.append("Could not detect model type. Please specify manually.")

        return merged, errors


# Convenience function for quick access
def parse_model_files(weights_path: Optional[str] = None, yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to parse model files.

    Args:
        weights_path: Path to .pt file (optional)
        yaml_path: Path to .yaml file (optional)

    Returns:
        Extracted metadata dictionary
    """
    weights = Path(weights_path) if weights_path else None
    yaml_file = Path(yaml_path) if yaml_path else None

    metadata, errors = ModelParser.extract_model_info(weights, yaml_file)

    if errors:
        logger.warning(f"Validation errors during parsing: {errors}")

    return metadata
