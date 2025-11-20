"""
Class index mapping utilities for handling different model class configurations.

Supports:
- COCO standard (80 classes)
- Custom trained models with different class orders
- Class remapping between models
"""
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# COCO 80 classes (standard)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ClassMapper:
    """Handle class index mapping between different model configurations."""

    def __init__(
        self,
        model_classes: List[str],
        reference_classes: Optional[List[str]] = None,
        is_coco_format: bool = False
    ):
        """
        Initialize class mapper.

        Args:
            model_classes: List of class names in model's order
            reference_classes: Reference class list to map to (e.g., COCO)
            is_coco_format: If True, assume model uses COCO class order
        """
        self.model_classes = model_classes
        self.reference_classes = reference_classes or COCO_CLASSES
        self.is_coco_format = is_coco_format

        # Create mapping dictionaries
        self.model_class_to_idx = {name: idx for idx, name in enumerate(model_classes)}
        self.idx_to_model_class = {idx: name for idx, name in enumerate(model_classes)}

        # Create forward and reverse mappings
        self.model_to_reference_map = self._build_mapping(
            model_classes, self.reference_classes
        )
        self.reference_to_model_map = self._build_mapping(
            self.reference_classes, model_classes
        )

        logger.info(
            f"ClassMapper initialized: {len(model_classes)} model classes, "
            f"{len(self.reference_classes)} reference classes"
        )

    def _build_mapping(
        self, source_classes: List[str], target_classes: List[str]
    ) -> Dict[int, int]:
        """
        Build index mapping from source to target classes.

        Args:
            source_classes: Source class list
            target_classes: Target class list

        Returns:
            Dict mapping source indices to target indices
        """
        mapping = {}
        target_name_to_idx = {name.lower(): idx for idx, name in enumerate(target_classes)}

        for src_idx, src_name in enumerate(source_classes):
            src_name_lower = src_name.lower()
            if src_name_lower in target_name_to_idx:
                mapping[src_idx] = target_name_to_idx[src_name_lower]
            else:
                # No mapping found - keep original index
                logger.warning(
                    f"No mapping found for class '{src_name}' (index {src_idx})"
                )

        return mapping

    def remap_predictions(
        self, predictions: Dict, to_reference: bool = True
    ) -> Dict:
        """
        Remap class indices in predictions.

        Args:
            predictions: Prediction dict with 'labels' key
            to_reference: If True, map model indices to reference indices
                         If False, map reference indices to model indices

        Returns:
            Predictions with remapped class indices
        """
        import torch
        import numpy as np

        mapping = self.model_to_reference_map if to_reference else self.reference_to_model_map

        if not mapping:
            return predictions  # No mapping needed

        # Deep copy predictions
        remapped = predictions.copy()

        # Remap labels
        if 'labels' in predictions:
            labels = predictions['labels']

            if isinstance(labels, torch.Tensor):
                new_labels = labels.clone()
                for src_idx, tgt_idx in mapping.items():
                    mask = labels == src_idx
                    new_labels[mask] = tgt_idx
                remapped['labels'] = new_labels

            elif isinstance(labels, np.ndarray):
                new_labels = labels.copy()
                for src_idx, tgt_idx in mapping.items():
                    mask = labels == src_idx
                    new_labels[mask] = tgt_idx
                remapped['labels'] = new_labels

            elif isinstance(labels, list):
                new_labels = []
                for label in labels:
                    new_labels.append(mapping.get(label, label))
                remapped['labels'] = new_labels

        return remapped

    def get_class_name(self, class_idx: int, use_model_classes: bool = True) -> str:
        """
        Get class name from index.

        Args:
            class_idx: Class index
            use_model_classes: If True, use model classes, else reference classes

        Returns:
            Class name
        """
        classes = self.model_classes if use_model_classes else self.reference_classes
        if 0 <= class_idx < len(classes):
            return classes[class_idx]
        return f"unknown_{class_idx}"

    def get_class_index(self, class_name: str) -> Optional[int]:
        """
        Get class index from name.

        Args:
            class_name: Class name

        Returns:
            Class index or None if not found
        """
        return self.model_class_to_idx.get(class_name)

    def create_labelmap(self) -> Dict[str, str]:
        """
        Create labelmap dict for database storage.

        Returns:
            Dict mapping string indices to class names
        """
        return {str(idx): name for idx, name in enumerate(self.model_classes)}

    @classmethod
    def from_labelmap(cls, labelmap: Dict[str, str]) -> "ClassMapper":
        """
        Create ClassMapper from labelmap dict.

        Args:
            labelmap: Dict mapping string indices to class names

        Returns:
            ClassMapper instance
        """
        # Sort by index to get ordered list
        indices = sorted([int(k) for k in labelmap.keys()])
        model_classes = [labelmap[str(i)] for i in indices]

        return cls(model_classes=model_classes)

    def is_subset_of_coco(self) -> Tuple[bool, List[str]]:
        """
        Check if model classes are a subset of COCO classes.

        Returns:
            (is_subset, missing_classes)
        """
        coco_names = set(c.lower() for c in COCO_CLASSES)
        model_names = set(c.lower() for c in self.model_classes)

        is_subset = model_names.issubset(coco_names)
        missing = list(model_names - coco_names)

        return is_subset, missing

    def get_coco_indices(self) -> List[int]:
        """
        Get COCO indices for model classes.

        Returns:
            List of COCO indices
        """
        coco_name_to_idx = {name.lower(): idx for idx, name in enumerate(COCO_CLASSES)}
        indices = []

        for model_class in self.model_classes:
            coco_idx = coco_name_to_idx.get(model_class.lower())
            if coco_idx is not None:
                indices.append(coco_idx)

        return indices


def detect_class_format(class_names: List[str]) -> str:
    """
    Detect if class names follow a known format.

    Args:
        class_names: List of class names

    Returns:
        'coco', 'coco_subset', 'custom', or 'unknown'
    """
    # Check if identical to COCO
    if class_names == COCO_CLASSES:
        return 'coco'

    # Check if subset of COCO
    coco_names_set = set(c.lower() for c in COCO_CLASSES)
    model_names_set = set(c.lower() for c in class_names)

    if model_names_set.issubset(coco_names_set):
        return 'coco_subset'

    # Check for common custom formats
    common_vehicle = {'car', 'truck', 'bus', 'motorcycle'}
    common_person = {'person', 'pedestrian', 'cyclist'}

    if model_names_set.issubset(common_vehicle):
        return 'custom_vehicle'
    elif model_names_set.issubset(common_person):
        return 'custom_person'

    return 'custom'


def create_class_mapping_report(
    model_classes: List[str],
    reference_classes: List[str] = None
) -> Dict:
    """
    Create a detailed report of class mapping.

    Args:
        model_classes: Model class names
        reference_classes: Reference class names (default: COCO)

    Returns:
        Mapping report dict
    """
    reference_classes = reference_classes or COCO_CLASSES
    mapper = ClassMapper(model_classes, reference_classes)

    is_subset, missing = mapper.is_subset_of_coco()

    report = {
        "num_model_classes": len(model_classes),
        "num_reference_classes": len(reference_classes),
        "format": detect_class_format(model_classes),
        "is_coco_subset": is_subset,
        "missing_from_coco": missing,
        "mapping": mapper.model_to_reference_map,
        "unmapped_indices": [
            idx for idx in range(len(model_classes))
            if idx not in mapper.model_to_reference_map
        ],
    }

    return report
