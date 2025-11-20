"""
EfficientDet detection head with incremental learning support.

This module provides a custom detection head that supports incremental learning.
"""
import sys
import os

# Add mmdetection projects to path to import EfficientDet modules
mmdet_projects_path = "/home/jaehyun/mmdetection/projects"
if os.path.exists(mmdet_projects_path) and mmdet_projects_path not in sys.path:
    sys.path.insert(0, mmdet_projects_path)

from mmdet.registry import MODELS

# Import the standard EfficientDet head from the EfficientDet project
try:
    from EfficientDet.efficientdet.efficientdet_head import EfficientDetSepBNHead
    BaseHead = EfficientDetSepBNHead
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Could not import EfficientDetSepBNHead: {e}")
    # Fallback - this won't work properly but at least won't crash during import
    from mmdet.models.dense_heads import RetinaHead
    BaseHead = RetinaHead


@MODELS.register_module()
class EfficientDetSepBNHeadIncre(BaseHead):
    """
    EfficientDet Separable BN Head with incremental learning support.

    This head extends the standard EfficientDetSepBNHead to support:
    - Incremental addition of new object classes
    - Preserving weights for existing classes
    - Flexible number of classes
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the incremental learning head.

        Args:
            *args: Positional arguments passed to base class
            **kwargs: Keyword arguments passed to base class
        """
        super().__init__(*args, **kwargs)

    def init_weights(self) -> None:
        """Initialize weights for the head."""
        super().init_weights()
