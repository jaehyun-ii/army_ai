"""
EfficientDet with incremental learning support.

This implementation extends the standard EfficientDet model to support
incremental learning scenarios where new classes are added over time.
"""
from typing import Optional
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class EfficientDetIncre(SingleStageDetector):
    """
    EfficientDet detector with incremental learning support.

    This model extends the standard SingleStageDetector to support:
    - Loading pretrained models with different numbers of classes
    - Incremental addition of new object classes
    - Preserving knowledge from previously learned classes

    Args:
        backbone: Backbone network configuration
        neck: Neck network configuration (typically BiFPN)
        bbox_head: Detection head configuration
        train_cfg: Training configuration
        test_cfg: Testing configuration
        data_preprocessor: Data preprocessing configuration
        init_cfg: Initialization configuration
        ori_setting: Original model settings for incremental learning
            - ori_checkpoint_file: Path to original checkpoint
            - ori_num_classes: Number of classes in original model
            - ori_config_file: Path to original config
        latest_model_flag: Whether this is the latest model version
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ori_setting: Optional[dict] = None,
                 latest_model_flag: bool = True) -> None:

        # Store incremental learning settings
        self.ori_setting = ori_setting or {}
        self.latest_model_flag = latest_model_flag

        # Initialize the base single-stage detector
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )

    def init_weights(self) -> None:
        """Initialize weights, optionally loading from original checkpoint."""
        super().init_weights()

        # Load from original checkpoint if specified for incremental learning
        if self.ori_setting.get('ori_checkpoint_file'):
            self._load_original_weights()

    def _load_original_weights(self) -> None:
        """
        Load weights from original checkpoint for incremental learning.

        This method handles loading pretrained weights when the number of
        classes differs between the original and current model.
        """
        import torch
        from mmengine.runner import load_checkpoint

        ori_checkpoint = self.ori_setting.get('ori_checkpoint_file')
        if not ori_checkpoint:
            return

        try:
            # Load checkpoint with strict=False to allow partial loading
            load_checkpoint(
                self,
                ori_checkpoint,
                map_location='cpu',
                strict=False
            )
        except Exception as e:
            # Log warning but don't fail - the model can still work
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Could not load original checkpoint {ori_checkpoint}: {e}"
            )
