"""
Attack-specific loss functions for adversarial attacks.

This module provides a registry system for managing different loss functions
tailored to specific attack types (e.g., universal noise, patch attacks).

Usage:
    # Get a loss function for a specific attack type
    >>> loss_fn = AttackLossRegistry.get('universal_noise', config)
    >>> total_loss, *components = loss_fn(predictions, targets, ...)

    # Register a new attack loss
    >>> @AttackLossRegistry.register('my_attack')
    >>> class MyAttackLoss(nn.Module):
    ...     pass
"""

from typing import Dict, Optional, Any, Callable
import torch
import torch.nn as nn

from .box_utils import box_iou, xywh2xyxy
from .detection_attack_loss import DetectionAttackLoss


class AttackLossRegistry:
    """
    Registry for attack-specific loss functions.

    This allows attacks to use specialized loss functions optimized for their
    particular attack strategy, rather than using generic training losses.
    """

    _registry: Dict[str, type] = {}
    _default_configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        attack_type: str,
        default_config: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Register a loss class for a specific attack type.

        Args:
            attack_type: Name of the attack type (e.g., 'universal_noise')
            default_config: Default configuration for the loss function

        Returns:
            Decorator function

        Example:
            >>> @AttackLossRegistry.register('my_attack', {'threshold': 0.1})
            >>> class MyAttackLoss(nn.Module):
            ...     def __init__(self, threshold=0.5):
            ...         super().__init__()
            ...         self.threshold = threshold
        """
        def decorator(loss_class: type) -> type:
            if not issubclass(loss_class, nn.Module):
                raise TypeError(
                    f"Registered loss class must be a subclass of nn.Module, "
                    f"got {loss_class}"
                )

            cls._registry[attack_type] = loss_class

            if default_config is not None:
                cls._default_configs[attack_type] = default_config

            return loss_class

        return decorator

    @classmethod
    def get(
        cls,
        attack_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Get an instance of the loss function for a specific attack type.

        Args:
            attack_type: Name of the attack type
            config: Configuration dict for the loss function. If None,
                   uses default config.

        Returns:
            Instantiated loss function (nn.Module)

        Raises:
            ValueError: If attack_type is not registered

        Example:
            >>> loss_fn = AttackLossRegistry.get('universal_noise', {
            ...     'iou_threshold': 0.1,
            ...     'top_k_boxes': 5
            ... })
        """
        if attack_type not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Attack type '{attack_type}' not registered. "
                f"Available types: {available}"
            )

        loss_class = cls._registry[attack_type]

        # Merge default config with provided config
        if config is None:
            config = {}

        final_config = cls._default_configs.get(attack_type, {}).copy()
        final_config.update(config)

        # Instantiate and return
        return loss_class(**final_config)

    @classmethod
    def list_registered(cls) -> list:
        """
        Get list of all registered attack types.

        Returns:
            List of registered attack type names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, attack_type: str) -> bool:
        """
        Check if an attack type is registered.

        Args:
            attack_type: Name of the attack type

        Returns:
            True if registered, False otherwise
        """
        return attack_type in cls._registry


# Register built-in attack losses
@AttackLossRegistry.register(
    attack_type='universal_noise',
    default_config={
        'iou_threshold': 0.1,
        'class_conf_threshold': 0.05,
        'top_k_boxes': -1,
        'top_n_classes': 3,
        'lambda_targeted': 1.0,
        'lambda_agnostic': 2.0
    }
)
class UniversalNoiseAttackLoss(DetectionAttackLoss):
    """
    Attack loss for Universal Noise attacks.

    Inherits from DetectionAttackLoss with default configurations
    optimized for universal perturbation attacks.
    """
    pass


@AttackLossRegistry.register(
    attack_type='adversarial_patch',
    default_config={
        'target_class': 0,  # Person class in COCO
    }
)
class AdversarialPatchAttackLoss(nn.Module):
    """
    Attack loss for Adversarial Patch attacks.

    Based on Naturalistic Adversarial Patch paper's MaxProbExtractor.
    Extracts maximum detection score (objectness × class_confidence) and
    minimizes it to reduce detection.

    This is simpler than DetectionAttackLoss - just finds the max detection score.
    """

    def __init__(self, target_class: int = 0):
        super().__init__()
        self.target_class = target_class

    def forward(
        self,
        inference_output: torch.Tensor,
        target_class_idx: int | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate patch attack loss: mean of maximum detection scores.

        Args:
            inference_output: Model predictions
                            - YOLO v5 format: (batch_size, num_proposals, 85)
                            - YOLO v8 format: (batch_size, num_proposals, 84)
            target_class_idx: Target class index (overrides self.target_class if provided)

        Returns:
            Loss scalar (mean of max detection scores across batch)
        """
        # Use provided target_class_idx if available, otherwise use initialization value
        target_class = target_class_idx if target_class_idx is not None else self.target_class

        batch_size = inference_output.shape[0]
        num_features = inference_output.shape[2]  # 84 or 85

        # Auto-detect YOLO version
        has_objectness = (num_features == 85)

        max_scores = []

        for b in range(batch_size):
            preds = inference_output[b]  # (num_proposals, 84/85)

            if has_objectness:
                # YOLO v5: [x, y, w, h, obj, class_0, ..., class_79]
                obj_scores = torch.sigmoid(preds[:, 4])  # (num_proposals,)
                class_scores = torch.sigmoid(preds[:, 5:])  # (num_proposals, 80)
            else:
                # YOLO v8: [x, y, w, h, class_0, ..., class_79]
                class_scores = torch.sigmoid(preds[:, 4:])  # (num_proposals, 80)
                obj_scores = class_scores.max(dim=1)[0]  # Use max class as proxy

            # Detection score = objectness × class_confidence
            # For target class (use runtime parameter if provided)
            target_class_scores = class_scores[:, target_class]
            detection_scores = obj_scores * target_class_scores

            # Get maximum detection score for this image
            max_score = detection_scores.max()
            max_scores.append(max_score)

        # Return mean of max scores across batch
        # Minimizing this reduces detection
        loss = torch.mean(torch.stack(max_scores))
        return loss


# Export public API
__all__ = [
    'AttackLossRegistry',
    'DetectionAttackLoss',
    'UniversalNoiseAttackLoss',
    'AdversarialPatchAttackLoss',
    'box_iou',
    'xywh2xyxy'
]
