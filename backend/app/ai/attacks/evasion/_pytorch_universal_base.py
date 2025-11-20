"""
Base class for PyTorch-based universal perturbation attacks.

This module provides common functionality shared by universal perturbation attacks
to reduce code duplication and improve maintainability.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from app.ai.attacks.attack import EvasionAttack

if TYPE_CHECKING:
    from app.ai.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class PyTorchUniversalPerturbationAttack(EvasionAttack):
    """
    Base class for PyTorch-based universal perturbation attacks.

    Provides common functionality for:
    - Perturbation initialization and management
    - PyTorch ↔ NumPy conversion
    - Epsilon constraint enforcement
    - Device management
    """

    def __init__(self, estimator: "OBJECT_DETECTOR_TYPE", **kwargs):
        """
        Initialize base universal perturbation attack.

        Args:
            estimator: A trained object detector
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(estimator=estimator, **kwargs)

        # Use ART estimator's device
        self.device = self.estimator.device

        # Get PyTorch model from estimator (already on correct device)
        self._torch_model = self.estimator.model

        # Check if all preprocessing is PyTorch-native (for performance optimization)
        self._all_framework_preprocessing = getattr(
            self.estimator, 'all_framework_preprocessing', False
        )
        if self._all_framework_preprocessing:
            logger.debug("Using all-PyTorch preprocessing for better performance")

        # Universal perturbation (PyTorch parameter)
        self._perturbation_torch: nn.Parameter | None = None
        # NumPy version for external use
        self._perturbation: np.ndarray | None = None

    def _initialize_perturbation(self, x: torch.Tensor) -> None:
        """
        Initialize universal perturbation tensor.

        Args:
            x: Input tensor to determine shape (B, C, H, W)
        """
        if self._perturbation_torch is None:
            self._perturbation_torch = nn.Parameter(
                torch.zeros(
                    1, x.shape[1], x.shape[2], x.shape[3],
                    dtype=torch.float32,
                    device=self.device
                )
            )
            logger.info(f"Initialized perturbation: {self._perturbation_torch.shape}")

    def _enforce_epsilon(self, eps: float) -> None:
        """
        Enforce epsilon constraint on perturbation.

        Args:
            eps: Maximum perturbation magnitude
        """
        if self._perturbation_torch is not None:
            with torch.no_grad():
                self._perturbation_torch.data.clamp_(-eps, eps)

    def _torch_to_numpy(self, x_torch: torch.Tensor, x_original: np.ndarray) -> np.ndarray:
        """
        Convert PyTorch tensor back to NumPy array matching original format.

        This handles:
        - Detaching from computation graph
        - Moving to CPU
        - Denormalization (if clip_values specified)
        - Channel ordering (channels_first/last)

        Args:
            x_torch: Torch tensor to convert
            x_original: Original NumPy array (for shape reference)

        Returns:
            NumPy array in the same format as x_original
        """
        x_np = x_torch.detach().cpu().numpy()

        # Denormalize if needed
        if self.estimator.clip_values is not None:
            min_val, max_val = self.estimator.clip_values
            if max_val > 1:
                x_np = x_np * max_val

        # Handle channels_first/last to match original format
        if not self.estimator.channels_first and x_np.ndim == 4:
            # (N, C, H, W) -> (N, H, W, C)
            x_np = np.transpose(x_np, (0, 2, 3, 1))

        return x_np.astype(np.float32)

    def _preprocess_and_convert(
        self,
        x: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess inputs using ART's standard preprocessing.

        Args:
            x: Input images (N, H, W, C) or (N, C, H, W) - NumPy

        Returns:
            Tuple of (preprocessed torch tensor, original numpy array)
        """
        x_torch, _ = self.estimator._preprocess_and_convert_inputs(
            x=x, y=None, fit=False, no_grad=True
        )
        logger.debug(f"Input converted to torch: {x_torch.shape}, device: {x_torch.device}")
        return x_torch, x

    @property
    def perturbation(self) -> np.ndarray | None:
        """Get the current universal perturbation (NumPy)."""
        return self._perturbation

    def set_perturbation(self, perturbation: np.ndarray) -> None:
        """
        Set a pre-trained universal perturbation.

        Args:
            perturbation: Perturbation to set (NumPy array)
        """
        self._perturbation = perturbation
        self._perturbation_torch = nn.Parameter(
            torch.from_numpy(perturbation).float().to(self.device)
        )
