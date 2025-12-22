"""
Learning Rate Scheduler utilities for adversarial attacks.

Provides a unified interface for PyTorch learning rate schedulers
that can be used across all attack implementations.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch.optim as optim

if TYPE_CHECKING:
    import torch.optim.lr_scheduler as lr_scheduler

logger = logging.getLogger(__name__)


class LRSchedulerFactory:
    """
    Factory class for creating PyTorch learning rate schedulers.

    Supports common scheduler types used in adversarial attack optimization.
    """

    SUPPORTED_SCHEDULERS = {
        "constant": "No scheduler (constant learning rate)",
        "step": "StepLR - Decay by gamma every step_size epochs",
        "exponential": "ExponentialLR - Exponential decay by gamma",
        "cosine": "CosineAnnealingLR - Cosine annealing",
        "linear": "LinearLR - Linear decay",
        "plateau": "ReduceLROnPlateau - Reduce on loss plateau",
    }

    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = "constant",
        scheduler_params: dict[str, Any] | None = None,
        max_iter: int = 500,
    ) -> lr_scheduler.LRScheduler | None:
        """
        Create a learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ("constant", "step", "exponential", "cosine", "linear", "plateau")
            scheduler_params: Parameters for the scheduler
            max_iter: Maximum number of iterations (used for cosine annealing)

        Returns:
            PyTorch LR scheduler or None if scheduler_type is "constant"

        Examples:
            >>> # Constant learning rate (no scheduler)
            >>> scheduler = create_scheduler(optimizer, "constant")

            >>> # Step decay: lr *= 0.1 every 100 epochs
            >>> scheduler = create_scheduler(
            ...     optimizer, "step",
            ...     {"step_size": 100, "gamma": 0.1}
            ... )

            >>> # Exponential decay: lr *= 0.95 every epoch
            >>> scheduler = create_scheduler(
            ...     optimizer, "exponential",
            ...     {"gamma": 0.95}
            ... )

            >>> # Cosine annealing with warm restarts
            >>> scheduler = create_scheduler(
            ...     optimizer, "cosine",
            ...     {"T_max": 50, "eta_min": 0.001}
            ... )

            >>> # Linear decay from initial_lr to end_factor * initial_lr
            >>> scheduler = create_scheduler(
            ...     optimizer, "linear",
            ...     {"start_factor": 1.0, "end_factor": 0.1, "total_iters": 500}
            ... )
        """
        if scheduler_params is None:
            scheduler_params = {}

        scheduler_type = scheduler_type.lower()

        if scheduler_type == "constant":
            logger.info("Using constant learning rate (no scheduler)")
            return None

        elif scheduler_type == "step":
            step_size = scheduler_params.get("step_size", max(1, max_iter // 10))
            gamma = scheduler_params.get("gamma", 0.1)

            logger.info(f"Creating StepLR scheduler: step_size={step_size}, gamma={gamma}")
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )

        elif scheduler_type == "exponential":
            gamma = scheduler_params.get("gamma", 0.95)

            logger.info(f"Creating ExponentialLR scheduler: gamma={gamma}")
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=gamma
            )

        elif scheduler_type == "cosine":
            T_max = scheduler_params.get("T_max", max_iter)
            eta_min = scheduler_params.get("eta_min", 0.0)

            logger.info(f"Creating CosineAnnealingLR scheduler: T_max={T_max}, eta_min={eta_min}")
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )

        elif scheduler_type == "linear":
            start_factor = scheduler_params.get("start_factor", 1.0)
            end_factor = scheduler_params.get("end_factor", 0.1)
            total_iters = scheduler_params.get("total_iters", max_iter)

            logger.info(
                f"Creating LinearLR scheduler: start_factor={start_factor}, "
                f"end_factor={end_factor}, total_iters={total_iters}"
            )
            return optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters
            )

        elif scheduler_type == "plateau":
            mode = scheduler_params.get("mode", "min")
            factor = scheduler_params.get("factor", 0.1)
            patience = scheduler_params.get("patience", max(1, max_iter // 20))
            threshold = scheduler_params.get("threshold", 1e-4)

            logger.info(
                f"Creating ReduceLROnPlateau scheduler: mode={mode}, factor={factor}, "
                f"patience={patience}, threshold={threshold}"
            )
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                verbose=True
            )

        else:
            logger.warning(
                f"Unknown scheduler type: {scheduler_type}. "
                f"Supported types: {list(LRSchedulerFactory.SUPPORTED_SCHEDULERS.keys())}"
            )
            return None

    @staticmethod
    def get_default_params(scheduler_type: str, max_iter: int = 500) -> dict[str, Any]:
        """
        Get default parameters for a scheduler type.

        Args:
            scheduler_type: Type of scheduler
            max_iter: Maximum number of iterations

        Returns:
            Dictionary of default parameters
        """
        scheduler_type = scheduler_type.lower()

        if scheduler_type == "step":
            return {
                "step_size": max(1, max_iter // 10),  # Decay every 10% of training
                "gamma": 0.1,  # Reduce by 10x
            }

        elif scheduler_type == "exponential":
            return {
                "gamma": 0.95,  # 5% decay per epoch
            }

        elif scheduler_type == "cosine":
            return {
                "T_max": max_iter,  # Full cosine cycle
                "eta_min": 0.0,  # Minimum LR
            }

        elif scheduler_type == "linear":
            return {
                "start_factor": 1.0,  # Start at full LR
                "end_factor": 0.1,  # End at 10% of initial LR
                "total_iters": max_iter,
            }

        elif scheduler_type == "plateau":
            return {
                "mode": "min",  # Minimize loss
                "factor": 0.1,  # Reduce by 10x
                "patience": max(1, max_iter // 20),  # Wait 5% of training
                "threshold": 1e-4,
            }

        else:
            return {}

    @staticmethod
    def validate_params(
        scheduler_type: str,
        scheduler_params: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Validate and fill in missing scheduler parameters with defaults.

        Args:
            scheduler_type: Type of scheduler
            scheduler_params: User-provided parameters (can be None or incomplete)

        Returns:
            Complete parameter dictionary with defaults filled in
        """
        if scheduler_params is None:
            scheduler_params = {}

        # Get defaults
        defaults = LRSchedulerFactory.get_default_params(scheduler_type)

        # Merge with user params (user params take precedence)
        validated = {**defaults, **scheduler_params}

        return validated


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "constant",
    scheduler_params: dict[str, Any] | None = None,
    max_iter: int = 500,
) -> lr_scheduler.LRScheduler | None:
    """
    Convenience function to create a learning rate scheduler.

    This is a wrapper around LRSchedulerFactory.create_scheduler() for easier importing.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        scheduler_params: Parameters for the scheduler
        max_iter: Maximum number of iterations

    Returns:
        PyTorch LR scheduler or None
    """
    return LRSchedulerFactory.create_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        scheduler_params=scheduler_params,
        max_iter=max_iter,
    )
