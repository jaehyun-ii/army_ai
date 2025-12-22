"""
Model loader - Legacy support for old custom model system.

NOTE: This module is deprecated. Use the new model loading system instead:
- app.api.v1.endpoints.simple_models (for uploading models)
- app.api.v1.endpoints.estimators (for loading and using estimators)

This file exists only for backward compatibility with old code that
imports ModelLoader.
"""
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    DEPRECATED: Model loader for custom models.

    Use the new estimator-based system instead:
    1. Upload model via /api/v1/simple-models/simple-upload
    2. Load estimator via /api/v1/estimators/load
    3. Run prediction via /api/v1/estimators/{estimator_id}/predict
    """

    def __init__(self):
        """Initialize model loader."""
        self.loaded_models: Dict[str, any] = {}
        logger.warning(
            "ModelLoader is deprecated. Use the new estimator-based system instead."
        )

    def load_model(
        self,
        model_id: str,
        weights_path: str,
        config: dict,
        adapter_path: Optional[str] = None,
    ) -> None:
        """
        Load a model (deprecated).

        Args:
            model_id: Unique model identifier
            weights_path: Path to model weights
            config: Model configuration
            adapter_path: Path to adapter module (deprecated)

        Raises:
            NotImplementedError: This method is deprecated
        """
        raise NotImplementedError(
            "Custom model loading is deprecated. "
            "Use the new estimator-based system:\n"
            "1. Upload via POST /api/v1/simple-models/simple-upload\n"
            "2. Load via POST /api/v1/estimators/load\n"
            "3. Predict via POST /api/v1/estimators/{estimator_id}/predict"
        )

    def get_model(self, model_id: str) -> Optional[any]:
        """
        Get a loaded model (deprecated).

        Args:
            model_id: Model identifier

        Returns:
            None (deprecated)
        """
        logger.warning(f"ModelLoader.get_model() is deprecated: {model_id}")
        return None

    def unload_model(self, model_id: str) -> None:
        """
        Unload a model (deprecated).

        Args:
            model_id: Model identifier
        """
        logger.warning(f"ModelLoader.unload_model() is deprecated: {model_id}")
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]


# Global instance for backward compatibility
model_loader = ModelLoader()
