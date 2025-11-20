"""
Service for loading models from the database and preparing them as ART estimators.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from pathlib import Path
import logging

from app import crud, schemas
from app.ai.estimators.object_detection import model_factory
from app.services.model_inference_service import model_inference_service

logger = logging.getLogger(__name__)

class EstimatorLoaderService:
    async def load_estimator_from_db(
        self,
        db: AsyncSession,
        model_id: UUID,
        estimator_id: str,
    ):
        """
        Loads a model from the database, creates an ART estimator,
        and registers it with the inference service.
        """
        if model_inference_service.is_loaded(estimator_id):
            logger.info(f"Estimator '{estimator_id}' is already loaded. Unloading and reloading with correct device.")
            model_inference_service.unregister_estimator(estimator_id)

        # 1. Get model from DB
        model = await crud.od_model.get(db, id=model_id)
        if not model:
            raise ValueError(f"Model with ID {model_id} not found")

        # 2. Get weights artifact
        if not model.artifacts:
            raise ValueError(f"Model {model_id} has no artifacts")
        
        weights_artifact = next((a for a in model.artifacts if a.artifact_type == "weights"), None)
        if not weights_artifact:
            raise ValueError(f"Model {model_id} has no weights artifact")

        # Try storage_key first (may be absolute path for external models)
        model_path = Path(weights_artifact.storage_key)
        if not model_path.exists():
            # Fall back to storage_path property (STORAGE_ROOT + storage_key + file_name)
            model_path = Path(weights_artifact.storage_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        # 3. Get parameters
        framework = schemas.EstimatorFramework(model.framework)
        if not model.inference_params or "estimator_type" not in model.inference_params:
            raise ValueError("Model is missing 'estimator_type' in inference_params")
        
        estimator_type = schemas.EstimatorType(model.inference_params["estimator_type"])
        
        class_names = ["person"] # Default, should be improved
        if model.labelmap:
            class_names = [model.labelmap[str(i)] for i in sorted([int(k) for k in model.labelmap.keys()])]

        input_size = [640, 640] # Default
        if model.input_spec and "shape" in model.input_spec and isinstance(model.input_spec["shape"], list):
            # Assuming shape is [H, W, C] or [W, H, C], we need [H, W]
            # Take the first two elements as H, W
            input_size = model.input_spec["shape"][:2]

        # 4. Use model factory to create estimator
        if framework != schemas.EstimatorFramework.PYTORCH:
            raise NotImplementedError("Only PyTorch framework is supported for auto-loading.")

        model_type_map = {
            schemas.EstimatorType.YOLO: 'yolo',
            schemas.EstimatorType.RT_DETR: 'rtdetr',
            schemas.EstimatorType.FASTER_RCNN: 'faster_rcnn',
            schemas.EstimatorType.EFFICIENTDET: 'efficientdet',
        }
        model_type = model_type_map.get(estimator_type)
        if not model_type:
            raise ValueError(f"Unsupported PyTorch estimator type: {estimator_type}")

        # Use CPU to avoid device mismatch issues
        # TODO: Fix GPU support in estimators
        device_type = "cpu"
        logger.info(f"Using device: {device_type}")

        # Use clip_values=(0, 255) to match ART's YOLO convention
        # YOLO models expect images in [0, 255] range (not normalized to [0, 1])
        # See: adversarial-robustness-toolbox/examples/get_started_yolo.py
        estimator = model_factory.load_model(
            model_path=str(model_path),
            model_type=model_type,
            class_names=class_names,
            input_size=input_size,
            device_type=device_type,
            clip_values=(0, 255),
        )

        # 5. Register with inference service
        model_inference_service.register_estimator(
            version_id=estimator_id, # Use the session-specific estimator_id
            estimator=estimator,
            class_names=class_names,
            source_model_id=str(model_id),
        )
        logger.info(f"Successfully loaded and registered estimator '{estimator_id}' from model '{model.name}'.")

        return {
            "estimator": estimator,
            "class_names": class_names,
            "model_path": str(model_path),
        }


estimator_loader = EstimatorLoaderService()
