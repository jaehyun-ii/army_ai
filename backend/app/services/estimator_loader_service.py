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
        if not model_path.is_absolute() or not model_path.exists():
            # Use storage_path property (STORAGE_ROOT/models/{storage_key}/{file_name})
            model_path = Path(weights_artifact.storage_path)
            logger.info(f"Using storage_path: {model_path}")

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. "
                    f"storage_key={weights_artifact.storage_key}, "
                    f"file_name={weights_artifact.file_name}"
                )

        # 3. Get parameters
        framework = schemas.EstimatorFramework(model.framework)

        # Try to get estimator_type from yaml_config.model_type first (more reliable)
        # Fall back to inference_params.estimator_type for backward compatibility
        estimator_type_str = None
        if (model.inference_params and
            "metadata" in model.inference_params and
            "yaml_config" in model.inference_params["metadata"] and
            "model_type" in model.inference_params["metadata"]["yaml_config"]):
            estimator_type_str = model.inference_params["metadata"]["yaml_config"]["model_type"]
            logger.info(f"Using model_type from yaml_config: {estimator_type_str}")
        elif model.inference_params and "estimator_type" in model.inference_params:
            estimator_type_str = model.inference_params["estimator_type"]
            logger.info(f"Using estimator_type from inference_params: {estimator_type_str}")
        else:
            raise ValueError("Model is missing 'estimator_type' in inference_params and 'model_type' in yaml_config")

        # Map string to enum (handle both formats: "yolo" and "YOLO")
        estimator_type_str = estimator_type_str.lower()
        if estimator_type_str == "yolo":
            estimator_type = schemas.EstimatorType.YOLO
        elif estimator_type_str == "rtdetr" or estimator_type_str == "rt-detr":
            estimator_type = schemas.EstimatorType.RT_DETR
        elif estimator_type_str == "faster_rcnn":
            estimator_type = schemas.EstimatorType.FASTER_RCNN
        elif estimator_type_str == "efficientdet":
            estimator_type = schemas.EstimatorType.EFFICIENTDET
        else:
            # Try to use as-is (for backward compatibility)
            try:
                estimator_type = schemas.EstimatorType(estimator_type_str)
            except ValueError:
                raise ValueError(f"Unsupported estimator type: {estimator_type_str}")
        
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

        # Use GPU if available, otherwise CPU
        import torch
        device_type = "gpu" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_type} (CUDA available: {torch.cuda.is_available()})")

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
