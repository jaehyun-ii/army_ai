"""
Estimator management endpoints for adversarial robustness testing.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
from uuid import UUID
import numpy as np
from pathlib import Path
import base64
import cv2
import logging

from app.database import get_db
from app import schemas
from app.ai.estimators.object_detection import model_factory
from app.services.model_inference_service import model_inference_service
from app.services.estimator_loader_service import estimator_loader

router = APIRouter()
logger = logging.getLogger(__name__)

# The local _loaded_estimators cache is REMOVED.
# All state is now managed in model_inference_service.


@router.post("/estimators/load", response_model=schemas.LoadEstimatorResponse)
async def load_estimator(
    request: schemas.LoadEstimatorRequest,
    db: AsyncSession = Depends(get_db)
) -> schemas.LoadEstimatorResponse:
    """
    Load an object detection estimator for adversarial testing.
    """
    estimator_id = request.estimator_id
    if model_inference_service.is_loaded(estimator_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Estimator '{estimator_id}' is already loaded"
        )

    if not request.model_id and not request.model_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either model_id or model_path must be provided"
        )

    try:
        estimator = None
        db_model_id = None
        model_path_str = None
        
        if request.model_id:
            # --- DB Loading Workflow ---
            db_model_id = UUID(request.model_id)
            loaded_data = await estimator_loader.load_estimator_from_db(
                db=db,
                model_id=db_model_id,
                estimator_id=estimator_id
            )
            estimator = loaded_data["estimator"]
            model_path_str = loaded_data["model_path"]
            db_model_id = request.model_id # Keep it as string for response

        else:
            # --- Path Loading Workflow ---
            model_path_str = request.model_path
            model_path = Path(model_path_str)
            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model file not found: {model_path_str}")

            # This logic remains from the original file, but now registers to the central service
            config = request.config or {}
            if "input_size" not in config:
                config["input_size"] = [640, 640]
            if "class_names" not in config:
                config["class_names"] = ["person"] # Minimal default

            if request.framework != schemas.EstimatorFramework.PYTORCH:
                raise NotImplementedError("Only PyTorch framework is supported for path loading.")

            model_type_map = {
                schemas.EstimatorType.YOLO: 'yolo',
                schemas.EstimatorType.RT_DETR: 'rtdetr',
                schemas.EstimatorType.FASTER_RCNN: 'faster_rcnn',
                schemas.EstimatorType.EFFICIENTDET: 'efficientdet',
            }
            model_type = model_type_map.get(request.estimator_type)
            if not model_type:
                raise ValueError(f"Unsupported PyTorch estimator type: {request.estimator_type}")

            estimator = model_factory.load_model(
                model_path=str(model_path),
                model_type=model_type,
                class_names=config.get("class_names"),
                input_size=config.get("input_size"),
                device_type="auto",
                clip_values=(0, 255),
            )
            
            # Register with the central service
            model_inference_service.register_estimator(
                version_id=estimator_id,
                estimator=estimator,
                class_names=config.get("class_names"),
                source_model_path=model_path_str,
            )

        if estimator is None:
            raise HTTPException(status_code=500, detail="Failed to create estimator instance")

        return schemas.LoadEstimatorResponse(
            estimator_id=estimator_id,
            status="loaded",
            message=f"Successfully loaded {request.framework.value} {request.estimator_type.value} estimator",
            framework=request.framework,
            estimator_type=request.estimator_type,
            model_id=db_model_id,
            model_path=model_path_str,
            supports_adversarial_attack=estimator.supports_adversarial_attack(),
        )

    except Exception as e:
        logger.error(f"Error loading estimator: {e}", exc_info=True)
        # Clean up if something went wrong
        if model_inference_service.is_loaded(estimator_id):
            model_inference_service.unregister_estimator(estimator_id)
        raise HTTPException(status_code=500, detail=f"Error loading estimator: {str(e)}")


@router.post("/estimators/{estimator_id}/predict", response_model=schemas.PredictResponse)
async def predict(
    estimator_id: str,
    request: schemas.PredictRequest
) -> schemas.PredictResponse:
    """
    Run prediction on an image using a loaded estimator.
    """
    if not model_inference_service.is_loaded(estimator_id):
        raise HTTPException(status_code=404, detail=f"Estimator '{estimator_id}' not found.")

    try:
        if request.image_base64:
            image = await model_inference_service.decode_image(request.image_base64)
        elif request.image_path:
            img_path = Path(request.image_path)
            if not img_path.exists():
                raise HTTPException(status_code=404, detail=f"Image file not found: {request.image_path}")
            image = cv2.imread(str(img_path))
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_path must be provided")

        # Use the central inference service
        result = await model_inference_service.run_inference(
            version_id=estimator_id,
            image=image,
            conf_threshold=request.conf_threshold,
            iou_threshold=0.45 # iou is not used in predict endpoint schema, using default
        )
        
        # Convert result to PredictResponse schema
        detections = []
        for det in result.detections:
            detections.append(
                schemas.Detection(
                    bbox=schemas.BBox(x1=det.bbox.x1, y1=det.bbox.y1, x2=det.bbox.x2, y2=det.bbox.y2),
                    class_id=det.class_id,
                    confidence=det.confidence
                )
            )

        return schemas.PredictResponse(
            estimator_id=estimator_id,
            num_detections=len(detections),
            detections=detections
        )
    except Exception as e:
        logger.error(f"Error running prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running prediction: {str(e)}")


@router.get("/estimators", response_model=schemas.EstimatorListResponse)
async def list_estimators() -> schemas.EstimatorListResponse:
    """
    List all currently loaded estimators.
    """
    loaded_ids = model_inference_service.get_loaded_estimator_ids()
    estimators_info = []
    for estimator_id in loaded_ids:
        estimator_data = model_inference_service._loaded_estimators[estimator_id] # Accessing private member, not ideal but necessary here
        estimator = estimator_data["estimator"]
        estimators_info.append({
            "estimator_id": estimator_id,
            "class_name": estimator.__class__.__name__,
            "framework": "pytorch", # Assuming pytorch for now
            "estimator_type": "yolo", # Assuming yolo for now
            "model_id": estimator_data.get("source_model_id"),
            "model_path": estimator_data.get("source_model_path"),
            "supports_adversarial_attack": estimator.supports_adversarial_attack(),
            "status": "loaded"
        })

    return schemas.EstimatorListResponse(
        count=len(estimators_info),
        estimators=estimators_info
    )


@router.delete("/estimators/{estimator_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unload_estimator(estimator_id: str):
    """
    Unload an estimator and free its resources.
    """
    if not model_inference_service.is_loaded(estimator_id):
        raise HTTPException(status_code=404, detail=f"Estimator '{estimator_id}' not found")
    
    model_inference_service.unregister_estimator(estimator_id)
    return None


@router.get("/estimators/{estimator_id}/status", response_model=schemas.EstimatorStatusResponse)
async def get_estimator_status(estimator_id: str) -> schemas.EstimatorStatusResponse:
    """
    Get the status of a loaded estimator.
    """
    if not model_inference_service.is_loaded(estimator_id):
        return schemas.EstimatorStatusResponse(
            estimator_id=estimator_id,
            status="not_loaded",
            message=f"Estimator '{estimator_id}' is not loaded"
        )
    
    estimator_data = model_inference_service._loaded_estimators[estimator_id]
    estimator = estimator_data["estimator"]
    return schemas.EstimatorStatusResponse(
        estimator_id=estimator_id,
        status="loaded",
        class_name=estimator.__class__.__name__,
        supports_adversarial_attack=estimator.supports_adversarial_attack(),
        message=f"Estimator '{estimator_id}' is loaded and ready"
    )