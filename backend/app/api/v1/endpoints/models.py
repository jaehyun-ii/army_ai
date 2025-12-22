"""
Model repository endpoints.
"""
import shutil
import uuid
import tempfile
import zipfile
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID
import logging

from app.database import get_db
from app import crud, schemas
from app.services.model_inference_service import model_inference_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=schemas.ODModelResponse, status_code=status.HTTP_201_CREATED)
async def upload_model_for_estimator(
    *,
    db: AsyncSession = Depends(get_db),
    name: str = Form(None),
    framework: schemas.EstimatorFramework = Form(None),
    estimator_type: schemas.EstimatorType = Form(None),
    weights_file: UploadFile = File(None),
    yaml_file: UploadFile = File(None),
    description: str = Form(None),
    labelmap_json: str = Form(None),
    input_size_json: str = Form(None),
):
    """
    Upload a model file and create a corresponding database entry for use with estimators.
    Now supports automatic metadata extraction from .pt and config files (.yaml or .py)!

    If both .pt and config files are provided, metadata will be automatically extracted.
    Manual fields (name, labelmap_json, etc.) can override auto-detected values.

    Config file formats:
    - .yaml/.yml: For YOLO models
    - .py: For MMDetection models (EfficientDet, etc.)
    """
    import json
    import tempfile
    from app.utils.model_parser import ModelParser

    if not weights_file and not yaml_file:
        raise HTTPException(
            status_code=400,
            detail="At least one of weights_file or yaml_file must be provided"
        )

    # Save files to temporary locations for parsing
    temp_weights_path = None
    temp_yaml_path = None

    try:
        if weights_file:
            temp_weights = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
            shutil.copyfileobj(weights_file.file, temp_weights)
            temp_weights.close()
            temp_weights_path = Path(temp_weights.name)
            weights_file.file.seek(0)  # Reset file pointer for later use

        if yaml_file:
            # Support both .yaml/.yml and .py config files (for MMDetection)
            file_ext = Path(yaml_file.filename).suffix or '.yaml'
            temp_yaml = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            shutil.copyfileobj(yaml_file.file, temp_yaml)
            temp_yaml.close()
            temp_yaml_path = Path(temp_yaml.name)
            yaml_file.file.seek(0)  # Reset file pointer

        # Extract metadata from files
        metadata, errors = ModelParser.extract_model_info(temp_weights_path, temp_yaml_path)

        # Use auto-detected values as defaults, allow manual override
        auto_class_names = metadata.get("class_names")
        auto_input_size = metadata.get("input_size")
        auto_model_type = metadata.get("model_type")  # Don't default to "yolo" yet

        # Parse manual inputs
        labelmap = None
        if labelmap_json:
            try:
                labelmap = json.loads(labelmap_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for labelmap.")
        elif auto_class_names:
            # Auto-generate labelmap from detected class names
            labelmap = {str(i): name for i, name in enumerate(auto_class_names)}

        input_size = None
        if input_size_json:
            try:
                input_size = json.loads(input_size_json)
                if not isinstance(input_size, list) or len(input_size) != 2:
                    raise ValueError()
            except (json.JSONDecodeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid format for input_size. Expected '[height, width]'."
                )
        elif auto_input_size:
            input_size = auto_input_size

        # Use manual name or derive from filename
        model_name = name
        if not model_name and weights_file:
            model_name = Path(weights_file.filename).stem
        elif not model_name and yaml_file:
            model_name = Path(yaml_file.filename).stem

        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")

        # Use manual estimator_type or auto-detected
        final_estimator_type = estimator_type
        if not final_estimator_type:
            # Map detected model_type to estimator_type
            type_mapping = {
                "yolo": schemas.EstimatorType.YOLO,
                "detr": schemas.EstimatorType.RT_DETR,
                "rtdetr": schemas.EstimatorType.RT_DETR,
                "rt_detr": schemas.EstimatorType.RT_DETR,
                "faster_rcnn": schemas.EstimatorType.FASTER_RCNN,
                "efficientdet": schemas.EstimatorType.EFFICIENTDET,
            }
            if auto_model_type:
                final_estimator_type = type_mapping.get(auto_model_type, schemas.EstimatorType.YOLO)
            else:
                # No model type detected, default to YOLO
                final_estimator_type = schemas.EstimatorType.YOLO

        final_framework = framework or schemas.EstimatorFramework.PYTORCH

        # Validation
        if not labelmap:
            raise HTTPException(
                status_code=400,
                detail="Could not extract class information. Please provide labelmap_json manually."
            )

    finally:
        # Clean up temp files
        if temp_weights_path and temp_weights_path.exists():
            temp_weights_path.unlink()
        if temp_yaml_path and temp_yaml_path.exists():
            temp_yaml_path.unlink()

    # Save the model file
    from app.core.config import settings
    storage_base = Path(settings.STORAGE_ROOT) / "models"
    model_dir = storage_base / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if weights_file:
        weights_path = model_dir / weights_file.filename
        with weights_path.open("wb") as buffer:
            shutil.copyfileobj(weights_file.file, buffer)

    if yaml_file:
        yaml_path = model_dir / yaml_file.filename
        with yaml_path.open("wb") as buffer:
            shutil.copyfileobj(yaml_file.file, buffer)

    # Construct input_spec
    input_spec = None
    if input_size:
        input_spec = {"shape": input_size + [3], "dtype": "float32"}

    # Construct inference_params with extracted metadata
    inference_params = {
        "estimator_type": final_estimator_type.value,
        "auto_detected": bool(metadata),
        "metadata": metadata.get("additional_info", {})
    }

    model_in = schemas.ODModelCreate(
        name=model_name,
        description=description or f"Auto-uploaded {auto_model_type or 'detection'} model",
        framework=final_framework.value,
        inference_params=inference_params,
        labelmap=labelmap,
        input_spec=input_spec,
    )
    db_model = await crud.od_model.create(db, obj_in=model_in)

    # Create artifact entries
    # storage_key should be the relative path from STORAGE_ROOT/models to the directory containing the file
    storage_key = f"{model_name}"

    if weights_file:
        from app.models.model_repo import ArtifactType
        artifact_in = schemas.ODModelArtifactCreate(
            model_id=db_model.id,
            artifact_type=ArtifactType.WEIGHTS.value,
            storage_key=storage_key,
            file_name=weights_file.filename,
        )
        await crud.model_artifact.create(db, obj_in=artifact_in)

    if yaml_file:
        from app.models.model_repo import ArtifactType
        artifact_in = schemas.ODModelArtifactCreate(
            model_id=db_model.id,
            artifact_type=ArtifactType.CONFIG.value,
            storage_key=storage_key,
            file_name=yaml_file.filename,
        )
        await crud.model_artifact.create(db, obj_in=artifact_in)

    await db.refresh(db_model)

    return db_model


# Model Versions endpoints removed - ODModel now contains merged version fields
# Use POST / and GET / endpoints instead

@router.get("/versions")
async def list_model_versions(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all models (unified structure - no separate versions table)."""
    from sqlalchemy import select
    from app.models.model_repo import ODModel

    result = await db.execute(
        select(ODModel)
        .filter(ODModel.deleted_at.is_(None))
        .offset(skip)
        .limit(limit)
    )
    models = result.scalars().all()

    # Format response
    return [
        {
            "id": str(m.id),
            "model_id": str(m.id),  # Same as id for backward compatibility
            "name": m.name,
            "framework": m.framework,
            "stage": m.stage,
            "created_at": m.created_at.isoformat()
        }
        for m in models
    ]


@router.get("/versions/{version_id}", response_model=schemas.ODModelResponse)
async def get_model_version(
    version_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> schemas.ODModelResponse:
    """Get a model by ID (version_id param kept for backward compatibility)."""
    model = await crud.od_model.get(db, id=version_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {version_id} not found",
        )
    return model


# OD Models
@router.post("/", response_model=schemas.ODModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    *,
    db: AsyncSession = Depends(get_db),
    model_in: schemas.ODModelCreate,
) -> schemas.ODModelResponse:
    """Create a new OD model."""
    model = await crud.od_model.create(db, obj_in=model_in)
    return model


@router.get("/", response_model=List[schemas.ODModelResponse])
async def list_models(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> List[schemas.ODModelResponse]:
    """List OD models."""
    models = await crud.od_model.get_multi(db, skip=skip, limit=limit)
    return models


@router.get("/{model_id}", response_model=schemas.ODModelResponse)
async def get_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> schemas.ODModelResponse:
    """Get an OD model by ID."""
    model = await crud.od_model.get(db, id=model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found",
        )
    return model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete an OD model and its artifacts."""
    from app.crud.model_repo import crud_model

    model = await crud_model.get_model(db, model_id=model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found",
        )

    # Delete model files from storage
    from app.core.config import settings
    model_dir = Path(settings.STORAGE_ROOT) / "models" / model.name
    if model_dir.exists():
        shutil.rmtree(model_dir)
        logger.info(f"Deleted model files: {model_dir}")

    # Soft delete from database (sets deleted_at timestamp)
    deleted = await crud_model.delete_model(db, model_id=model_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model from database"
        )

    logger.info(f"Model deleted: {model_id}")

    return None  # FastAPI will automatically return 204 No Content


# Model Artifacts
@router.post("/artifacts", response_model=schemas.ODModelArtifactResponse, status_code=status.HTTP_201_CREATED)
async def create_model_artifact(
    *,
    db: AsyncSession = Depends(get_db),
    artifact_in: schemas.ODModelArtifactCreate,
) -> schemas.ODModelArtifactResponse:
    """Create a new model artifact."""
    artifact = await crud.model_artifact.create(db, obj_in=artifact_in)
    return artifact


# Model Inference
@router.post("/{model_id}/predict", response_model=schemas.PredictResponse)
async def predict_with_model(
    model_id: UUID,
    request: schemas.PredictRequest,
    db: AsyncSession = Depends(get_db),
) -> schemas.PredictResponse:
    """
    Run inference using a model directly (auto-load estimator if needed).

    This endpoint automatically loads the model as an estimator and runs prediction.
    """
    # Get model from database
    model = await crud.od_model.get(db, id=model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    # Use model ID as estimator ID
    estimator_id = str(model_id)

    # Check if estimator is already loaded
    if not model_inference_service.is_loaded(estimator_id):
        logger.info(f"Auto-loading estimator for model {model_id}")

        # Load estimator from database using the service
        from app.services.estimator_loader_service import estimator_loader

        try:
            await estimator_loader.load_estimator_from_db(
                db=db,
                model_id=model_id,
                estimator_id=estimator_id
            )
            logger.info(f"Successfully loaded estimator {estimator_id}")

        except Exception as e:
            logger.error(f"Failed to load estimator: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )

    # Run prediction
    try:
        # Load image from path or base64
        import cv2
        import numpy as np

        if request.image_path:
            image = cv2.imread(request.image_path)
            if image is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to load image from path: {request.image_path}"
                )
        elif request.image_base64:
            image = await model_inference_service.decode_image(request.image_base64)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either image_path or image_base64 must be provided"
            )

        result = await model_inference_service.run_inference(
            version_id=estimator_id,
            image=image,
            conf_threshold=request.conf_threshold or 0.25,
            iou_threshold=request.iou_threshold or 0.45
        )

        return schemas.PredictResponse(
            estimator_id=estimator_id,
            num_detections=len(result.detections),
                detections=[
                    schemas.Detection(
                        bbox=schemas.YoloBBox(
                            x_center=(det.bbox.x1 + det.bbox.x2) / 2,
                            y_center=(det.bbox.y1 + det.bbox.y2) / 2,
                            width=max(0.0, det.bbox.x2 - det.bbox.x1),
                            height=max(0.0, det.bbox.y2 - det.bbox.y1)
                        ),
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence
                    )
                    for det in result.detections
                ]
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/{model_id}/download")
async def download_model_archive(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Download model files (weights + config) as a ZIP archive.

    Returns a ZIP file containing:
    - Model weights file (.pt)
    - Config file (.yaml, .yml, or .py)
    - README with model metadata
    """
    # Get model from database
    model = await crud.od_model.get(db, id=model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found",
        )

    # Get model artifacts
    from sqlalchemy import select
    from app.models.model_repo import ODModelArtifact

    result = await db.execute(
        select(ODModelArtifact)
        .filter(
            ODModelArtifact.model_id == model_id,
            ODModelArtifact.deleted_at.is_(None)
        )
    )
    artifacts = result.scalars().all()

    if not artifacts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No artifacts found for this model"
        )

    # Get model directory
    from app.core.config import settings
    model_dir = Path(settings.STORAGE_ROOT) / "models" / artifacts[0].storage_key

    if not model_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model files not found in storage"
        )

    # Create temporary ZIP file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_zip_path = Path(temp_zip.name)
    temp_zip.close()

    try:
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all artifact files
            for artifact in artifacts:
                file_path = model_dir / artifact.file_name
                if file_path.exists():
                    zipf.write(file_path, arcname=artifact.file_name)
                    logger.info(f"Added {artifact.file_name} to archive")

            # Create README with model metadata
            readme_content = f"""# {model.name}

## Model Information
- **Name**: {model.name}
- **Framework**: {model.framework}
- **Task**: {model.task}
- **Stage**: {model.stage}
- **Description**: {model.description or 'N/A'}

## Model Specifications
- **Input Spec**: {model.input_spec or 'N/A'}
- **Label Map**: {model.labelmap or 'N/A'}

## Inference Parameters
{model.inference_params or 'N/A'}

## Files Included
"""
            for artifact in artifacts:
                readme_content += f"- {artifact.file_name} ({artifact.artifact_type})\n"

            # Add README to ZIP
            zipf.writestr('README.md', readme_content)

        # Return ZIP file
        zip_filename = f"{model.name}.zip"

        return FileResponse(
            path=str(temp_zip_path),
            filename=zip_filename,
            media_type='application/zip',
            background=None  # File will be deleted by cleanup
        )

    except Exception as e:
        # Clean up temp file on error
        if temp_zip_path.exists():
            temp_zip_path.unlink()
        logger.error(f"Failed to create model archive: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model archive: {str(e)}"
        )
