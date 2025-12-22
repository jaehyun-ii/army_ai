"""
Attack dataset endpoints for creating adversarial attack datasets.

Supports:
- Noise attacks (FGSM, PGD): Single-step workflow
- Patch attacks: Two-step workflow (generate patch → apply patch)
"""
from fastapi import APIRouter, Depends, Body, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from uuid import UUID

from app.database import get_db
from app import schemas
# from app.api import deps  # Temporarily disabled - no auth
from app.services.noise_attack_service import noise_attack_service
from app.services.patch_attack_service import patch_attack_service

router = APIRouter()


@router.post("/noise", status_code=status.HTTP_202_ACCEPTED)
async def create_noise_attack_dataset(
    background_tasks: BackgroundTasks,
    attack_name: str = Body(..., description="Name for the attacked dataset"),
    attack_method: str = Body(..., description="Attack method: 'fgsm', 'pgd', 'universal_noise', or 'noise_osfd'"),
    base_dataset_id: UUID = Body(..., description="Source dataset to attack"),
    model_id: UUID = Body(..., description="Target model for attack"),
    epsilon: Optional[float] = Body(None, ge=0.1, le=255.0, description="Maximum perturbation (0-255 scale, optional if intensity_level provided)"),
    alpha: Optional[float] = Body(None, ge=0.01, le=50.0, description="Step size (0-255 scale, auto-configured if None)"),
    iterations: Optional[int] = Body(None, ge=1, le=1000, description="Number of iterations (auto-configured if None)"),
    intensity_level: Optional[str] = Body(None, description="Attack intensity level: 'weak', 'medium', or 'strong' (auto-configures epsilon and iterations)"),
    target_class: Optional[str] = Body(None, description="Target class to attack (e.g., 'person'). If None, attack all objects."),
    session_id: Optional[str] = Body(None, description="SSE session ID for progress updates"),
    db: AsyncSession = Depends(get_db),
    # current_user: schemas.UserResponse = Depends(deps.get_current_user),  # Temporarily disabled
):
    """
    Create a noise-based attacked dataset using FGSM, PGD, Universal Noise, or Noise OSFD.

    **Workflow (Single-step):**
    1. Load base_dataset images
    2. Load model as estimator
    3. Apply noise attack to all images
    4. Save attacked images to new dataset
    5. Create AttackDataset2D record

    **Parameters:**
    - **attack_name**: Name for the attacked dataset
    - **attack_method**: "fgsm", "pgd", "universal_noise", or "noise_osfd"
    - **base_dataset_id**: UUID of the dataset to attack
    - **model_id**: UUID of the target model
    - **epsilon**: Maximum perturbation in [0, 255] range (optional if intensity_level provided)
    - **alpha**: Step size in [0, 255] range (auto-configured based on epsilon if None)
    - **iterations**: Number of iterations (auto-configured based on attack_method if None)
    - **intensity_level**: Attack intensity preset ("weak", "medium", "strong") - auto-configures epsilon and iterations
      - weak: Lower epsilon, fewer iterations (stealth-focused)
      - medium: Balanced epsilon and iterations (default)
      - strong: Higher epsilon, more iterations (effectiveness-focused)
    - **session_id**: (Optional) SSE session ID for real-time progress updates

    **Returns:**
    - **attack_dataset**: AttackDataset2D record
    - **output_dataset_id**: UUID of the created output dataset
    - **storage_path**: Path to the attacked images
    - **statistics**: Attack statistics (processed/failed images)

    **Example FGSM (with intensity level):**
    ```json
    {
      "attack_name": "FGSM_Person_Attack",
      "attack_method": "fgsm",
      "base_dataset_id": "uuid-dataset-123",
      "model_id": "uuid-model-456",
      "intensity_level": "medium"
    }
    ```

    **Example PGD (manual parameters):**
    ```json
    {
      "attack_name": "PGD_Custom_Attack",
      "attack_method": "pgd",
      "base_dataset_id": "uuid-dataset-123",
      "model_id": "uuid-model-456",
      "epsilon": 8.0,
      "iterations": 20
    }
    ```

    **Example Universal Noise (PyTorch with pseudo-GT):**
    ```json
    {
      "attack_name": "Universal_Noise_Attack",
      "attack_method": "universal_noise",
      "base_dataset_id": "uuid-dataset-123",
      "model_id": "uuid-model-456",
      "epsilon": 10.0,
      "alpha": 1.0,
      "iterations": 50
    }
    ```

    **Example Noise OSFD (PyTorch with RRB augmentation):**
    ```json
    {
      "attack_name": "Noise_OSFD_Attack",
      "attack_method": "noise_osfd",
      "base_dataset_id": "uuid-dataset-123",
      "model_id": "uuid-model-456",
      "epsilon": 10.0,
      "alpha": 1.0,
      "iterations": 30
    }
    ```
    """
    # Generate a session ID if not provided
    if not session_id:
        import time
        session_id = f"noise_{int(time.time() * 1000)}"

    # Define background task wrapper to create new DB session
    async def _create_noise_attack_background():
        """Background task to create noise attack dataset with its own DB session"""
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as bg_db:
            try:
                await noise_attack_service.create_noise_attack_dataset(
                    db=bg_db,
                    attack_name=attack_name,
                    attack_method=attack_method,
                    base_dataset_id=base_dataset_id,
                    model_id=model_id,
                    epsilon=epsilon,
                    alpha=alpha,
                    iterations=iterations,
                    intensity_level=intensity_level,
                    target_class=target_class,
                    session_id=session_id,
                    current_user_id=None,
                )
            except Exception as e:
                # Log error but don't crash - SSE will report the error
                print(f"[Background] Noise attack failed: {e}")
                import traceback
                traceback.print_exc()

    # Schedule background task
    background_tasks.add_task(_create_noise_attack_background)

    # Return immediately with status
    return {
        "status": "processing",
        "session_id": session_id,
        "message": "Noise attack started in background. Use SSE to monitor progress.",
        "sse_url": f"/api/v1/attack-datasets/sse/{session_id}"
    }


@router.get("")
async def list_attack_datasets(
    skip: int = 0,
    limit: int = 100,
    attack_type: str = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List all attack datasets.

    **Parameters:**
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    - **attack_type**: Filter by attack type (optional)

    **Returns:**
    List of AttackDataset2D records with actual image counts
    """
    from app import crud
    from sqlalchemy import select, func
    from app.models.dataset_2d import AttackDataset2D, Image2D
    import logging

    logger = logging.getLogger(__name__)

    # Build query
    query = select(AttackDataset2D).where(AttackDataset2D.deleted_at.is_(None))

    if attack_type:
        query = query.where(AttackDataset2D.attack_type == attack_type)

    query = query.offset(skip).limit(limit).order_by(AttackDataset2D.created_at.desc())

    # Execute query
    result = await db.execute(query)
    attack_datasets = result.scalars().all()

    # Calculate actual image counts from output datasets
    output_list = []
    for ds in attack_datasets:
        # Calculate actual image count from output_dataset_id
        image_count = 0
        if ds.output_dataset_id:
            try:
                count_stmt = (
                    select(func.count(Image2D.id))
                    .where(
                        Image2D.dataset_id == ds.output_dataset_id,
                        Image2D.deleted_at.is_(None)
                    )
                )
                count_result = await db.execute(count_stmt)
                image_count = count_result.scalar() or 0
            except Exception as e:
                logger.warning(f"Failed to count images for attack dataset {ds.id}: {e}")
                # Fallback to parameters.processed_images if count fails
                image_count = ds.parameters.get("processed_images", 0) if ds.parameters else 0

        # Convert to dict and add image_count
        ds_dict = {
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "attack_type": ds.attack_type,
            "target_class": ds.target_class,
            "target_model_id": ds.target_model_id,
            "base_dataset_id": ds.base_dataset_id,
            "output_dataset_id": ds.output_dataset_id,
            "patch_id": ds.patch_id,
            "parameters": ds.parameters or {},
            "created_by": ds.created_by,
            "created_at": ds.created_at,
            "image_count": image_count,  # Add actual image count
        }

        # Update parameters with actual image count
        if ds_dict["parameters"] is not None:
            ds_dict["parameters"]["processed_images"] = image_count

        output_list.append(ds_dict)

    return output_list


@router.get("/{attack_dataset_id}", response_model=schemas.AttackDataset2DResponse)
async def get_attack_dataset(
    attack_dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific attack dataset by ID.

    **Parameters:**
    - **attack_dataset_id**: UUID of the attack dataset

    **Returns:**
    AttackDataset2D record
    """
    from app import crud
    from fastapi import HTTPException

    attack_dataset = await crud.attack_dataset_2d.get(db, id=attack_dataset_id)
    if not attack_dataset:
        raise HTTPException(status_code=404, detail=f"Attack dataset {attack_dataset_id} not found")

    return attack_dataset


@router.delete("/{attack_dataset_id}")
async def delete_attack_dataset(
    attack_dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an attack dataset by ID (soft delete).

    **Parameters:**
    - **attack_dataset_id**: UUID of the attack dataset

    **Returns:**
    Success message
    """
    from app import crud
    from fastapi import HTTPException

    attack_dataset = await crud.attack_dataset_2d.get(db, id=attack_dataset_id)
    if not attack_dataset:
        raise HTTPException(status_code=404, detail=f"Attack dataset {attack_dataset_id} not found")

    await crud.attack_dataset_2d.remove(db, id=attack_dataset_id)
    await db.commit()

    return {"message": "Attack dataset deleted successfully"}


@router.get("/{attack_dataset_id}/download")
async def download_attack_dataset(
    attack_dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Download attack dataset as a ZIP file.

    **Parameters:**
    - **attack_dataset_id**: UUID of the attack dataset

    **Returns:**
    ZIP file containing all attacked images

    **Workflow:**
    1. Load AttackDataset2D record
    2. Get output_dataset_id
    3. Load all images from output dataset
    4. Create ZIP archive
    5. Return ZIP file for download
    """
    from app import crud
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse
    from pathlib import Path
    from app.core.config import settings
    import zipfile
    import io
    import logging

    logger = logging.getLogger(__name__)

    # Load attack dataset
    attack_dataset = await crud.attack_dataset_2d.get(db, id=attack_dataset_id)
    if not attack_dataset:
        raise HTTPException(status_code=404, detail=f"Attack dataset {attack_dataset_id} not found")

    # Get output dataset
    output_dataset_id = attack_dataset.output_dataset_id
    if not output_dataset_id:
        raise HTTPException(status_code=404, detail="Attack dataset has no output dataset")

    output_dataset = await crud.dataset_2d.get(db, id=output_dataset_id)
    if not output_dataset:
        raise HTTPException(status_code=404, detail=f"Output dataset {output_dataset_id} not found")

    # Get all images from output dataset
    from sqlalchemy import select
    from app.models.dataset_2d import Image2D

    query = select(Image2D).where(
        Image2D.dataset_id == output_dataset_id,
        Image2D.deleted_at.is_(None)
    )
    result = await db.execute(query)
    images = result.scalars().all()

    if not images:
        raise HTTPException(status_code=404, detail="No images found in output dataset")

    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    storage_root = Path(settings.STORAGE_ROOT)

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add dataset metadata
        metadata = {
            "attack_dataset_id": str(attack_dataset_id),
            "attack_dataset_name": attack_dataset.name,
            "attack_type": attack_dataset.attack_type.value if hasattr(attack_dataset.attack_type, 'value') else str(attack_dataset.attack_type),
            "base_dataset_id": str(attack_dataset.base_dataset_id),
            "output_dataset_id": str(output_dataset_id),
            "parameters": attack_dataset.parameters,
            "image_count": len(images),
            "created_at": attack_dataset.created_at.isoformat() if attack_dataset.created_at else None
        }

        import json
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

        # Add all images
        for idx, image in enumerate(images):
            image_path = storage_root / image.storage_key

            if not image_path.exists():
                logger.warning(f"Image file not found: {image.storage_key}")
                continue

            # Add image to ZIP with original filename
            zip_file.write(str(image_path), arcname=f"images/{image.file_name}")

        logger.info(f"Created ZIP with {len(images)} images for attack dataset {attack_dataset_id}")

    # Prepare ZIP for download
    zip_buffer.seek(0)

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={attack_dataset.name}_{attack_dataset_id}.zip"
        }
    )


@router.get("/sse/{session_id}")
async def attack_dataset_events(session_id: str):
    """
    Server-Sent Events endpoint for real-time attack progress updates.

    **Usage:**
    1. Generate a unique session_id (e.g., UUID)
    2. Open EventSource connection to this endpoint
    3. Pass the same session_id to the attack creation endpoint
    4. Receive real-time progress updates

    **Example (JavaScript):**
    ```javascript
    const sessionId = crypto.randomUUID();
    const eventSource = new EventSource(`/api/v1/attack-datasets/sse/${sessionId}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log(data.type, data.message);
    };

    // Then call the attack endpoint with this sessionId
    fetch('/api/v1/attack-datasets/noise', {
      method: 'POST',
      body: JSON.stringify({ ..., session_id: sessionId })
    });
    ```

    **Event Types:**
    - `status`: General status updates
    - `progress`: Progress updates with current/total
    - `info`: Informational messages
    - `warning`: Warning messages
    - `error`: Error messages
    - `success`: Success completion
    """
    from fastapi.responses import StreamingResponse
    import logging

    logger = logging.getLogger(__name__)

    # Create SSE session when client connects
    # Both services share the same SSE manager instance
    noise_attack_service.sse_manager.create_session(session_id)
    logger.info(f"SSE endpoint: Created session {session_id}")

    return StreamingResponse(
        noise_attack_service.sse_manager.event_stream(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/patch", status_code=status.HTTP_202_ACCEPTED)
async def apply_patch_to_dataset(
    background_tasks: BackgroundTasks,
    attack_name: str = Body(..., description="Name for the attacked dataset"),
    patch_id: UUID = Body(..., description="ID of the patch to apply"),
    base_dataset_id: UUID = Body(..., description="Dataset to apply the patch to"),
    patch_scale: float = Body(30.0, ge=1.0, le=100.0, description="Patch scale relative to bbox area (%)"),
    session_id: Optional[str] = Body(None, description="SSE session ID for progress updates"),
    db: AsyncSession = Depends(get_db),
    # current_user: schemas.UserResponse = Depends(deps.get_current_user),  # Temporarily disabled
):
    """
    Apply an existing patch to a dataset (Step 2 of patch attack workflow).

    **Prerequisites:**
    Must have generated a patch first using `POST /api/v1/patches/generate`

    **Workflow:**
    1. Load Patch2D record and patch file
    2. Load base_dataset images and annotations
    3. Apply patch to detection bboxes matching target_class
    4. Save patched images to new dataset
    5. Create AttackDataset2D record

    **Parameters:**
    - **attack_name**: Name for the attacked dataset
    - **patch_id**: UUID of the patch (from Step 1)
    - **base_dataset_id**: UUID of the dataset to apply the patch to
    - **patch_scale**: Patch size as percentage of bbox area (default: 30%, range: 1-100%)
    - **session_id**: (Optional) SSE session ID for real-time progress updates

    **Returns:**
    - **attack_dataset**: AttackDataset2D record
    - **output_dataset_id**: UUID of the created output dataset
    - **storage_path**: Path to the patched images
    - **statistics**: Attack statistics (processed/failed images)

    **Example:**
    ```json
    {
      "attack_name": "Person_Patch_Attack_Dataset",
      "patch_id": "uuid-patch-xyz",
      "base_dataset_id": "uuid-target-dataset-789",
      "patch_scale": 30.0
    }
    ```

    **Note:**
    The patch will be applied to all detection bboxes matching the target_class from the patch record.
    """
    # Generate a session ID if not provided
    if not session_id:
        import time
        session_id = f"patch_attack_{int(time.time() * 1000)}"

    # Define background task wrapper to create new DB session
    async def _apply_patch_background():
        """Background task to apply patch to dataset with its own DB session"""
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as bg_db:
            try:
                await patch_attack_service.apply_patch_to_dataset(
                    db=bg_db,
                    attack_name=attack_name,
                    patch_id=patch_id,
                    base_dataset_id=base_dataset_id,
                    patch_scale=patch_scale,
                    session_id=session_id,
                    current_user_id=None,
                )
            except Exception as e:
                # Log error but don't crash - SSE will report the error
                print(f"[Background] Patch attack failed: {e}")
                import traceback
                traceback.print_exc()

    # Schedule background task
    background_tasks.add_task(_apply_patch_background)

    # Return immediately with status
    return {
        "status": "processing",
        "session_id": session_id,
        "message": "Patch attack started in background. Use SSE to monitor progress.",
        "sse_url": f"/api/v1/attack-datasets/sse/{session_id}"
    }
