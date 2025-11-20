"""
Patch endpoints for generating and managing adversarial patches.

This is Step 1 of the patch attack workflow.
Step 2 (applying patches) is in attack_datasets.py
"""
from fastapi import APIRouter, Depends, Body, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from uuid import UUID

from app.database import get_db
from app import schemas
# from app.api import deps  # Temporarily disabled - no auth
from app.services.patch_service import patch_service

router = APIRouter()


@router.post("/generate", status_code=status.HTTP_201_CREATED)
async def generate_patch(
    patch_name: str = Body(..., description="Name for the patch"),
    attack_method: str = Body(..., description="Attack method: 'patch', 'dpatch', or 'robust_dpatch'"),
    source_dataset_id: UUID = Body(..., description="Dataset to train the patch on"),
    model_id: UUID = Body(..., description="Target model for attack"),
    target_class: str = Body(..., description="Target class name (e.g., 'person')"),
    patch_size: int = Body(100, ge=50, le=300, description="Size of the patch (height/width in pixels)"),
    learning_rate: float = Body(5.0, ge=0.1, le=20.0, description="Learning rate for optimization"),
    iterations: int = Body(500, ge=50, le=2000, description="Number of optimization iterations"),
    session_id: Optional[str] = Body(None, description="SSE session ID for progress updates"),
    db: AsyncSession = Depends(get_db),
    # current_user: schemas.UserResponse = Depends(deps.get_current_user),  # Temporarily disabled
):
    """
    Generate an adversarial patch (Step 1 of patch attack workflow).

    **Workflow:**
    1. Load source_dataset images containing target_class
    2. Load model as estimator
    3. Train adversarial patch using ART
    4. Save patch file to /storage/patches/
    5. Create Patch2D database record

    **Parameters:**
    - **patch_name**: Name for the patch
    - **attack_method**:
      - `"patch"`: AdversarialPatchPyTorch (general patch with transformations)
      - `"dpatch"`: DPatch (object detector specific)
      - `"robust_dpatch"`: RobustDPatch (robust to cropping, rotation, brightness)
    - **source_dataset_id**: UUID of dataset containing target class images
    - **model_id**: UUID of the target model
    - **target_class**: Target class name (must exist in model's labelmap)
    - **patch_size**: Size of the square patch in pixels (default: 100)
    - **learning_rate**: Optimization learning rate (default: 5.0)
    - **iterations**: Number of training iterations (default: 500)
    - **session_id**: (Optional) SSE session ID for real-time progress updates

    **Returns:**
    - **patch**: Patch2D record with metadata
    - **patch_file_path**: Path to the saved patch image

    **Example (RobustDPatch):**
    ```json
    {
      "patch_name": "Person_Invisibility_Patch",
      "attack_method": "robust_dpatch",
      "source_dataset_id": "uuid-training-dataset-123",
      "model_id": "uuid-yolo-456",
      "target_class": "person",
      "patch_size": 100,
      "learning_rate": 5.0,
      "iterations": 500
    }
    ```

    **After patch generation:**
    Use the returned `patch.id` in Step 2 to apply the patch to a dataset:
    ```
    POST /api/v1/attack-datasets/patch
    {
      "patch_id": "<patch.id>",
      "base_dataset_id": "<target-dataset-id>",
      ...
    }
    ```
    """
    patch = await patch_service.generate_patch(
        db=db,
        patch_name=patch_name,
        attack_method=attack_method,
        source_dataset_id=source_dataset_id,
        model_id=model_id,
        target_class=target_class,
        patch_size=patch_size,
        learning_rate=learning_rate,
        iterations=iterations,
        session_id=session_id,
        current_user_id=None,  # Temporarily disabled auth
    )

    return {
        "patch": patch,
        "patch_file_path": f"/storage/{patch.storage_key}",
    }


@router.get("", response_model=List[schemas.Patch2DResponse])
async def list_patches(
    skip: int = 0,
    limit: int = 100,
    target_class: str = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List all generated patches.

    **Parameters:**
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    - **target_class**: Filter by target class (optional)

    **Returns:**
    List of Patch2D records
    """
    from app import crud
    from sqlalchemy import select
    from app.models.dataset_2d import Patch2D

    # Build query
    query = select(Patch2D).where(Patch2D.deleted_at.is_(None))

    if target_class:
        query = query.where(Patch2D.target_class == target_class)

    query = query.offset(skip).limit(limit).order_by(Patch2D.created_at.desc())

    # Execute query
    result = await db.execute(query)
    patches = result.scalars().all()

    return patches


@router.get("/{patch_id}", response_model=schemas.Patch2DResponse)
async def get_patch(
    patch_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific patch by ID.

    **Parameters:**
    - **patch_id**: UUID of the patch

    **Returns:**
    Patch2D record
    """
    from app import crud
    from fastapi import HTTPException

    patch = await crud.patch_2d.get(db, id=patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found")

    return patch


@router.delete("/{patch_id}")
async def delete_patch(
    patch_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a patch by ID (soft delete).

    **Parameters:**
    - **patch_id**: UUID of the patch

    **Returns:**
    Success message
    """
    from app import crud
    from fastapi import HTTPException

    patch = await crud.patch_2d.get(db, id=patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found")

    await crud.patch_2d.remove(db, id=patch_id)
    await db.commit()

    return {"message": "Patch deleted successfully"}


@router.get("/{patch_id}/image")
async def get_patch_image(
    patch_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get patch image file directly.

    **Parameters:**
    - **patch_id**: UUID of the patch

    **Returns:**
    PNG image file
    """
    from app import crud
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from pathlib import Path
    from app.core.config import settings

    patch = await crud.patch_2d.get(db, id=patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found")

    # Construct full path
    storage_root = Path(settings.STORAGE_ROOT)
    patch_path = storage_root / patch.storage_key

    if not patch_path.exists():
        raise HTTPException(status_code=404, detail=f"Patch file not found: {patch.storage_key}")

    return FileResponse(
        path=str(patch_path),
        media_type="image/png",
        filename=patch.file_name or f"patch_{patch_id}.png"
    )


@router.get("/{patch_id}/download")
async def download_patch(
    patch_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Download patch file.

    **Parameters:**
    - **patch_id**: UUID of the patch

    **Returns:**
    PNG file download
    """
    from app import crud
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from pathlib import Path
    from app.core.config import settings

    patch = await crud.patch_2d.get(db, id=patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found")

    # Construct full path
    storage_root = Path(settings.STORAGE_ROOT)
    patch_path = storage_root / patch.storage_key

    if not patch_path.exists():
        raise HTTPException(status_code=404, detail=f"Patch file not found: {patch.storage_key}")

    return FileResponse(
        path=str(patch_path),
        media_type="application/octet-stream",
        filename=patch.file_name or f"patch_{patch_id}.png",
        headers={"Content-Disposition": f"attachment; filename={patch.file_name or f'patch_{patch_id}.png'}"}
    )


@router.post("/{patch_id}/preview")
async def preview_patch_on_image(
    patch_id: UUID,
    image_id: UUID = Body(..., description="Image ID to preview patch on"),
    patch_scale: float = Body(30.0, ge=1.0, le=100.0, description="Patch scale (% of image)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Preview patch applied to a specific image.

    **Parameters:**
    - **patch_id**: UUID of the patch
    - **image_id**: UUID of the image to apply patch to
    - **patch_scale**: Patch size as percentage of image area (1-100%)

    **Returns:**
    - **image_data**: Base64 encoded preview image
    - **image_mime_type**: MIME type (image/png)
    - **patch_applied**: Success flag

    **Example:**
    ```json
    {
      "image_id": "uuid-image-123",
      "patch_scale": 30.0
    }
    ```
    """
    from app import crud
    from fastapi import HTTPException
    from pathlib import Path
    from app.core.config import settings
    import cv2
    import numpy as np
    import base64

    # Load patch
    patch = await crud.patch_2d.get(db, id=patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found")

    storage_root = Path(settings.STORAGE_ROOT)
    patch_path = storage_root / patch.storage_key

    if not patch_path.exists():
        raise HTTPException(status_code=404, detail=f"Patch file not found")

    patch_bgr = cv2.imread(str(patch_path))
    if patch_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to load patch image")

    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

    # Load image
    image = await crud.image_2d.get(db, id=image_id)
    if not image:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

    image_path = storage_root / image.storage_key

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to load image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Calculate patch size based on image area
    img_height, img_width = img_rgb.shape[:2]
    img_area = img_height * img_width
    patch_area = img_area * (patch_scale / 100.0)
    patch_size = int(np.sqrt(patch_area))

    # Resize patch
    patch_resized = cv2.resize(patch_rgb, (patch_size, patch_size))

    # Place patch at center
    center_y = img_height // 2
    center_x = img_width // 2
    patch_y1 = max(0, center_y - patch_size // 2)
    patch_x1 = max(0, center_x - patch_size // 2)
    patch_y2 = min(img_height, patch_y1 + patch_size)
    patch_x2 = min(img_width, patch_x1 + patch_size)

    # Adjust patch size if needed
    actual_patch_height = patch_y2 - patch_y1
    actual_patch_width = patch_x2 - patch_x1
    patch_resized = cv2.resize(patch_resized, (actual_patch_width, actual_patch_height))

    # Apply patch
    preview_img = img_rgb.copy()
    preview_img[patch_y1:patch_y2, patch_x1:patch_x2] = patch_resized

    # Encode to base64
    preview_bgr = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', preview_bgr)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "image_data": image_base64,
        "image_mime_type": "image/png",
        "patch_applied": True,
        "patch_location": {
            "x": patch_x1,
            "y": patch_y1,
            "width": actual_patch_width,
            "height": actual_patch_height
        }
    }


@router.get("/sse/{session_id}")
async def patch_generation_events(session_id: str):
    """
    Server-Sent Events endpoint for real-time patch generation progress updates.

    **Usage:**
    1. Generate a unique session_id (e.g., UUID)
    2. Open EventSource connection to this endpoint
    3. Pass the same session_id to the patch generation endpoint
    4. Receive real-time progress updates

    **Example (JavaScript):**
    ```javascript
    const sessionId = crypto.randomUUID();
    const eventSource = new EventSource(`/api/v1/patches/sse/${sessionId}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log(data.type, data.message);
    };

    // Then call the patch generation endpoint with this sessionId
    fetch('/api/v1/patches/generate', {
      method: 'POST',
      body: JSON.stringify({ ..., session_id: sessionId })
    });
    ```

    **Event Types:**
    - `status`: General status updates
    - `info`: Informational messages
    - `warning`: Warning messages
    - `error`: Error messages
    - `success`: Success completion
    """
    from fastapi.responses import StreamingResponse
    import logging

    logger = logging.getLogger(__name__)

    # Create SSE session when client connects
    patch_service.sse_manager.create_session(session_id)
    logger.info(f"SSE endpoint: Created session {session_id}")

    return StreamingResponse(
        patch_service.sse_manager.event_stream(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
