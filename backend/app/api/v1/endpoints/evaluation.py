"""
API endpoints for evaluation operations.
"""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks, Body
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel as PydanticBaseModel
import cv2
import numpy as np
from pathlib import Path
import hashlib

from app.database import get_db
from app.crud import evaluation as crud_evaluation
from app.schemas.evaluation import (
    EvalRunCreate,
    EvalRunUpdate,
    EvalRunResponse,
    EvalRunListResponse,
    EvalItemCreate,
    EvalItemUpdate,
    EvalItemResponse,
    EvalItemListResponse,
    EvalListCreate,
    EvalListUpdate,
    EvalListResponse,
    EvalListListResponse,
    EvalListItemCreate,
    EvalListItemResponse,
    EvalRunPairResponse,
    EvalRunPairDeltaResponse,
    EvalStatus,
    EvalPhase,
    EvalDatasetType,
)
from app.services.evaluation_service import evaluation_service

router = APIRouter()


# ========== Request Models ==========

class ExecuteEvalRequest(PydanticBaseModel):
    """Request body for executing evaluation."""
    session_id: Optional[str] = None


# ========== Evaluation Run Endpoints ==========

@router.post("/runs", response_model=EvalRunResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation_run(
    eval_run: EvalRunCreate,
    db: AsyncSession = Depends(get_db),
    # current_user: User = Depends(get_current_user),  # TODO: Add auth
):
    """
    Create a new evaluation run.

    - **pre_attack**: Requires base_dataset_id
    - **post_attack**: Requires attack_dataset_id (base_dataset_id auto-validated)
    """
    try:
        # Auto-populate base_dataset_id from attack_dataset if needed
        if eval_run.attack_dataset_id and not eval_run.base_dataset_id:
            from app import crud
            # Get attack dataset and extract base_dataset_id
            attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
            if attack_dataset and attack_dataset.base_dataset_id:
                eval_run.base_dataset_id = attack_dataset.base_dataset_id

        db_eval_run = await crud_evaluation.create_eval_run(
            db=db,
            eval_run=eval_run,
            # created_by=current_user.id,  # TODO: Add auth
        )
        return db_eval_run
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/runs/{run_id}", response_model=EvalRunResponse)
async def get_evaluation_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get evaluation run by ID."""
    db_eval_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=run_id)
    if not db_eval_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )
    return db_eval_run


@router.get("/runs", response_model=EvalRunListResponse)
async def list_evaluation_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    phase: Optional[EvalPhase] = None,
    status_filter: Optional[EvalStatus] = Query(None, alias="status"),
    model_id: Optional[UUID] = None,
    base_dataset_id: Optional[UUID] = None,
    attack_dataset_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List evaluation runs with filters and pagination.

    Filters:
    - **phase**: pre_attack or post_attack
    - **status**: queued, running, completed, failed, aborted
    - **model_id**: Filter by model version
    - **base_dataset_id**: Filter by base dataset
    - **attack_dataset_id**: Filter by attack dataset
    """
    skip = (page - 1) * page_size
    items, total = await crud_evaluation.get_eval_runs(
        db=db,
        skip=skip,
        limit=page_size,
        phase=phase,
        status=status_filter,
        model_id=model_id,
        base_dataset_id=base_dataset_id,
        attack_dataset_id=attack_dataset_id,
    )
    return EvalRunListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.patch("/runs/{run_id}", response_model=EvalRunResponse)
async def update_evaluation_run(
    run_id: UUID,
    eval_run_update: EvalRunUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update evaluation run (status, metrics, timestamps, etc.)."""
    db_eval_run = await crud_evaluation.update_eval_run(
        db=db,
        eval_run_id=run_id,
        eval_run_update=eval_run_update,
    )
    if not db_eval_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )
    return db_eval_run


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Soft delete evaluation run."""
    success = await crud_evaluation.delete_eval_run(db=db, eval_run_id=run_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )


@router.post("/runs/{run_id}/execute", response_model=EvalRunResponse)
async def execute_evaluation_run(
    run_id: UUID,
    background_tasks: BackgroundTasks,
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0),
    request_body: ExecuteEvalRequest = Body(default=ExecuteEvalRequest()),
    db: AsyncSession = Depends(get_db),
):
    """
    Execute an evaluation run in the background.

    The evaluation will run asynchronously and update the run status.
    - Initial status will be set to 'queued' then 'running'
    - Upon completion, status will be 'completed' with metrics
    - On failure, status will be 'failed'

    If session_id is provided, real-time logs will be streamed via SSE.
    Connect to /evaluation/runs/events/{session_id} before calling this endpoint.
    """
    # Check if evaluation run exists
    db_eval_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=run_id)
    if not db_eval_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )

    # Check if already running or completed
    if db_eval_run.status in [EvalStatus.RUNNING, EvalStatus.COMPLETED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation run is already {db_eval_run.status}",
        )

    # Update status to queued
    await crud_evaluation.update_eval_run(
        db=db,
        eval_run_id=run_id,
        eval_run_update=EvalRunUpdate(status=EvalStatus.QUEUED),
    )
    await db.commit()
    await db.refresh(db_eval_run)

    # Schedule evaluation in background
    # Note: We need to create a new session for background tasks
    session_id = request_body.session_id

    async def run_evaluation():
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as bg_db:
            try:
                await evaluation_service.execute_evaluation(
                    db=bg_db,
                    eval_run_id=run_id,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    session_id=session_id,
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Background evaluation failed: {e}", exc_info=True)

    background_tasks.add_task(run_evaluation)

    return db_eval_run


@router.get("/runs/events/{session_id}")
async def evaluation_events(session_id: str):
    """
    SSE endpoint for real-time evaluation logs.

    Connect to this endpoint BEFORE calling /runs/{run_id}/execute with the same session_id.

    Receives:
    - Status updates (queued, running, completed, failed)
    - Progress updates (loading dataset, running inference, calculating metrics)
    - Info messages (dataset info, image counts, metric results)
    - Success/Error notifications
    - Completion notification

    Example event format:
    data: {"type": "status", "message": "평가 시작 중...", "timestamp": "..."}
    data: {"type": "info", "message": "총 100개 이미지 발견", "timestamp": "..."}
    data: {"type": "complete", "message": "평가 완료", "eval_run_id": "...", "timestamp": "..."}
    """
    # Create session for this SSE connection
    evaluation_service.sse_manager.create_session(session_id)

    return StreamingResponse(
        evaluation_service.sse_manager.event_stream(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


# ========== Evaluation Item Endpoints ==========

@router.post("/items", response_model=EvalItemResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation_item(
    eval_item: EvalItemCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new evaluation item (per-image result)."""
    try:
        db_eval_item = await crud_evaluation.create_eval_item(db=db, eval_item=eval_item)
        return db_eval_item
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/items/bulk", response_model=List[EvalItemResponse], status_code=status.HTTP_201_CREATED)
async def create_evaluation_items_bulk(
    eval_items: List[EvalItemCreate],
    db: AsyncSession = Depends(get_db),
):
    """Create multiple evaluation items in bulk."""
    try:
        db_eval_items = await crud_evaluation.create_eval_items_bulk(db=db, eval_items=eval_items)
        return db_eval_items
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/items/{item_id}", response_model=EvalItemResponse)
async def get_evaluation_item(
    item_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get evaluation item by ID.

    PHASE 2 UPDATE: Returns computed file_name/storage_key from image_2d.
    """
    db_eval_item = await crud_evaluation.get_eval_item(db=db, eval_item_id=item_id)
    if not db_eval_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation item not found",
        )

    # PHASE 2: Populate computed fields from image_2d relationship
    response_dict = {
        **db_eval_item.__dict__,
        "file_name": db_eval_item.image_2d.file_name if db_eval_item.image_2d else None,
        "storage_key": db_eval_item.image_2d.storage_key if db_eval_item.image_2d else None,
    }
    return EvalItemResponse(**response_dict)


@router.get("/runs/{run_id}/items", response_model=EvalItemListResponse)
async def list_evaluation_items(
    run_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    dataset_type: Optional[EvalDatasetType] = Query(None, description="Filter by dataset type: 'base' or 'attack'"),
    db: AsyncSession = Depends(get_db),
):
    """
    List all evaluation items for a specific run.

    - **dataset_type**: Optional filter for 'base' or 'attack' items

    PHASE 2 UPDATE: Returns computed file_name/storage_key from image_2d.
    """
    skip = (page - 1) * page_size
    items, total = await crud_evaluation.get_eval_items(
        db=db,
        run_id=run_id,
        skip=skip,
        limit=page_size,
        dataset_type=dataset_type,  # NEW: Pass filter to CRUD
    )

    # PHASE 2: Populate computed fields from image_2d relationship
    response_items = []
    for item in items:
        item_dict = {
            **item.__dict__,
            "file_name": item.image_2d.file_name if item.image_2d else None,
            "storage_key": item.image_2d.storage_key if item.image_2d else None,
        }
        response_items.append(EvalItemResponse(**item_dict))

    return EvalItemListResponse(
        items=response_items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.patch("/items/{item_id}", response_model=EvalItemResponse)
async def update_evaluation_item(
    item_id: UUID,
    eval_item_update: EvalItemUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update evaluation item."""
    db_eval_item = await crud_evaluation.update_eval_item(
        db=db,
        eval_item_id=item_id,
        eval_item_update=eval_item_update,
    )
    if not db_eval_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation item not found",
        )
    return db_eval_item


@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation_item(
    item_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Soft delete evaluation item."""
    success = await crud_evaluation.delete_eval_item(db=db, eval_item_id=item_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation item not found",
        )


# ========== Class Metrics Endpoints ==========

@router.get("/runs/{run_id}/class-metrics")
async def get_evaluation_class_metrics(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get per-class metrics for an evaluation run.

    Returns metrics for each class including:
    - AP metrics (map, map50, map75)
    - Precision, Recall, F1
    - Ground truth count and prediction count
    """
    # Check if evaluation run exists
    eval_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=run_id)
    if not eval_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )

    # Get class metrics
    class_metrics = await crud_evaluation.get_eval_class_metrics(db=db, run_id=run_id)

    # Format response
    return [
        {
            "class_name": cm.class_name,
            "metrics": cm.metrics,
        }
        for cm in class_metrics
    ]


# ========== Visualization Endpoints ==========

def get_class_color(class_name: str) -> tuple:
    """
    Generate consistent color for a class name using hash.
    Same class name will always return the same color.
    """
    # Hash the class name
    hash_value = int(hashlib.md5(class_name.encode()).hexdigest(), 16)

    # Generate RGB values from hash (using HSV for better color distribution)
    hue = (hash_value % 360) / 360.0
    saturation = 0.7 + (hash_value % 30) / 100.0  # 0.7-1.0
    value = 0.8 + (hash_value % 20) / 100.0  # 0.8-1.0

    # Convert HSV to RGB
    import colorsys
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    # Convert to BGR for OpenCV (0-255)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def draw_bounding_boxes(image: np.ndarray, detections: List[dict], title: str = "") -> np.ndarray:
    """
    Draw bounding boxes on image with consistent colors per class.

    Args:
        image: Input image
        detections: List of detection dicts with bbox and class_name
        title: Title to display on image

    Returns:
        Image with bounding boxes drawn
    """
    img = image.copy()
    height, width = img.shape[:2]

    # Draw title if provided
    if title:
        cv2.rectangle(img, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)

    for det in detections:
        bbox = det.get('bbox', {})
        class_name = det.get('class_name', 'unknown')
        confidence = det.get('confidence', 0.0)

        # Get consistent color for this class
        color = get_class_color(class_name)

        # Extract coordinates (handle both formats)
        if 'x1' in bbox:
            x1_raw, y1_raw = float(bbox['x1']), float(bbox['y1'])
            x2_raw, y2_raw = float(bbox['x2']), float(bbox['y2'])

            if max(x1_raw, y1_raw, x2_raw, y2_raw) <= 1.0:
                x1 = int(x1_raw * width)
                y1 = int(y1_raw * height)
                x2 = int(x2_raw * width)
                y2 = int(y2_raw * height)
            else:
                x1, y1, x2, y2 = int(x1_raw), int(y1_raw), int(x2_raw), int(y2_raw)
        elif 'x_center' in bbox:
            x_center, y_center = bbox['x_center'], bbox['y_center']
            w, h = bbox['width'], bbox['height']
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)
        else:
            continue

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)

        # Draw label text with black outline for better visibility on any background color
        text_x, text_y = x1 + 5, y1 - 5
        # Draw black outline (thicker)
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 3)
        # Draw white text on top
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)

    return img


@router.get("/items/{item_id}/visualize")
async def visualize_evaluation_item(
    item_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Visualize evaluation item with bounding boxes drawn on the image.

    Shows both ground truth (if available) and predictions with consistent colors per class.
    """
    # Get evaluation item
    eval_item = await crud_evaluation.get_eval_item(db=db, eval_item_id=item_id)
    if not eval_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation item not found"
        )

    # Get image path (PHASE 2: storage_key now from image_2d only)
    from app.core.config import settings

    if not eval_item.image_2d_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation item has no associated image"
        )

    from app import crud
    image_2d = await crud.image_2d.get(db=db, id=eval_item.image_2d_id)
    if not image_2d:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )

    storage_key = image_2d.storage_key
    if not storage_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage key"
        )

    # Construct absolute path
    image_path = Path(settings.STORAGE_ROOT) / storage_key

    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found: {image_path}"
        )

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read image"
        )

    # Draw predictions
    predictions = eval_item.prediction if eval_item.prediction else []
    if isinstance(predictions, dict):
        predictions = [predictions]
    elif not isinstance(predictions, list):
        predictions = []

    # Log for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Visualizing item {item_id}: {len(predictions)} predictions")
    if predictions:
        logger.info(f"First prediction sample: {predictions[0]}")

    # Draw bounding boxes with error handling
    try:
        img_with_boxes = draw_bounding_boxes(img, predictions, None)  # Remove title bar
    except Exception as e:
        logger.error(f"Failed to draw bounding boxes: {e}")
        # Return original image if drawing fails
        img_with_boxes = img

    # Encode image to JPEG
    success, encoded_img = cv2.imencode('.jpg', img_with_boxes)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode image"
        )

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@router.get("/runs/{run_id}/sample-images")
async def get_sample_visualizations(
    run_id: UUID,
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    """
    Get sample visualization URLs for an evaluation run.

    Returns a list of evaluation item IDs that can be visualized.
    """
    # Get evaluation items
    items, _ = await crud_evaluation.get_eval_items(
        db=db,
        run_id=run_id,
        skip=0,
        limit=limit
    )

    return {
        "run_id": str(run_id),
        "sample_items": [
            {
                "item_id": str(item.id),
                "file_name": item.file_name,
                "visualization_url": f"/api/v1/evaluation/items/{item.id}/visualize"
            }
            for item in items
        ]
    }


@router.get("/runs/{run_id}/images-with-predictions")
async def get_images_with_predictions(
    run_id: UUID,
    dataset_type: Optional[EvalDatasetType] = Query(None, description="Filter by dataset type: 'base' or 'attack'"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get evaluation images with their prediction results for comparison.

    Returns all images with:
    - Original image storage key
    - Ground truth boxes
    - Predicted boxes
    - Metrics (IoU, confidence scores)

    PHASE 2 UPDATE: Accepts dataset_type filter parameter. No pagination.
    """
    # Get evaluation run to check if it exists
    eval_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=run_id)
    if not eval_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found"
        )

    # Determine base_dataset_id for classification
    base_dataset_id = None
    if eval_run.phase == "post_attack" and eval_run.attack_dataset_id:
        # Get attack dataset to find base_dataset_id
        from app import crud as app_crud
        attack_dataset = await app_crud.attack_dataset_2d.get(db=db, id=eval_run.attack_dataset_id)
        if attack_dataset:
            base_dataset_id = attack_dataset.base_dataset_id
    elif eval_run.phase == "pre_attack":
        base_dataset_id = eval_run.base_dataset_id

    # Get all evaluation items (no pagination)
    items = await crud_evaluation.get_all_eval_items(
        db=db,
        run_id=run_id,
        dataset_type=dataset_type
    )

    # Format response
    # PHASE 2: items already have image_2d loaded via joinedload
    images_data = []
    for item in items:
        # PHASE 2: Get file_name and storage_key from image_2d relationship
        file_name = None
        storage_key = None
        dataset_type_computed = None  # 'base' or 'attack'

        if item.image_2d:
            file_name = item.image_2d.file_name
            storage_key = item.image_2d.storage_key
            # Determine dataset type by comparing dataset_id
            if base_dataset_id and item.image_2d.dataset_id:
                if item.image_2d.dataset_id == base_dataset_id:
                    dataset_type_computed = "base"
                else:
                    dataset_type_computed = "attack"

        # Use item.dataset_type if available, otherwise use computed
        final_dataset_type = item.dataset_type.value if item.dataset_type else dataset_type_computed

        images_data.append({
            "item_id": str(item.id),
            "file_name": file_name,  # PHASE 2: From image_2d
            "storage_key": storage_key,  # PHASE 2: From image_2d
            "dataset_type": final_dataset_type,
            "ground_truth_boxes": item.ground_truth or [],
            "predicted_boxes": item.prediction or [],
            "metrics": item.metrics or {},
            "visualization_url": f"/api/v1/evaluation/items/{item.id}/visualize"
        })

    return {
        "run_id": str(run_id),
        "phase": eval_run.phase,
        "items": images_data,
        "total": len(images_data)
    }


@router.get("/runs/compare-images")
async def compare_evaluation_images(
    clean_run_id: UUID = Query(..., description="Clean/baseline evaluation run ID"),
    adv_run_id: UUID = Query(..., description="Adversarial evaluation run ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Compare images from clean and adversarial evaluation runs side-by-side.

    Matches images by filename and returns paired comparison data with:
    - Clean image with predictions
    - Adversarial image with predictions
    - Performance delta (mAP drop, IoU changes, etc.)
    """
    # Get both evaluation runs
    clean_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=clean_run_id)
    adv_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=adv_run_id)

    if not clean_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clean evaluation run {clean_run_id} not found"
        )

    if not adv_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adversarial evaluation run {adv_run_id} not found"
        )

    # Get all items from both runs to match by filename
    clean_items, _ = await crud_evaluation.get_eval_items(
        db=db, run_id=clean_run_id, skip=0, limit=10000
    )
    adv_items, _ = await crud_evaluation.get_eval_items(
        db=db, run_id=adv_run_id, skip=0, limit=10000
    )

    # Create filename lookup for adversarial items
    adv_items_map = {item.file_name: item for item in adv_items}

    # Match items and create comparison pairs
    from app import crud as app_crud
    comparison_pairs = []
    for clean_item in clean_items:
        adv_item = adv_items_map.get(clean_item.file_name)
        if adv_item:
            # Get storage_key from image_2d table
            clean_storage_key = None
            adv_storage_key = None

            try:
                if clean_item.image_2d_id:
                    clean_img = await app_crud.image_2d.get(db=db, id=clean_item.image_2d_id)
                    if clean_img:
                        clean_storage_key = clean_img.storage_key

                if adv_item.image_2d_id:
                    adv_img = await app_crud.image_2d.get(db=db, id=adv_item.image_2d_id)
                    if adv_img:
                        adv_storage_key = adv_img.storage_key
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to get images for comparison: {e}")

            # Calculate performance delta
            clean_metrics = clean_item.metrics or {}
            adv_metrics = adv_item.metrics or {}

            clean_map = clean_metrics.get("map", 0.0)
            adv_map = adv_metrics.get("map", 0.0)
            map_drop = clean_map - adv_map
            map_drop_percentage = (map_drop / clean_map * 100) if clean_map > 0 else 0.0

            comparison_pairs.append({
                "file_name": clean_item.file_name,
                "clean": {
                    "item_id": str(clean_item.id),
                    "storage_key": clean_storage_key,
                    "predicted_boxes": clean_item.prediction or [],
                    "ground_truth_boxes": clean_item.ground_truth or [],
                    "metrics": clean_metrics,
                    "visualization_url": f"/api/v1/evaluation/items/{clean_item.id}/visualize"
                },
                "adversarial": {
                    "item_id": str(adv_item.id),
                    "storage_key": adv_storage_key,
                    "predicted_boxes": adv_item.prediction or [],
                    "ground_truth_boxes": adv_item.ground_truth or [],
                    "metrics": adv_metrics,
                    "visualization_url": f"/api/v1/evaluation/items/{adv_item.id}/visualize"
                },
                "delta": {
                    "map_drop": map_drop,
                    "map_drop_percentage": map_drop_percentage,
                    "detection_count_change": len(adv_item.prediction or []) - len(clean_item.prediction or [])
                }
            })

    # Paginate results
    skip = (page - 1) * page_size
    paginated_pairs = comparison_pairs[skip:skip + page_size]

    return {
        "clean_run_id": str(clean_run_id),
        "adv_run_id": str(adv_run_id),
        "clean_run_name": clean_run.name,
        "adv_run_name": adv_run.name,
        "comparisons": paginated_pairs,
        "total": len(comparison_pairs),
        "page": page,
        "page_size": page_size
    }


# ========== Robustness Analysis Endpoints ==========

@router.post("/runs/compare-robustness")
async def compare_robustness(
    clean_run_id: UUID = Body(..., description="Evaluation run ID for clean/baseline dataset"),
    adv_run_id: UUID = Body(..., description="Evaluation run ID for adversarial dataset"),
    db: AsyncSession = Depends(get_db),
):
    """
    Calculate robustness metrics by comparing clean vs adversarial evaluation runs.

    Returns:
        Dictionary containing:
        - delta_map: Absolute drop in mAP
        - drop_percentage: Percentage drop in mAP
        - robustness_ratio: AP_adv / AP_clean (1.0 = fully robust)
        - delta metrics for various performance indicators
        - per-class robustness breakdown
    """
    from app.services.metrics_calculator import calculate_robustness_metrics

    # Get both evaluation runs
    clean_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=clean_run_id)
    adv_run = await crud_evaluation.get_eval_run(db=db, eval_run_id=adv_run_id)

    if not clean_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clean evaluation run {clean_run_id} not found"
        )

    if not adv_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adversarial evaluation run {adv_run_id} not found"
        )

    # Check if both runs are completed
    if clean_run.status != EvalStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Clean evaluation run is not completed (status: {clean_run.status})"
        )

    if adv_run.status != EvalStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Adversarial evaluation run is not completed (status: {adv_run.status})"
        )

    # Get metrics from both runs
    clean_metrics = clean_run.metrics_summary or {}
    adv_metrics = adv_run.metrics_summary or {}

    # Calculate overall robustness metrics
    overall_robustness = calculate_robustness_metrics(clean_metrics, adv_metrics)

    # Get per-class metrics for both runs
    clean_class_metrics = await crud_evaluation.get_eval_class_metrics(db=db, run_id=clean_run_id)
    adv_class_metrics = await crud_evaluation.get_eval_class_metrics(db=db, run_id=adv_run_id)

    # Build per-class robustness metrics
    per_class_robustness = {}
    clean_class_map = {cm.class_name: cm.metrics for cm in clean_class_metrics}
    adv_class_map = {cm.class_name: cm.metrics for cm in adv_class_metrics}

    all_classes = set(clean_class_map.keys()).union(set(adv_class_map.keys()))

    for class_name in all_classes:
        clean_class = clean_class_map.get(class_name, {})
        adv_class = adv_class_map.get(class_name, {})

        per_class_robustness[class_name] = calculate_robustness_metrics(clean_class, adv_class)

    # Get attack info if available
    attack_info = None
    if adv_run.attack_dataset_id:
        from app.crud import attack_dataset_2d as crud_attack
        attack_dataset = await crud_attack.get(db, id=adv_run.attack_dataset_id)

        if attack_dataset:
            attack_info = {
                "id": str(attack_dataset.id),
                "name": attack_dataset.name,
                "attack_type": attack_dataset.attack_type.value if hasattr(attack_dataset.attack_type, 'value') else attack_dataset.attack_type,
                "parameters": attack_dataset.parameters,
                "target_class": attack_dataset.target_class,
            }

    # Build visualization data
    visualization_data = {
        # 1. Overall comparison for bar charts
        "overall_comparison": [
            {"metric": "mAP", "clean": clean_metrics.get("map", 0.0), "adversarial": adv_metrics.get("map", 0.0)},
            {"metric": "mAP@50", "clean": clean_metrics.get("map50", 0.0), "adversarial": adv_metrics.get("map50", 0.0)},
            {"metric": "Precision", "clean": clean_metrics.get("precision", 0.0), "adversarial": adv_metrics.get("precision", 0.0)},
            {"metric": "Recall", "clean": clean_metrics.get("recall", 0.0), "adversarial": adv_metrics.get("recall", 0.0)},
        ],

        # 2. Performance drops for horizontal bar charts
        "drops": [
            {
                "metric": "mAP",
                "drop_percentage": overall_robustness.get("drop_percentage", 0.0),
                "delta": overall_robustness.get("delta_map", 0.0)
            },
            {
                "metric": "mAP@50",
                "drop_percentage": ((clean_metrics.get("map50", 0.0) - adv_metrics.get("map50", 0.0)) / clean_metrics.get("map50", 1.0) * 100) if clean_metrics.get("map50", 0.0) > 0 else 0.0,
                "delta": overall_robustness.get("delta_map50", 0.0)
            },
            {
                "metric": "Precision",
                "drop_percentage": ((clean_metrics.get("precision", 0.0) - adv_metrics.get("precision", 0.0)) / clean_metrics.get("precision", 1.0) * 100) if clean_metrics.get("precision", 0.0) > 0 else 0.0,
                "delta": overall_robustness.get("delta_precision", 0.0)
            },
            {
                "metric": "Recall",
                "drop_percentage": ((clean_metrics.get("recall", 0.0) - adv_metrics.get("recall", 0.0)) / clean_metrics.get("recall", 1.0) * 100) if clean_metrics.get("recall", 0.0) > 0 else 0.0,
                "delta": overall_robustness.get("delta_recall", 0.0)
            },
        ],

        # 3. Per-class comparison
        "per_class_comparison": [
            {
                "class_name": class_name,
                "clean_map": clean_class_map.get(class_name, {}).get("map", 0.0),
                "adv_map": adv_class_map.get(class_name, {}).get("map", 0.0),
                "drop_percentage": per_class_robustness[class_name].get("drop_percentage", 0.0),
                "robustness_ratio": per_class_robustness[class_name].get("robustness_ratio", 0.0),
            }
            for class_name in sorted(all_classes)
        ],

        # 4. Summary cards
        "summary": {
            "max_drop": {
                "metric": "mAP",
                "value": overall_robustness.get("drop_percentage", 0.0),
                "severity": "critical" if overall_robustness.get("drop_percentage", 0.0) > 70 else "high" if overall_robustness.get("drop_percentage", 0.0) > 50 else "medium"
            },
            "most_vulnerable_class": {
                "class": max(per_class_robustness.items(), key=lambda x: x[1].get("drop_percentage", 0.0))[0] if per_class_robustness else None,
                "drop": max(per_class_robustness.items(), key=lambda x: x[1].get("drop_percentage", 0.0))[1].get("drop_percentage", 0.0) if per_class_robustness else 0.0,
            } if per_class_robustness else None,
            "overall_robustness_ratio": overall_robustness.get("robustness_ratio", 0.0),
        }
    }

    return {
        "clean_run_id": str(clean_run_id),
        "adv_run_id": str(adv_run_id),
        "clean_run_name": clean_run.name,
        "adv_run_name": adv_run.name,
        "overall_robustness": overall_robustness,
        "per_class_robustness": per_class_robustness,
        "attack_info": attack_info,
        "visualization_data": visualization_data,
        "model_id": str(clean_run.model_id),
        "comparison_timestamp": datetime.now().isoformat(),
    }


# Include visualization router
from app.api.v1.endpoints import evaluation_viz
router.include_router(evaluation_viz.router, tags=["Evaluation Visualization"])

# Note: Evaluation Lists endpoints removed as they are not used by frontend
# The database tables (eval_lists, eval_list_items) and views (eval_run_pairs, eval_run_pairs_delta)
# remain in the database schema for potential future use.
