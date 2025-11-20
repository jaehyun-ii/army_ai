"""
Real-time performance measurement API endpoints (aligned with database schema).
Only rt_capture_runs and rt_frames tables exist in database.
Camera and rt_inferences tables do not exist in schema.
"""
from typing import List, Optional
from uuid import UUID
import logging
import json

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, schemas
from app.database import get_db
from app.services.realtime_service import realtime_service

router = APIRouter()
logger = logging.getLogger(__name__)


# ========== RT Capture Run Endpoints ==========

@router.post("/runs", response_model=schemas.RTCaptureRunResponse, status_code=status.HTTP_201_CREATED)
async def create_capture_run(
    *,
    db: AsyncSession = Depends(get_db),
    run_in: schemas.RTCaptureRunCreate,
) -> schemas.RTCaptureRunResponse:
    """Create a new RT capture run."""
    run = await crud.rt_capture_run.create(db, obj_in=run_in)
    return run


@router.get("/runs/{run_id}", response_model=schemas.RTCaptureRunResponse)
async def get_capture_run(
    *,
    db: AsyncSession = Depends(get_db),
    run_id: UUID,
) -> schemas.RTCaptureRunResponse:
    """Get an RT capture run by ID."""
    run = await crud.rt_capture_run.get(db, id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Capture run not found")
    return run


@router.get("/runs", response_model=List[schemas.RTCaptureRunResponse])
async def list_capture_runs(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> List[schemas.RTCaptureRunResponse]:
    """List all RT capture runs."""
    runs = await crud.rt_capture_run.get_multi(db, skip=skip, limit=limit)
    return runs


@router.put("/runs/{run_id}", response_model=schemas.RTCaptureRunResponse)
async def update_capture_run(
    *,
    db: AsyncSession = Depends(get_db),
    run_id: UUID,
    run_in: schemas.RTCaptureRunUpdate,
) -> schemas.RTCaptureRunResponse:
    """Update an RT capture run."""
    run = await crud.rt_capture_run.get(db, id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Capture run not found")
    run = await crud.rt_capture_run.update(db, db_obj=run, obj_in=run_in)
    return run


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_capture_run(
    *,
    db: AsyncSession = Depends(get_db),
    run_id: UUID,
):
    """Soft delete an RT capture run."""
    success = await crud.rt_capture_run.soft_delete(db, id=run_id)
    if not success:
        raise HTTPException(status_code=404, detail="Capture run not found")


# ========== RT Frame Endpoints ==========

@router.post("/frames", response_model=schemas.RTFrameResponse, status_code=status.HTTP_201_CREATED)
async def create_frame(
    *,
    db: AsyncSession = Depends(get_db),
    frame_in: schemas.RTFrameCreate,
) -> schemas.RTFrameResponse:
    """Create a new RT frame."""
    frame = await crud.rt_frame.create(db, obj_in=frame_in)
    return frame


@router.get("/frames/{frame_id}", response_model=schemas.RTFrameResponse)
async def get_frame(
    *,
    db: AsyncSession = Depends(get_db),
    frame_id: UUID,
) -> schemas.RTFrameResponse:
    """Get an RT frame by ID."""
    frame = await crud.rt_frame.get(db, id=frame_id)
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")
    return frame


@router.get("/frames", response_model=List[schemas.RTFrameResponse])
async def list_frames(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> List[schemas.RTFrameResponse]:
    """List all RT frames."""
    frames = await crud.rt_frame.get_multi(db, skip=skip, limit=limit)
    return frames


@router.put("/frames/{frame_id}", response_model=schemas.RTFrameResponse)
async def update_frame(
    *,
    db: AsyncSession = Depends(get_db),
    frame_id: UUID,
    frame_in: schemas.RTFrameUpdate,
) -> schemas.RTFrameResponse:
    """Update an RT frame."""
    frame = await crud.rt_frame.get(db, id=frame_id)
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")
    frame = await crud.rt_frame.update(db, db_obj=frame, obj_in=frame_in)
    return frame


@router.delete("/frames/{frame_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_frame(
    *,
    db: AsyncSession = Depends(get_db),
    frame_id: UUID,
):
    """Soft delete an RT frame."""
    success = await crud.rt_frame.soft_delete(db, id=frame_id)
    if not success:
        raise HTTPException(status_code=404, detail="Frame not found")




# ========== Real-time Webcam Capture Endpoints ==========

@router.get("/webcam/list")
async def list_webcams():
    """
    List available webcam devices.

    Returns:
        List of available cameras with device paths
    """
    try:
        cameras = realtime_service.get_available_cameras()
        return {
            "cameras": cameras,
            "count": len(cameras)
        }
    except Exception as e:
        logger.error(f"Error listing webcams: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list webcam devices"
        )


@router.post("/webcam/sessions/start")
async def create_and_start_webcam_session(
    *,
    db: AsyncSession = Depends(get_db),
    run_name: str = Query(..., description="Session name"),
    device: str = Query("0", description="Camera device index"),
    model_id: UUID = Query(..., description="Model version ID for inference"),
    fps_target: float = Query(30.0, description="Target FPS"),
    window_seconds: int = Query(300, description="Capture duration in seconds"),
    conf_threshold: float = Query(0.25, description="Confidence threshold"),
    iou_threshold: float = Query(0.45, description="IOU threshold")
):
    """
    Create a new webcam session and start capture.

    This endpoint:
    1. Creates an RTCaptureRun (no camera table in DB schema)
    2. Starts the webcam capture

    Args:
        run_name: Name for the capture session
        device: Camera device index (0, 1, 2, ...)
        model_id: Model version to use for inference
        fps_target: Target frames per second
        window_seconds: Duration in seconds
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS

    Returns:
        Session information with run_id
    """
    # Create RT capture run (camera_id removed - not in DB schema)
    run_data = schemas.RTCaptureRunCreate(
        model_id=model_id,
        window_seconds=window_seconds,
        fps_target=fps_target,
        frames_expected=int(window_seconds * fps_target),
        notes=f"{run_name} (device: {device})"
    )
    run = await crud.rt_capture_run.create(db, obj_in=run_data)
    logger.info(f"Created capture run: {run.id}")

    # Start capture
    try:
        result = await realtime_service.start_capture(
            db=db,
            camera_device=device,
            model_id=model_id,
            run_id=run.id,
            fps_target=fps_target,
            window_seconds=window_seconds
        )

        return {
            "run_id": str(run.id),
            "model_id": str(model_id),
            "device": device,
            "status": "started",
            "message": f"Webcam session '{run_name}' started successfully",
            "stream_url": f"/api/v1/realtime/webcam/stream/{run.id}",
            "mjpeg_url": f"/api/v1/realtime/webcam/stream-mjpeg/{run.id}"
        }

    except Exception as e:
        logger.error(f"Error starting webcam session: {e}", exc_info=True)
        # Clean up the run record
        await crud.rt_capture_run.soft_delete(db, id=run.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start webcam session: {str(e)}"
        )


@router.get("/webcam/stream/{run_id}")
async def stream_webcam_inference(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    model_id: UUID = Query(..., description="Model ID for inference"),
    max_frames: Optional[int] = Query(None, description="Maximum frames to capture"),
    conf_threshold: float = Query(0.25, description="Confidence threshold"),
    iou_threshold: float = Query(0.45, description="IOU threshold")
):
    """
    Stream real-time inference results from webcam (SSE format).
    """
    estimator_id = f"stream__{run_id}__{model_id}"
    from app.services.estimator_loader_service import estimator_loader
    from app.services.model_inference_service import model_inference_service

    try:
        await estimator_loader.load_estimator_from_db(
            db=db,
            model_id=model_id,
            estimator_id=estimator_id
        )

        async def generate():
            try:
                async for result in realtime_service.capture_and_infer(
                    run_id=run_id,
                    estimator_id=estimator_id,
                    max_frames=max_frames,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                ):
                    yield f"data: {json.dumps(result)}\n\n"
            except Exception as e:
                error_data = {"error": str(e), "run_id": str(run_id), "status": "error"}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    finally:
        if model_inference_service.is_loaded(estimator_id):
            model_inference_service.unregister_estimator(estimator_id)
            logger.info(f"Unloaded temporary estimator '{estimator_id}' for stream")


@router.get("/webcam/stream-mjpeg/{run_id}")
async def stream_webcam_mjpeg(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    model_id: UUID = Query(..., description="Model ID"),
    conf_threshold: float = Query(0.25, description="Confidence threshold"),
    draw_boxes: bool = Query(True, description="Draw bounding boxes on frames")
):
    """
    Stream real-time video with detection overlay (MJPEG format).
    """
    estimator_id = f"mjpeg__{run_id}__{model_id}"
    from app.services.estimator_loader_service import estimator_loader
    from app.services.model_inference_service import model_inference_service
    import cv2
    import time
    from concurrent.futures import ThreadPoolExecutor

    try:
        await estimator_loader.load_estimator_from_db(
            db=db,
            model_id=model_id,
            estimator_id=estimator_id
        )

        executor = ThreadPoolExecutor(max_workers=1)

        def run_inference_sync(frame, conf):
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    model_inference_service.run_inference(
                        version_id=estimator_id,
                        image=frame,
                        conf_threshold=conf,
                        iou_threshold=0.45
                    )
                )
            finally:
                loop.close()

        def generate_frames():
            run_id_str = str(run_id)
            cap = realtime_service.active_captures.get(run_id_str)
            if not cap:
                logger.error(f"No active capture for run {run_id}")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                if draw_boxes:
                    try:
                        result = executor.submit(run_inference_sync, frame, conf_threshold).result()
                        h, w = frame.shape[:2]
                        for det in result.detections:
                            x1 = int(det.bbox.x1 * w)
                            y1 = int(det.bbox.y1 * h)
                            x2 = int(det.bbox.x2 * w)
                            y2 = int(det.bbox.y2 * h)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'{det.class_name} {det.confidence:.2f}'
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    except Exception as e:
                        logger.error(f"Detection error on frame: {e}")

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.03)

        return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    finally:
        if model_inference_service.is_loaded(estimator_id):
            model_inference_service.unregister_estimator(estimator_id)
            logger.info(f"Unloaded temporary estimator '{estimator_id}' for MJPEG stream")
