"""
Global camera endpoints for simplified frontend integration.
Wraps realtime service with a single global session.
"""
from typing import Optional
from uuid import UUID, uuid4
import logging

from fastapi import APIRouter, HTTPException, status, Query, Depends
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, ConfigDict

from app.database import get_db
from app import crud, schemas
from app.services.realtime_service import realtime_service
from app.services.estimator_loader_service import estimator_loader
from app.services.model_inference_service import model_inference_service, DetectionResult

router = APIRouter()
logger = logging.getLogger(__name__)


# Global session state
class GlobalCameraState:
    """Manages single global camera session."""
    def __init__(self):
        self.run_id: Optional[UUID] = None
        self.estimator_id: Optional[str] = None  # Using estimator_id now
        self.model_id: Optional[UUID] = None # Keep original model id for reference
        self.device: str = "0"
        self.is_active: bool = False


global_camera = GlobalCameraState()
GLOBAL_ESTIMATOR_ID = "global_camera_estimator"


class StartDetectionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str  # Format: "{model_id}.pt"
    confidence_threshold: float = 0.25


class StartCameraRequest(BaseModel):
    device: Optional[str] = "0"


# ========== Camera Control Endpoints ========== 

@router.post("/start")
async def start_camera(
    request: StartCameraRequest = StartCameraRequest(),
    db: AsyncSession = Depends(get_db)
):
    """
    Start global camera session.
    """
    if global_camera.is_active and global_camera.run_id:
        logger.info("Camera already active")
        return {
            "status": "already_active",
            "message": "Camera is already running",
            "run_id": str(global_camera.run_id)
        }

    try:
        # Do not load any model automatically - wait for user to select via detection/start
        model_id = None
        model_name = "none"

        # Create RT capture run
        run_data = schemas.RTCaptureRunCreate(
            model_id=model_id,
            window_seconds=300,
            fps_target=30.0,
            frames_expected=9000,
            notes="Global camera session"
        )
        run = await crud.rt_capture_run.create(db, obj_in=run_data)
        logger.info(f"Created global camera run: {run.id}")

        # Start capture with selected device
        await realtime_service.start_capture(
            db=db,
            camera_device=request.device,
            run_id=run.id,
            model_id=model_id,
            fps_target=30.0,
            window_seconds=300
        )

        # Update global state
        global_camera.run_id = run.id
        global_camera.device = request.device
        global_camera.is_active = True

        logger.info(f"Global camera started: run_id={run.id}, model={model_name}, device={request.device}")

        return {
            "status": "started",
            "message": "Camera started successfully (no model loaded - select model via detection/start)",
            "run_id": str(run.id),
            "model_id": None,
            "estimator_id": None,
            "device": request.device
        }

    except Exception as e:
        logger.error(f"Failed to start camera: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start camera: {str(e)}"
        )


@router.post("/stop")
async def stop_camera():
    """
    Stop global camera session.
    """
    if not global_camera.is_active or not global_camera.run_id:
        return {
            "status": "not_running",
            "message": "Camera is not running"
        }

    try:
        # Stop capture
        await realtime_service.stop_capture(global_camera.run_id)

        # Unload estimator if it was loaded
        if global_camera.estimator_id and model_inference_service.is_loaded(global_camera.estimator_id):
            model_inference_service.unregister_estimator(global_camera.estimator_id)
            logger.info(f"Unloaded estimator '{global_camera.estimator_id}'")

        # Clear stats
        run_id_str = str(global_camera.run_id)
        if run_id_str in realtime_service.session_stats:
            del realtime_service.session_stats[run_id_str]

        # Reset global state
        global_camera.run_id = None
        global_camera.estimator_id = None
        global_camera.model_id = None
        global_camera.is_active = False

        logger.info("Global camera stopped")

        return {
            "status": "stopped",
            "message": "Camera stopped successfully"
        }

    except Exception as e:
        logger.error(f"Failed to stop camera: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop camera: {str(e)}"
        )


@router.get("/list")
async def list_cameras():
    """
    Get list of available camera devices.
    """
    try:
        cameras = realtime_service.get_available_cameras(max_check=10)
        logger.info(f"Found {len(cameras)} available cameras")
        return {
            "cameras": cameras,
            "count": len(cameras)
        }
    except Exception as e:
        logger.error(f"Failed to list cameras: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list cameras: {str(e)}"
        )


# ========== Stream Endpoints ========== 

@router.get("/stream")
async def stream_camera(
    draw_boxes: bool = Query(False, description="Whether to draw detection boxes on stream")
):
    """
    Get MJPEG video stream with optional detection overlay.
    """
    logger.info(f"Stream requested with draw_boxes={draw_boxes}, estimator_id={global_camera.estimator_id}")

    if not global_camera.is_active or not global_camera.run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera is not running. Start camera first."
        )

    if draw_boxes and not global_camera.estimator_id:
        logger.warning("Detection requested but no model loaded - will show raw camera feed")
        draw_boxes = False

    logger.info(f"Final draw_boxes value: {draw_boxes}")

    try:
        import cv2
        import time
        from concurrent.futures import ThreadPoolExecutor

        run_id_str = str(global_camera.run_id)
        
        executor = ThreadPoolExecutor(max_workers=1)

        def run_inference_sync(frame, estimator_id, conf, iou) -> DetectionResult:
            """Run inference synchronously using thread pool."""
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    model_inference_service.run_inference(
                        version_id=estimator_id,
                        image=frame,
                        conf_threshold=conf,
                        iou_threshold=iou
                    )
                )
                return result
            finally:
                loop.close()

        def generate_frames():
            """Generate MJPEG frames with detection overlay."""
            cap = realtime_service.active_captures.get(run_id_str)
            if not cap:
                logger.error(f"No active capture for run {run_id_str}")
                return

            # Initialize FPS tracking
            frame_count = 0
            start_time = time.time()
            fps_update_interval = 1.0  # Update FPS every second
            last_fps_update = start_time
            total_detections = 0
            inference_times = []

            while global_camera.is_active:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue

                    frame_count += 1
                    current_time = time.time()

                    # Update FPS statistics periodically
                    if current_time - last_fps_update >= fps_update_interval:
                        elapsed = current_time - start_time
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

                        # Update session stats for SSE stream
                        realtime_service.session_stats[run_id_str] = {
                            "fps": round(current_fps, 1),
                            "total_frames": frame_count,
                            "total_detections": total_detections,
                            "avg_inference_time": round(avg_inference_time, 2),
                            "timestamp": current_time
                        }
                        last_fps_update = current_time

                    if draw_boxes and global_camera.estimator_id:
                        try:
                            inference_start = time.time()
                            result = executor.submit(
                                run_inference_sync,
                                frame,
                                global_camera.estimator_id,
                                0.25,
                                0.45
                            ).result()
                            inference_time = (time.time() - inference_start) * 1000  # Convert to ms

                            h, w = frame.shape[:2]
                            num_detections = len(result.detections) if result.detections else 0

                            # Track statistics
                            total_detections += num_detections
                            inference_times.append(inference_time)
                            # Keep only last 30 inference times for average
                            if len(inference_times) > 30:
                                inference_times.pop(0)

                            if num_detections > 0:
                                logger.info(f"Drawing {num_detections} detections on frame (h={h}, w={w})")

                            # Generate distinct colors for each class
                            def get_class_color(class_id: int):
                                """Generate a distinct color for each class using HSV color space."""
                                import numpy as np
                                # Use golden ratio for better color distribution
                                golden_ratio = 0.618033988749895
                                hue = (class_id * golden_ratio) % 1.0
                                # Convert HSV to BGR for OpenCV
                                hsv = np.array([[[int(hue * 180), 200, 255]]], dtype=np.uint8)
                                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                                return tuple(map(int, bgr))

                            for det in result.detections:
                                # Bounding boxes are normalized (0-1); convert to pixel coordinates for overlay
                                x1 = int(det.bbox.x1 * w)
                                y1 = int(det.bbox.y1 * h)
                                x2 = int(det.bbox.x2 * w)
                                y2 = int(det.bbox.y2 * h)

                                logger.info(f"Drawing box: ({x1}, {y1}) to ({x2}, {y2}) - {det.class_name} {det.confidence:.2f}")

                                # Get color for this class
                                color = get_class_color(det.class_id)

                                # Draw bounding box with class-specific color
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                                # Prepare label text
                                label = f'{det.class_name} {det.confidence:.2f}'

                                # Get text size for background box
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                font_thickness = 2
                                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                                # Draw filled rectangle background for label
                                label_y1 = max(y1 - text_height - baseline - 8, 0)
                                label_y2 = y1
                                cv2.rectangle(frame, (x1, label_y1), (x1 + text_width + 8, label_y2), color, -1)

                                # Draw white text on colored background
                                cv2.putText(frame, label, (x1 + 4, y1 - baseline - 4), font, font_scale, (255, 255, 255), font_thickness)

                        except Exception as e:
                            logger.error(f"Detection error on frame: {e}", exc_info=True)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.03)

                except Exception as e:
                    logger.error(f"Error generating frame: {e}")
                    break
            
            executor.shutdown(wait=False)

        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    except Exception as e:
        logger.error(f"Failed to stream camera: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream camera: {str(e)}"
        )


@router.get("/status")
async def get_camera_status():
    """Get current camera status and statistics."""
    if not global_camera.is_active:
        return {
            "is_active": False,
            "message": "Camera is not running"
        }

    run_id_str = str(global_camera.run_id)
    stats = realtime_service.session_stats.get(run_id_str, {})

    return {
        "is_active": True,
        "run_id": run_id_str,
        "model_id": str(global_camera.model_id) if global_camera.model_id else None,
        "estimator_id": global_camera.estimator_id,
        "device": global_camera.device,
        "stats": stats
    }


# ========== Detection Control Endpoints ==========

@router.post("/detection/start")
async def start_detection(
    request: StartDetectionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start real-time detection with selected model.
    Camera must be started first.
    """
    if not global_camera.is_active or not global_camera.run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera is not running. Start camera first."
        )

    if global_camera.estimator_id and model_inference_service.is_loaded(global_camera.estimator_id):
        logger.info(f"Detection already running with estimator: {global_camera.estimator_id}")
        return {
            "status": "already_active",
            "message": "Detection is already running",
            "estimator_id": global_camera.estimator_id
        }

    try:
        # Extract model_id from model_path (format: "{model_id}.pt")
        model_id_str = request.model_path.replace('.pt', '')
        try:
            model_id = UUID(model_id_str)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model_path format. Expected UUID.pt, got: {request.model_path}"
            )

        # Load the model using estimator service
        estimator_id = f"camera_detection_{uuid4().hex[:8]}"

        logger.info(f"Loading model {model_id} for detection")

        # Load estimator from database
        await estimator_loader.load_estimator_from_db(
            db=db,
            model_id=model_id,
            estimator_id=estimator_id
        )

        # Update global camera state
        global_camera.estimator_id = estimator_id
        global_camera.model_id = model_id

        logger.info(f"Detection started with estimator: {estimator_id}")

        return {
            "status": "started",
            "message": "Detection started successfully",
            "estimator_id": estimator_id,
            "model_id": str(model_id),
            "confidence_threshold": request.confidence_threshold
        }

    except Exception as e:
        logger.error(f"Failed to start detection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start detection: {str(e)}"
        )


@router.post("/detection/stop")
async def stop_detection():
    """
    Stop real-time detection but keep camera running.
    """
    if not global_camera.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera is not running"
        )

    if not global_camera.estimator_id:
        return {
            "status": "not_running",
            "message": "Detection is not running"
        }

    try:
        # Unload the estimator
        if model_inference_service.is_loaded(global_camera.estimator_id):
            model_inference_service.unregister_estimator(global_camera.estimator_id)
            logger.info(f"Unloaded estimator: {global_camera.estimator_id}")

        # Clear estimator state
        old_estimator = global_camera.estimator_id
        global_camera.estimator_id = None

        logger.info(f"Detection stopped: {old_estimator}")

        return {
            "status": "stopped",
            "message": "Detection stopped successfully",
            "estimator_id": old_estimator
        }

    except Exception as e:
        logger.error(f"Failed to stop detection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop detection: {str(e)}"
        )


# ========== Statistics Streaming Endpoint ==========

@router.get("/stats/stream")
async def stream_stats():
    """
    SSE stream for real-time performance statistics.
    """
    if not global_camera.is_active or not global_camera.run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera is not running"
        )

    async def generate_stats():
        """Generate SSE events with performance metrics."""
        import asyncio
        import json

        run_id_str = str(global_camera.run_id)

        try:
            while global_camera.is_active:
                # Get current stats from realtime_service
                stats = realtime_service.session_stats.get(run_id_str, {})

                # Add detection status
                stats_data = {
                    "fps": stats.get("fps", 0),
                    "total_frames": stats.get("total_frames", 0),
                    "detections": stats.get("total_detections", 0),
                    "processing_time": stats.get("avg_inference_time", 0),
                    "is_detecting": global_camera.estimator_id is not None,
                    "timestamp": stats.get("timestamp", "")
                }

                # Format as SSE
                yield f"data: {json.dumps(stats_data)}\n\n"

                await asyncio.sleep(1.0)  # Update every second

        except Exception as e:
            logger.error(f"Error streaming stats: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stats(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ========== Capture Endpoints ==========

@router.post("/capture/start")
async def start_capture_session(db: AsyncSession = Depends(get_db)):
    """
    Start a timed capture session for dataset collection.
    Captures frames for 5 seconds and saves them to the database.
    """
    if not global_camera.is_active or not global_camera.run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera is not running. Start camera first."
        )

    if not global_camera.estimator_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Detection is not running. Start detection first."
        )

    try:
        import cv2
        import asyncio
        from datetime import datetime, timezone
        import io
        from pathlib import Path
        from app.core.config import settings

        run_id_str = str(global_camera.run_id)
        cap = realtime_service.active_captures.get(run_id_str)

        # Create storage directory for this capture run
        storage_dir = settings.get_storage_path("rt_captures", run_id_str)
        storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created storage directory: {storage_dir}")

        if not cap:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No active camera capture found"
            )

        logger.info(f"Starting 5-second frame capture for run {global_camera.run_id}")

        # Capture configuration
        capture_duration = 5.0  # seconds
        target_fps = 5  # capture 5 frames per second
        frame_interval = 1.0 / target_fps

        captured_frames = []
        start_time = asyncio.get_event_loop().time()

        # Get the max seq_no for this run_id to avoid UNIQUE constraint violation
        from sqlalchemy import select, func
        from app.models.realtime import RTFrame

        max_seq_result = await db.execute(
            select(func.max(RTFrame.seq_no))
            .where(RTFrame.run_id == global_camera.run_id)
            .where(RTFrame.deleted_at.is_(None))
        )
        max_seq = max_seq_result.scalar()
        seq_no = (max_seq or 0) + 1  # Start from next available seq_no

        logger.info(f"Starting capture from seq_no {seq_no} (max existing: {max_seq})")

        # Capture frames for specified duration
        while (asyncio.get_event_loop().time() - start_time) < capture_duration:
            ret, frame = cap.read()
            if ret and frame is not None:
                # Run detection on frame
                detections_list = []
                try:
                    # Verify estimator is loaded
                    if not global_camera.estimator_id:
                        logger.error(f"No estimator loaded for detection on frame {seq_no}")
                    elif not model_inference_service.is_loaded(global_camera.estimator_id):
                        logger.error(f"Estimator {global_camera.estimator_id} not loaded in inference service")
                    else:
                        # Run inference
                        logger.debug(f"Running detection on frame {seq_no} with estimator {global_camera.estimator_id}")
                        detection_result = await model_inference_service.run_inference(
                            version_id=global_camera.estimator_id,
                            image=frame,
                            conf_threshold=0.25,
                            iou_threshold=0.45
                        )

                        # Store detection info for annotations
                        for det in detection_result.detections:
                            detections_list.append({
                                "class_id": det.class_id,
                                "class_name": det.class_name,
                                "confidence": det.confidence,
                                "bbox": {
                                    "x1": det.bbox.x1,
                                    "y1": det.bbox.y1,
                                    "x2": det.bbox.x2,
                                    "y2": det.bbox.y2
                                }
                            })

                        logger.info(f"Frame {seq_no}: detected {len(detections_list)} objects")

                except Exception as e:
                    logger.error(f"Failed to run detection on frame {seq_no}: {e}", exc_info=True)

                # Save frame to database
                # Note: Use 'metadata_' as the key because RTFrameCreate has validation_alias="metadata_"
                frame_data = {
                    "run_id": global_camera.run_id,
                    "seq_no": seq_no,
                    "captured_at": datetime.now(timezone.utc),
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "mime_type": "image/jpeg",
                    "metadata_": {
                        "estimator_id": global_camera.estimator_id,
                        "model_id": str(global_camera.model_id) if global_camera.model_id else None,
                        "detections": detections_list,
                        "num_detections": len(detections_list)
                    }
                }

                # Encode frame to JPEG for storage
                ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ret_encode:
                    try:
                        # Save image file to disk
                        file_name = f"frame_{seq_no:05d}.jpg"
                        file_path = storage_dir / file_name

                        with open(file_path, 'wb') as f:
                            f.write(buffer.tobytes())

                        # Update frame_data with storage_key (relative path from STORAGE_ROOT)
                        relative_path = f"rt_captures/{run_id_str}/{file_name}"
                        frame_data["storage_key"] = relative_path

                        # Save frame metadata to database
                        frame_record = await crud.rt_frame.create(db, obj_in=schemas.RTFrameCreate(**frame_data))
                        captured_frames.append(frame_record.id)

                        # Save detections as annotations
                        for det in detections_list:
                            try:
                                # Stored detections are normalized xyxy; convert to normalized center-width-height
                                x1 = det["bbox"]["x1"]
                                y1 = det["bbox"]["y1"]
                                x2 = det["bbox"]["x2"]
                                y2 = det["bbox"]["y2"]

                                bbox_width = max(0.0, x2 - x1)
                                bbox_height = max(0.0, y2 - y1)
                                bbox_x = x1 + bbox_width / 2
                                bbox_y = y1 + bbox_height / 2

                                annotation_data = schemas.AnnotationCreate(
                                    rt_frame_id=frame_record.id,
                                    annotation_type="bbox",
                                    class_name=det["class_name"],
                                    class_index=det["class_id"],
                                    bbox_x=bbox_x,
                                    bbox_y=bbox_y,
                                    bbox_width=bbox_width,
                                    bbox_height=bbox_height,
                                    confidence=det["confidence"],
                                    metadata_={"source": "realtime_capture"}
                                )
                                await crud.annotation.create(db, obj_in=annotation_data)
                            except Exception as e:
                                logger.error(f"Failed to save annotation for frame {seq_no}: {e}")

                        logger.debug(f"Saved frame {seq_no} to {file_path} with {len(detections_list)} annotations")
                        seq_no += 1
                    except Exception as e:
                        logger.error(f"Failed to save frame {seq_no}: {e}")
                        # Continue capturing even if one frame fails
                else:
                    logger.warning(f"Failed to encode frame {seq_no}")

                await asyncio.sleep(frame_interval)

        logger.info(f"Captured {len(captured_frames)} new frames for run {global_camera.run_id}")

        # Get total frame count for this run (including previous captures)
        total_frames_result = await db.execute(
            select(func.count(RTFrame.id))
            .where(RTFrame.run_id == global_camera.run_id)
            .where(RTFrame.deleted_at.is_(None))
        )
        total_frames = total_frames_result.scalar() or 0

        # Update capture run with actual frame count
        from app.models.realtime import RTRunStatus
        from sqlalchemy import update as sql_update

        # IMPORTANT: Update frames_expected to match total stored frames
        # Note: We don't mark as "completed" because this is an ongoing capture session
        await db.execute(
            sql_update(crud.rt_capture_run.model)
            .where(crud.rt_capture_run.model.id == global_camera.run_id)
            .values(frames_expected=total_frames)
        )

        await db.commit()
        logger.info(f"Updated capture run {global_camera.run_id}: total_frames={total_frames}, new_frames={len(captured_frames)}")

        return {
            "status": "completed",
            "message": f"Successfully captured {len(captured_frames)} new frames (total: {total_frames})",
            "run_id": str(global_camera.run_id),
            "frames_captured": len(captured_frames),
            "total_frames": total_frames,
            "duration_seconds": capture_duration
        }

    except Exception as e:
        logger.error(f"Failed to capture frames: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start capture: {str(e)}"
        )


@router.get("/captures")
async def list_captures(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    List all capture runs with their frame counts.
    """
    try:
        from sqlalchemy import select, func
        from app.models.realtime import RTCaptureRun, RTFrame

        # Query capture runs with frame counts
        # Only show runs that have actual captured frames (exclude streaming-only sessions)
        query = (
            select(
                RTCaptureRun,
                func.count(RTFrame.id).label('frame_count')
            )
            .outerjoin(RTFrame, RTCaptureRun.id == RTFrame.run_id)
            .where(RTFrame.deleted_at.is_(None))
            .group_by(RTCaptureRun.id)
            .having(func.count(RTFrame.id) > 0)  # Only show runs with frames
            .order_by(RTCaptureRun.created_at.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(query)
        rows = result.all()

        captures = []
        for run, frame_count in rows:
            # Calculate detection statistics from annotations table
            from app.models.annotation import Annotation

            # Get all annotations for this run's frames
            annotations_query = (
                select(Annotation)
                .join(RTFrame, Annotation.rt_frame_id == RTFrame.id)
                .where(RTFrame.run_id == run.id)
            )
            annotations_result = await db.execute(annotations_query)
            annotations = annotations_result.scalars().all()

            total_detections = len(annotations)
            total_confidence = sum(float(ann.confidence) for ann in annotations if ann.confidence)
            confidence_count = sum(1 for ann in annotations if ann.confidence)

            # Aggregate confidence per class
            class_stats: dict[str, dict[str, float | int | str | None]] = {}
            for ann in annotations:
                class_key = ann.class_name or str(ann.class_index)
                stats = class_stats.setdefault(
                    class_key,
                    {
                        "class_id": ann.class_index,
                        "class_name": ann.class_name,
                        "count": 0,
                        "confidence_sum": 0.0,
                        "confidence_count": 0,
                    },
                )
                stats["count"] = int(stats["count"]) + 1  # type: ignore[arg-type]
                if ann.confidence is not None:
                    stats["confidence_sum"] = float(stats["confidence_sum"]) + float(ann.confidence)  # type: ignore[arg-type]
                    stats["confidence_count"] = int(stats["confidence_count"]) + 1  # type: ignore[arg-type]

            avg_confidence_by_class = [
                {
                    "class_id": stats["class_id"],
                    "class_name": stats["class_name"],
                    "count": stats["count"],
                    "avg_confidence": (
                        float(stats["confidence_sum"]) / int(stats["confidence_count"])  # type: ignore[arg-type]
                        if stats["confidence_count"]
                        else None
                    ),
                }
                for stats in class_stats.values()
            ]
            avg_confidence_by_class.sort(key=lambda item: item["count"] or 0, reverse=True)  # type: ignore[arg-type]

            avg_confidence = (total_confidence / confidence_count) if confidence_count > 0 else None

            captures.append({
                "id": str(run.id),
                "model_id": str(run.model_id) if run.model_id else None,
                "status": run.status,
                "created_at": run.started_at.isoformat() if run.started_at else None,  # Frontend expects created_at
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "ended_at": run.ended_at.isoformat() if run.ended_at else None,
                "window_seconds": run.window_seconds,
                "frames_expected": run.frames_expected,
                "frames_captured": frame_count,
                "total_images": frame_count,  # Frontend expects total_images
                "total_detections": total_detections,
                "avg_confidence_overall": avg_confidence,
                "avg_confidence_by_class": avg_confidence_by_class,
                "fps_target": float(run.fps_target) if run.fps_target else None,
                "notes": run.notes
            })

        return {
            "captures": captures,
            "total": len(captures),
            "skip": skip,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Failed to list captures: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list captures: {str(e)}"
        )


@router.get("/captures/{capture_id}")
async def get_capture_details(
    capture_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific capture session including all frames.
    """
    try:
        from sqlalchemy import select, func
        from app.models.realtime import RTCaptureRun, RTFrame

        # Get capture run
        run = await crud.rt_capture_run.get(db, id=capture_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Capture run {capture_id} not found"
            )

        # Get all frames for this run
        query = (
            select(RTFrame)
            .where(RTFrame.run_id == capture_id)
            .order_by(RTFrame.seq_no)
        )
        result = await db.execute(query)
        frames = result.scalars().all()

        # Format frame data and calculate detection stats
        from app.models.annotation import Annotation

        images = []
        total_detections = 0
        total_confidence = 0.0
        confidence_count = 0
        class_stats: dict[str, dict[str, float | int | str | None]] = {}

        for frame in frames:
            # Get annotations from database
            annotations_query = select(Annotation).where(Annotation.rt_frame_id == frame.id)
            annotations_result = await db.execute(annotations_query)
            annotations = annotations_result.scalars().all()

            # Format detections from annotations table
            detections = []
            for ann in annotations:
                detections.append({
                    "class_id": ann.class_index,
                    "class_name": ann.class_name,
                    "confidence": float(ann.confidence) if ann.confidence else 0.0,
                    "bbox": {
                        "x": float(ann.bbox_x) if ann.bbox_x else 0,
                        "y": float(ann.bbox_y) if ann.bbox_y else 0,
                        "width": float(ann.bbox_width) if ann.bbox_width else 0,
                        "height": float(ann.bbox_height) if ann.bbox_height else 0
                    }
                })

                total_detections += 1
                if ann.confidence:
                    total_confidence += float(ann.confidence)
                    confidence_count += 1

                class_key = ann.class_name or str(ann.class_index)
                stats = class_stats.setdefault(
                    class_key,
                    {
                        "class_id": ann.class_index,
                        "class_name": ann.class_name,
                        "count": 0,
                        "confidence_sum": 0.0,
                        "confidence_count": 0,
                    },
                )
                stats["count"] = int(stats["count"]) + 1  # type: ignore[arg-type]
                if ann.confidence is not None:
                    stats["confidence_sum"] = float(stats["confidence_sum"]) + float(ann.confidence)  # type: ignore[arg-type]
                    stats["confidence_count"] = int(stats["confidence_count"]) + 1  # type: ignore[arg-type]

            images.append({
                "id": str(frame.id),
                "frame_number": frame.seq_no,
                "captured_at": frame.captured_at.isoformat() if frame.captured_at else None,
                "width": frame.width,
                "height": frame.height,
                "metadata": frame.metadata_,
                "detections": detections
            })

        avg_confidence = (total_confidence / confidence_count) if confidence_count > 0 else None
        avg_confidence_by_class = [
            {
                "class_id": stats["class_id"],
                "class_name": stats["class_name"],
                "count": stats["count"],
                "avg_confidence": (
                    float(stats["confidence_sum"]) / int(stats["confidence_count"])  # type: ignore[arg-type]
                    if stats["confidence_count"]
                    else None
                ),
            }
            for stats in class_stats.values()
        ]
        avg_confidence_by_class.sort(key=lambda item: item["count"] or 0, reverse=True)  # type: ignore[arg-type]

        # Build response
        capture_data = {
            "id": str(run.id),
            "model_id": str(run.model_id) if run.model_id else None,
            "status": run.status,
            "created_at": run.started_at.isoformat() if run.started_at else None,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "ended_at": run.ended_at.isoformat() if run.ended_at else None,
            "window_seconds": run.window_seconds,
            "frames_expected": run.frames_expected,
            "total_images": len(frames),
            "total_detections": total_detections,
            "avg_confidence_overall": avg_confidence,
            "avg_confidence_by_class": avg_confidence_by_class,
            "fps_target": float(run.fps_target) if run.fps_target else None,
            "notes": run.notes
        }

        return {
            "capture": capture_data,
            "images": images
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get capture details: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get capture details: {str(e)}"
        )


@router.delete("/captures/{capture_id}")
async def delete_capture(
    capture_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a capture session and all associated frames.
    """
    try:
        # Check if capture exists
        run = await crud.rt_capture_run.get(db, id=capture_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Capture run {capture_id} not found"
            )

        # Delete the run (frames will be cascade deleted due to FK constraint)
        deleted = await crud.rt_capture_run.hard_delete(db, id=capture_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete capture run {capture_id}"
            )

        logger.info(f"Deleted capture run {capture_id}")

        return {
            "status": "deleted",
            "message": f"Capture session {capture_id} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete capture: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete capture: {str(e)}"
        )


@router.get("/captures/{capture_id}/image/{frame_number}")
async def get_frame_image(
    capture_id: UUID,
    frame_number: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific frame image from storage.
    """
    try:
        from sqlalchemy import select
        from app.models.realtime import RTFrame
        from app.core.config import settings
        from pathlib import Path

        # Get the frame
        query = (
            select(RTFrame)
            .where(RTFrame.run_id == capture_id)
            .where(RTFrame.seq_no == frame_number)
        )
        result = await db.execute(query)
        frame = result.scalar_one_or_none()

        if not frame:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Frame {frame_number} not found in capture {capture_id}"
            )

        # Check if image file exists
        if not frame.storage_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Frame {frame_number} has no stored image (storage_key is null)"
            )

        # Get full file path
        file_path = settings.get_storage_path() / frame.storage_key

        if not file_path.exists():
            logger.error(f"Image file not found: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image file not found at {frame.storage_key}"
            )

        # Read and return the image
        with open(file_path, 'rb') as f:
            image_data = f.read()

        # Determine content type from mime_type or default to jpeg
        content_type = frame.mime_type or "image/jpeg"

        return Response(
            content=image_data,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "Content-Disposition": f"inline; filename=frame_{frame_number}.jpg"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get frame image: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get frame image: {str(e)}"
        )
