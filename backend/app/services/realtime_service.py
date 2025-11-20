"""
Real-time video capture and inference service.
"""
import cv2
import numpy as np
import asyncio
import time
from typing import Optional, Dict, Any, AsyncGenerator
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.services.model_inference_service import model_inference_service
from app.crud.model_repo import crud_model
from app.models.realtime import RTRunStatus

logger = logging.getLogger(__name__)


class RealtimeService:
    """Service for real-time webcam capture and inference."""

    def __init__(self):
        self.active_captures = {}  # run_id -> VideoCapture mapping
        self.session_stats = {}  # run_id -> stats mapping

    def get_available_cameras(self, max_check: int = 10) -> list[Dict[str, Any]]:
        """
        Get list of available camera devices.

        Args:
            max_check: Maximum number of camera indices to check

        Returns:
            List of available cameras with their info
        """
        available_cameras = []

        for index in range(max_check):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                available_cameras.append({
                    "device_id": str(index),
                    "device_path": f"/dev/video{index}",
                    "name": f"Camera {index}",
                    "width": width,
                    "height": height,
                    "fps": fps if fps > 0 else 30.0,
                    "is_available": True
                })
                cap.release()

        logger.info(f"Found {len(available_cameras)} available cameras")
        return available_cameras

    async def start_capture(
        self,
        db: AsyncSession,
        camera_device: str,
        run_id: UUID,
        model_id: Optional[UUID] = None,
        fps_target: float = 30.0,
        window_seconds: int = 5,
    ) -> Dict[str, Any]:
        """
        Start real-time video capture from webcam.
        """
        if model_id:
            # Verify model exists, but do not load it here.
            # Loading is handled by the endpoint layer.
            model = await crud_model.get_model(db, model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")

        # Clean up any existing captures for this camera device
        had_existing_captures = len(self.active_captures) > 0
        for existing_run_id, existing_cap in list(self.active_captures.items()):
            try:
                logger.info(f"Releasing existing capture for run {existing_run_id}")
                existing_cap.release()
                del self.active_captures[existing_run_id]
                logger.info(f"Released existing capture for run {existing_run_id}")
            except Exception as e:
                logger.warning(f"Error releasing existing capture: {e}")

        # Give the camera a moment to release
        if had_existing_captures:
            logger.info("Waiting for camera to be fully released...")
            time.sleep(1.5)

        # Open video capture
        try:
            try:
                device_index = int(camera_device)
                cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            except ValueError:
                cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera device: {camera_device}")

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, fps_target)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            logger.info(f"Camera opened successfully on device {camera_device}")
            self.active_captures[str(run_id)] = cap
            logger.info(f"Started capture for run {run_id} on device {camera_device}")

            return {
                "run_id": str(run_id),
                "camera_device": camera_device,
                "model_id": str(model_id),
                "fps_target": fps_target,
                "window_seconds": window_seconds,
                "status": "started"
            }

        except Exception as e:
            logger.error(f"Error starting capture: {e}")
            raise

    async def capture_and_infer(
        self,
        run_id: UUID,
        estimator_id: str,
        max_frames: Optional[int] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Capture frames and run inference in real-time.
        """
        run_id_str = str(run_id)
        cap = self.active_captures.get(run_id_str)

        if not cap:
            raise ValueError(f"No active capture for run {run_id}")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_count}")
                    break

                frame_count += 1
                capture_time = time.time()

                try:
                    inference_start = time.time()
                    result = await model_inference_service.run_inference(
                        version_id=estimator_id,
                        image=frame,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold
                    )
                    inference_time = (time.time() - inference_start) * 1000

                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0

                    detections = []
                    for det in result.detections:
                        detections.append({
                            "bbox": {"x1": det.bbox.x1, "y1": det.bbox.y1, "x2": det.bbox.x2, "y2": det.bbox.y2},
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence
                        })

                    yield {
                        "run_id": run_id_str,
                        "frame_number": frame_count,
                        "timestamp": capture_time,
                        "detections": detections,
                        "num_detections": len(detections),
                        "inference_time_ms": inference_time,
                        "current_fps": round(current_fps, 2),
                        "frame_shape": frame.shape
                    }

                except Exception as e:
                    logger.error(f"Inference error on frame {frame_count}: {e}")
                    yield {
                        "run_id": run_id_str,
                        "frame_number": frame_count,
                        "timestamp": capture_time,
                        "error": str(e),
                        "status": "failed"
                    }

                await asyncio.sleep(0.001)

        finally:
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            logger.info(
                f"Capture complete: {frame_count} frames in {total_time:.2f}s (avg FPS: {avg_fps:.2f})"
            )

    async def stop_capture(self, run_id: UUID) -> Dict[str, Any]:
        """
        Stop an active capture.
        """
        run_id_str = str(run_id)
        cap = self.active_captures.get(run_id_str)

        if not cap:
            raise ValueError(f"No active capture for run {run_id}")

        cap.release()
        del self.active_captures[run_id_str]
        logger.info(f"Stopped capture for run {run_id}")

        return {"run_id": run_id_str, "status": "stopped"}

    async def get_camera_info(self, camera_device: str) -> Dict[str, Any]:
        """
        Get camera device information.
        """
        try:
            cap = cv2.VideoCapture(camera_device)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera device: {camera_device}")

            info = {
                "device": camera_device,
                "backend": cap.getBackendName(),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
            }
            cap.release()
            return info
        except Exception as e:
            logger.error(f"Error getting camera info: {e}")
            raise

    async def list_available_cameras(self) -> list:
        """
        List available camera devices (cross-platform).
        """
        available = []
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    available.append({
                        "device": str(i),
                        "index": i,
                        "backend": cap.getBackendName(),
                        "name": f"Camera {i}",
                        "resolution": f"{width}x{height}",
                        "fps": fps
                    })
                    cap.release()
            except Exception:
                continue
        return available

# Global service instance
realtime_service = RealtimeService()
