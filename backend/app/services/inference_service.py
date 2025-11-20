"""
Model inference orchestration services.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.core.exceptions import NotFoundError, ValidationError
from app.services.estimator_loader_service import estimator_loader
from app.services.model_inference_service import model_inference_service

logger = logging.getLogger(__name__)


class InferenceService:
    """Coordinate model inference workflows."""

    async def load_model(self, db: AsyncSession, model_id: UUID) -> str:
        """Load a model into memory for inference."""
        estimator_id = f"inference__{model_id}"
        if not model_inference_service.is_loaded(estimator_id):
            logger.info(f"Loading model {model_id} as estimator '{estimator_id}'")
            await estimator_loader.load_estimator_from_db(
                db=db,
                model_id=model_id,
                estimator_id=estimator_id
            )
        return estimator_id

    async def unload_model(self, model_id: UUID):
        """Unload a model from memory."""
        estimator_id = f"inference__{model_id}"
        if model_inference_service.is_loaded(estimator_id):
            model_inference_service.unregister_estimator(estimator_id)
            logger.info(f"Unloaded temporary estimator '{estimator_id}'")

    async def run_inference(
        self,
        db: AsyncSession,
        model_id: UUID,
        image_ids: List[UUID],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Run inference for a list of image IDs and return detailed detections.
        This function assumes the model has already been loaded.
        """
        estimator_id = f"inference__{model_id}"
        if not model_inference_service.is_loaded(estimator_id):
            raise RuntimeError(f"Model {model_id} (estimator '{estimator_id}') is not loaded. Call load_model first.")

        logger.info(
            "Running inference on %s images with model %s (using estimator '%s')",
            len(image_ids),
            model_id,
            estimator_id,
        )

        results: List[Dict[str, Any]] = []
        for image_id in image_ids:
            image = await crud.image_2d.get(db, id=image_id)
            if not image:
                logger.warning("Image %s not found, skipping", image_id)
                continue

            # Convert storage_key to absolute path
            image_path = Path(image.storage_key)
            if not image_path.is_absolute():
                from app.core.config import settings
                image_path = Path(settings.STORAGE_ROOT) / image.storage_key

            if not image_path.exists():
                logger.warning("Image file not found: %s (tried: %s)", image.storage_key, image_path)
                continue

            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning("Could not read image: %s", image.storage_key)
                continue

            try:
                # Run inference using the new service
                inference_result = await model_inference_service.run_inference(
                    version_id=estimator_id,
                    image=img,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
            except Exception as exc:
                logger.error("Inference failed for image %s: %s", image_id, exc)
                results.append({
                    "image_id": str(image_id),
                    "detections": [],
                    "status": "error",
                    "error": str(exc),
                })
                continue

            # Convert results
            detections = []
            for det in inference_result.detections:
                detections.append({
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": {
                        "x1": det.bbox.x1,
                        "y1": det.bbox.y1,
                        "x2": det.bbox.x2,
                        "y2": det.bbox.y2
                    },
                })

            results.append({
                "image_id": str(image_id),
                "file_name": image.file_name,
                "detections": detections,
                "inference_time_ms": inference_result.inference_time_ms,
                "image_width": img.shape[1],
                "image_height": img.shape[0],
                "status": "success",
            })

        return results

    async def batch_inference(
        self,
        db: AsyncSession,
        model_id: UUID,
        dataset_id: UUID,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Run inference for all images in a dataset in batches.
        """
        images = await crud.image_2d.get_by_dataset(db, dataset_id=dataset_id)
        if not images:
            raise ValidationError(detail=f"No images in dataset {dataset_id}")

        image_ids = [img.id for img in images]
        logger.info(
            "Running batch inference on %s images from dataset %s",
            len(image_ids),
            dataset_id,
        )

        all_results: List[Dict[str, Any]] = []
        for i in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[i : i + batch_size]
            # This now uses the refactored run_inference, which handles its own estimator lifecycle
            batch_results = await self.run_inference(
                db=db,
                model_id=model_id,
                image_ids=batch_ids,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            all_results.extend(batch_results)

        successful = sum(1 for r in all_results if r["status"] == "success")
        failed = sum(1 for r in all_results if r["status"] == "error")
        total_detections = sum(len(r["detections"]) for r in all_results)
        avg_inference_time = (
            sum(r["inference_time_ms"] for r in all_results) / len(all_results)
            if all_results
            else 0
        )

        return {
            "dataset_id": str(dataset_id),
            "model_id": str(model_id),
            "total_images": len(image_ids),
            "successful": successful,
            "failed": failed,
            "total_detections": total_detections,
            "avg_inference_time_ms": avg_inference_time,
            "results": all_results,
        }


inference_service = InferenceService()
