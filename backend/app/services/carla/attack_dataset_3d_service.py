"""
AttackDataset3D service for managing 3D attack datasets.

3D 적대적 공격 데이터셋 생성 및 관리 로직을 처리합니다.
"""
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset_3d import AttackDataset3D, Dataset3D
from app.models.dataset_2d import AttackType
from app.core.logging import get_logger
from .dataset_3d_service import Dataset3DService

logger = get_logger(__name__)


class AttackDataset3DService:
    """3D 공격 데이터셋 서비스."""

    @staticmethod
    async def create_attack_dataset(
        payload: Dict,
        storage_path: str,
        processed_images: Optional[int],
        loc_total: Optional[int],
        batch_total: Optional[int],
        db: AsyncSession
    ) -> Optional[AttackDataset3D]:
        """
        AttackDataset3D 및 출력 Dataset3D를 생성합니다.

        Args:
            payload: 요청 페이로드
            storage_path: Storage 경로
            processed_images: 처리된 이미지 수
            loc_total: 총 위치 수
            batch_total: 총 배치 수
            db: DB 세션

        Returns:
            생성된 AttackDataset3D 객체 또는 None
        """
        try:
            # 타임스탬프 기반 이름 생성
            timestamp_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_dataset_name = payload.get("dataset_name", "unknown")
            attack_dataset_name = f"{base_dataset_name}_attack_{timestamp_suffix}"
            output_dataset_name = f"{base_dataset_name}_output_{timestamp_suffix}"

            # 1) 출력 Dataset3D 생성
            dataset_metadata = {
                "attack_method": payload.get("attack_method"),
                "patch_name": payload.get("patch_name"),
                "map_name": payload.get("map_name"),
                "weather_name": payload.get("weather_name"),
                "time_name": payload.get("time_name"),
                "object_name": payload.get("object_name"),
                "image_width": payload.get("image_width"),
                "image_height": payload.get("image_height"),
                "loc_total": loc_total,
                "batch_total": batch_total,
                "processed_images": processed_images,
                "storage_path": storage_path,
                "source": "carla_simulator",
            }

            output_dataset = await Dataset3DService.create_dataset(
                dataset_name=output_dataset_name,
                storage_path=storage_path,
                metadata=dataset_metadata,
                db=db,
            )

            if not output_dataset:
                logger.error("Failed to create output dataset")
                return None

            # 2) 이미지 및 어노테이션 저장
            image_count, annotation_count = await Dataset3DService.save_images_and_annotations(
                dataset=output_dataset,
                storage_path=storage_path,
                db=db,
                parse_annotations=True
            )
            logger.info(f"Saved {image_count} images and {annotation_count} annotations for attack dataset")

            # 3) AttackDataset3D 생성 (패치 이름을 문자열로 저장)
            attack_method = payload.get("attack_method", "ACTIVE")
            patch_name = payload.get("patch_name", "unknown_pattern")

            attack_dataset = AttackDataset3D(
                name=attack_dataset_name,
                description=f"Attack dataset with {attack_method} patch '{patch_name}' applied",
                attack_type=AttackType.PATCH,
                output_dataset_id=output_dataset.id,
                target_class=payload.get("object_name"),
                patch_id=None,  # Patch3D를 생성하지 않으므로 NULL
                parameters={
                    "attack_method": attack_method,
                    "patch_name": patch_name,  # 패치 이름 문자열로 저장
                    "map_name": payload.get("map_name"),
                    "weather_name": payload.get("weather_name"),
                    "time_name": payload.get("time_name"),
                    "object_name": payload.get("object_name"),
                    "image_width": payload.get("image_width"),
                    "image_height": payload.get("image_height"),
                    "loc_total": loc_total,
                    "batch_total": batch_total,
                    "storage_path": storage_path,
                    "processed_images": processed_images,
                    "output_dataset_name": output_dataset_name,
                    "output_dataset_id": str(output_dataset.id),
                    "source": "carla_simulator",
                },
            )

            db.add(attack_dataset)
            await db.commit()
            await db.refresh(attack_dataset)

            logger.info(f"Created attack dataset {attack_dataset_name} (ID: {attack_dataset.id})")
            return attack_dataset

        except Exception as e:
            logger.error(f"Failed to create attack dataset: {e}", exc_info=True)
            await db.rollback()
            return None
