"""
Dataset3D service for managing 3D datasets.

3D 데이터셋 생성 및 관리 로직을 처리합니다.
"""
import mimetypes
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset_3d import Dataset3D, Image3D
from app.core.logging import get_logger
from app.core.config import settings
from .annotation_parser import AnnotationParser

logger = get_logger(__name__)


class Dataset3DService:
    """3D 데이터셋 서비스."""

    @staticmethod
    async def create_dataset(
        dataset_name: str,
        storage_path: str,
        metadata: Dict,
        db: AsyncSession,
        owner_id: Optional[UUID] = None
    ) -> Optional[Dataset3D]:
        """
        Dataset3D 레코드를 DB에 저장.

        Args:
            dataset_name: 데이터셋 이름
            storage_path: Storage 경로
            metadata: 메타데이터 (map, weather, object 등)
            db: DB 세션
            owner_id: 소유자 ID

        Returns:
            생성된 Dataset3D 객체 또는 None
        """
        try:
            dataset = Dataset3D(
                name=dataset_name,
                description=f"3D dataset generated from CARLA simulator",
                owner_id=owner_id,
                storage_path=storage_path,
                metadata_=metadata
            )

            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)

            logger.info(f"Saved dataset {dataset_name} to DB (ID: {dataset.id})")
            return dataset

        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            await db.rollback()
            return None

    @staticmethod
    async def save_images_and_annotations(
        dataset: Dataset3D,
        storage_path: str,
        db: AsyncSession,
        parse_annotations: bool = True
    ) -> Tuple[int, int]:
        """
        이미지 파일을 스캔하여 Image3D 레코드를 생성하고, 어노테이션을 파싱합니다.

        Args:
            dataset: Dataset3D 객체
            storage_path: Storage 경로 (예: /storage/3d/k2_tank)
            db: DB 세션
            parse_annotations: 어노테이션 파싱 여부

        Returns:
            (저장된 이미지 개수, 저장된 어노테이션 개수)
        """
        image_count = 0
        annotation_count = 0

        storage_key_base = storage_path.replace("/storage/", "") if storage_path.startswith("/storage/") else None
        if not storage_key_base:
            logger.warning(f"Invalid storage_path format: {storage_path}")
            return 0, 0

        try:
            base_dir = Path(settings.STORAGE_ROOT) / storage_key_base
            if not base_dir.exists():
                logger.warning(f"Image base directory does not exist: {base_dir}")
                return 0, 0

            # 이미지 파일 스캔
            from PIL import Image
            image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
            image_records = []

            for file_path in base_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in image_exts:
                    continue

                storage_key = str(file_path.relative_to(settings.STORAGE_ROOT))
                mime_type, _ = mimetypes.guess_type(file_path.name)

                # 이미지 파일 열어서 크기 가져오기
                width, height = None, None
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                except Exception as img_err:
                    logger.warning(f"Failed to read image size for {file_path.name}: {img_err}")

                image_records.append(Image3D(
                    dataset_id=dataset.id,
                    file_name=file_path.name,
                    storage_key=storage_key,
                    mime_type=mime_type or "image/png",
                    width=width,
                    height=height,
                ))

            if image_records:
                db.add_all(image_records)
                await db.commit()
                await db.refresh(dataset)
                image_count = len(image_records)
                logger.info(f"Saved {image_count} Image3D records for dataset {dataset.id}")

                # 어노테이션 파싱
                if parse_annotations:
                    # file_name -> image_3d_id 매핑 생성
                    image_id_map = {img.file_name: img.id for img in image_records}

                    # 어노테이션 디렉토리 찾기
                    annotations_dir = base_dir / "annotations"
                    if annotations_dir.exists():
                        annotation_count = await AnnotationParser.parse_and_save(
                            annotations_dir=annotations_dir,
                            dataset_id=dataset.id,
                            image_id_map=image_id_map,
                            db=db
                        )
                        logger.info(f"Saved {annotation_count} annotations for dataset {dataset.id}")
                    else:
                        logger.warning(
                            f"Annotations directory not found: {annotations_dir}. "
                            f"Annotations may be created later by CARLA. "
                            f"Use POST /api/v1/carla/datasets_3d/{dataset.id}/parse-annotations to parse them manually."
                        )
            else:
                logger.warning(f"No image files found under {base_dir} for dataset {dataset.id}")

        except Exception as e:
            logger.error(f"Failed to save images and annotations: {e}", exc_info=True)

        return image_count, annotation_count

    @staticmethod
    def convert_carla_path_to_storage(image_path: str) -> Optional[str]:
        """
        CARLA 백엔드의 경로를 메인 백엔드의 storage 경로로 변환.

        볼륨 마운트로 인해 CARLA의 /workspace/exports가 메인 백엔드의 storage/3d/에 마운트됨.
        따라서 다운로드 없이 경로만 변환하면 됨.

        Args:
            image_path: CARLA 경로 (예: /static/exports/k2_tank/image.jpg)

        Returns:
            메인 백엔드의 storage 경로 (예: /storage/3d/k2_tank/image.jpg)
        """
        try:
            logger.debug(f"Converting CARLA path: {image_path}")

            # /static/exports/{folder_name}/... -> /storage/3d/exports/{folder_name}/...
            if image_path.startswith("/static/exports/"):
                relative_path = image_path.replace("/static/exports/", "")
                storage_path = f"/storage/3d/exports/{relative_path}"
                logger.debug(f"Converted (static/exports): {image_path} -> {storage_path}")
                return storage_path

            # /exports/{folder_name}/... -> /storage/3d/exports/{folder_name}/...
            elif image_path.startswith("/exports/"):
                relative_path = image_path.replace("/exports/", "")
                storage_path = f"/storage/3d/exports/{relative_path}"
                logger.debug(f"Converted (exports): {image_path} -> {storage_path}")
                return storage_path

            # /workspace/exports/{folder_name}/... -> /storage/3d/exports/{folder_name}/...
            elif image_path.startswith("/workspace/exports/"):
                relative_path = image_path.replace("/workspace/exports/", "")
                storage_path = f"/storage/3d/exports/{relative_path}"
                logger.debug(f"Converted (workspace): {image_path} -> {storage_path}")
                return storage_path

            # /storage/3d/... 형식은 이미 변환된 경로
            elif image_path.startswith("/storage/3d/"):
                logger.debug(f"Already in storage format: {image_path}")
                return image_path

            # 알 수 없는 형식
            else:
                logger.warning(f"Unexpected image path format (cannot convert): {image_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to convert path '{image_path}': {e}")
            return None

    @staticmethod
    def infer_image_count(metadata: Optional[Dict]) -> int:
        """
        메타데이터에서 이미지 개수를 추론합니다.

        Args:
            metadata: 데이터셋 메타데이터

        Returns:
            추론된 이미지 개수
        """
        try:
            if not metadata or not isinstance(metadata, dict):
                return 0

            # 명시적 카운트 확인
            for key in ("total_images", "processed_images", "image_count"):
                value = metadata.get(key)
                if isinstance(value, int) and value >= 0:
                    return value

            # CARLA 생성 정보로 추론
            loc_total = metadata.get("loc_total")
            batch_total = metadata.get("batch_total")
            if isinstance(loc_total, int) and isinstance(batch_total, int):
                return max(loc_total * batch_total, 0)
        except Exception as exc:
            logger.warning(f"Failed to infer image count from metadata: {exc}")

        return 0
