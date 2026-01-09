"""
CARLA annotation parsing service.

CARLA에서 생성된 COCO 형식 어노테이션 또는 YOLO 형식 어노테이션을 DB에 저장합니다.
"""
import csv
import json
from pathlib import Path
from typing import Dict, Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.annotation import Annotation
from app.models.dataset_3d import Dataset3D
from app.core.logging import get_logger

logger = get_logger(__name__)


class AnnotationParser:
    """CARLA 어노테이션 파싱 서비스."""

    @staticmethod
    async def parse_yolo_txt_and_save(
        txt_file: Path,
        dataset_id: UUID,
        image_id_map: Dict[str, UUID],
        db: AsyncSession,
        class_names: Optional[List[str]] = None
    ) -> int:
        """
        YOLO 형식 txt 파일을 파싱하여 DB에 저장.

        파일 형식: filename,class_id,x_center,y_center,width,height
        좌표는 정규화된 값 (0~1 범위)

        Args:
            txt_file: YOLO 형식 txt 파일 경로
            dataset_id: 데이터셋 ID
            image_id_map: 파일명 -> Image3D ID 매핑
            db: DB 세션
            class_names: 클래스 이름 리스트 (없으면 class_0, class_1, ... 형식으로 생성)

        Returns:
            저장된 어노테이션 개수
        """
        try:
            logger.info(f"Parsing YOLO format annotations from {txt_file}")

            annotation_records = []
            class_ids_found = set()

            with open(txt_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('filename')
                    class_id = int(row.get('class_id', 0))
                    x_center = float(row.get('x_center', 0))
                    y_center = float(row.get('y_center', 0))
                    width = float(row.get('width', 0))
                    height = float(row.get('height', 0))

                    class_ids_found.add(class_id)

                    # Image3D ID 찾기
                    image_3d_id = image_id_map.get(filename)
                    if not image_3d_id:
                        logger.warning(f"Image3D not found for file {filename}")
                        continue

                    # 클래스 이름 결정
                    if class_names and 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    # YOLO 정규화 좌표를 bbox 파라미터로 저장
                    # x_center, y_center, width, height (모두 0~1 범위)
                    annotation_records.append(Annotation(
                        image_3d_id=image_3d_id,
                        annotation_type="bbox_normalized",
                        class_name=class_name,
                        class_index=class_id,
                        bbox_x=x_center,
                        bbox_y=y_center,
                        bbox_width=width,
                        bbox_height=height,
                        confidence=1.0,  # Ground truth annotation
                        is_crowd=False,
                        area=width * height,  # normalized area
                    ))

            # Dataset3D 메타데이터에 classes 저장
            try:
                dataset_row = await db.execute(
                    select(Dataset3D).where(Dataset3D.id == dataset_id, Dataset3D.deleted_at.is_(None))
                )
                dataset = dataset_row.scalar_one_or_none()
                if dataset:
                    metadata = dataset.metadata_ or {}

                    # class_names가 제공되지 않았으면 발견된 class_id로부터 생성
                    if not class_names:
                        max_class_id = max(class_ids_found) if class_ids_found else 0
                        class_names = [f"class_{i}" for i in range(max_class_id + 1)]

                    metadata["classes"] = class_names
                    dataset.metadata_ = metadata
                    logger.info(f"Updated Dataset3D {dataset_id} metadata.classes with {len(class_names)} categories")
                else:
                    logger.warning(f"Dataset3D not found for updating classes (id={dataset_id})")
            except Exception as meta_error:
                logger.error(f"Failed to update Dataset3D metadata.classes: {meta_error}", exc_info=True)

            # DB에 저장
            if annotation_records:
                db.add_all(annotation_records)
                await db.commit()
                logger.info(f"Saved {len(annotation_records)} YOLO format annotations to database")
                return len(annotation_records)
            else:
                logger.warning("No YOLO annotations to save")
                return 0

        except Exception as e:
            logger.error(f"Failed to parse YOLO txt annotations: {e}", exc_info=True)
            await db.rollback()
            return 0

    @staticmethod
    async def parse_and_save(
        annotations_dir: Path,
        dataset_id: UUID,
        image_id_map: Dict[str, UUID],
        db: AsyncSession
    ) -> int:
        """
        CARLA에서 생성된 어노테이션 파일을 파싱하여 DB에 저장.
        JSON (COCO 형식) 또는 TXT (YOLO 형식)를 자동으로 감지하여 처리합니다.

        Args:
            annotations_dir: 어노테이션 파일이 있는 디렉토리
            dataset_id: 데이터셋 ID
            image_id_map: 파일명 -> Image3D ID 매핑
            db: DB 세션

        Returns:
            저장된 어노테이션 개수
        """
        try:
            # YOLO 형식 TXT 파일 찾기 (우선순위)
            # 1) annotations 폴더 안에서 찾기
            txt_files = list(annotations_dir.glob("*_yolo.txt"))

            # 2) annotations 폴더가 비어있으면 상위 폴더에서 찾기 (CARLA가 상위에 생성하는 경우)
            if not txt_files:
                parent_dir = annotations_dir.parent
                txt_files = list(parent_dir.glob("*_yolo.txt"))
                if txt_files:
                    logger.info(f"Found YOLO file in parent directory: {txt_files[0]}")

            if txt_files:
                annotation_file = txt_files[0]
                logger.info(f"Found YOLO format annotation file: {annotation_file}")
                return await AnnotationParser.parse_yolo_txt_and_save(
                    txt_file=annotation_file,
                    dataset_id=dataset_id,
                    image_id_map=image_id_map,
                    db=db
                )

            dataset_updated = False
            # COCO 형식 JSON 파일 찾기
            json_files = list(annotations_dir.glob("*.json"))
            if not json_files:
                logger.warning(f"No annotation files (JSON or YOLO TXT) found in {annotations_dir}")
                return 0

            annotation_file = json_files[0]
            logger.info(f"Parsing COCO format annotations from {annotation_file}")

            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)

            # COCO 형식: {"images": [...], "annotations": [...], "categories": [...]}
            annotations = coco_data.get("annotations", [])
            images = coco_data.get("images", [])
            categories = coco_data.get("categories", [])

            # image_id -> file_name 매핑
            image_map = {img["id"]: img["file_name"] for img in images}

            # category_id -> category_name 매핑
            category_map = {cat["id"]: cat.get("name", f"class_{cat['id']}") for cat in categories}
            class_names: List[str] = [cat.get("name", f"class_{cat['id']}") for cat in categories]

            # Dataset3D 메타데이터에 classes 저장 (2D 업로드 시 class.txt 처리와 유사한 형태)
            try:
                dataset_row = await db.execute(
                    select(Dataset3D).where(Dataset3D.id == dataset_id, Dataset3D.deleted_at.is_(None))
                )
                dataset = dataset_row.scalar_one_or_none()
                if dataset:
                    metadata = dataset.metadata_ or {}
                    metadata["classes"] = class_names
                    dataset.metadata_ = metadata
                    dataset_updated = True
                    logger.info(f"Updated Dataset3D {dataset_id} metadata.classes with {len(class_names)} categories")
                else:
                    logger.warning(f"Dataset3D not found for updating classes (id={dataset_id})")
            except Exception as meta_error:
                logger.error(f"Failed to update Dataset3D metadata.classes: {meta_error}", exc_info=True)

            annotation_records = []
            for ann in annotations:
                image_id_coco = ann.get("image_id")
                file_name = image_map.get(image_id_coco)

                if not file_name:
                    logger.warning(f"Image ID {image_id_coco} not found in images list")
                    continue

                image_3d_id = image_id_map.get(file_name)
                if not image_3d_id:
                    logger.warning(f"Image3D not found for file {file_name}")
                    continue

                # COCO bbox: [x, y, width, height] (픽셀 좌표)
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    logger.warning(f"Invalid bbox format for annotation {ann.get('id')}")
                    continue

                x, y, w, h = bbox
                category_id = ann.get("category_id")
                class_name = category_map.get(category_id, f"class_{category_id}")
                area = ann.get("area", w * h)
                is_crowd = ann.get("iscrowd", 0) == 1

                # Annotation 레코드 생성
                annotation_records.append(Annotation(
                    image_3d_id=image_3d_id,
                    annotation_type="bbox",
                    class_name=class_name,
                    class_index=category_id,
                    bbox_x=x,
                    bbox_y=y,
                    bbox_width=w,
                    bbox_height=h,
                    confidence=1.0,  # Ground truth annotation
                    is_crowd=is_crowd,
                    area=area,
                ))

            # DB에 저장
            try:
                if annotation_records:
                    db.add_all(annotation_records)
                    logger.info(f"Prepared {len(annotation_records)} annotations for database save")

                if dataset_updated or annotation_records:
                    await db.commit()
                    logger.info(f"Committed annotations ({len(annotation_records)}) and metadata update ({dataset_updated})")
                else:
                    logger.warning("No annotations to save and no metadata updates to commit")

                return len(annotation_records)
            except Exception as db_error:
                # Handle schema mismatch errors (e.g., missing image_3d_id column)
                if "image_3d_id" in str(db_error) and "does not exist" in str(db_error):
                    logger.error(
                        "Database schema is outdated. Missing 'image_3d_id' column in annotations table. "
                        "Please run database migration: "
                        "docker exec -i army_ai_postgres_dev psql -U postgres -d army_ai < "
                        "/home/jaehyun/army_ai/database/migrations/add_image_3d_id_to_annotations.sql"
                    )
                    await db.rollback()  # Rollback the failed transaction
                    logger.warning("Skipping annotation storage due to schema mismatch. Dataset creation will continue.")
                    return 0
                else:
                    # Re-raise unexpected errors
                    logger.error(f"Unexpected database error while saving annotations: {db_error}")
                    await db.rollback()
                    raise

        except Exception as e:
            logger.error(f"Failed to parse CARLA annotations: {e}", exc_info=True)
            return 0
