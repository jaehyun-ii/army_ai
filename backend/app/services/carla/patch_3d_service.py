"""
Patch3D service for managing 3D adversarial patches.

3D 적대적 패치 생성 및 관리 로직을 처리합니다.
"""
import hashlib
from pathlib import Path
from typing import Dict, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset_3d import Patch3D
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class Patch3DService:
    """3D 패치 서비스."""

    @staticmethod
    async def save_patch_metadata(
        image_path: str,
        patch_name: str,
        method: str,
        target_class: str,
        model_id: Optional[UUID],
        db: AsyncSession,
        source_dataset_id: Optional[UUID] = None,
        training_metadata: Optional[Dict] = None
    ) -> Optional[Patch3D]:
        """
        패치 메타데이터를 DB에 저장.

        볼륨 마운트로 인해 이미지는 이미 storage/3d/에 존재하므로 다운로드 불필요.

        Args:
            image_path: 변환된 storage 경로 (예: /storage/3d/Active_3d_k2_tank/attack_pattern/k2_tank_bestval.png)
            patch_name: 패치 이름
            method: 공격 방법 (ACTIVE, DTA)
            target_class: 타겟 클래스
            model_id: 타겟 모델 ID
            db: DB 세션
            source_dataset_id: 소스 데이터셋 ID (선택)
            training_metadata: 학습 메타데이터 (epoch, loss 등)

        Returns:
            생성된 Patch3D 객체 또는 None
        """
        try:
            # 1. storage 경로에서 파일 정보 추출
            # /storage/3d/Active_3d_k2_tank/attack_pattern/k2_tank_bestval.png
            storage_key = image_path.replace("/storage/", "")  # 3d/Active_3d_k2_tank/...
            file_name = Path(image_path).name  # k2_tank_bestval.png
            file_path = Path(settings.STORAGE_ROOT) / storage_key

            # 2. 파일이 실제로 존재하는지 확인하고 정보 수집
            if file_path.exists():
                size_bytes = file_path.stat().st_size
                with open(file_path, "rb") as f:
                    file_data = f.read()
                    sha256_hash = hashlib.sha256(file_data).hexdigest()
            else:
                logger.warning(f"Patch file not found at {file_path}, creating DB entry without file info")
                size_bytes = None
                sha256_hash = None

            # 3. DB에 Patch3D 레코드 생성
            patch = Patch3D(
                name=patch_name,
                description=f"3D adversarial patch generated using {method} method",
                target_model_id=model_id,
                source_dataset_id=source_dataset_id,
                target_class=target_class,
                method=method,
                storage_key=storage_key,
                file_name=file_name,
                size_bytes=size_bytes,
                sha256=sha256_hash,
                hyperparameters={
                    "attack_method": method,
                    "source": "carla_simulator",
                    **(training_metadata or {})
                }
            )

            db.add(patch)
            await db.commit()
            await db.refresh(patch)

            logger.info(f"Saved patch {patch_name} (ID: {patch.id})")
            return patch

        except Exception as e:
            logger.error(f"Failed to save patch metadata: {e}")
            await db.rollback()
            return None
