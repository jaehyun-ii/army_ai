"""
CARLA 3D 차량 모델 관리 API
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from typing import List, Optional
from pathlib import Path
import shutil
import uuid
import os
import logging

from app.db.session import get_db
from app.models.vehicle_model import VehicleModel, VehicleCategory, ModelFormat
from app.schemas.vehicle_model import (
    VehicleModelResponse,
    VehicleModelListResponse,
    VehicleModelCreate,
    VehicleModelUpdate
)
from app.core.config import settings
from app.api.v1.dependencies import require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {".fbx", ".obj", ".gltf", ".glb", ".dae", ".blend", ".mtl", ".png", ".jpg", ".jpeg", ".tga", ".dds"}
ALLOWED_MODEL_EXTENSIONS = {".fbx", ".obj", ".gltf", ".glb", ".dae", ".blend"}


def get_model_format(filename: str) -> str:
    """파일명에서 모델 포맷 추출"""
    ext = Path(filename).suffix.lower()
    format_map = {
        ".fbx": "fbx",
        ".obj": "obj",
        ".gltf": "gltf",
        ".glb": "glb",
        ".dae": "dae",
        ".blend": "blend",
    }
    return format_map.get(ext, "fbx")


@router.post("/upload", response_model=VehicleModelResponse, status_code=status.HTTP_201_CREATED)
async def upload_vehicle_model(
    model_name: str = Form(..., description="모델 이름"),
    display_name: Optional[str] = Form(None, description="한글 표시 이름"),
    category: str = Form(..., description="카테고리"),
    description: Optional[str] = Form(None, description="설명"),
    files: List[UploadFile] = File(..., description="모델 파일들 (메인 모델 + 텍스처)"),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_admin),
):
    """
    3D 차량 모델 업로드

    - model_name: 영문, 숫자, -, _만 사용 가능
    - category: 군용차량, 항공기, 개인장비, 건축물, 기타
    - files: 메인 모델 파일 + 텍스처/머티리얼 파일들
    """
    try:
        # 1. 모델명 검증
        if not model_name or not model_name.replace("_", "").replace("-", "").isalnum():
            raise HTTPException(
                status_code=400,
                detail="모델 이름은 영문자, 숫자, -, _만 사용 가능합니다"
            )

        # 2. 중복 체크
        existing = await db.execute(
            select(VehicleModel).where(
                VehicleModel.name == model_name,
                VehicleModel.deleted_at.is_(None)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail=f"모델 이름 '{model_name}'이(가) 이미 존재합니다"
            )

        # 3. 메인 모델 파일 찾기
        main_model_file = None
        texture_files = []

        for file in files:
            ext = Path(file.filename).suffix.lower()
            if ext in ALLOWED_MODEL_EXTENSIONS:
                if main_model_file is None:
                    main_model_file = file
                else:
                    texture_files.append(file)  # 추가 모델 파일도 텍스처로 처리
            elif ext in ALLOWED_EXTENSIONS:
                texture_files.append(file)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"허용되지 않는 파일 형식: {ext}"
                )

        if not main_model_file:
            raise HTTPException(
                status_code=400,
                detail="메인 3D 모델 파일(.fbx, .obj, .gltf 등)을 찾을 수 없습니다"
            )

        # 4. 저장 경로 생성
        model_dir = Path(settings.STORAGE_ROOT) / "simulator" / "Vehicle" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 5. 메인 모델 파일 저장
        main_file_path = model_dir / main_model_file.filename
        with open(main_file_path, "wb") as buffer:
            content = await main_model_file.read()
            buffer.write(content)

        file_size = main_file_path.stat().st_size
        model_format = get_model_format(main_model_file.filename)

        # 6. 텍스처/머티리얼 파일 저장
        texture_filenames = []
        for texture_file in texture_files:
            texture_path = model_dir / texture_file.filename
            with open(texture_path, "wb") as buffer:
                content = await texture_file.read()
                buffer.write(content)
            texture_filenames.append(texture_file.filename)
            logger.info(f"Saved texture file: {texture_file.filename}")

        # 7. storage_key 생성
        storage_key = f"simulator/Vehicle/{model_name}/{main_model_file.filename}"

        # 8. DB에 저장
        vehicle_model = VehicleModel(
            id=uuid.uuid4(),
            name=model_name,
            display_name=display_name or model_name,
            category=VehicleCategory(category) if category in [c.value for c in VehicleCategory] else VehicleCategory.OTHER,
            format=ModelFormat(model_format),
            storage_key=storage_key,
            file_name=main_model_file.filename,
            file_size=file_size,
            description=description,
            textures=texture_filenames if texture_filenames else None,
            status="ready",
        )

        db.add(vehicle_model)
        await db.commit()
        await db.refresh(vehicle_model)

        logger.info(f"Vehicle model uploaded successfully: {model_name} ({vehicle_model.id})")

        return vehicle_model

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload vehicle model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"모델 업로드 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("", response_model=VehicleModelListResponse)
async def list_vehicle_models(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    format: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    3D 차량 모델 목록 조회

    - category: 카테고리 필터
    - format: 파일 포맷 필터
    - search: 이름 검색
    """
    try:
        # Base query
        stmt = select(VehicleModel).where(VehicleModel.deleted_at.is_(None))

        # Filters
        if category:
            stmt = stmt.where(VehicleModel.category == category)
        if format:
            stmt = stmt.where(VehicleModel.format == format)
        if search:
            stmt = stmt.where(
                or_(
                    VehicleModel.name.ilike(f"%{search}%"),
                    VehicleModel.display_name.ilike(f"%{search}%")
                )
            )

        # Total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar() or 0

        # Paginated results
        stmt = stmt.order_by(VehicleModel.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        models = result.scalars().all()

        return VehicleModelListResponse(
            total=total,
            items=models
        )

    except Exception as e:
        logger.error(f"Failed to list vehicle models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"모델 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/{model_id}", response_model=VehicleModelResponse)
async def get_vehicle_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """3D 차량 모델 상세 조회"""
    model = await db.get(VehicleModel, model_id)

    if not model or model.deleted_at is not None:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")

    return model


@router.get("/{model_id}/download")
async def download_vehicle_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """3D 차량 모델 다운로드"""
    model = await db.get(VehicleModel, model_id)

    if not model or model.deleted_at is not None:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")

    file_path = Path(settings.STORAGE_ROOT) / model.storage_key

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다")

    return FileResponse(
        path=file_path,
        filename=model.file_name,
        media_type="application/octet-stream"
    )


@router.patch("/{model_id}", response_model=VehicleModelResponse)
async def update_vehicle_model(
    model_id: uuid.UUID,
    update_data: VehicleModelUpdate,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_admin),
):
    """3D 차량 모델 정보 수정"""
    model = await db.get(VehicleModel, model_id)

    if not model or model.deleted_at is not None:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(model, field, value)

    await db.commit()
    await db.refresh(model)

    return model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vehicle_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_admin),
):
    """3D 차량 모델 삭제 (Soft delete)"""
    model = await db.get(VehicleModel, model_id)

    if not model or model.deleted_at is not None:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")

    # Soft delete
    model.deleted_at = func.now()
    await db.commit()

    logger.info(f"Vehicle model deleted (soft): {model.name} ({model_id})")

    return None
