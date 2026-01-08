"""
CARLA 3D 차량 모델 Pydantic 스키마
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class VehicleModelBase(BaseModel):
    """기본 스키마"""
    name: str = Field(..., description="모델 이름 (영문, 숫자, -, _만 사용)")
    display_name: Optional[str] = Field(None, description="한글 표시 이름")
    category: str = Field(..., description="카테고리 (군용차량, 항공기, 개인장비, 건축물, 기타)")
    description: Optional[str] = Field(None, description="모델 설명")


class VehicleModelCreate(VehicleModelBase):
    """생성 요청 스키마"""
    format: str = Field(..., description="파일 포맷 (fbx, obj, gltf, glb, dae, blend)")
    storage_key: str = Field(..., description="Storage key")
    file_name: str = Field(..., description="파일명")
    file_size: Optional[int] = Field(None, description="파일 크기 (bytes)")
    polygons: Optional[int] = Field(None, description="폴리곤 수")
    vertices: Optional[int] = Field(None, description="버텍스 수")
    textures: Optional[List[str]] = Field(None, description="텍스처 파일 목록")
    materials: Optional[Dict[str, Any]] = Field(None, description="머티리얼 정보")
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")


class VehicleModelUpdate(BaseModel):
    """수정 요청 스키마"""
    display_name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VehicleModelResponse(VehicleModelBase):
    """응답 스키마"""
    id: UUID
    format: str
    storage_key: str
    file_name: str
    file_size: Optional[int] = None
    polygons: Optional[int] = None
    vertices: Optional[int] = None
    textures: Optional[List[str]] = None
    materials: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class VehicleModelListResponse(BaseModel):
    """목록 응답 스키마"""
    total: int
    items: List[VehicleModelResponse]
