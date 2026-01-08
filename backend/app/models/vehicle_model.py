"""
CARLA 시뮬레이터용 3D 차량 모델 관리
"""
from sqlalchemy import Column, String, Integer, DateTime, Text, Enum as SQLEnum, BIGINT
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.db.base_class import Base
import uuid
import enum


class VehicleCategory(str, enum.Enum):
    """차량 카테고리"""
    MILITARY_VEHICLE = "군용차량"
    AIRCRAFT = "항공기"
    EQUIPMENT = "개인장비"
    BUILDING = "건축물"
    OTHER = "기타"


class ModelFormat(str, enum.Enum):
    """3D 모델 포맷"""
    FBX = "fbx"
    OBJ = "obj"
    GLTF = "gltf"
    GLB = "glb"
    DAE = "dae"
    BLEND = "blend"


class VehicleModel(Base):
    """
    CARLA 시뮬레이터용 3D 차량 모델
    storage/simulator/Vehicle/ 경로에 저장
    """
    __tablename__ = "vehicle_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, unique=True, index=True)
    display_name = Column(String(200), nullable=True)  # 한글 표시 이름
    category = Column(SQLEnum(VehicleCategory), nullable=False, default=VehicleCategory.OTHER)
    format = Column(SQLEnum(ModelFormat), nullable=False)

    # 파일 정보
    storage_key = Column(Text, nullable=False)  # simulator/Vehicle/{name}/{file}
    file_name = Column(String(1024), nullable=False)
    file_size = Column(BIGINT, nullable=True)  # bytes

    # 모델 메타데이터
    polygons = Column(Integer, nullable=True)  # 폴리곤 수
    vertices = Column(Integer, nullable=True)  # 버텍스 수
    textures = Column(JSONB, nullable=True)  # 텍스처 파일 목록
    materials = Column(JSONB, nullable=True)  # 머티리얼 정보

    # 메타데이터
    description = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)

    # 상태 관리
    status = Column(String(50), default="ready")  # ready, processing, error

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<VehicleModel(id={self.id}, name={self.name}, category={self.category})>"
