"""
Pydantic schemas for request/response validation.
"""
from app.schemas.user import UserCreate, UserUpdate, UserResponse, Token, TokenData
from app.schemas.dataset_2d import (
    Dataset2DCreate,
    Dataset2DUpdate,
    Dataset2DResponse,
    DatasetSummaryResponse,
    DatasetStatisticsResponse,
    ImageCreate,
    ImageResponse,
    ImageListResponse,
    Patch2DCreate,
    Patch2DResponse,
    PatchGenerationRequest,
    AttackDataset2DCreate,
    AttackDataset2DResponse,
)
from app.schemas.dataset_3d import (
    Dataset3DCreate,
    Dataset3DUpdate,
    Dataset3DResponse,
    Image3DCreate,
    Image3DResponse,
    Image3DListResponse,
    Patch3DCreate,
    Patch3DResponse,
    AttackDataset3DCreate,
    AttackDataset3DResponse,
)
from app.schemas.model_repo import (
    ODModelCreate,
    ODModelResponse,
    ODModelArtifactCreate,
    ODModelArtifactResponse,
)
from app.schemas.realtime import (
    RTCaptureRunCreate,
    RTCaptureRunUpdate,
    RTCaptureRunResponse,
    RTFrameCreate,
    RTFrameUpdate,
    RTFrameResponse,
)
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
)
from app.schemas.evaluation import (
    EvalRunCreate,
    EvalRunUpdate,
    EvalRunResponse,
    EvalItemCreate,
    EvalItemUpdate,
    EvalItemResponse,
    # EvalClassMetricsCreate,  # Removed - table not in use
    # EvalClassMetricsUpdate,  # Removed - table not in use
    # EvalClassMetricsResponse,  # Removed - table not in use
)
from app.schemas.annotation import (
    AnnotationCreate,
    AnnotationResponse,
    AnnotationDetectionInfo,
)
from app.schemas.estimator import (
    EstimatorFramework,
    EstimatorType,
    LoadEstimatorRequest,
    LoadEstimatorResponse,
    PredictRequest,
    PredictResponse,
    EstimatorListResponse,
    EstimatorStatusResponse,
    BBox,
    YoloBBox,
    Detection,
)
from app.schemas.backup import (
    BackupCreate,
    BackupUpdate,
    BackupResponse,
    BackupListResponse,
)

__all__ = [
    # User
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "Token",
    "TokenData",
    # 2D Datasets
    "Dataset2DCreate",
    "Dataset2DUpdate",
    "Dataset2DResponse",
    "DatasetSummaryResponse",
    "DatasetStatisticsResponse",
    "ImageCreate",
    "ImageResponse",
    "ImageListResponse",
    "Patch2DCreate",
    "Patch2DResponse",
    "PatchGenerationRequest",
    "AttackDataset2DCreate",
    "AttackDataset2DResponse",
    # 3D Datasets
    "Dataset3DCreate",
    "Dataset3DUpdate",
    "Dataset3DResponse",
    "Image3DCreate",
    "Image3DResponse",
    "Image3DListResponse",
    "Patch3DCreate",
    "Patch3DResponse",
    "AttackDataset3DCreate",
    "AttackDataset3DResponse",
    # Model Repo
    "ODModelCreate",
    "ODModelResponse",
    "ODModelArtifactCreate",
    "ODModelArtifactResponse",
    # Real-time Performance
    "RTCaptureRunCreate",
    "RTCaptureRunUpdate",
    "RTCaptureRunResponse",
    "RTFrameCreate",
    "RTFrameUpdate",
    "RTFrameResponse",
    # Experiments
    "ExperimentCreate",
    "ExperimentUpdate",
    "ExperimentResponse",
    # Evaluations
    "EvalRunCreate",
    "EvalRunUpdate",
    "EvalRunResponse",
    "EvalItemCreate",
    "EvalItemUpdate",
    "EvalItemResponse",
    # "EvalClassMetricsCreate",  # Removed - table not in use
    # "EvalClassMetricsUpdate",  # Removed - table not in use
    # "EvalClassMetricsResponse",  # Removed - table not in use
    # Annotations
    "AnnotationCreate",
    "AnnotationResponse",
    "AnnotationDetectionInfo",
    # Estimators
    "EstimatorFramework",
    "EstimatorType",
    "LoadEstimatorRequest",
    "LoadEstimatorResponse",
    "PredictRequest",
    "PredictResponse",
    "EstimatorListResponse",
    "EstimatorStatusResponse",
    "BBox",
    "YoloBBox",
    "Detection",
]
