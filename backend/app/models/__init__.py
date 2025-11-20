"""
SQLAlchemy ORM models (aligned with database schema).
Removed models: Camera, RTInference, InferenceMetadata, ImageDetection, DatasetClassStatistics
These tables do not exist in the database schema.
"""
from app.models.user import User
from app.models.dataset_2d import Dataset2D, Image2D, Patch2D, AttackDataset2D
from app.models.dataset_3d import Dataset3D, Image3D
from app.models.realtime import RTCaptureRun, RTFrame
from app.models.model_repo import (
    ODModel,
    ODModelArtifact,
    # ODModelClass,  # Removed - use od_models.labelmap instead
    # ODModelDeployment,  # Disabled - table not in use
)
from app.models.audit import AuditLog
from app.models.evaluation import (
    EvalRun,
    EvalItem,
    # EvalClassMetrics,  # Removed - table not in use
    EvalList,
    EvalListItem,
)
from app.models.experiment import Experiment
from app.models.annotation import Annotation, AnnotationType

__all__ = [
    "User",
    "Dataset2D",
    "Image2D",
    "Dataset3D",
    "Image3D",
    "Patch2D",
    "AttackDataset2D",
    "RTCaptureRun",
    "RTFrame",
    "ODModel",
    "ODModelArtifact",
    # "ODModelClass",  # Removed - use od_models.labelmap instead
    # "ODModelDeployment",  # Disabled - table not in use
    "AuditLog",
    "EvalRun",
    "EvalItem",
    # "EvalClassMetrics",  # Removed - table not in use
    "EvalList",
    "EvalListItem",
    "Experiment",
    "Annotation",
    "AnnotationType",
]
