"""
CRUD operations (aligned with database schema).
Removed: Camera, RTInference (tables do not exist in DB schema)
"""
from pydantic import BaseModel
from app.crud.base import CRUDBase
from app.models.user import User
from app.models.dataset_2d import Dataset2D, Image2D, Patch2D, AttackDataset2D
from app.models.realtime import RTCaptureRun, RTFrame
from app.models.model_repo import ODModel, ODModelArtifact
from app.schemas.user import UserCreate, UserUpdate
from app.schemas.dataset_2d import (
    Dataset2DCreate,
    Dataset2DUpdate,
    ImageCreate,
    Patch2DCreate,
    AttackDataset2DCreate,
)
from app.schemas.realtime import (
    RTCaptureRunCreate,
    RTCaptureRunUpdate,
    RTFrameCreate,
    RTFrameUpdate,
)
from app.schemas.model_repo import (
    ODModelCreate,
    ODModelArtifactCreate,
)


# User CRUD
class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """CRUD operations for User model."""

    pass


# Dataset 2D CRUD
class CRUDDataset2D(CRUDBase[Dataset2D, Dataset2DCreate, Dataset2DUpdate]):
    """CRUD operations for Dataset2D model."""

    pass


# Image 2D CRUD
class CRUDImage2D(CRUDBase[Image2D, ImageCreate, BaseModel]):
    """CRUD operations for Image2D model."""

    async def get_by_dataset(self, db, *, dataset_id, skip: int = 0, limit: int = 100):
        """Get images by dataset ID."""
        from uuid import UUID
        from sqlalchemy import select
        result = await db.execute(
            select(Image2D)
            .where(Image2D.dataset_id == dataset_id, Image2D.deleted_at.is_(None))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


# Patch 2D CRUD
class CRUDPatch2D(CRUDBase[Patch2D, Patch2DCreate, BaseModel]):
    """CRUD operations for Patch2D model."""

    pass


# Attack Dataset 2D CRUD
class CRUDAttackDataset2D(CRUDBase[AttackDataset2D, AttackDataset2DCreate, BaseModel]):
    """CRUD operations for AttackDataset2D model."""

    pass


# OD Model CRUD
class CRUDODModel(CRUDBase[ODModel, ODModelCreate, BaseModel]):
    """CRUD operations for ODModel."""

    pass




# OD Model Artifact CRUD
class CRUDODModelArtifact(CRUDBase[ODModelArtifact, ODModelArtifactCreate, BaseModel]):
    """CRUD operations for ODModelArtifact."""

    async def get_by_version(self, db, *, version_id):
        """Get all artifacts for a model version."""
        from sqlalchemy import select, and_
        from uuid import UUID
        result = await db.execute(
            select(ODModelArtifact).where(
                and_(
                    ODModelArtifact.model_id == version_id,
                    ODModelArtifact.deleted_at.is_(None)
                )
            )
        )
        return list(result.scalars().all())


# RT Capture Run CRUD
class CRUDRTCaptureRun(CRUDBase[RTCaptureRun, RTCaptureRunCreate, RTCaptureRunUpdate]):
    """CRUD operations for RTCaptureRun model."""

    pass


# RT Frame CRUD
class CRUDRTFrame(CRUDBase[RTFrame, RTFrameCreate, RTFrameUpdate]):
    """CRUD operations for RTFrame model."""

    pass


# Instantiate CRUD objects
user = CRUDUser(User)
dataset_2d = CRUDDataset2D(Dataset2D)
image_2d = CRUDImage2D(Image2D)
patch_2d = CRUDPatch2D(Patch2D)
attack_dataset_2d = CRUDAttackDataset2D(AttackDataset2D)
od_model = CRUDODModel(ODModel)
model_artifact = CRUDODModelArtifact(ODModelArtifact)
rt_capture_run = CRUDRTCaptureRun(RTCaptureRun)
rt_frame = CRUDRTFrame(RTFrame)

# Import new CRUD modules
from app.crud.experiment import experiment
from app.crud.evaluation import EvalRun
from app.crud.annotation import annotation
