"""
CRUD operations for model repository
"""
from typing import List, Optional, Dict, Any
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
import hashlib
import os
from datetime import datetime

from app.models.model_repo import (
    ODModel,
    ODModelArtifact,
    ModelFramework,
    ModelStage,
    ArtifactType
)


class CRUDModel:
    """CRUD operations for models."""

    async def create_model(
        self,
        db: AsyncSession,
        *,
        name: str,
        task: str = "object-detection",
        owner_id: Optional[UUID] = None,
        description: Optional[str] = None
    ) -> ODModel:
        """Create a new model (legacy method - without version info)."""
        db_model = ODModel(
            name=name,
            task=task,
            owner_id=owner_id,
            description=description,
            version="1.0.0",  # Default version
            framework=ModelFramework.CUSTOM  # Default framework
        )
        db.add(db_model)
        await db.flush()
        await db.refresh(db_model)
        return db_model

    async def create_model_with_version(
        self,
        db: AsyncSession,
        *,
        name: str,
        version: str,
        framework: ModelFramework,
        task: str = "object-detection",
        owner_id: Optional[UUID] = None,
        description: Optional[str] = None,
        framework_version: Optional[str] = None,
        input_spec: Optional[Dict] = None,
        labelmap: Optional[Dict] = None,
        inference_params: Optional[Dict] = None,
        stage: ModelStage = ModelStage.DRAFT,
        created_by: Optional[UUID] = None
    ) -> ODModel:
        """Create a new model with all version information in one record."""
        kwargs = {
            "name": name,
            "version": version,
            "framework": framework,
            "task": task,
            "stage": stage,
        }

        # Add optional fields
        if owner_id is not None:
            kwargs["owner_id"] = owner_id
        if description is not None:
            kwargs["description"] = description
        if framework_version is not None:
            kwargs["framework_version"] = framework_version
        if input_spec is not None:
            kwargs["input_spec"] = input_spec
        if labelmap is not None:
            kwargs["labelmap"] = labelmap
        if inference_params is not None:
            kwargs["inference_params"] = inference_params
        if created_by is not None:
            kwargs["created_by"] = created_by

        db_model = ODModel(**kwargs)
        db.add(db_model)
        await db.flush()
        await db.refresh(db_model)
        return db_model

    async def get_model(self, db: AsyncSession, model_id: UUID) -> Optional[ODModel]:
        """Get model by ID."""
        result = await db.execute(
            select(ODModel).where(
                and_(
                    ODModel.id == model_id,
                    ODModel.deleted_at.is_(None)
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_model_by_name(self, db: AsyncSession, name: str) -> Optional[ODModel]:
        """Get model by name."""
        result = await db.execute(
            select(ODModel).where(
                and_(
                    ODModel.name == name,
                    ODModel.deleted_at.is_(None)
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_models(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[ODModel]:
        """List all models."""
        result = await db.execute(
            select(ODModel)
            .where(ODModel.deleted_at.is_(None))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_model(self, db: AsyncSession, model_id: UUID) -> bool:
        """Soft delete a model by setting deleted_at timestamp."""
        result = await db.execute(
            select(ODModel).where(
                and_(
                    ODModel.id == model_id,
                    ODModel.deleted_at.is_(None)
                )
            )
        )
        db_model = result.scalar_one_or_none()

        if db_model:
            db_model.deleted_at = datetime.utcnow()
            await db.flush()
            await db.commit()
            return True
        return False


# CRUDModelVersion has been removed - all version data is now in ODModel
# CRUDModelClass has been removed - use od_models.labelmap instead


class CRUDModelArtifact:
    """CRUD operations for model artifacts."""

    async def get_by_version(
        self,
        db: AsyncSession,
        version_id: UUID
    ) -> List[ODModelArtifact]:
        """Get all artifacts for a model version."""
        result = await db.execute(
            select(ODModelArtifact).where(
                and_(
                    ODModelArtifact.model_id == version_id,
                    ODModelArtifact.deleted_at.is_(None)
                )
            )
        )
        return list(result.scalars().all())

    async def create_artifact(
        self,
        db: AsyncSession,
        *,
        model_id: UUID,  # Direct link to model (merged table)
        artifact_type: ArtifactType,
        storage_key: str,
        file_name: str,
        file_path: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> ODModelArtifact:
        """Create a model artifact."""
        if model_id is None:
            raise ValueError("model_id must be provided")

        # Calculate file info if file_path provided
        size_bytes = None
        sha256 = None

        if file_path and os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)

            # Calculate SHA256
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            sha256 = sha256_hash.hexdigest()

        # Build kwargs, excluding None values for JSONB fields
        kwargs = {
            "artifact_type": artifact_type,
            "storage_key": storage_key,
            "file_name": file_name,
        }

        # Add model reference
        if model_id is not None:
            kwargs["model_id"] = model_id

        if size_bytes is not None:
            kwargs["size_bytes"] = size_bytes
        if sha256 is not None:
            kwargs["sha256"] = sha256
        if content_type is not None:
            kwargs["content_type"] = content_type

        db_artifact = ODModelArtifact(**kwargs)
        db.add(db_artifact)
        await db.flush()
        await db.refresh(db_artifact)
        return db_artifact


# Deployment CRUD disabled (table not in use)
# class CRUDModelDeployment:
#     """CRUD operations for model deployments."""
#
#     async def create_deployment(
#         self,
#         db: AsyncSession,
#         *,
#         model_id: UUID,
#         endpoint_url: Optional[str] = None,
#         runtime: Optional[Dict] = None,
#         region: Optional[str] = None,
#         is_active: bool = True
#     ) -> ODModelDeployment:
#         """Create a model deployment."""
#         db_deployment = ODModelDeployment(
#             model_id=model_id,
#             endpoint_url=endpoint_url,
#             runtime=runtime,
#             region=region,
#             is_active=is_active
#         )
#         db.add(db_deployment)
#         await db.flush()
#         await db.refresh(db_deployment)
#         return db_deployment
#
#     async def deactivate_deployment(
#         self,
#         db: AsyncSession,
#         deployment_id: UUID
#     ) -> Optional[ODModelDeployment]:
#         """Deactivate a deployment."""
#         result = await db.execute(
#             select(ODModelDeployment).where(
#                 ODModelDeployment.id == deployment_id
#             )
#         )
#         db_deployment = result.scalar_one_or_none()
#
#         if db_deployment:
#             db_deployment.is_active = False
#             await db.flush()
#             await db.refresh(db_deployment)
#
#         return db_deployment


# Create instances
crud_model = CRUDModel()
crud_model_artifact = CRUDModelArtifact()
# crud_model_class = CRUDModelClass()  # Removed - use od_models.labelmap instead
# crud_model_deployment = CRUDModelDeployment()  # Disabled - table not in use
