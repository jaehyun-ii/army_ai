"""
API v1 router (aligned with database schema).
Removed: images endpoint (ImageDetection table does not exist in DB schema)

PHASE 2 UPDATE: Added tags endpoint for experiment tag management.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    storage,
    datasets_2d,
    dataset_service,
    annotations,
    realtime,
    camera,
    models,
    evaluation,
    experiments,
    tags,  # PHASE 2 NEW
    system_stats,
    system_logs,
    users,
    estimators,
    attack_datasets,
    patches,
)

api_router = APIRouter()

# Include endpoint routers
# Authentication
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# Storage
api_router.include_router(storage.router, prefix="/storage", tags=["Storage"])

# Dataset CRUD (database operations only)
api_router.include_router(datasets_2d.datasets_router, prefix="/datasets-2d", tags=["2D Datasets"])

# Dataset Service (upload, file management, business logic)
api_router.include_router(dataset_service.router, prefix="/dataset-service", tags=["Dataset Service"])

# Annotations
api_router.include_router(annotations.router, prefix="/annotations", tags=["Annotations"])

# Core endpoints
api_router.include_router(realtime.router, prefix="/realtime", tags=["Real-time Performance"])
api_router.include_router(camera.router, prefix="/camera", tags=["Camera (Global Session)"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
api_router.include_router(tags.router, prefix="/tags", tags=["Experiment Tags"])  # PHASE 2 NEW

# Estimators
api_router.include_router(estimators.router, prefix="/adversarial", tags=["Estimators"])

# Attack Datasets
api_router.include_router(attack_datasets.router, prefix="/attack-datasets", tags=["Attack Datasets"])

# Patches
api_router.include_router(patches.router, prefix="/patches", tags=["Patches"])

# System Monitoring
api_router.include_router(system_stats.router, prefix="/system", tags=["System Statistics"])
api_router.include_router(system_logs.router, prefix="/system-logs", tags=["System Logs"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
