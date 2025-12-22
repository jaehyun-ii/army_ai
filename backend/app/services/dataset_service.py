"""Convenience re-exports for dataset services.

This module keeps backward-compatible imports while encouraging callers to
depend directly on the dedicated upload/statistics services.
"""
from app.services.dataset_upload_service import (  # noqa: F401
    DatasetUploadService,
    dataset_upload_service,
)
from app.services.dataset_management_service import (  # noqa: F401
    DatasetStatisticsService,
    dataset_statistics_service,
)

__all__ = [
    "DatasetUploadService",
    "dataset_upload_service",
    "DatasetStatisticsService",
    "dataset_statistics_service",
]

