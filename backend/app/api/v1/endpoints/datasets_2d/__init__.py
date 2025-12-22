"""Datasets v1 router package."""

from fastapi import APIRouter

from . import images, metadata, stats, datasets


datasets_router = APIRouter()

datasets_router.include_router(datasets.router, prefix="", tags=["2D Datasets"])
datasets_router.include_router(images.router, prefix="", tags=["2D Datasets"])
datasets_router.include_router(metadata.router, prefix="", tags=["2D Datasets"])
datasets_router.include_router(stats.router, prefix="", tags=["2D Datasets"])


__all__ = [
    "datasets_router",
]

