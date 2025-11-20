"""
FastAPI main application.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
import asyncio

from app.core.config import settings
from app.api.v1.router import api_router
from app.core.logging import setup_logging, get_logger
from app.core.cache import cache_manager

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Track active streaming tasks
active_tasks = set()

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""## 🚀 Adversarial Vision Platform API

    A comprehensive platform for adversarial AI research and testing.

    ### Features
    - 📊 **Dataset Management**: Upload and manage 2D/3D datasets
    - 🤖 **Model Repository**: Store and version AI models
    - ⚔️ **Adversarial Attacks**: Generate attacks using FGSM, PGD, etc.
    - 📈 **Evaluation**: Benchmark model robustness
    - 🔄 **Real-time Processing**: Stream processing capabilities
    ### Authentication
    Most endpoints require Bearer token authentication.
    """,
    version="1.0.0",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_tags=[
        {"name": "2D Datasets", "description": "Manage 2D image datasets"},
        {"name": "3D Datasets", "description": "Manage 3D datasets"},
        {"name": "Models", "description": "Model repository operations"},
        {"name": "Adversarial Attacks", "description": "Generate adversarial examples"},
        {"name": "Evaluation", "description": "Model evaluation and benchmarking"},
        {"name": "Real-time", "description": "Real-time capture and processing"},
        {"name": "System", "description": "System metrics and health"},
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_PREFIX)

# Mount storage directory for static file serving (images, models, etc.)
storage_path = Path(settings.STORAGE_ROOT)
if storage_path.exists():
    app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")
    logger.info(f"Mounted storage directory: {storage_path}")
else:
    logger.warning(f"Storage directory not found: {storage_path}")


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint.

    Returns:
        Health status including app name, version, and environment.
    """
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {settings.APP_NAME}...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API docs available at: http://localhost:8000{settings.API_PREFIX}/docs")

    # Initialize cache
    await cache_manager.initialize()
    logger.info(f"Cache initialized: {settings.CACHE_TYPE}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info(f"Shutting down {settings.APP_NAME}...")

    # Cancel all active streaming tasks
    if active_tasks:
        logger.info(f"Cancelling {len(active_tasks)} active streaming task(s)...")
        for task in active_tasks:
            if not task.done():
                task.cancel()

        # Wait briefly for tasks to cancel
        await asyncio.gather(*active_tasks, return_exceptions=True)
        logger.info("All streaming tasks cancelled")

    # Close cache connection
    await cache_manager.close()
    logger.info("Cache connection closed")


if __name__ == "__main__":
    import uvicorn
    import os

    # In development mode, use shorter timeout for faster hot reload
    timeout_graceful_shutdown = 1 if settings.ENVIRONMENT.value == "development" else 30

    # Get the backend root directory (parent of app/)
    backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_dir = os.path.join(backend_root, "app")

    # Only watch app/ directory, not storage/ or logs/
    reload_dirs = [app_dir] if settings.DEBUG else None

    logger.info(f"Starting uvicorn with reload_dirs: {reload_dirs}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        reload_dirs=reload_dirs,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
        log_level="info"
    )
