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
from app.core.middleware import SystemLoggingMiddleware, MaintenanceModeMiddleware

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
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add Maintenance Mode Middleware (must be before logging)
app.add_middleware(MaintenanceModeMiddleware)

# Add System Logging Middleware
app.add_middleware(SystemLoggingMiddleware)

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
    logger.info(f"API docs available at: http://localhost:{settings.BACKEND_PORT}{settings.API_PREFIX}/docs")

    # Initialize manual file in storage
    try:
        from pathlib import Path
        import shutil

        storage_root = Path(settings.STORAGE_ROOT)
        manual_dir = storage_root / "manual"
        manual_path = manual_dir / "manual.pdf"

        if not manual_path.exists():
            # Try to find source manual
            backend_root = Path(__file__).parent.parent
            source_paths = [
                backend_root / "manual.pdf",  # Symlink in backend folder
                backend_root.parent / "manual.pdf",  # Original in project root
            ]

            source_manual = None
            for source_path in source_paths:
                if source_path.exists():
                    source_manual = source_path
                    break

            if source_manual:
                manual_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_manual, manual_path)
                logger.info(f"✓ Initialized manual file: {manual_path}")
            else:
                logger.warning("Manual file not found in source locations")
        else:
            logger.info(f"✓ Manual file already exists: {manual_path}")
    except Exception as e:
        logger.error(f"Failed to initialize manual file: {e}")

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

    # 환경변수에서 포트 읽기 (docker에서는 컨테이너 내부 포트 8000 사용)
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.DEBUG,
        reload_dirs=reload_dirs,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
        log_level="info"
    )
