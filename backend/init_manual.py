#!/usr/bin/env python3
"""
Initialize manual file in storage.
This script copies the manual PDF to storage/manual/manual.pdf
"""
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_manual():
    """Copy manual file to storage if it doesn't exist."""
    # Get paths
    backend_root = Path(__file__).parent
    project_root = backend_root.parent

    # Source: project root or backend folder (symlink)
    source_paths = [
        backend_root / "manual.pdf",  # Symlink in backend folder
        project_root / "manual.pdf",  # Original in project root
    ]

    source_manual = None
    for path in source_paths:
        if path.exists():
            source_manual = path
            logger.info(f"Found source manual at: {source_manual}")
            break

    if not source_manual:
        logger.error("Source manual file not found!")
        return False

    # Destination: storage/manual/manual.pdf
    # Try to determine storage root
    storage_root = Path("/storage")  # Docker container path
    if not storage_root.exists():
        storage_root = project_root / "storage"  # Local development path

    manual_dir = storage_root / "manual"
    dest_manual = manual_dir / "manual.pdf"

    # Create directory if needed
    try:
        manual_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created manual directory: {manual_dir}")
    except PermissionError as e:
        logger.error(f"Permission denied creating directory: {e}")
        return False

    # Copy file if it doesn't exist
    if dest_manual.exists():
        logger.info(f"Manual already exists at: {dest_manual}")
        return True

    try:
        shutil.copy2(source_manual, dest_manual)
        logger.info(f"Copied manual from {source_manual} to {dest_manual}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy manual: {e}")
        return False


if __name__ == "__main__":
    success = init_manual()
    if success:
        logger.info("✓ Manual initialization completed")
    else:
        logger.error("✗ Manual initialization failed")
        exit(1)
