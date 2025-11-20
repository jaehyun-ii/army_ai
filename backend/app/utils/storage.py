"""
Storage utility for handling file uploads and local storage.
"""
import os
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID
import hashlib
from datetime import datetime
from PIL import Image


class StorageManager:
    """Manage local file storage for datasets."""

    def __init__(self, base_storage_path: Optional[str] = None):
        """Initialize storage manager.

        Args:
            base_storage_path: Base path for storing datasets.
                If None, uses settings.STORAGE_ROOT from centralized config.
        """
        if base_storage_path is None:
            from app.core.config import settings
            base_storage_path = settings.STORAGE_ROOT

        self.storage_root = Path(base_storage_path)
        self.base_path = self.storage_root / "datasets"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_dataset_folder(
        self,
        source_folder: str,
        dataset_name: str,
        owner_id: Optional[UUID] = None
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Save uploaded dataset folder to storage.

        Args:
            source_folder: Path to the uploaded folder
            dataset_name: Name of the dataset
            owner_id: Optional owner UUID

        Returns:
            Tuple of (storage_path, metadata, image_list)
        """
        source_path = Path(source_folder)
        if not source_path.exists():
            raise FileNotFoundError(f"Source folder not found: {source_folder}")

        # Create unique storage path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_folder_name = f"{dataset_name}_{timestamp}"
        storage_path = self.base_path / dataset_folder_name

        # Copy entire folder to storage
        shutil.copytree(source_path, storage_path)

        # Load metadata if exists
        metadata = self._load_metadata(storage_path)

        # Scan and process images
        image_list = self._scan_images(storage_path, dataset_folder_name)

        return str(storage_path), metadata, image_list

    def _load_metadata(self, dataset_path: Path) -> Dict[str, Any]:
        """Load metadata from dataset folder.

        Args:
            dataset_path: Path to dataset folder

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Look for common metadata files
        metadata_files = [
            "metadata.json",
            "dataset.json",
            "info.json",
            "README.json"
        ]

        for filename in metadata_files:
            metadata_file = dataset_path / filename
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    break
                except json.JSONDecodeError:
                    continue

        # Add file count information
        if not metadata:
            metadata = {}

        return metadata

    def _scan_images(
        self,
        dataset_path: Path,
        dataset_folder_name: str
    ) -> List[Dict[str, Any]]:
        """Scan folder for image files.

        Args:
            dataset_path: Path to dataset folder
            dataset_folder_name: Name of dataset folder

        Returns:
            List of image information dictionaries
        """
        from app.core.config import settings
        image_extensions = settings.get_allowed_image_extensions()
        image_list = []

        # Walk through directory
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file

                if file_path.suffix.lower() in image_extensions:
                    # Get relative path from dataset root
                    relative_path = file_path.relative_to(dataset_path)
                    # Use relative path from STORAGE_ROOT for storage_key
                    storage_key = str(file_path.relative_to(self.storage_root))

                    # Get image dimensions
                    width, height = None, None
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                    except Exception:
                        pass

                    # Get file metadata
                    file_stat = file_path.stat()

                    image_info = {
                        "file_name": file,
                        "storage_key": storage_key,
                        "width": width,
                        "height": height,
                        "mime_type": self._get_mime_type(file_path.suffix),
                        "size_bytes": file_stat.st_size,
                        "relative_path": str(relative_path),
                    }

                    image_list.append(image_info)

        return image_list

    def _get_mime_type(self, extension: str) -> str:
        """Get MIME type from file extension.

        Args:
            extension: File extension (e.g., '.jpg')

        Returns:
            MIME type string
        """
        from app.core.config import settings
        return settings.get_mime_type_from_extension(extension)

    def delete_dataset_folder(self, storage_path: str) -> bool:
        """Delete dataset folder from storage.

        Args:
            storage_path: Path to dataset folder

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(storage_path)
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting dataset folder: {e}")
            return False

    def get_dataset_info(self, storage_path: str) -> Dict[str, Any]:
        """Get information about a stored dataset.

        Args:
            storage_path: Path to dataset folder

        Returns:
            Dataset information dictionary
        """
        from app.core.config import settings

        path = Path(storage_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {storage_path}")

        # Count files using allowed extensions from centralized config
        image_extensions = settings.get_allowed_image_extensions()
        image_count = sum(
            1 for f in path.rglob('*')
            if f.is_file() and f.suffix.lower() in image_extensions
        )

        # Get folder size
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

        return {
            "path": str(path),
            "image_count": image_count,
            "total_size_bytes": total_size,
            "created_at": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
        }


# Global storage manager instance
storage_manager = StorageManager()
