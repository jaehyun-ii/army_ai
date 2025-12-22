"""
GAN weights management service.

Handles downloading, caching, and validation of GAN model weights
for naturalistic adversarial patch generation.
"""
import os
import logging
from pathlib import Path
from typing import Optional
import hashlib
import urllib.request
from tqdm import tqdm

from app.core.config import settings

logger = logging.getLogger(__name__)


class GANWeightsService:
    """Service for managing GAN model weights."""

    def __init__(self, storage_root: Optional[str] = None):
        """
        Initialize GAN weights service.

        Args:
            storage_root: Root directory for storing weights (default: settings.STORAGE_ROOT)
        """
        self.storage_root = Path(storage_root or settings.STORAGE_ROOT)
        self.weights_dir = self.storage_root / "gan_weights"
        self.biggan_dir = self.weights_dir / "biggan"

        # Create directories if they don't exist
        self.biggan_dir.mkdir(parents=True, exist_ok=True)

    def get_biggan_weights_path(self) -> Path:
        """
        Get path to BigGAN generator weights.

        Returns:
            Path to G_ema.pth file
        """
        return self.biggan_dir / "G_ema.pth"

    def biggan_weights_exist(self) -> bool:
        """
        Check if BigGAN weights are downloaded.

        Returns:
            True if weights exist, False otherwise
        """
        weights_path = self.get_biggan_weights_path()
        return weights_path.exists() and weights_path.stat().st_size > 0

    def download_biggan_weights(self, force: bool = False) -> Path:
        """
        Download BigGAN pretrained weights if not already cached.

        Args:
            force: Force re-download even if weights exist

        Returns:
            Path to downloaded weights

        Raises:
            RuntimeError: If download fails
        """
        weights_path = self.get_biggan_weights_path()

        # Skip if already exists and not forcing
        if self.biggan_weights_exist() and not force:
            logger.info(f"BigGAN weights already exist at {weights_path}")
            return weights_path

        logger.info("Downloading BigGAN pretrained weights...")

        # BigGAN weights URL - from NAP's download_weights.sh
        # This is the BigGAN-deep 128x128 model trained on ImageNet
        # Note: NAP uses tar file, but we only need the .pth file inside
        # If downloading fails, manually copy from NAP project:
        # cp NaturalisticAdversarialPatches/GANLatentDiscovery/.../G_ema.pth storage/gan_weights/biggan/
        biggan_url = (
            "https://www.dropbox.com/scl/fi/yo49fqo4jskuc69/G_ema.pth?rlkey=xxxxx&dl=1"
        )

        try:
            # Download with progress bar
            logger.info(f"Downloading from {biggan_url}")

            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with DownloadProgressBar(
                unit='B',
                unit_scale=True,
                miniters=1,
                desc='BigGAN weights'
            ) as t:
                urllib.request.urlretrieve(
                    biggan_url,
                    weights_path,
                    reporthook=t.update_to
                )

            logger.info(f"BigGAN weights downloaded to {weights_path}")

            # Verify file size (should be ~140MB)
            file_size_mb = weights_path.stat().st_size / (1024 * 1024)
            logger.info(f"Downloaded file size: {file_size_mb:.1f} MB")

            if file_size_mb < 100:
                raise RuntimeError(
                    f"Downloaded file seems too small ({file_size_mb:.1f} MB). "
                    "Download may be incomplete."
                )

            return weights_path

        except Exception as e:
            # Clean up partial download
            if weights_path.exists():
                weights_path.unlink()

            logger.error(f"Failed to download BigGAN weights: {e}")
            raise RuntimeError(
                f"Failed to download BigGAN weights: {e}. "
                "Please check your internet connection and try again."
            )

    def validate_weights(self, weights_path: Path) -> bool:
        """
        Validate that weights file is valid.

        Args:
            weights_path: Path to weights file

        Returns:
            True if valid, False otherwise
        """
        if not weights_path.exists():
            return False

        try:
            import torch
            # Try to load the state dict
            state_dict = torch.load(weights_path, map_location='cpu')
            return isinstance(state_dict, dict) and len(state_dict) > 0
        except Exception as e:
            logger.error(f"Invalid weights file: {e}")
            return False

    def ensure_biggan_weights(self) -> Path:
        """
        Ensure BigGAN weights are available, downloading if necessary.

        Returns:
            Path to valid BigGAN weights

        Raises:
            RuntimeError: If weights cannot be obtained
        """
        if not self.biggan_weights_exist():
            logger.info("BigGAN weights not found, downloading...")
            weights_path = self.download_biggan_weights()
        else:
            weights_path = self.get_biggan_weights_path()

        # Validate weights
        if not self.validate_weights(weights_path):
            logger.warning("Weights validation failed, re-downloading...")
            weights_path = self.download_biggan_weights(force=True)

            if not self.validate_weights(weights_path):
                raise RuntimeError(
                    "Failed to obtain valid BigGAN weights. "
                    "Please check the download source."
                )

        return weights_path

    def get_storage_info(self) -> dict:
        """
        Get storage information for GAN weights.

        Returns:
            Dictionary with storage statistics
        """
        biggan_exists = self.biggan_weights_exist()
        biggan_size = 0

        if biggan_exists:
            biggan_size = self.get_biggan_weights_path().stat().st_size

        return {
            "weights_dir": str(self.weights_dir),
            "biggan": {
                "exists": biggan_exists,
                "path": str(self.get_biggan_weights_path()),
                "size_mb": biggan_size / (1024 * 1024) if biggan_size > 0 else 0
            },
            "total_size_mb": biggan_size / (1024 * 1024) if biggan_size > 0 else 0
        }


# Global instance
gan_weights_service = GANWeightsService()
