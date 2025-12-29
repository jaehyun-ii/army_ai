"""
BigGAN loader for naturalistic adversarial patch generation.

This module provides utilities to load BigGAN generator and optional deformator
for naturalistic patch generation.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn

from . import BigGAN, utils
from .latent_deformator import LatentDeformator, DeformatorType

logger = logging.getLogger(__name__)


class ConditionedBigGAN(nn.Module):
    """
    BigGAN conditioned on specific ImageNet classes.

    Wraps BigGAN to always generate images from specified classes.
    """
    def __init__(self, big_gan, target_classes=(239,)):
        super(ConditionedBigGAN, self).__init__()
        self.big_gan = big_gan

        self.target_classes = nn.Parameter(
            torch.tensor(target_classes, dtype=torch.int64),
            requires_grad=False
        )

        self.dim_z = self.big_gan.dim_z
        self.dim_shift = self.dim_z

    def set_classes(self, cl):
        """Set target classes for generation."""
        try:
            cl[0]
        except Exception:
            cl = [cl]
        self.target_classes.data = torch.tensor(cl, dtype=torch.int64)

    def mixed_classes(self, batch_size):
        """Get random target classes for batch."""
        if len(self.target_classes.data.shape) == 0:
            return self.target_classes.repeat(batch_size).cuda()
        else:
            import numpy as np
            return torch.from_numpy(
                np.random.choice(self.target_classes.cpu(), [batch_size])
            ).cuda()

    def forward(self, z, classes=None):
        """Generate images from latent vector z."""
        if classes is None:
            classes = self.mixed_classes(z.shape[0]).to(z.device)
        return self.big_gan(z, self.big_gan.shared(classes))

    def gen_shifted(self, z, shift):
        """Generate images with shifted latent vector."""
        return self.forward(z + shift)


def make_biggan_config(weights_root: str, config_path: Optional[str] = None) -> dict:
    """
    Create BigGAN configuration from config file.

    Args:
        weights_root: Path to BigGAN weights
        config_path: Path to generator_config.json (defaults to module directory)

    Returns:
        BigGAN configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / 'generator_config.json'

    with open(config_path) as f:
        config = json.load(f)

    config['weights_root'] = weights_root
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config['skip_init'] = True
    config['no_optim'] = True

    return config


def load_biggan(
    weights_path: str,
    target_class: int = 259,
    device: str = 'cuda',
    deformator_path: Optional[str] = None,
    deformator_type: str = 'ortho'
) -> Tuple[ConditionedBigGAN, Optional[LatentDeformator]]:
    """
    Load BigGAN generator and optional deformator.

    Args:
        weights_path: Path to BigGAN generator weights (.pth file)
        target_class: ImageNet class ID for generation (default: 259 = Pomeranian)
        device: Device to load model on ('cuda' or 'cpu')
        deformator_path: Optional path to deformator weights
        deformator_type: Deformator type ('ortho', 'linear', 'fc', 'id', 'proj')

    Returns:
        Tuple of (generator, deformator)
        - generator: ConditionedBigGAN model
        - deformator: LatentDeformator or None if not loaded

    Example:
        >>> generator, deformator = load_biggan(
        ...     '/storage/gan_weights/biggan/G_ema.pth',
        ...     target_class=259
        ... )
        >>> z = torch.randn(1, 120).cuda()
        >>> image = generator(z)  # Shape: (1, 3, 128, 128)
    """
    logger.info(f"Loading BigGAN from {weights_path} for class {target_class}")

    # Create BigGAN configuration
    config = make_biggan_config(weights_path)

    # Create generator
    G = BigGAN.Generator(**config)

    # Load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"BigGAN weights not found at {weights_path}. "
            f"Please run GAN weights download first."
        )

    state_dict = torch.load(weights_path, map_location='cpu')
    G.load_state_dict(state_dict, strict=True)

    # Wrap in conditioned wrapper
    generator = ConditionedBigGAN(G, target_class)
    generator = generator.to(device).eval()

    logger.info(f"BigGAN loaded successfully (dim_z={generator.dim_z})")

    # Load deformator if specified
    deformator = None
    if deformator_path and os.path.exists(deformator_path):
        logger.info(f"Loading deformator from {deformator_path}")

        # Map deformator type string to enum
        deformator_type_map = {
            'fc': DeformatorType.FC,
            'linear': DeformatorType.LINEAR,
            'id': DeformatorType.ID,
            'ortho': DeformatorType.ORTHO,
            'proj': DeformatorType.PROJECTIVE
        }
        deformator_type_enum = deformator_type_map.get(
            deformator_type.lower(),
            DeformatorType.ORTHO
        )

        deformator = LatentDeformator(
            shift_dim=generator.dim_shift,
            type=deformator_type_enum
        )
        deformator.load_state_dict(
            torch.load(deformator_path, map_location='cpu')
        )
        deformator = deformator.to(device).eval()

        annotation_path = Path(deformator_path).parent.parent / "human_annotation.txt"
        annotation = _load_human_annotation(annotation_path)
        if annotation:
            setattr(deformator, "annotation", annotation)
            logger.info(f"Loaded {len(annotation)} human-annotated directions")

        logger.info("Deformator loaded successfully")

    return generator, deformator


def _load_human_annotation(annotation_path: Path) -> dict[str, int]:
    annotation_dict: dict[str, int] = {}
    if not annotation_path.is_file():
        return annotation_dict

    with annotation_path.open() as source:
        for line in source.readlines():
            if ": " not in line:
                continue
            idx_str, annotation = line.split(": ", 1)
            annotation = annotation.strip()
            if not annotation:
                continue
            annotation_key = annotation.replace(" ", "_")
            suffix = 1
            unique_key = annotation_key
            while unique_key in annotation_dict:
                suffix += 1
                unique_key = f"{annotation_key}_{suffix}"
            try:
                annotation_dict[unique_key] = int(idx_str)
            except ValueError:
                continue
    return annotation_dict


def generate_from_latent(
    generator: ConditionedBigGAN,
    latent: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
    deformator: Optional[LatentDeformator] = None
) -> torch.Tensor:
    """
    Generate image from latent vector with optional shift.

    Args:
        generator: BigGAN generator
        latent: Base latent vector (shape: [batch, dim_z])
        shift: Optional shift vector (shape: [batch, dim_z] or [dim_z])
        deformator: Optional deformator for latent transformation

    Returns:
        Generated image tensor (shape: [batch, 3, H, W])
    """
    if shift is not None:
        if deformator is not None:
            # Apply deformator to shift
            shift = deformator(shift)

        # Ensure shift has correct shape
        if shift.dim() == 1:
            shift = shift.unsqueeze(0).expand(latent.size(0), -1)

        return generator.gen_shifted(latent, shift)
    else:
        return generator(latent)
