# aegis/core/config_models.py

from pydantic import BaseModel, Field, validator
from typing import List, Literal, Tuple, Optional, Union, Any

def parse_epsilon(value: Any) -> float:
    """
    Parses a user-friendly epsilon value into a float.
    - If it's a float, return it directly.
    - If it's a string like "5/255", perform the division.
    - If it's a string or int like "5" or 5, assume division by 255.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value) / 255.0
    if isinstance(value, str):
        if "/" in value:
            try:
                num, den = value.split('/')
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(f"Invalid epsilon fraction '{value}': {e}")
        else:
            return float(value) / 255.0
    raise TypeError(f"Epsilon must be a float, int, or string, not {type(value)}")

# --- This new model defines the structure for a single, fully-specified pose ---
class SinglePoseConfig(BaseModel):
    ref_image_path: str
    mask_path: Optional[str] = None
    depth_map_path: Optional[str] = None
    extrinsics_inv_path: Optional[str] = None
    intrinsics: Optional[Union[str, List[List[float]]]] = None

# --- These models define the specific parameters for each attack's "params" block ---
class RendererWeightsConfig(BaseModel):
    """Configuration for paths to pre-trained renderer models."""
    ntr: str # Path to the Neural Texture Renderer weights
    # We can add more here later if needed, e.g., for other renderers

class BaseAttackParams(BaseModel):
    img_size: Tuple[int, int] = (480, 640)

class PatchAttackParams(BaseAttackParams):
    type: Literal["patch"]
    shape: Literal["square", "circle"] = "square"
    placement: Literal["on_car", "off_car"] = "off_car"
    relative_size: float = Field(..., gt=0.0, le=1.0)
    learnable_patch_resolution: int = Field(..., gt=0)
    patch_camouflage_lambda: float = Field(..., ge=0.0) 
    position_learning_rate: Optional[float] = None
    translation_limit_pixels: int = Field(..., ge=0)

class Universal2DParams(BaseAttackParams):
    type: Literal["noise","universal_2d"]
    epsilon: float

    _normalize_epsilon = validator('epsilon', pre=True, allow_reuse=True)(parse_epsilon)

class OSFDParams(BaseAttackParams):
    type: Literal["osfd"]
    epsilon: float
    amplification_factor_k: float = Field(..., gt=0)
    feature_layers_to_attack: List[int]

    _normalize_epsilon = validator('epsilon', pre=True, allow_reuse=True)(parse_epsilon)

class SinglePlaneParams(BaseAttackParams):
    type: Literal["single_plane"]
    resolution: int = Field(..., gt=0)
    initialization_type: str = "random"

class TriplaneParams(BaseAttackParams):
    type: Literal["triplane"]
    resolution: int = Field(..., gt=0)
    initialization_type: str = "random"

