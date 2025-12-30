# aegis/core/config.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

from .config_models import (
    SinglePoseConfig, PatchAttackParams, Universal2DParams, OSFDParams,
    SinglePlaneParams, TriplaneParams, RendererWeightsConfig
)

# A Union of all possible attack parameter models.
# Pydantic will use the 'type' field to automatically select the correct one.
AttackParamsUnion = Union[
    PatchAttackParams, Universal2DParams, OSFDParams,
    SinglePlaneParams, TriplaneParams 
]

# Defines the structure of the 'attack' block in the YAML.
class AttackDefinition(BaseModel):
    params: AttackParamsUnion = Field(..., discriminator='type')
    loss_params: Dict[str, Any] = Field(default_factory=dict)
    renderer_weights: Optional[RendererWeightsConfig] = None

class DatasetConfig(BaseModel):
    # The path is now part of this config block
    path: Union[str, List[str], SinglePoseConfig, List[SinglePoseConfig]]
    
    # Target resolution for resizing images, masks, and depth maps
    height: int = 480
    width: int = 640
    
    # Original resolution of the dataset (for correct camera intrinsic scaling)
    original_height: int = 480
    original_width: int = 640
    
    # For NTR_ReRendering_Dataset, the list of target colors can now be configured
    ntr_target_colors: Optional[List[str]] = None

# Defines the structure of the 'training' block in the YAML.
class TrainingConfig(BaseModel):
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-2
    vis_every_n_epochs: Optional[int] = 1
    
# The top-level configuration model for the entire YAML file.
class AttackConfig(BaseModel):
    attack_name: str
    attack: AttackDefinition
    victims: List[Dict[str, Any]]
    dataset: DatasetConfig
    output_path: str
    training: TrainingConfig
    clear_output_dir_on_start: bool = False