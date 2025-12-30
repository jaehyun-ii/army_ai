# aegis/attacks/triplane_3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os

from .base_attack import BaseAttack
from ..models.renderers.ntr import NTR
from ..models.renderers.tpm import TriPlaneSampler
from ..losses.DetectionLoss import DetectionLoss
from ..models.detectors.differentiable_api import get_differentiable_yolo_output, get_differentiable_dino_output
from ..utils.helpers import get_yolo_pseudo_gt 

class TriplaneAttack(BaseAttack):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._initialize_learnable_parameters()
        self._initialize_helpers()

    def _initialize_learnable_parameters(self):
        """Creates the three adversarial texture planes."""
        params = self.config.attack.params
        size = params.resolution
        init_type = params.initialization_type

        if init_type == 'random':
            base_texture = torch.rand(1, 3, size, size, device=self.device)
        elif init_type.endswith('.pt'):
            print(f"Warning: Loading from file '{init_type}' is not fully implemented yet. Using random.")
            base_texture = torch.rand(1, 3, size, size, device=self.device)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        self.adv_textures = nn.ParameterDict({
            'xy': nn.Parameter(base_texture.clone()),
            'yz': nn.Parameter(base_texture.clone()),
            'xz': nn.Parameter(base_texture.clone()),
        })

    def _initialize_helpers(self):
        """Loads the NTR, TPM, and the loss function."""
        # --- Load Renderers ---
        ntr_weights_path = self.config.attack.renderer_weights.ntr
        self.ntr_model = NTR().to(self.device)
        self.ntr_model.load_state_dict(torch.load(ntr_weights_path, map_location=self.device))
        self.ntr_model.eval()

        for param in self.ntr_model.parameters():
            param.requires_grad = False

        self.tpm = TriPlaneSampler().to(self.device)
        self.tpm.eval()

        for param in self.tpm.parameters():
             param.requires_grad = False

        # --- Load Loss Function ---
        self.loss_fn = DetectionLoss(**self.config.attack.loss_params)

    def apply_attack(self, clean_images: torch.Tensor, batch: dict, victim_models: dict) -> torch.Tensor:
        """Renders the car with the adversarial triplane textures."""
        params = self.config.attack.params
        
        # Apply random transformations for robustness
        texture_params = {
            'roll': random.uniform(-20, 20), 'pitch': random.uniform(-20, 20),
            'yaw': random.uniform(-20, 20), 'scale': random.uniform(0.8, 1.2),
            'x_shift': random.uniform(-0.15, 0.15), 'y_shift': random.uniform(-0.15, 0.15)
        }
        
        projected_texture = self.tpm(
            self.adv_textures['xy'], self.adv_textures['yz'], self.adv_textures['xz'],
            batch['depth_map'], batch['intrinsics'], batch['extrinsics_inv'],
            texture_transform_params=texture_params
        )
        rendered_cars = self.ntr_model(batch['ref_image'], projected_texture)
        
        # Combine with background and resize to the standard size for detectors
        adversarial_images = torch.clamp(rendered_cars + batch['background'], 0, 1)
        target_size = (self.config.dataset.height, self.config.dataset.width)
        return F.interpolate(adversarial_images, size=target_size, mode='bilinear', align_corners=False)

    def attack_step(self, batch, victim_models):
        """Performs one full optimization step."""
        # 1. Apply the attack to get adversarial images
        adv_yolo = self.apply_attack(None, batch, victim_models)
        
        # 2. Get pseudo ground-truth boxes from the clean image
        # This helper function encapsulates the logic to avoid repetition
        target_size = (self.config.dataset.height, self.config.dataset.width)
        clean_images_yolo = F.interpolate(
            torch.clamp(batch['ref_image'] + batch['background'], 0, 1),
            size=target_size, mode='bilinear', align_corners=False
        )
        pseudo_gts = get_yolo_pseudo_gt(clean_images_yolo, victim_models, self.device)

        # 3. Calculate loss across the ensemble
        total_loss_for_step = torch.tensor(0.0, device=self.device)
        step_stats = {}

        for name, model_data in victim_models.items():
            config = model_data['config']
            if name not in pseudo_gts: continue

            # Get the correct differentiable output based on model type
            if config['type'] == 'yolo' or config['type'] == 'rtdetr':
                inference_output = get_differentiable_yolo_output(model_data['model'].model, adv_yolo)
            elif config['type'] == 'dino':
                inference_output = get_differentiable_dino_output(
                    model_data['model'], model_data['processor'], adv_yolo, config['prompt']
                )
            else:
                continue
            
            # Calculate loss for this model
            loss_total, _, _ = self.loss_fn(
                inference_output, pseudo_gts[name], config['car_class_id']
            )
            total_loss_for_step += loss_total
            step_stats[f'{name}_loss'] = loss_total.item()

        step_stats['total_loss'] = total_loss_for_step.item()
        return total_loss_for_step, step_stats

    def post_step_update(self):
        """Clamps the texture values to the valid [0, 1] range."""
        with torch.no_grad():
            for tex in self.adv_textures.values():
                tex.data.clamp_(0, 1)

    def save_artifact(self, path):
        """Saves the dictionary of texture planes."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data_to_save = {key: val.detach().cpu() for key, val in self.adv_textures.items()}
        torch.save(data_to_save, path)