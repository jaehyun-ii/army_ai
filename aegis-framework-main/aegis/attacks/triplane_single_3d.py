# aegis/attacks/triplane_single_3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os

from .base_attack import BaseAttack
from ..models.renderers.ntr import NTR
from ..models.renderers.tpm import TriplanarMapping 
from ..losses.DetectionLoss import DetectionLoss 
from ..models.detectors.differentiable_api import get_differentiable_yolo_output
from ..utils.helpers import get_yolo_pseudo_gt

class TriplaneSingleAttack(BaseAttack):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._initialize_learnable_parameters()
        self._initialize_helpers()

    def _initialize_learnable_parameters(self):
        """Creates the single adversarial texture plane."""
        params = self.config.attack.params
        size = params.resolution
        init_type = params.initialization_type

        if init_type == 'random':
            texture = torch.rand(1, 3, size, size, device=self.device)
        elif os.path.isfile(init_type):
            print(f"Loading initial texture from: {init_type}")
            texture = torch.load(init_type, map_location=self.device)
            if texture.dim() == 3:
                texture = texture.unsqueeze(0)
            if texture.shape[2:] != (size, size):
                texture = F.interpolate(texture, size=(size, size), mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unknown initialization type or invalid file path: '{init_type}'")

        self.adv_texture = nn.Parameter(texture)

    def _initialize_helpers(self):
        """Loads the NTR, TPM, and the loss function."""
        # --- Load Renderers ---
        ntr_weights_path = self.config.attack.renderer_weights.ntr
        self.ntr_model = NTR().to(self.device)
        self.ntr_model.load_state_dict(torch.load(ntr_weights_path, map_location=self.device))
        self.ntr_model.eval()

        self.tpm = TriplanarMapping().to(self.device)
        self.tpm.eval()

        self.ntr_model.eval()
        for param in self.ntr_model.parameters():
            param.requires_grad = False
    
        self.tpm = TriplanarMapping().to(self.device)
        self.tpm.eval()
        for param in self.tpm.parameters():
             param.requires_grad = False

        # --- Load Loss Function ---
        self.loss_fn = DetectionLoss(**self.config.attack.loss_params)

    def apply_attack(self, clean_images: torch.Tensor, batch: dict, victim_models: dict) -> torch.Tensor:
        """Renders the car with the adversarial single-plane texture."""
        params = self.config.attack.params
        
        # This TPM doesn't take random transform params in its forward pass, so we don't need them.
        projected_texture = self.tpm(
            self.adv_texture,
            batch['depth_map'], batch['intrinsics'], batch['extrinsics_inv']
        )
        rendered_cars = self.ntr_model(batch['ref_image'], projected_texture)
        
        adversarial_images = torch.clamp(rendered_cars + batch['background'], 0, 1)
        return F.interpolate(adversarial_images, size=tuple(params.img_size), mode='bilinear', align_corners=False)

    def attack_step(self, batch, victim_models):
        """Performs one full optimization step."""
        # 1. Apply the attack
        adv_yolo = self.apply_attack(None, batch, victim_models)
        
        # 2. Get pseudo ground-truth
        clean_images_yolo = F.interpolate(
            torch.clamp(batch['ref_image'] + batch['background'], 0, 1),
            size=tuple(self.config.attack.params.img_size), mode='bilinear', align_corners=False
        )
        pseudo_gts = get_yolo_pseudo_gt(clean_images_yolo, victim_models, self.device)

        # 3. Calculate ensemble loss
        total_loss_for_step = torch.tensor(0.0, device=self.device)
        step_stats = {}

        for name, model_data in victim_models.items():
            config = model_data['config']
            if name not in pseudo_gts: continue
            
            # This attack was originally YOLO-only, but we can easily support more
            if config['type'] in ['yolo', 'rtdetr']:
                inference_output = get_differentiable_yolo_output(model_data['model'].model, adv_yolo)
            else:
                print(f"Warning: SinglePlaneAttack currently only supports YOLO-family models. Skipping {name}.")
                continue
            
            # The original loss function needs the target class index passed in
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
            self.adv_texture.data.clamp_(0, 1)

    def save_artifact(self, path):
        """Saves the final adversarial texture tensor."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.adv_texture.detach().cpu(), path)