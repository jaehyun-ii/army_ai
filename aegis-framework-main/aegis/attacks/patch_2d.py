# aegis/attacks/patch_2d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
import os

from .base_attack import BaseAttack
from ..losses.DetectionLoss import DetectionLoss
from ..models.detectors.differentiable_api import get_differentiable_yolo_output, get_differentiable_dino_output
from ..models.renderers.PatchApplier import DifferentiablePasteApplier
from ..utils.helpers import get_yolo_pseudo_gt, DirectPastePatchApplier

class PatchAttack(BaseAttack):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._initialize_learnable_parameters()
        self._initialize_helpers()

    def _initialize_learnable_parameters(self):
        params = self.config.attack.params
        res = params.learnable_patch_resolution
        self.adv_patch = nn.Parameter(torch.rand(1, 3, res, res, device=self.device))

    def _initialize_helpers(self):
        params = self.config.attack.params
        self.loss_fn = DetectionLoss(**self.config.attack.loss_params)
        self.patch_applier = DifferentiablePasteApplier(
            placement=params.placement,
            relative_size=params.relative_size,
            translation_limit_pixels=params.translation_limit_pixels
        ).to(self.device)

    def get_optimizer_configs(self) -> List[Dict[str, Any]]:
        """Override to provide two separate optimizer configurations."""
        params = self.config.attack.params
        
        # Config for the patch texture (uses default LR)
        patch_config = {'name': 'patch', 'params': [self.adv_patch]}
        
        # Config for the position network
        pos_config = {'name': 'position', 'params': list(self.patch_applier.parameters())}
        if params.position_learning_rate:
            pos_config['lr'] = params.position_learning_rate
        
        return [patch_config, pos_config]

    def apply_attack(self, clean_images: torch.Tensor, batch: dict, victim_models: dict) -> torch.Tensor:
        # Patch is applied to full-resolution images
        return self.patch_applier(clean_images, self.adv_patch, batch['mask'])

    def attack_step(self, batch, victim_models):
        params = self.config.attack.params
        img_size = tuple(params.img_size)

        clean_images = torch.clamp(batch['ref_image'] + batch['background'], 0, 1)
        
        adversarial_images_full_res, final_mask, transformed_patch = self.apply_attack(clean_images, batch, victim_models)
        adversarial_images_yolo = F.interpolate(adversarial_images_full_res, size=img_size, mode='bilinear', align_corners=False)

        clean_images_yolo = F.interpolate(clean_images, size=img_size, mode='bilinear', align_corners=False)
        pseudo_gts = get_yolo_pseudo_gt(clean_images_yolo, victim_models, self.device)
        
        # Standard ensemble loss calculation
        total_loss_for_step = torch.tensor(0.0, device=self.device)
        step_stats = {}
        for name, model_data in victim_models.items():
            config = model_data['config']
            if name not in pseudo_gts: continue
            
            if config['type'] in ['yolo', 'rtdetr']:
                inference_output = get_differentiable_yolo_output(model_data['model'].model, adversarial_images_yolo)
            elif config['type'] == 'dino':
                inference_output = get_differentiable_dino_output(model_data['model'], model_data['processor'], adversarial_images_yolo, config['prompt'])
            else: continue
                
            loss_total, _, _ = self.loss_fn(inference_output, pseudo_gts[name], config['car_class_id'])
            total_loss_for_step += loss_total
            step_stats[f'{name}_loss'] = loss_total.item()
        
        target_background = batch['background'] * final_mask
        patch_pixels = transformed_patch * final_mask
        camouflage_loss = F.mse_loss(patch_pixels, target_background)

        total_loss_for_step += params.patch_camouflage_lambda * camouflage_loss
        step_stats['total_loss'] = total_loss_for_step.item()
        return total_loss_for_step, step_stats

    def post_step_update(self):
        with torch.no_grad():
            self.adv_patch.data.clamp_(0, 1)

    def save_artifact(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save patch and config together as in original script
        attack_config = {
            'patch_shape': self.config.attack.params.shape,
            'placement': self.config.attack.params.placement,
            'relative_size': self.config.attack.params.relative_size,
            'learnable_patch_resolution': self.config.attack.params.learnable_patch_resolution,
        }
        data_to_save = {
            'patch': self.adv_patch.detach().cpu(),
            'config': attack_config
        }
        torch.save(data_to_save, path)