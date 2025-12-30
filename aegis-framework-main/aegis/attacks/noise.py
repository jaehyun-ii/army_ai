# aegis/attacks/noise.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .base_attack import BaseAttack
from ..losses.DetectionLoss import DetectionLoss
from ..models.detectors.differentiable_api import get_differentiable_yolo_output, get_differentiable_dino_output
from ..utils.helpers import get_yolo_pseudo_gt

class NoiseAttack(BaseAttack):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._initialize_learnable_parameters()
        self._initialize_helpers()

    def _initialize_learnable_parameters(self):
        params = self.config.attack.params
        img_size = tuple(params.img_size)
        
        self.universal_perturbation = nn.Parameter(
            torch.zeros(1, 3, img_size[0], img_size[1], device=self.device)
        )
        self.epsilon = params.epsilon

    def _initialize_helpers(self):
        self.loss_fn = DetectionLoss(**self.config.attack.loss_params)

    def apply_attack(self, clean_images: torch.Tensor, batch: dict, victim_models: dict) -> torch.Tensor:
        # Create a mask from the pseudo-GT to apply perturbation only on the car
        pseudo_gts = get_yolo_pseudo_gt(clean_images, victim_models, self.device)
        
        if not pseudo_gts:
            print("Warning: Pseudo-GT dictionary is empty. Cannot create a mask for NoiseAttack.")
            return clean_images

        # Get the name of the first model in the dictionary
        first_model_name = next(iter(pseudo_gts))
        gt_boxes_for_mask = pseudo_gts[first_model_name]

        mask = torch.zeros_like(clean_images, device=self.device)
        for b in range(clean_images.shape[0]):
            if torch.sum(gt_boxes_for_mask[b]) > 0:
                x1, y1, x2, y2 = gt_boxes_for_mask[b].int()
                h, w = clean_images.shape[2:]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x1 < x2 and y1 < y2:
                    mask[b, :, y1:y2, x1:x2] = 1.0
        
        return torch.clamp(clean_images + (self.universal_perturbation * mask), 0, 1)

    def attack_step(self, batch, victim_models):
        params = self.config.attack.params
        img_size = tuple(params.img_size)
        
        clean_images = torch.clamp(batch['ref_image'] + batch['background'], 0, 1)
        clean_images = F.interpolate(clean_images, size=img_size, mode='bilinear', align_corners=False)
        
        # We need victim_models inside `apply_attack` for masking, so pass it in the batch dict
        adversarial_images = self.apply_attack(clean_images, batch, victim_models)

        # Re-calculate pseudo_gts for the loss function
        pseudo_gts = get_yolo_pseudo_gt(clean_images, victim_models, self.device)
        
        # Standard ensemble loss calculation
        total_loss_for_step = torch.tensor(0.0, device=self.device)
        step_stats = {}
        for name, model_data in victim_models.items():
            config = model_data['config']
            if name not in pseudo_gts: continue
            
            if config['type'] in ['yolo', 'rtdetr']:
                inference_output = get_differentiable_yolo_output(model_data['model'].model, adversarial_images)
            elif config['type'] == 'dino':
                inference_output = get_differentiable_dino_output(model_data['model'], model_data['processor'], adversarial_images, config['prompt'])
            else: continue
                
            loss_total, _, _ = self.loss_fn(inference_output, pseudo_gts[name], config['car_class_id'])
            total_loss_for_step += loss_total
            step_stats[f'{name}_loss'] = loss_total.item()

        step_stats['total_loss'] = total_loss_for_step.item()
        return total_loss_for_step, step_stats

    def post_step_update(self):
        with torch.no_grad():
            self.universal_perturbation.data.clamp_(-self.epsilon, self.epsilon)

    def save_artifact(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.universal_perturbation.detach().cpu(), path)