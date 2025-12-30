# aegis/attacks/osfd.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import collections
import os

from .base_attack import BaseAttack
from ..utils.helpers import FeatureExtractor, RRBAugmentation

class NoiseOSFDAttack(BaseAttack):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._initialize_learnable_parameters()
        self._initialize_helpers()

    def _initialize_learnable_parameters(self):
        """Creates the universal perturbation as a learnable parameter."""
        params = self.config.attack.params
        img_size = tuple(params.img_size) 
        
        self.universal_perturbation = nn.Parameter(
            torch.zeros(1, 3, img_size[0], img_size[1], device=self.device)
        )
        self.epsilon = params.epsilon

    def _initialize_helpers(self):
        """Initializes the augmentation module."""
        self.rrb_augmentation = RRBAugmentation().to(self.device)
        self.mse_loss = nn.MSELoss()

    def apply_attack(self, clean_images: torch.Tensor, batch: dict, victim_models: dict) -> torch.Tensor:
        """Adds the universal perturbation to the clean images."""
        return torch.clamp(clean_images + self.universal_perturbation, 0, 1)

    def attack_step(self, batch, victim_models):
        """Performs one step of the OSFD attack."""
        params = self.config.attack.params
        img_size = tuple(params.img_size)
        
        # OSFD only works on a single surrogate model
        if len(victim_models) != 1:
            raise ValueError("OSFD attack requires exactly one surrogate model in the config.")
        
        surrogate_name = list(victim_models.keys())[0]
        surrogate_model = victim_models[surrogate_name]['model'].model # Get the nn.Module

        # Prepare clean images
        clean_images = torch.clamp(batch['ref_image'] + batch['background'], 0, 1)
        clean_images = F.interpolate(clean_images, size=img_size, mode='bilinear', align_corners=False)

        # Apply the perturbation
        adversarial_images = self.apply_attack(clean_images, batch, victim_models)

        # We need a new FeatureExtractor for each step to ensure hooks are fresh
        feature_extractor = FeatureExtractor(surrogate_model, params.feature_layers_to_attack)

        with torch.no_grad():
            benign_features = feature_extractor(clean_images)
        
        # Apply augmentation to the adversarial batch
        adversarial_images_aug = self.rrb_augmentation(adversarial_images)
        adv_features_aug = feature_extractor(adversarial_images_aug)

        # Calculate loss by comparing augmented adversarial features to amplified benign features
        total_loss_for_step = torch.tensor(0.0, device=self.device)
        for layer_idx in params.feature_layers_to_attack:
            target_features = params.amplification_factor_k * benign_features[layer_idx]
            adv_features = adv_features_aug[layer_idx]

            if adv_features.shape != target_features.shape:
                # Get H, W from the target feature map
                target_size = target_features.shape[2:]
                # Use bilinear interpolation to differentiably resize
                adv_features = F.interpolate(adv_features, size=target_size, mode='bilinear', align_corners=False)
            
            layer_loss = self.mse_loss(adv_features, target_features)
            total_loss_for_step += layer_loss
        
        # OSFD aims to MAXIMIZE the difference, so we backpropagate the NEGATIVE loss.
        loss_to_backward = -total_loss_for_step

        feature_extractor.remove_hooks()
        
        stats = {
            'total_loss': total_loss_for_step.item(), # Log the positive value for clarity
            'grad_norm': self.universal_perturbation.grad.norm().item() if self.universal_perturbation.grad is not None else 0
        }
        return loss_to_backward, stats

    def post_step_update(self):
        """Enforces the L-infinity norm constraint on the perturbation."""
        with torch.no_grad():
            self.universal_perturbation.data.clamp_(-self.epsilon, self.epsilon)

    def save_artifact(self, path):
        """Saves the final universal perturbation tensor."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.universal_perturbation.detach().cpu(), path)