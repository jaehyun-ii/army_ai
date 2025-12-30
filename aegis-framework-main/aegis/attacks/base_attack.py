# aegis/attacks/base_attack.py

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

class BaseAttack(ABC, nn.Module):
    """
    Abstract Base Class for all adversarial attack strategies.

    This class defines the interface that the core Trainer expects for any
    given attack. Each specific attack (e.g., PatchAttack, TriplaneAttack)
    must implement the abstract methods defined here.
    """
    def __init__(self, config, device: torch.device):
        """
        Initializes the attack strategy.

        Args:
            config: The validated Pydantic configuration object for the attack.
            device: The torch device (e.g., 'cuda' or 'cpu') to run on.
        """
        super().__init__()
        self.config = config
        self.device = device
        
        # These methods must be called by the concrete implementation's __init__
        # self._initialize_learnable_parameters()
        # self._initialize_helpers()

    @abstractmethod
    def _initialize_learnable_parameters(self):
        """
        Creates the nn.Parameter(s) that will be optimized during the attack.
        For example, this is where the adversarial patch tensor or the
        3D texture planes would be created.
        """
        pass

    @abstractmethod
    def _initialize_helpers(self):
        """
        Loads any necessary components for the attack, such as renderers
        (NTR, TPM), patch appliers, or specific loss functions.
        This is called once at the beginning of the training process.
        """
        pass
        
    @abstractmethod
    def apply_attack(self, clean_images: torch.Tensor, batch: Dict, victim_models: Dict) -> torch.Tensor:
        """
        Applies the adversarial pattern to a batch of clean images.

        Args:
            clean_images: A tensor of clean, original images.
            batch: The full data batch, which may contain additional information
                   like masks, depth maps, etc., needed for the attack.
            victim_models: A dictionary of loaded victim models. This is required
                           for attacks that need a guide model for placement (e.g., noise).

        Returns:
            A tensor of adversarial images ready to be fed to victim models.
        """
        pass

    def get_learnable_parameters(self) -> List[nn.Parameter]:
        """
        Returns a list of the parameters that the Trainer should pass to the optimizer.
        
        Returns:
            A list of nn.Parameter objects that require gradients.
        """
        # This default implementation works for most cases.
        return [p for p in self.parameters() if p.requires_grad]

    def get_optimizer_configs(self) -> List[Dict[str, Any]]:
        """
        Returns a list of configurations for one or more optimizers.
        Each configuration is a dictionary containing 'params' and an optional 'lr'.
        If 'lr' is not provided, the default learning rate from the main config is used.
        
        Example for a single optimizer:
            return [{'params': self.parameters()}]
        
        Example for multiple optimizers:
            return [
                {'name': 'patch', 'params': [self.adv_patch]},
                {'name': 'position', 'params': self.applier.parameters(), 'lr': 1e-4}
            ]
        """
        return [{'name': 'default', 'params': list(p for p in self.parameters() if p.requires_grad)}]

    def calculate_gradient_norm(self, optimizer_name: str = None) -> float:
        """
        Calculates the L2 norm of gradients for all learnable parameters,
        or for a specific optimizer group if a name is provided.
        """
        total_norm = 0.0
        params_to_check = []
        
        optimizer_configs = self.get_optimizer_configs()
        if optimizer_name:
            # Find the specific parameter group
            for config in optimizer_configs:
                if config.get('name') == optimizer_name:
                    params_to_check = config['params']
                    break
        else:
            # Default to all parameters if no name is given
            for config in optimizer_configs:
                params_to_check.extend(config['params'])

        if not params_to_check:
             # Fallback for old attacks that don't name their optimizers
             params_to_check = self.get_learnable_parameters()

        for p in params_to_check:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** 0.5

    @abstractmethod
    def attack_step(self, batch: Dict, victim_models: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Performs one full optimization step for a given batch of data.

        This method should orchestrate:
        1. Applying the attack to generate adversarial images.
        2. Calculating the total loss across the ensemble of victim models.
        
        Args:
            batch: The current batch of data from the DataLoader.
            victim_models: A dictionary of loaded victim models.

        Returns:
            A tuple containing:
            - The final loss tensor for this step (for backpropagation).
            - A dictionary of statistics to be logged (e.g., {'total_loss': 1.23}).
        """
        pass

    @abstractmethod
    def post_step_update(self):
        """
        Performs any necessary actions after the optimizer's step.
        The most common use is clamping the values of the learnable parameters
        (e.g., ensuring pixel values of a patch remain between 0 and 1).
        """
        pass

    @abstractmethod
    def save_artifact(self, path: str):
        """
        Saves the final, trained adversarial artifact to a specified path.

        Args:
            path: The file path where the artifact should be saved.
        """
        pass