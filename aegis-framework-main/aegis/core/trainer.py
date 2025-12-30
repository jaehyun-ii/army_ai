# aegis/core/trainer.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil
import collections

# Import from our framework's modules
from .config import AttackConfig
from ..models.detectors.victim_factory import VictimFactory
from ..attacks.base_attack import BaseAttack
from ..data.data_factory import DataLoaderFactory
from ..utils.visualization import visualize_proof 

class Trainer:
    """
    The core training class that orchestrates the adversarial attack process.
    """
    def __init__(self, config: AttackConfig):
        """
        Initializes the Trainer.

        Args:
            config: The validated Pydantic configuration object for the attack.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- 1. Load Victim Models ---
        self.victim_models = VictimFactory.load_models(self.config.victims, self.device)

        # --- 2. Instantiate the Attack Strategy ---
        self.attack_strategy: BaseAttack = self._get_attack_strategy()

        # --- 3. Setup Dataset ---
        self.data_loader, self.dataset = DataLoaderFactory.create(self.config)
        # Get a fixed sample for consistent visualization
        self.vis_sample = self.dataset[0] 

        self.optimizers = {}
        self.schedulers = {}
        
        optimizer_configs = self.attack_strategy.get_optimizer_configs()
        print("--- Initializing Optimizers ---")
        
        for opt_config in optimizer_configs:
            name = opt_config.get('name', 'default')
            # Use the specific LR if provided, otherwise fall back to the main training LR
            lr = opt_config.get('lr', self.config.training.learning_rate)
            
            self.optimizers[name] = optim.AdamW(opt_config['params'], lr=lr)
            self.schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers[name],
                T_max=self.config.training.epochs,
                eta_min=1e-5
            )
            print(f"  - Optimizer '{name}' initialized with LR: {lr}")
        
        # --- 5. Prepare Output Directory ---
        self.output_dir = os.path.dirname(self.config.output_path)

        if self.config.clear_output_dir_on_start:
            if os.path.isdir(self.output_dir):
                print(f"WARNING: `clear_output_dir_on_start` is True. Deleting directory: {self.output_dir}")
                shutil.rmtree(self.output_dir)
            else:
                print(f"INFO: `clear_output_dir_on_start` is True, but directory does not exist yet: {self.output_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved in: {self.output_dir}")

    def _get_dataset(self):
        attack_type = self.config.attack.params.type
        if attack_type == 'patch':
            return AttackDatasetPatch(base_path=self.config.dataset_path)
        else:
            return AttackDataset(base_path=self.config.dataset_path)
        

    def _get_attack_strategy(self) -> BaseAttack:
        """
        Factory method to instantiate the correct attack class based on the config.
        """
        attack_type = self.config.attack.params.type
        print(f"Initializing attack strategy: {attack_type}")
        
        if attack_type == 'triplane':
            from ..attacks.triplane_3d import TriplaneAttack
            return TriplaneAttack(self.config, self.device)
        elif attack_type == 'single_plane' or attack_type == 'active':
            from ..attacks.triplane_single_3d import TriplaneSingleAttack
            return TriplaneSingleAttack(self.config, self.device)
        elif attack_type == 'patch':
            from ..attacks.patch_2d import PatchAttack
            return PatchAttack(self.config, self.device)
        elif attack_type == 'noise':
            from ..attacks.noise import NoiseAttack
            return NoiseAttack(self.config, self.device)
        elif attack_type == 'osfd' or attack_type == 'noise_osfd':
            from ..attacks.noise_osfd import NoiseOSFDAttack
            return NoiseOSFDAttack(self.config, self.device)
        else:
            raise ValueError(f"Unknown attack type specified in config: '{attack_type}'")

    def train(self):
        """
        The main training loop.
        """
        print(f"--- Starting Attack: {self.config.attack_name} ---")
        
        # Initial visualization before training
        visualize_proof(-1, self.attack_strategy, self.victim_models, self.vis_sample, self.device, self.config)

        for epoch in range(self.config.training.epochs):
            epoch_stats = collections.defaultdict(list)
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.config.training.epochs}")

            for batch in pbar:
                # Move all tensors in the batch to the correct device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # --- The Core Attack Step ---
                # All complex logic is delegated to the attack strategy object
                loss, step_stats = self.attack_strategy.attack_step(batch, self.victim_models)
                
                # --- Standard Optimization ---
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                    
                loss.backward()
                
                grad_norm = self.attack_strategy.calculate_gradient_norm()
                step_stats['grad_norm'] = grad_norm # Add it to our stats for this step
                
                grad_stats = {}
                for name in self.optimizers.keys():
                    grad_norm = self.attack_strategy.calculate_gradient_norm(optimizer_name=name)
                    step_stats[f'{name}_grad_norm'] = grad_norm
                    grad_stats[f'{name}_grad'] = f"{grad_norm:.4f}"

                for optimizer in self.optimizers.values():
                    optimizer.step()
                
                # --- Post-Optimization Update ---
                # e.g., clamping patch values
                self.attack_strategy.post_step_update()

                # --- Logging ---
                for key, val in step_stats.items():
                    epoch_stats[key].append(val)
                pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.4f}")

            # --- End of Epoch ---
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Log average stats for the epoch
            avg_loss = sum(epoch_stats['total_loss']) / len(epoch_stats['total_loss'])
            avg_grad_norm = sum(epoch_stats['grad_norm']) / len(epoch_stats['grad_norm'])
                                                                
            print(f"\n--- Epoch {epoch + 1} Finished ---")
            print(f"  Average Total Loss: {avg_loss:.4f}")
            print(f"  Average Grad Norm: {avg_grad_norm:.4f}")
            # You can add more detailed logging here for individual model losses if returned in step_stats
            
            # Generate and save proof-of-concept visualization
            vis_freq = self.config.training.vis_every_n_epochs
            if vis_freq and (epoch + 1) % vis_freq == 0:
                print(f"Generating visualization for epoch {epoch + 1}...")
                visualize_proof(epoch, self.attack_strategy, self.victim_models, self.vis_sample, self.device, self.config)

        # --- Attack Finished ---
        print("--- Attack Finished ---")
        final_artifact_path = self.config.output_path
        self.attack_strategy.save_artifact(final_artifact_path)
        print(f"Adversarial artifact saved to: {final_artifact_path}")