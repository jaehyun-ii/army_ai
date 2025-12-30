# aegis/models/detectors/victim_factory.py

import torch
from ultralytics import YOLO
from typing import List, Dict, Any

# Optional import for GroundingDINO
_DINO_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    _DINO_AVAILABLE = True
except ImportError:
    print("Warning: 'transformers' library not found. GroundingDINO support is disabled.")

class VictimFactory:
    """A factory class for loading various object detection models."""

    @staticmethod
    def load_models(model_configs: List[Dict[str, Any]], device: torch.device) -> Dict:
        """
        Loads a dictionary of victim models based on a list of configurations.

        Args:
            model_configs: A list of dictionaries, where each dict defines a
                           model's name, type, path, etc.
            device: The torch device to load the models onto.

        Returns:
            A dictionary where keys are model names and values are dictionaries
            containing the loaded model, its config, and any other necessary
            components (like a text processor for DINO).
        """
        victim_models = {}
        print("--- Loading Victim Model Ensemble ---")
        
        for config in model_configs:
            name = config.get('name')
            model_type = config.get('type')
            if not name or not model_type:
                print(f"  [!] Skipping model config due to missing 'name' or 'type': {config}")
                continue

            try:
                if model_type in ['yolo', 'rtdetr']:
                    model = YOLO(config['path']).to(device)
                    model.model.eval()
                    victim_models[name] = {'model': model, 'config': config}
                    print(f"  [+] Loaded ULTRALYTICS '{name}' ({model_type}) successfully.")
                
                elif model_type == 'dino':
                    if not _DINO_AVAILABLE:
                        print(f"  [!] Skipping DINO model '{name}': 'transformers' library not installed.")
                        continue
                    processor = AutoProcessor.from_pretrained(config['path'])
                    model = AutoModelForZeroShotObjectDetection.from_pretrained(config['path']).to(device)
                    model.eval()
                    victim_models[name] = {
                        'model': model,
                        'processor': processor,
                        'config': config
                    }
                    print(f"  [+] Loaded DINO '{name}' successfully.")
                
                else:
                    print(f"  [!] Skipping '{name}': Unsupported model type '{model_type}'.")

            except Exception as e:
                print(f"  [!] CRITICAL: Failed to load '{name}': {e}. Skipping.")
        
        if not victim_models:
            raise RuntimeError("FATAL: No victim models could be loaded. Please check your configuration.")
            
        print("-" * 35)
        return victim_models