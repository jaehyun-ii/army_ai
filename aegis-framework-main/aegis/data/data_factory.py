# DEFINITIVE FIX for aegis/data/data_factory.py

import os
import torch
from torch.utils.data import DataLoader, Dataset
from .attack_dataset import AttackDataset, AttackDatasetPatch
from .single_image_dataset import SingleImageDataset
from ..core.config_models import SinglePoseConfig # Import the Pydantic model

class DataLoaderFactory:
    @staticmethod
    def create(config) -> (DataLoader, Dataset):
        dataset_config = config.dataset
        attack_params = config.attack.params
        batch_size = config.training.batch_size
        
        # This is the Union[str, List[str], SinglePoseConfig, List[SinglePoseConfig]] field
        path_config = dataset_config.path
        
        data_sources = []
        is_directory_dataset = False
        dataset_instance = None

        # --- REVISED AND SIMPLIFIED LOGIC ---

        if isinstance(path_config, str):
            # The path is a string. Check if it's a directory or a file.
            if os.path.isdir(path_config):
                # Case 1: Full dataset directory
                print(f"Loading data from directory: {path_config}")
                if attack_params.type == 'patch':
                    dataset_instance = AttackDatasetPatch(config=dataset_config)
                else:
                    dataset_instance = AttackDataset(config=dataset_config)
                is_directory_dataset = True
            elif os.path.isfile(path_config):
                # Case 2: A single image file path string (for 2D attacks)
                print(f"Loading data from single image file: {path_config}")
                if 'plane' in attack_params.type:
                    raise ValueError(f"Attack type '{attack_params.type}' requires 3D data. Use a pose dictionary/config.")
                data_sources = [path_config]
            else:
                raise FileNotFoundError(f"The specified dataset path string does not exist: {path_config}")

        elif isinstance(path_config, SinglePoseConfig):
            # Case 3: A single, already-parsed SinglePoseConfig object
            print("Loading data from a single pose configuration object.")
            data_sources = [path_config]
            
        elif isinstance(path_config, list):
            if not path_config: raise ValueError("`dataset.path` list cannot be empty.")
            
            # The Pydantic model ensures all items in the list are of the same parsed type
            if isinstance(path_config[0], str):
                # Case 4: A list of image path strings
                print(f"Loading data from a list of {len(path_config)} image files.")
                if 'plane' in attack_params.type:
                    raise ValueError(f"Attack type '{attack_params.type}' requires 3D data. Use a list of pose dictionaries.")
                data_sources = path_config
            elif isinstance(path_config[0], SinglePoseConfig):
                # Case 5: A list of pose objects
                print(f"Loading data from a list of {len(path_config)} pose configurations.")
                data_sources = path_config
            else:
                # This case should ideally not be reachable due to Pydantic validation
                raise TypeError("List elements in `dataset.path` have an unexpected type.")
        
        else:
             # This case should also not be reachable
             raise TypeError(f"The `dataset.path` field has an unsupported type: {type(path_config)}")

        # --- Finalize DataLoader creation ---
        if is_directory_dataset:
            return DataLoader(dataset_instance, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True), dataset_instance
        else:
            target_size = (dataset_config.width, dataset_config.height)
            dataset_instance = SingleImageDataset(data_sources=data_sources, target_size=target_size)
            return DataLoader(dataset_instance, batch_size=min(batch_size, len(data_sources)), shuffle=False), dataset_instance