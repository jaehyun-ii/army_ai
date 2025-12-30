# REVISED aegis/data/single_image_dataset.py

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List, Union
from ..core.config_models import SinglePoseConfig
from ultralytics import YOLO # Already imported

# --- MODIFIED GUIDE MODEL HELPER ---
_GUIDE_SEG_MODEL = None
def get_guide_segmentation_model(device):
    """
    Loads a YOLOv8 segmentation model for high-quality mask generation.
    This model is only loaded once and when it's actually needed.
    """
    global _GUIDE_SEG_MODEL
    if _GUIDE_SEG_MODEL is None:
        print("Loading guide segmentation model (yolov8s-seg.pt)...")
        # Use the segmentation variant of YOLOv8
        _GUIDE_SEG_MODEL = YOLO('yolov8s-seg.pt').to(device)
    return _GUIDE_SEG_MODEL

class SingleImageDataset(Dataset):
    def __init__(self, data_sources: Union[List[str], List[SinglePoseConfig]], target_size=(640, 480)):
        self.data_sources = data_sources
        self.width, self.height = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.guide_model = None # Lazy-load the segmentation model

    def __len__(self):
        return len(self.data_sources)

    def _load_and_process_mask(self, mask_path: str) -> torch.Tensor:
        # This helper function remains the same
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None: raise IOError(f"Could not read provided mask file: {mask_path}")
        mask_resized = cv2.resize(mask_img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(mask_resized > 0).unsqueeze(0).float()

    # --- THIS IS THE CRUCIAL, UPGRADED FUNCTION ---
    def _generate_mask_from_image(self, img_bgr_resized: np.ndarray) -> torch.Tensor:
        """
        Generates a high-quality segmentation mask when one isn't provided.
        Uses a YOLOv8-seg model to find the largest car and draw its polygon mask.
        """
        if self.guide_model is None:
            self.guide_model = get_guide_segmentation_model(self.device)
        
        with torch.no_grad():
            results = self.guide_model.predict(img_bgr_resized, verbose=False)
        
        mask_np = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Find the car detection with the largest area
        best_instance = None
        max_area = 0
        car_class_index = 2 # COCO index for 'car'
        
        for res in results:
            if res.masks is None: # Handle cases where no objects are found
                continue
                
            for i in range(len(res.boxes)):
                box = res.boxes[i]
                if int(box.cls) == car_class_index:
                    # Calculate area from the bounding box for simplicity to find the largest car
                    x1, y1, x2, y2 = box.xyxy[0]
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        # Store the index of the best instance
                        best_instance = i

        # If a car was found, draw its segmentation mask
        if best_instance is not None:
            # Get the polygon coordinates for the best car instance
            polygon = results[0].masks.xy[best_instance]
            
            # Use OpenCV to draw the filled polygon on our blank mask
            # The coordinates are float, so we need to convert them to int32
            polygon_int = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask_np, [polygon_int], 255)
            
        return torch.from_numpy(mask_np > 0).unsqueeze(0).float()

    def __getitem__(self, idx):
        # The rest of the __getitem__ method remains exactly the same as the previous version.
        # It correctly calls either _load_and_process_mask or the new, improved _generate_mask_from_image.
        source = self.data_sources[idx]
        ref_path, mask_tensor = None, None
        depth_map, extrinsics_inv, intrinsics = np.zeros((self.height, self.width)), np.eye(4), np.eye(3)

        if isinstance(source, SinglePoseConfig):
            ref_path = source.ref_image_path
            if source.depth_map_path: depth_map = np.load(source.depth_map_path)
            if source.extrinsics_inv_path: extrinsics_inv = np.load(source.extrinsics_inv_path)
            if source.intrinsics:
                intrinsics = np.load(source.intrinsics) if isinstance(source.intrinsics, str) else np.array(source.intrinsics)
            if source.mask_path:
                mask_tensor = self._load_and_process_mask(source.mask_path)
        else:
            ref_path = source

        img_bgr = cv2.imread(ref_path)
        if img_bgr is None: raise IOError(f"Could not read ref_image: {ref_path}")
        img_bgr_resized = cv2.resize(img_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if mask_tensor is None:
            mask_tensor = self._generate_mask_from_image(img_bgr_resized)

        depth_map_resized = cv2.resize(depth_map, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_RGB2BGR)
        img_norm = img_rgb.astype(np.float32) / 255.0
        mask_3ch = mask_tensor.numpy().transpose(1, 2, 0)

        return {
            'ref_image': torch.from_numpy(img_norm * mask_3ch).permute(2, 0, 1),
            'background': torch.from_numpy(img_norm * (1 - mask_3ch)).permute(2, 0, 1),
            'mask': mask_tensor,
            'depth_map': torch.from_numpy(depth_map_resized.astype(np.float32)).unsqueeze(0),
            'intrinsics': torch.from_numpy(intrinsics.astype(np.float32)),
            'extrinsics_inv': torch.from_numpy(extrinsics_inv.astype(np.float32)),
        }