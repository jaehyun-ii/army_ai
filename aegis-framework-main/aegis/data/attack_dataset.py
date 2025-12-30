import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
# The original resolution of the collected data
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

class AttackDataset(Dataset):
    def __init__(self, config):
        self.config = config
        base_path = config.path
        gray_path = os.path.join(base_path, 'reference_gray')
        self.ref_dir = os.path.join(gray_path, 'xref')
        self.mask_dir = os.path.join(gray_path, 'xm')
        self.depth_dir = os.path.join(gray_path, 'xd')
        self.extrin_inv_dir = os.path.join(gray_path, 'extrin_inv')
        
        self.file_list = [f for f in os.listdir(self.ref_dir) if f.endswith('.png')]

        intrinsics_path = os.path.join(base_path, "intrinsics.npy")
        try:
            # ### EDIT: Load original intrinsics and scale them to the target resolution ###
            orig_intrinsics = np.load(intrinsics_path)
            scaled_intrinsics = orig_intrinsics.copy()
            scaled_intrinsics[0, 0] *= self.config.width / self.config.original_width
            scaled_intrinsics[1, 1] *= self.config.height / self.config.original_height
            scaled_intrinsics[0, 2] *= self.config.width / self.config.original_width
            scaled_intrinsics[1, 2] *= self.config.height / self.config.original_height
            self.intrinsics = torch.from_numpy(scaled_intrinsics.astype(np.float32))
        except FileNotFoundError:
            print(f"FATAL ERROR: intrinsics.npy not found at {intrinsics_path}"); exit()

        print(f"AttackDataset initialized for {IMAGE_WIDTH}x{IMAGE_HEIGHT}. Found {len(self.file_list)} samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_base = os.path.splitext(file_name)[0]
        
        ref_path = os.path.join(self.ref_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        depth_path = os.path.join(self.depth_dir, f"{file_base}.npy")
        extrin_inv_path = os.path.join(self.extrin_inv_dir, f"{file_base}.npy")
        
        ref_img_bgr = cv2.imread(ref_path)
        ref_img = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        depth_map = np.load(depth_path)
        extrinsics_inv = np.load(extrin_inv_path)

        # ### EDIT: Resize all image-like data to the target resolution ###
        target_size = (self.config.width, self.config.height)
        ref_img = cv2.resize(ref_img, target_size, interpolation=cv2.INTER_AREA)
        mask_img = cv2.resize(mask_img, target_size, interpolation=cv2.INTER_NEAREST)
        depth_map = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_LINEAR)
        
        ref_img_norm = ref_img.astype(np.float32) / 255.0
        mask_3ch = np.expand_dims(mask_img > 0, axis=2).astype(np.float32)
        ref_masked = ref_img_norm * mask_3ch
        background = ref_img_norm * (1 - mask_3ch)
        
        return {
            'ref_image': torch.from_numpy(ref_masked).permute(2, 0, 1),
            'background': torch.from_numpy(background).permute(2, 0, 1),
            'depth_map': torch.from_numpy(depth_map.astype(np.float32)).unsqueeze(0),
            'intrinsics': self.intrinsics, # Return the pre-scaled intrinsics
            'extrinsics_inv': torch.from_numpy(extrinsics_inv.astype(np.float32)),
        }

class AttackDatasetPatch(AttackDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        file_name = self.file_list[idx]
        mask_path = os.path.join(self.mask_dir, file_name)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_size = (self.config.width, self.config.height)
        mask_img = cv2.resize(mask_img, target_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask_img > 0).unsqueeze(0).float()
        data['mask'] = mask
        return data
