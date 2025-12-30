import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import random


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
# The original resolution of the collected data
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640

class NTR_ReRendering_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        base_path = config.path
        ref_gray_path = os.path.join(base_path, 'reference_gray')
        self.ref_dir = os.path.join(ref_gray_path, 'xref')
        self.mask_dir = os.path.join(ref_gray_path, 'xm')
        self.depth_dir = os.path.join(ref_gray_path, 'xd')
        self.extrin_inv_dir = os.path.join(ref_gray_path, 'extrin_inv')

        intrinsics_path = os.path.join(base_path, "intrinsics.npy")
        try:
            # ### EDIT: Load original intrinsics and scale them to the target resolution ###
            orig_intrinsics = np.load(intrinsics_path)
            scaled_intrinsics = orig_intrinsics.copy()
            # Scale fx, fy (focal lengths)
            scaled_intrinsics[0, 0] *= IMAGE_WIDTH / ORIGINAL_WIDTH
            scaled_intrinsics[1, 1] *= IMAGE_HEIGHT / ORIGINAL_HEIGHT
            # Scale cx, cy (principal point)
            scaled_intrinsics[0, 2] *= IMAGE_WIDTH / ORIGINAL_WIDTH
            scaled_intrinsics[1, 2] *= IMAGE_HEIGHT / ORIGINAL_HEIGHT
            self.intrinsics = torch.from_numpy(scaled_intrinsics.astype(np.float32))
        except FileNotFoundError:
            print(f"FATAL ERROR: intrinsics.npy not found at {intrinsics_path}."); exit()

        default_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black']
        self.target_colors = self.config.ntr_target_colors if self.config.ntr_target_colors is not None else default_colors
        self.target_dirs = {color: os.path.join(base_path, f'target_{color}', 'xref') for color in self.target_colors}
        self.target_rgb = {
            'red':[1,0,0],'green':[0,1,0],'blue':[0,0,1],'cyan':[0,1,1],'magenta':[1,0,1],'yellow':[1,1,0],'white':[1,1,1],'black':[0,0,0]
        }
        
        self.samples = []
        file_list = [f for f in os.listdir(self.ref_dir) if f.endswith('.png')]
        for file_name in file_list:
            for color in self.target_colors:
                self.samples.append((file_name, color))
        
        print(f"NTR Dataset initialized for {IMAGE_WIDTH}x{IMAGE_HEIGHT} training. Found {len(file_list)} poses, {len(self.samples)} total samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, target_color = self.samples[idx]
        file_base = os.path.splitext(file_name)[0]

        ref_path = os.path.join(self.ref_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        depth_path = os.path.join(self.depth_dir, f"{file_base}.npy")
        extrin_inv_path = os.path.join(self.extrin_inv_dir, f"{file_base}.npy")
        gt_path = os.path.join(self.target_dirs[target_color], file_name)

        ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        depth_map = np.load(depth_path)
        extrinsics_inv = np.load(extrin_inv_path)
        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)

        if ref_img is None or mask_img is None or gt_img is None:
            return self.__getitem__(random.randint(0, len(self)-1))

        # ### EDIT: Resize all image-like data to the target resolution ###
        target_size = (self.config.width, self.config.height)
        ref_img = cv2.resize(ref_img, target_size, interpolation=cv2.INTER_AREA)
        gt_img = cv2.resize(gt_img, target_size, interpolation=cv2.INTER_AREA)
        mask_img = cv2.resize(mask_img, target_size, interpolation=cv2.INTER_NEAREST)
        depth_map = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_LINEAR)

        ref_img_norm = ref_img.astype(np.float32) / 255.0
        gt_img_norm = gt_img.astype(np.float32) / 255.0
        mask_3ch = np.expand_dims(mask_img > 0, axis=2).astype(np.float32)
        
        ref_masked = ref_img_norm * mask_3ch
        gt_masked = gt_img_norm * mask_3ch
        
        texture_color_rgb = self.target_rgb[target_color]
        exp_texture_np = np.ones((64, 64, 3), dtype=np.float32) * np.array(texture_color_rgb, dtype=np.float32)

        return {
            'ref_image': torch.from_numpy(ref_masked).permute(2, 0, 1),
            'exp_texture': torch.from_numpy(exp_texture_np).permute(2, 0, 1),
            'ground_truth': torch.from_numpy(gt_masked).permute(2, 0, 1),
            'depth_map': torch.from_numpy(depth_map.astype(np.float32)).unsqueeze(0),
            'intrinsics': self.intrinsics, # Return the pre-scaled intrinsics
            'extrinsics_inv': torch.from_numpy(extrinsics_inv.astype(np.float32)),
        }
