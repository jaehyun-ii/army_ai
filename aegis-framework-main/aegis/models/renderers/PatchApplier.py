import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import os


class PlacementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 13 * 18, 128), nn.ReLU(True),
            nn.Linear(128, 2)
        )
        self.fc[-1].weight.data.normal_(mean=0.0, std=0.01)
        self.fc[-1].bias.data.zero_()

    def forward(self, x):
        xs = self.conv(x)
        xs = xs.view(xs.size(0), -1)
        return torch.tanh(self.fc(xs))

class DifferentiablePasteApplier(nn.Module):
    def __init__(self, placement='on_car', relative_size=0.4, translation_limit_pixels=50):
        super().__init__()
        self.placement = placement
        self.relative_size = relative_size
        self.translation_limit = translation_limit_pixels
        self.placement_net = PlacementNet()

    def forward(self, image_batch, patch, car_mask_batch):
        bs, _, img_h, img_w = image_batch.shape
        device = image_batch.device

        constraint_mask_b = car_mask_batch if self.placement == 'on_car' else 1.0 - car_mask_batch
        loc_net_input = torch.cat([image_batch, constraint_mask_b], dim=1)
        
        # This now returns a value already in the range [-1, 1]
        normalized_offset_params = self.placement_net(loc_net_input)

        patch_canvases_b, mask_canvases_b, thetas = [], [], []

        for i in range(bs):
            car_mask_i = car_mask_batch[i]
            if not torch.any(car_mask_i):
                patch_canvases_b.append(torch.zeros(1, 3, img_h, img_w, device=device))
                mask_canvases_b.append(torch.zeros(1, 1, img_h, img_w, device=device))
                thetas.append(torch.eye(2, 3, device=device))
                continue

            rows, cols = torch.where(car_mask_i[0])
            ymin, ymax, xmin, xmax = rows.min(), rows.max(), cols.min(), cols.max()
            bbox_h, bbox_w = ymax - ymin, xmax - xmin
            
            patch_target_size = int(torch.min(bbox_h, bbox_w).float() * self.relative_size)
            patch_target_size = max(patch_target_size, 2)
            
            resized_patch_4d = F.interpolate(patch, size=(patch_target_size, patch_target_size), mode='bilinear', align_corners=False)
            patch_silhouette = torch.ones_like(resized_patch_4d[:, :1, :, :])

            y_pad_start, x_pad_start = (img_h - patch_target_size) // 2, (img_w - patch_target_size) // 2
            padding = (x_pad_start, img_w - (x_pad_start + patch_target_size), y_pad_start, img_h - (y_pad_start + patch_target_size))
            
            p_canvas, m_canvas = F.pad(resized_patch_4d, padding), F.pad(patch_silhouette, padding)
            patch_canvases_b.append(p_canvas)
            mask_canvases_b.append(m_canvas)

            if self.placement == 'on_car':
                center_y_px, center_x_px = rows.float().mean(), cols.float().mean()
            else:
                center_y_px = ymax.float() - (patch_target_size / 2)
                center_x_px = cols.float().mean() - (bbox_w / 2) - (patch_target_size / 2)

            # The tanh is gone from here. We just scale the normalized offset.
            offset_x = self.translation_limit * normalized_offset_params[i, 0]
            offset_y = self.translation_limit * normalized_offset_params[i, 1]
            final_center_x = center_x_px + offset_x
            final_center_y = center_y_px + offset_y

            displacement_x = final_center_x - (img_w / 2)
            displacement_y = final_center_y - (img_h / 2)
            tx, ty = displacement_x / (img_w / 2), displacement_y / (img_h / 2)
            
            ident = torch.eye(2, 2, device=device)
            trans = torch.stack([-tx, -ty]).unsqueeze(1)
            theta = torch.cat([ident, trans], dim=1)
            thetas.append(theta)

        patch_canvases_b = torch.cat(patch_canvases_b, dim=0)
        mask_canvases_b = torch.cat(mask_canvases_b, dim=0)
        thetas_b = torch.stack(thetas)

        grid = F.affine_grid(thetas_b, (bs, 1, img_h, img_w), align_corners=False)
        transformed_patch_canvas = F.grid_sample(patch_canvases_b, grid, align_corners=False)
        transformed_mask_canvas = F.grid_sample(mask_canvases_b, grid, align_corners=False)

        final_mask = transformed_mask_canvas * constraint_mask_b
        attacked_image = image_batch * (1.0 - final_mask) + transformed_patch_canvas * final_mask
        
        return attacked_image, final_mask, transformed_patch_canvas