# file: model/tpm.py (MODIFIED with Texture Transformations)

import torch
import torch.nn as nn
import torch.nn.functional as F

class TriplanarMapping(nn.Module):
    """
    A differentiable module to perform triplanar mapping, now with
    support for 3D texture transformations (rotation, scale, translation).
    """
    def __init__(self):
        super().__init__()
        self.sensor_to_actor_rotation = torch.tensor([
            [0., 0., 1.], [1., 0., 0.], [0.,-1., 0.]
        ], dtype=torch.float32)

    def forward(self, adv_texture, depth_map, camera_intrinsics, camera_extrinsics_inv, 
                texture_transform_params=None): # New optional argument
        """
        Args:
            adv_texture (...): The adversarial texture(s).
            ...
            texture_transform_params (dict, optional): A dictionary of parameters
                to transform the texture in 3D space. Keys can include 'roll', 
                'pitch', 'yaw', 'scale', 'x_shift', 'y_shift', etc.
        """
        B, _, H_img, W_img = depth_map.shape
        device = depth_map.device

        if self.sensor_to_actor_rotation.device != device:
            self.sensor_to_actor_rotation = self.sensor_to_actor_rotation.to(device)

        # Step 1 & 2: Unproject and convert to Camera Actor Space (as before)
        i = torch.arange(W_img, device=device); j = torch.arange(H_img, device=device)
        u_coords, v_coords = torch.meshgrid(i, j, indexing='xy')
        pixels = torch.stack((u_coords, v_coords, torch.ones_like(u_coords)), dim=-1).float().view(1, H_img*W_img, 3).transpose(1, 2).expand(B, -1, -1)
        K_inv = torch.inverse(camera_intrinsics)
        points_camera_sensor_lh = (K_inv @ pixels) * depth_map.view(B, 1, -1)
        points_camera_actor_lh = self.sensor_to_actor_rotation @ points_camera_sensor_lh
        points_camera_actor_h_lh = F.pad(points_camera_actor_lh, (0, 0, 0, 1), value=1.0)
        
        # Step 3 & 4: Transform to Vehicle Actor Space (as before)
        camera_actor_to_vehicle_actor_lh = torch.inverse(camera_extrinsics_inv)
        points_vehicle_actor_h_lh = camera_actor_to_vehicle_actor_lh @ points_camera_actor_h_lh
        points_vehicle_actor_lh = points_vehicle_actor_h_lh[:, :3, :]
        points_vehicle_actor_rh = points_vehicle_actor_lh.clone()
        points_vehicle_actor_rh[:, 1, :] *= -1 
        
        if texture_transform_params:
            # Fetch parameters with defaults
            roll = torch.deg2rad(torch.tensor(texture_transform_params.get('roll', 0.0), device=device))
            pitch = torch.deg2rad(torch.tensor(texture_transform_params.get('pitch', 0.0), device=device))
            yaw = torch.deg2rad(torch.tensor(texture_transform_params.get('yaw', 0.0), device=device))
            scale = texture_transform_params.get('scale', 1.0)
            x_shift = texture_transform_params.get('x_shift', 0.0)
            y_shift = texture_transform_params.get('y_shift', 0.0)
            # You can add scale_ratio and flip here as well if needed

            # Create rotation matrices
            Rx = torch.tensor([[1, 0, 0], [0, torch.cos(roll), -torch.sin(roll)], [0, torch.sin(roll), torch.cos(roll)]], device=device)
            Ry = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)], [0, 1, 0], [-torch.sin(pitch), 0, torch.cos(pitch)]], device=device)
            Rz = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0], [torch.sin(yaw), torch.cos(yaw), 0], [0, 0, 1]], device=device)
            
            # Combine rotation and scale into a single transformation matrix
            # Note: The order of rotations (e.g., ZYX) can be changed if needed
            transform_matrix = (Rz @ Ry @ Rx) * scale
            
            # Apply the transformation to the 3D object coordinates
            # This rotates and scales the "texture projector" relative to the car
            transformed_points = transform_matrix @ points_vehicle_actor_rh
            
            # Apply translation (shift) after rotation/scale
            shift_vector = torch.tensor([x_shift, y_shift, 0.0], device=device).view(1, 3, 1)
            final_points = transformed_points + shift_vector
        else:
            final_points = points_vehicle_actor_rh

        # The rest of the function now uses the transformed points
        points_object_img_rh = final_points.view(B, 3, H_img, W_img)

        # Step 5: Calculate normals and sample texture (as before)
        d_y = points_object_img_rh[:, :, 1:, :] - points_object_img_rh[:, :, :-1, :]
        d_x = points_object_img_rh[:, :, :, 1:] - points_object_img_rh[:, :, :, :-1]
        d_y = F.pad(d_y, (0, 0, 0, 1), 'replicate'); d_x = F.pad(d_x, (0, 1, 0, 0), 'replicate')
        normals = F.normalize(torch.cross(d_y, d_x, dim=1), p=2, dim=1)
        weights = normals.abs().pow(2); weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        w_x, w_y, w_z = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        world_texture_scale = 1.5
        coords_object_permuted = points_object_img_rh.permute(0, 2, 3, 1)
        uvs_normalized = (coords_object_permuted / world_texture_scale) % 1.0
        uvs_normalized = uvs_normalized * 2.0 - 1.0
        
        grid_xy = uvs_normalized[..., [0, 1]]; grid_yz = uvs_normalized[..., [1, 2]]; grid_xz = uvs_normalized[..., [0, 2]]
        adv_texture_batched = adv_texture.expand(B, -1, -1, -1) if adv_texture.shape[0] == 1 and B > 1 else adv_texture
        tex_xy = F.grid_sample(adv_texture_batched, grid_xy, mode='bilinear', align_corners=False)
        tex_yz = F.grid_sample(adv_texture_batched, grid_yz, mode='bilinear', align_corners=False)
        tex_xz = F.grid_sample(adv_texture_batched, grid_xz, mode='bilinear', align_corners=False)
        
        projected_texture = (tex_xy * w_z + tex_yz * w_x + tex_xz * w_y)
        return projected_texture

class TriPlaneSampler(nn.Module):
    """
    A differentiable module to perform triplanar mapping using three separate
    2D textures for the XY, YZ, and XZ planes. Includes support for 3D
    texture transformations (rotation, scale, translation).
    """
    def __init__(self):
        super().__init__()
        self.sensor_to_actor_rotation = torch.tensor([
            [0., 0., 1.], [1., 0., 0.], [0.,-1., 0.]
        ], dtype=torch.float32)

    def forward(self, adv_texture_xy, adv_texture_yz, adv_texture_xz, 
                depth_map, camera_intrinsics, camera_extrinsics_inv,
                texture_transform_params=None):
        """
        Args:
            adv_texture_xy (torch.Tensor): The adversarial texture for the XY plane.
            adv_texture_yz (torch.Tensor): The adversarial texture for the YZ plane.
            adv_texture_xz (torch.Tensor): The adversarial texture for the XZ plane.
            ...
            texture_transform_params (dict, optional): A dictionary of parameters
                to transform the texture in 3D space. Keys can include 'roll', 
                'pitch', 'yaw', 'scale', 'x_shift', 'y_shift', etc.
        """
        B, _, H_img, W_img = depth_map.shape
        device = depth_map.device

        if self.sensor_to_actor_rotation.device != device:
            self.sensor_to_actor_rotation = self.sensor_to_actor_rotation.to(device)

        # Step 1 & 2: Unproject and convert to Camera Actor Space
        i = torch.arange(W_img, device=device); j = torch.arange(H_img, device=device)
        u_coords, v_coords = torch.meshgrid(i, j, indexing='xy')
        pixels = torch.stack((u_coords, v_coords, torch.ones_like(u_coords)), dim=-1).float().view(1, H_img*W_img, 3).transpose(1, 2).expand(B, -1, -1)
        K_inv = torch.inverse(camera_intrinsics)
        points_camera_sensor_lh = (K_inv @ pixels) * depth_map.view(B, 1, -1)
        points_camera_actor_lh = self.sensor_to_actor_rotation @ points_camera_sensor_lh
        points_camera_actor_h_lh = F.pad(points_camera_actor_lh, (0, 0, 0, 1), value=1.0)
        
        # Step 3 & 4: Transform to Vehicle Actor Space
        camera_actor_to_vehicle_actor_lh = torch.inverse(camera_extrinsics_inv)
        points_vehicle_actor_h_lh = camera_actor_to_vehicle_actor_lh @ points_camera_actor_h_lh
        points_vehicle_actor_lh = points_vehicle_actor_h_lh[:, :3, :]
        points_vehicle_actor_rh = points_vehicle_actor_lh.clone()
        points_vehicle_actor_rh[:, 1, :] *= -1 
        
        if texture_transform_params:
            roll = torch.deg2rad(torch.tensor(texture_transform_params.get('roll', 0.0), device=device))
            pitch = torch.deg2rad(torch.tensor(texture_transform_params.get('pitch', 0.0), device=device))
            yaw = torch.deg2rad(torch.tensor(texture_transform_params.get('yaw', 0.0), device=device))
            scale = texture_transform_params.get('scale', 1.0)
            x_shift = texture_transform_params.get('x_shift', 0.0)
            y_shift = texture_transform_params.get('y_shift', 0.0)
            Rx = torch.tensor([[1, 0, 0], [0, torch.cos(roll), -torch.sin(roll)], [0, torch.sin(roll), torch.cos(roll)]], device=device)
            Ry = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)], [0, 1, 0], [-torch.sin(pitch), 0, torch.cos(pitch)]], device=device)
            Rz = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0], [torch.sin(yaw), torch.cos(yaw), 0], [0, 0, 1]], device=device)
            transform_matrix = (Rz @ Ry @ Rx) * scale
            transformed_points = transform_matrix @ points_vehicle_actor_rh
            shift_vector = torch.tensor([x_shift, y_shift, 0.0], device=device).view(1, 3, 1)
            final_points = transformed_points + shift_vector
        else:
            final_points = points_vehicle_actor_rh


        points_object_img_rh = final_points.view(B, 3, H_img, W_img)

        # Step 5: Calculate normals and blending weights
        d_y = points_object_img_rh[:, :, 1:, :] - points_object_img_rh[:, :, :-1, :]
        d_x = points_object_img_rh[:, :, :, 1:] - points_object_img_rh[:, :, :, :-1]
        d_y = F.pad(d_y, (0, 0, 0, 1), 'replicate'); d_x = F.pad(d_x, (0, 1, 0, 0), 'replicate')
        normals = F.normalize(torch.cross(d_y, d_x, dim=1), p=2, dim=1)
        weights = normals.abs().pow(2); weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        w_x, w_y, w_z = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        # Step 6: Sample from the Three Planes using transformed coordinates
        world_texture_scale = 1.5
        coords_object_permuted = points_object_img_rh.permute(0, 2, 3, 1)
        uvs_normalized = (coords_object_permuted / world_texture_scale) % 1.0
        uvs_grid_format = uvs_normalized * 2.0 - 1.0
        
        grid_xy = uvs_grid_format[..., [0, 1]]
        grid_yz = uvs_grid_format[..., [1, 2]]
        grid_xz = uvs_grid_format[..., [0, 2]]
        
        tex_xy = adv_texture_xy.expand(B, -1, -1, -1) if adv_texture_xy.shape[0] == 1 and B > 1 else adv_texture_xy
        tex_yz = adv_texture_yz.expand(B, -1, -1, -1) if adv_texture_yz.shape[0] == 1 and B > 1 else adv_texture_yz
        tex_xz = adv_texture_xz.expand(B, -1, -1, -1) if adv_texture_xz.shape[0] == 1 and B > 1 else adv_texture_xz
        
        sampled_color_xy = F.grid_sample(tex_xy, grid_xy, mode='bilinear', align_corners=False)
        sampled_color_yz = F.grid_sample(tex_yz, grid_yz, mode='bilinear', align_corners=False)
        sampled_color_xz = F.grid_sample(tex_xz, grid_xz, mode='bilinear', align_corners=False)
        
        # Step 7: Blend the Results using Normal Weights
        # w_z weights the XY plane (top-down), w_x weights YZ (side), w_y weights XZ (front-back)
        projected_texture = (sampled_color_xy * w_z) + (sampled_color_yz * w_x) + (sampled_color_xz * w_y)
        
        return projected_texture
