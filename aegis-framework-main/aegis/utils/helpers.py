import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from typing import List, Dict, Any, Optional, Union

def xywh2xyxy(x):
    """Converts bbox format from [x, y, w, h] to [x1, y1, x2, y2], supporting torch.Tensor and np.ndarray."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def select_device(device="", batch_size=0, newline=True):
    """Selects the device for running models, handling CPU, GPU, and MPS with optional batch size divisibility check."""
    s = f"YOLOv3 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(",", "")), (
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
        )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)

def get_yolo_pseudo_gt(clean_images_yolo: torch.Tensor, victim_models: dict, device: torch.device) -> dict:
    """
    Generates pseudo ground-truth bounding boxes for a batch of clean images.

    It uses a "guide" model to find the target object. The guide is chosen with the
    following priority:
    1. A preferred model named 'yolo12x' if it exists in the victim list.
    2. The first model listed in the victim configuration if 'yolo12x' is not present.
    """
    pseudo_gts = {}
    
    # --- START OF NEW, ROBUST GUIDE SELECTION LOGIC ---
    preferred_guide_name = 'yolo12x'
    guide_model_name = None
    
    if not victim_models:
        raise ValueError("Cannot generate pseudo-GTs: The victim_models dictionary is empty.")

    # Prioritize the preferred guide model
    if preferred_guide_name in victim_models:
        guide_model_name = preferred_guide_name
    else:
        # Fallback to the first model in the configuration
        guide_model_name = next(iter(victim_models))
        print(f"INFO: Preferred guide model '{preferred_guide_name}' not found. "
              f"Using the first available model '{guide_model_name}' as the guide for pseudo-GT generation.")
    
    # --- END OF NEW LOGIC ---
    
    guide_model_data = victim_models[guide_model_name]
    guide_model = guide_model_data['model']
    
    # Ensure the guide model config has the car_class_id
    if 'car_class_id' not in guide_model_data['config']:
        raise ValueError(f"Guide model '{guide_model_name}' is missing the 'car_class_id' in its configuration.")
    car_id = guide_model_data['config']['car_class_id']
    
    with torch.no_grad():
        results = guide_model.predict(clean_images_yolo, verbose=False)
        gt_list = []
        for res in results:
            # Filter detections for the specified car class
            car_dets = res.boxes.data[res.boxes.data[:, -1] == car_id]
            if len(car_dets) > 0:
                # Get the detection with the highest confidence
                best_det = car_dets[torch.argmax(car_dets[:, 4])]
                gt_list.append(best_det[:4])
            else:
                # If no car is detected, use a zero-box placeholder
                gt_list.append(torch.zeros(4, device=device))
        
        shared_gt = torch.stack(gt_list)
        
        # Assign this shared ground truth to all models in the ensemble
        for name in victim_models.keys():
            pseudo_gts[name] = shared_gt
            
    return pseudo_gts


class FeatureExtractor(nn.Module):
    """
    A more robust feature extractor for Ultralytics models that avoids
    triggering the non-differentiable inference head.
    """
    def __init__(self, model: nn.Module, layers_to_hook: List[int]):
        super().__init__()
        # We need the sequential part of the model (the actual layers)
        if not hasattr(model, 'model') or not isinstance(model.model, nn.Sequential):
            raise TypeError("The provided model must have a `.model` attribute that is an `nn.Sequential` module.")
        self.model_layers = model.model
        self.layers_to_hook = layers_to_hook
        self.features = {}
        self.hooks = []

        for layer_idx in self.layers_to_hook:
            try:
                layer_module = self.model_layers[layer_idx]
                hook = layer_module.register_forward_hook(self.save_features_hook(layer_idx))
                self.hooks.append(hook)
            except IndexError:
                raise ValueError(f"Layer index {layer_idx} is out of bounds for the model with {len(self.model_layers)} layers.")

    def save_features_hook(self, layer_id):
        def fn(_, __, output):
            # The output of some layers can be a tuple (e.g., from C2f).
            # We are interested in the main feature tensor, which is usually the first element.
            if isinstance(output, tuple):
                self.features[layer_id] = output[0]
            else:
                self.features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Performs a differentiable forward pass through the model's backbone and neck,
        stopping before the final detection head to avoid inference-mode errors.
        """
        self.features = {}
        
        # The detection head is almost always the last layer in `model.model`.
        # By iterating up to the second-to-last layer, we run the backbone and neck
        # but completely skip the problematic `Detect` head.
        num_layers_to_run = len(self.model_layers) - 1
        
        for i in range(num_layers_to_run):
            # Only run the forward pass for the layers we need.
            # This is efficient and avoids the head.
            if i in self.layers_to_hook or i < max(self.layers_to_hook):
                 x = self.model_layers[i](x)
            else:
                # If we've already passed the last layer we need to hook, we can stop.
                break
                
        # The hooks will have already saved the features we care about.
        return self.features

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

class RRBAugmentation(nn.Module):
    """
    Applies Random Resize, Rotation, and Blur.
    This version includes a fix to ensure output dimensions are compatible with
    models like YOLOv8 that have architectural constraints (e.g., divisibility by 32).
    """
    def __init__(self, rotation_degrees=7.0, resize_scale=(0.9, 1.1), blur_kernel=3, blur_sigma=(0.1, 2.0), stride=32):
        super().__init__()
        self.rotation_degrees = rotation_degrees
        self.resize_scale = resize_scale
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.stride = stride # The architectural constraint, usually 32 for YOLO models.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        augmented_batch = []
        original_size = x.shape[-2:] # H, W

        for img in x:
            # 1. Rotation (unchanged)
            angle = torch.rand(1).item() * 2 * self.rotation_degrees - self.rotation_degrees
            img = TF.rotate(img, angle)
            
            # 2. Random Resizing (unchanged)
            scale = torch.rand(1).item() * (self.resize_scale[1] - self.resize_scale[0]) + self.resize_scale[0]
            new_size_h, new_size_w = int(original_size[0] * scale), int(original_size[1] * scale)
            img = F.interpolate(img.unsqueeze(0), size=(new_size_h, new_size_w), mode='bilinear', align_corners=False)

            # --- START OF DEFINITIVE FIX ---
            # 3. Letterbox Padding to a compatible size
            # Calculate the target dimensions that are divisible by the stride
            target_h = ((original_size[0] + self.stride - 1) // self.stride) * self.stride
            target_w = ((original_size[1] + self.stride - 1) // self.stride) * self.stride

            # Calculate the padding needed to reach the target size
            pad_h = target_h - new_size_h
            pad_w = target_w - new_size_w
            
            # Use asymmetric padding (pad to one side), similar to how YOLO dataloaders work
            # This is simpler and avoids centering issues.
            # Padding format is (pad_left, pad_right, pad_top, pad_bottom)
            padding = [0, pad_w, 0, pad_h]
            
            # Pad the image. The value 0.0 corresponds to black padding.
            img = F.pad(img, padding, "constant", 0.0)
            # --- END OF DEFINITIVE FIX ---

            # 4. Gaussian Blur (unchanged)
            # Squeeze and unsqueeze to match TF.gaussian_blur expected input shape
            img = TF.gaussian_blur(img.squeeze(0), kernel_size=[self.blur_kernel, self.blur_kernel], sigma=self.blur_sigma)
            
            augmented_batch.append(img)
            
        return torch.stack(augmented_batch)

class DirectPastePatchApplier(nn.Module):
    def __init__(self, patch_shape='circle', placement='on_car', relative_size=0.15):
        super().__init__()
        self.shape = patch_shape
        self.placement = placement
        self.relative_size = relative_size # Patch size as a fraction of car's smaller dimension (width or height)

    def get_patch_shape_mask(self, size_h, size_w, device):
        """Creates a shape mask (e.g., circle) of a specific size on the fly."""
        if self.shape == 'circle':
            center_y, center_x = size_h // 2, size_w // 2
            Y, X = torch.meshgrid(torch.arange(size_h, device=device), torch.arange(size_w, device=device), indexing='ij')
            dist_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = (dist_from_center <= min(size_h, size_w) / 2).float()
        elif self.shape == 'square':
            mask = torch.ones(size_h, size_w, device=device)
        else:
            raise ValueError(f"Unsupported patch shape: {self.shape}")
        return mask.unsqueeze(0) # (1, H, W)

    def forward(self, image_batch, patch, car_mask_batch):
        bs, _, img_h, img_w = image_batch.shape
        device = image_batch.device
        
        # This will be the final list of attacked images
        final_attacked_images = []

        # We must loop over the batch because each image has a different car size and placement
        for i in range(bs):
            image = image_batch[i]
            car_mask = car_mask_batch[i]

            # --- 1. Get Car BBox from Mask ---
            if not torch.any(car_mask):
                final_attacked_images.append(image)
                continue # No car in this image, skip patching
            
            rows = torch.any(car_mask[0], axis=1)
            cols = torch.any(car_mask[0], axis=0)
            ymin, ymax = torch.where(rows)[0][[0, -1]]
            xmin, xmax = torch.where(cols)[0][[0, -1]]
            
            bbox_h = ymax - ymin
            bbox_w = xmax - xmin
            if bbox_h <= 0 or bbox_w <= 0:
                final_attacked_images.append(image)
                continue

            # --- 2. Determine Patch Size and Resize Patch ---
            car_short_side = torch.min(bbox_h, bbox_w).float()
            patch_target_size = int(car_short_side * self.relative_size)
            
            # Ensure patch is at least 2x2 pixels for interpolation
            patch_target_size = max(patch_target_size, 2)
            
            # Differentiably resize the LEARNABLE patch to the target size for this specific image
            resized_patch = F.interpolate(patch, size=(patch_target_size, patch_target_size), mode='bilinear', align_corners=False)

            # Create the shape mask for the RESIZED patch
            patch_shape_mask = self.get_patch_shape_mask(patch_target_size, patch_target_size, device) # (1, pH, pW)
            shaped_patch = resized_patch * patch_shape_mask # (C, pH, pW)

            # --- 3. Determine Placement Coordinates ---
            if self.placement == 'on_car':
                # Place on the right side of the car, centered vertically
                # Random jitter is added for robustness
                # offset_x = int(torch.rand(1).item() * 0.2 * bbox_w)
                # offset_y = int((torch.rand(1).item() - 0.5) * 0.4 * bbox_h) # vertical jitter

                offset_x = 0 # int(torch.rand(1).item() * 0.2 * bbox_w)
                offset_y = 0 # int((torch.rand(1).item() - 0.5) * 0.4 * bbox_h) # vertical jitter

                paste_x1 = xmax - patch_target_size - offset_x
                paste_y1 = ymin + (bbox_h // 2) - (patch_target_size // 2) + offset_y
                
                # The area where the patch can be applied is the car itself
                apply_mask = car_mask
            
            elif self.placement == 'off_car':
                # Place on the ground to the left of the car
                # offset_x = int((torch.rand(1).item() * 0.2 + 0.1) * bbox_w) # jitter away from car
                # offset_y = int(torch.rand(1).item() * 0.1 * bbox_h) # slight vertical jitter

                offset_x = int(0.15 * bbox_w) # int((torch.rand(1).item() * 0.2 + 0.1) * bbox_w) # jitter away from car
                offset_y = 0 # int(torch.rand(1).item() * 0.1 * bbox_h) # slight vertical jitter

                paste_x1 = xmin - patch_target_size - offset_x
                paste_y1 = ymax - patch_target_size - offset_y # slightly above the car's bottom edge
                
                # The area where the patch can be applied is the background
                apply_mask = 1.0 - car_mask
            
            else:
                 raise ValueError(f"Unsupported placement: {self.placement}")

            # Clamp coordinates to be within image bounds
            paste_x1 = torch.clamp(torch.tensor(paste_x1), 0, img_w - patch_target_size)
            paste_y1 = torch.clamp(torch.tensor(paste_y1), 0, img_h - patch_target_size)
            
            # --- 4. Create Full-Size Canvases and Paste ---
            # Create a full-image-sized tensor of zeros
            patch_canvas = torch.zeros_like(image)
            # Paste the shaped patch onto this canvas at the calculated location
            patch_canvas[:, paste_y1:paste_y1+patch_target_size, paste_x1:paste_x1+patch_target_size] = shaped_patch

            # Do the same for the mask
            mask_canvas = torch.zeros(1, img_h, img_w, device=device)
            mask_canvas[:, paste_y1:paste_y1+patch_target_size, paste_x1:paste_x1+patch_target_size] = patch_shape_mask

            # --- 5. Apply to Image ---
            # The final combined mask ensures the patch only appears in its shape AND in the allowed region (car or background)
            combined_mask = mask_canvas * apply_mask
            
            attacked_image = image * (1.0 - combined_mask) + patch_canvas * combined_mask
            final_attacked_images.append(attacked_image)
            
        return torch.stack(final_attacked_images)
