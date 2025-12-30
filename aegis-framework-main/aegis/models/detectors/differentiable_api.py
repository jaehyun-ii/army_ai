# aegis/models/detectors/differentiable_api.py

import torch
import torch.nn.functional as F
from ...utils.helpers import xywh2xyxy 

# --- YOLOv8 / RT-DETR Differentiable Forward Pass ---
def get_differentiable_yolo_output(model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a differentiable forward pass for a YOLO-style model.
    This function bypasses non-differentiable operations like NMS by manually
    decoding the raw output from the detection heads.
    """
    # Set model to train mode to get raw feature maps
    model.train()
    raw_feature_maps = model(input_tensor)
    model.eval() # Set back to eval mode

    head = model.model[-1]
    box_split_size, cls_split_size = head.reg_max * 4, head.nc
    
    output_components = [fm.view(fm.shape[0], fm.shape[1], -1).permute(0, 2, 1) for fm in raw_feature_maps]
    inference_output = torch.cat(output_components, dim=1)
    
    box_raw, cls_raw = inference_output.split((box_split_size, cls_split_size), dim=-1)

    # --- Decode boxes ---
    strides = head.stride.to(input_tensor.device)
    anchor_points_list, strides_tensor_list = [], []
    for i, stride in enumerate(strides):
        h, w = raw_feature_maps[i].shape[2:]
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=input_tensor.device), torch.arange(w, device=input_tensor.device), indexing='ij')
        anchor_points_list.append(torch.stack((grid_x, grid_y), 2).view(1, -1, 2))
        strides_tensor_list.append(torch.full((1, h * w, 1), stride, device=input_tensor.device))
    
    anchor_points = torch.cat(anchor_points_list, dim=1).expand(input_tensor.shape[0], -1, -1)
    strides_tensor = torch.cat(strides_tensor_list, dim=1).expand(input_tensor.shape[0], -1, -1)
    
    box_dist_softmax = box_raw.view(input_tensor.shape[0], -1, 4, head.reg_max).softmax(3)
    reg_max_range = torch.arange(head.reg_max, device=input_tensor.device, dtype=torch.float)
    box_offsets = (box_dist_softmax * reg_max_range).sum(3)
    
    lt, rb = box_offsets.chunk(2, -1)
    box_decoded_xyxy = torch.cat((anchor_points - lt, anchor_points + rb), -1) * strides_tensor
    
    # Use max class logit as a proxy for objectness score for the loss function
    max_class_logits, _ = torch.max(cls_raw, dim=-1, keepdim=True)
    
    # Return in a standardized format: [boxes, objectness, class_scores]
    return torch.cat([box_decoded_xyxy, max_class_logits, cls_raw], dim=-1)


# --- GroundingDINO Differentiable Forward Pass ---
def get_differentiable_dino_output(
    model: torch.nn.Module, 
    processor, 
    input_tensor: torch.Tensor, 
    prompt: str, 
    num_coco_classes: int = 80
) -> torch.Tensor:
    """Performs a differentiable forward pass for a GroundingDINO model."""
    device = input_tensor.device
    bs, _, h, w = input_tensor.shape
    
    # Differentiable normalization and resizing
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    resized_input = F.interpolate(input_tensor, size=(800, 800), mode='bilinear', align_corners=False)
    normalized_input = (resized_input - mean) / std

    batched_prompts = [prompt] * bs
    tokenized_text = processor.tokenizer(batched_prompts, return_tensors="pt", padding=True).to(device)

    outputs = model(pixel_values=normalized_input, input_ids=tokenized_text['input_ids'], attention_mask=tokenized_text['attention_mask'])

    logits = outputs.logits
    pred_boxes_cxcywh_norm = outputs.pred_boxes
    
    img_size_tensor = torch.tensor([w, h, w, h], device=device).float()
    pred_boxes_xyxy_abs = xywh2xyxy(pred_boxes_cxcywh_norm * img_size_tensor)

    # Aggregate multi-token logits into a single objectness score
    aggregated_logits, _ = torch.max(logits, dim=-1)
    objectness_logits = aggregated_logits.unsqueeze(-1)
    
    # Pad class scores to match the format expected by the loss function
    padded_cls_raw = torch.full((bs, logits.shape[1], num_coco_classes), -10.0, device=device)
    padded_cls_raw[:, :, 0] = aggregated_logits # Assuming the prompt 'car' maps to class 0
    
    return torch.cat([pred_boxes_xyxy_abs, objectness_logits, padded_cls_raw], dim=-1)