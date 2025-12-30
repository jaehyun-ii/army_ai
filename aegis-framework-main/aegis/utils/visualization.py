# aegis/utils/visualization.py

import torch
import os
import cv2
import numpy as np
from torchvision.utils import save_image
from PIL import Image

from ..attacks.base_attack import BaseAttack 

def visualize_proof(epoch: int, attack_strategy: BaseAttack, victim_models: dict, vis_sample: dict, device: torch.device, config):
    """
    Generates and saves a proof-of-concept image showing the attack's effect.
    """
    vis_folder = os.path.join(os.path.dirname(config.output_path), "visualizations")
    os.makedirs(vis_folder, exist_ok=True)
    
    # --- 1. Save the raw artifact(s) ---
    # This part needs to know a little about the attack type
    attack_type = config.attack.params.type
    if attack_type == 'triplane':
        # This is a bit of a special case due to multiple textures
        for name, texture in attack_strategy.adv_textures.items():
            save_image(texture, os.path.join(vis_folder, f"epoch_{epoch+1:03d}_texture_{name}.png"))
    elif hasattr(attack_strategy, 'adv_patch'):
        save_image(attack_strategy.adv_patch, os.path.join(vis_folder, f"epoch_{epoch+1:03d}_patch.png"))
    elif hasattr(attack_strategy, 'universal_perturbation'):
        # Normalize perturbation for better viewing
        pert = attack_strategy.universal_perturbation.detach().cpu()
        normalized_pert = (pert + config.attack.params.epsilon) / (2 * config.attack.params.epsilon)
        save_image(normalized_pert, os.path.join(vis_folder, f"epoch_{epoch+1:03d}_perturbation.png"))
        
    # --- 2. Generate the attacked image ---
    with torch.no_grad():
        # Prepare a single-item batch for visualization
        vis_batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v] for k, v in vis_sample.items()}
        clean_image = torch.clamp(vis_batch['ref_image'] + vis_batch['background'], 0, 1)
        
        # Use the strategy's own method to apply the attack
        if hasattr(attack_strategy, 'adv_patch'):
            attacked_image_tensor , _ , _ = attack_strategy.apply_attack(clean_image, vis_batch, victim_models)
        else:
            attacked_image_tensor = attack_strategy.apply_attack(clean_image, vis_batch, victim_models)

    # --- 3. Run inference and draw boxes for each victim model ---
    for model_name, model_data in victim_models.items():
        model_config = model_data['config']
        
        with torch.no_grad():
            # This part is copied and adapted from your original scripts
            if model_config['type'] in ['yolo', 'rtdetr']:
                results = model_data['model'].predict(attacked_image_tensor, verbose=False)[0]
                final_detections = results.boxes.data
                names = model_data['model'].model.names
            elif model_config['type'] == 'dino':
                # DINO-specific inference logic
                pil_image = Image.fromarray((attacked_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                processor, victim_model = model_data['processor'], model_data['model']
                inputs = processor(images=pil_image, text=model_config['prompt'], return_tensors="pt").to(device)
                outputs = victim_model(**inputs)
                processed_results = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=0.25, target_sizes=[pil_image.size[::-1]]
                )[0]
                boxes = processed_results['boxes']
                scores = processed_results['scores']
                # Create a standardized format
                final_detections = torch.cat([boxes, scores.unsqueeze(1), torch.zeros_like(scores).unsqueeze(1)], dim=1) if len(boxes) > 0 else torch.empty((0, 6))
                names = {0: 'car'} # Or get from config

        # --- Draw the boxes on the image ---
        img_np_rgb = (attacked_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_to_draw = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        
        if len(final_detections) > 0:
            for det in final_detections:
                x1, y1, x2, y2, conf, cls = det
                label = f"{names[int(cls)]} {conf:.2f}"
                color = (0, 0, 255) # Red
                cv2.rectangle(img_to_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img_to_draw, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        proof_filename = f"epoch_{epoch+1:03d}_proof_{model_name}.png"
        cv2.imwrite(os.path.join(vis_folder, proof_filename), img_to_draw)