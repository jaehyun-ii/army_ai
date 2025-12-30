import torch
import torch.nn as nn
from ..utils.helpers import box_iou

class DetectionLoss(nn.Module):
    """
    A highly configurable loss function for adversarial attacks, allowing fine-grained control
    over how boxes and classes are selected and weighted.
    """
    def __init__(self, 
                 iou_threshold=0.1,      # Min IoU for a box proposal to be considered.
                 class_conf_threshold=0.05, # Min confidence for a class within a box to be considered for agnostic loss.
                 top_k_boxes=5,          # Number of boxes to use in the loss. -1 for all passing the IoU threshold.
                 top_n_classes=3,        # Number of classes per box to sum for agnostic loss. -1 for all passing the confidence threshold.
                 lambda_targeted=1.0, 
                 lambda_agnostic=1.5):
        super().__init__()
        # --- Validate and Store Hyperparameters ---
        if not 0.0 <= iou_threshold < 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if not 0.0 <= class_conf_threshold < 1.0:
            raise ValueError("class_conf_threshold must be between 0.0 and 1.0")
            
        self.iou_threshold = iou_threshold
        self.class_conf_threshold = class_conf_threshold
        self.top_k_boxes = top_k_boxes
        self.top_n_classes = top_n_classes
        self.lambda_targeted = lambda_targeted
        self.lambda_agnostic = lambda_agnostic

    def forward(self, inference_output, pseudo_gt_boxes, target_class_idx):
        device = inference_output.device
        batch_loss_targeted, batch_loss_agnostic = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        bs = inference_output.shape[0]

        for b in range(bs):
            gt_box_single = pseudo_gt_boxes[b].unsqueeze(0)
            if torch.sum(gt_box_single) == 0: continue
            
            preds = inference_output[b]
            if preds is None or len(preds) == 0: continue
            
            pred_boxes, obj_scores, class_scores = preds[:,:4], torch.sigmoid(preds[:,4]), torch.sigmoid(preds[:,5:])
            
            # --- Step 1: Filter Boxes by IoU ---
            ious = box_iou(pred_boxes, gt_box_single).squeeze(-1)
            keep_mask = ious >= self.iou_threshold
            if not torch.any(keep_mask):
                continue
            
            filtered_obj_scores = obj_scores[keep_mask]
            filtered_class_scores = class_scores[keep_mask]
            filtered_ious = ious[keep_mask]
            
            # --- Targeted Loss Calculation (on filtered proposals) ---
            target_class_conf = filtered_class_scores[:, target_class_idx]
            targeted_scores = filtered_obj_scores * target_class_conf * filtered_ious

            # --- Agnostic Loss Calculation (with new class selection logic) ---
            
            # Create a mask for classes with confidence > threshold
            class_keep_mask = filtered_class_scores >= self.class_conf_threshold
            # Apply the mask, zeroing out low-confidence classes
            thresholded_class_scores = filtered_class_scores * class_keep_mask.float()

            # ---  Sub-Step 2b - Select Classes (Top-N or All) ---
            if self.top_n_classes == -1:
                # Use ALL classes that passed the threshold
                sum_class_conf = torch.sum(thresholded_class_scores, dim=1)
            else:
                # Find the top_n_classes from the thresholded scores
                # Note: We take topk from thresholded scores. If fewer than N classes pass the
                # threshold, the sum will correctly include only those that did.
                
                # Ensure we don't ask for more classes than are available
                n_classes_available = thresholded_class_scores.shape[1]
                n = min(self.top_n_classes, n_classes_available)

                top_n_class_conf_vals, _ = torch.topk(thresholded_class_scores, n, dim=1)
                sum_class_conf = torch.sum(top_n_class_conf_vals, dim=1)
            
            agnostic_scores = filtered_obj_scores * sum_class_conf * filtered_ious

            # --- Step 3: Select Boxes (Top-K or All) ---
            num_filtered_boxes = len(targeted_scores)
            if self.top_k_boxes == -1:
                k = num_filtered_boxes
            else:
                k = min(self.top_k_boxes, num_filtered_boxes)

            if k > 0:
                if self.top_k_boxes == -1:
                    sum_targeted_scores = torch.sum(targeted_scores)
                    sum_agnostic_scores = torch.sum(agnostic_scores)
                else:
                    top_k_targeted_boxes, _ = torch.topk(targeted_scores, k)
                    sum_targeted_scores = torch.sum(top_k_targeted_boxes)
                    
                    top_k_agnostic_boxes, _ = torch.topk(agnostic_scores, k)
                    sum_agnostic_scores = torch.sum(top_k_agnostic_boxes)
                
                batch_loss_targeted += sum_targeted_scores
                batch_loss_agnostic += sum_agnostic_scores
        
        # --- Final loss calculation ---
        final_targeted_loss = (batch_loss_targeted / bs) if bs > 0 else torch.tensor(0.0, device=device)
        final_agnostic_loss = (batch_loss_agnostic / bs) if bs > 0 else torch.tensor(0.0, device=device)
        
        total_loss = (self.lambda_targeted * final_targeted_loss) + (self.lambda_agnostic * final_agnostic_loss)
        return total_loss, final_targeted_loss, final_agnostic_loss