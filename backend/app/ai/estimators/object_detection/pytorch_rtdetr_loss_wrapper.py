# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2025
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
PyTorch-specific RT-DETR loss wrapper for app.ai.
"""

import torch


class PyTorchRTDETRLossWrapper(torch.nn.Module):
    """Wrapper for RT-DETR models to handle loss dict format."""

    def __init__(self, model, device="cpu"):
        super().__init__()
        self.model = model
        self.device = device
        self._head_patched = False
        self._cdn_patched = False
        self.return_raw_predictions = False  # Flag for attack loss mode

        try:
            from ultralytics.models.rtdetr import RTDETRPredictor
            from ultralytics.models.utils.loss import RTDETRDetectionLoss

            # Initialize predictor to get args
            self.predictor = RTDETRPredictor()
            self.model.args = self.predictor.args

            # Initialize criterion for RT-DETR
            # RT-DETR uses different parameters than YOLO
            nc = getattr(model, 'nc', 80)  # number of classes
            self.model.criterion = RTDETRDetectionLoss(nc=nc)

            # Move criterion's internal tensors to the specified device
            device_obj = torch.device(device)
            if hasattr(self.model.criterion, 'device'):
                self.model.criterion.device = device_obj

            # Patch get_cdn_group to be device-safe
            self._patch_cdn_group()

            # Patch RT-DETR head to use stored batch during forward
            self._patch_rtdetr_head()

        except ImportError as e:
            raise ImportError(
                "The 'ultralytics' package is required for RT-DETR models but not installed."
            ) from e

    def _patch_cdn_group(self):
        """Patch get_cdn_group to ensure all tensors are on the same device."""
        if self._cdn_patched:
            return

        try:
            import logging
            from ultralytics.models.utils import ops
            import torch as torch_module

            # Store the original function
            original_get_cdn_group = ops.get_cdn_group

            def get_cdn_group_device_safe(batch, num_classes, num_queries, class_embed, num_dn=100, *args, **kwargs):
                """
                Device-safe wrapper for get_cdn_group.
                Calls the original function but ensures all tensors are on the correct device.

                The issue is that get_cdn_group creates intermediate tensors without specifying device,
                causing them to default to CPU even when inputs are on CUDA.
                """
                if batch is None:
                    return None, None, None, None

                # Extract actual device from batch tensors or class_embed
                actual_device = None
                if isinstance(batch, dict):
                    if 'bboxes' in batch and hasattr(batch['bboxes'], 'device'):
                        actual_device = batch['bboxes'].device
                    elif 'cls' in batch and hasattr(batch['cls'], 'device'):
                        actual_device = batch['cls'].device
                    elif 'img' in batch and hasattr(batch['img'], 'device'):
                        actual_device = batch['img'].device

                # Fallback to class_embed device
                if actual_device is None and hasattr(class_embed, 'device'):
                    actual_device = class_embed.device

                # If we still don't have a device, return None
                if actual_device is None:
                    return None, None, None, None

                # Move all batch tensors to the target device BEFORE calling get_cdn_group
                batch_on_device = {}
                if isinstance(batch, dict):
                    for key, value in batch.items():
                        if hasattr(value, 'to'):
                            # It's a tensor, move to device
                            batch_on_device[key] = value.to(actual_device)
                        else:
                            # Not a tensor (e.g., gt_groups list), keep as is
                            batch_on_device[key] = value
                else:
                    batch_on_device = batch

                # Also ensure class_embed is on the correct device
                if hasattr(class_embed, 'to') and class_embed.device != actual_device:
                    class_embed = class_embed.to(actual_device)

                # Monkey-patch torch.tensor to use the correct device during get_cdn_group execution
                # This fixes the internal tensor creation that doesn't specify device
                original_torch_tensor = torch_module.tensor

                def tensor_with_device(*args, **kwargs_inner):
                    if 'device' not in kwargs_inner:
                        kwargs_inner['device'] = actual_device
                    return original_torch_tensor(*args, **kwargs_inner)

                torch_module.tensor = tensor_with_device

                try:
                    result = original_get_cdn_group(batch_on_device, num_classes, num_queries, class_embed, num_dn, *args, **kwargs)
                except Exception as e:
                    import logging
                    logging.warning(f"get_cdn_group failed, disabling CDN: {e}")
                    return None, None, None, None
                finally:
                    # Restore original torch.tensor
                    torch_module.tensor = original_torch_tensor

                # Ensure all returned tensors are on the correct device
                if result is None or result == (None, None, None, None):
                    return None, None, None, None

                dn_embed, dn_bbox, attn_mask, dn_meta = result

                # Move tensors to device if needed (should already be on correct device, but double-check)
                if dn_embed is not None and hasattr(dn_embed, 'device') and dn_embed.device != actual_device:
                    dn_embed = dn_embed.to(actual_device)
                if dn_bbox is not None and hasattr(dn_bbox, 'device') and dn_bbox.device != actual_device:
                    dn_bbox = dn_bbox.to(actual_device)
                if attn_mask is not None and hasattr(attn_mask, 'device') and attn_mask.device != actual_device:
                    attn_mask = attn_mask.to(actual_device)

                return dn_embed, dn_bbox, attn_mask, dn_meta

            # Replace the function globally
            ops.get_cdn_group = get_cdn_group_device_safe
            self._cdn_patched = True
            logging.info("Successfully patched get_cdn_group to be device-safe")

        except Exception as e:
            import logging
            logging.warning(f"Failed to patch get_cdn_group: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def _patch_rtdetr_head(self):
        """Patch RT-DETR head forward to use stored batch attribute and disable CDN for adversarial attacks."""
        if self._head_patched:
            return

        try:
            import types

            # Get the head layer (last layer)
            if hasattr(self.model, 'model') and hasattr(self.model.model, '__getitem__'):
                head_layer = self.model.model[-1]

                # Check if it's RT-DETR head
                if 'RTDETR' in head_layer.__class__.__name__:
                    # Store original forward as a separate attribute
                    head_layer._original_forward = head_layer.forward

                    # Create wrapped forward that uses stored batch
                    def forward_with_stored_batch(self, x, batch=None):
                        # Use stored batch if not provided
                        if batch is None and hasattr(self, '_stored_batch'):
                            batch = self._stored_batch
                        # Call the original forward with proper batch
                        # Now get_cdn_group is patched to be device-safe
                        return self._original_forward(x, batch)

                    # Bind the function as a method to the head layer
                    head_layer.forward = types.MethodType(forward_with_stored_batch, head_layer)
                    self._head_patched = True

        except Exception as e:
            # If patching fails, continue without it
            import logging
            logging.warning(f"Failed to patch RT-DETR head: {e}")

    def forward(self, images, targets=None):
        """
        Forward pass with training/inference mode handling.

        :param images: Input images tensor
        :param targets: List of target dicts with 'boxes', 'labels', 'scores' (for training)
        :return: Loss dict (training) or predictions list (inference)
        """
        if self.training:
            # Training mode - compute loss
            if targets is None:
                raise ValueError("Targets should not be None when training.")

            device_obj = torch.device(self.device)

            # Align model/loss buffers with incoming tensor device
            self.model.to(images.device)
            if hasattr(self.model, "criterion"):
                criterion = self.model.criterion
                if hasattr(criterion, "device"):
                    criterion.device = images.device
                if hasattr(criterion, "to"):
                    criterion.to(images.device)

                # Move all criterion tensors to the correct device
                # This is important for RT-DETR CDN
                for attr_name in dir(criterion):
                    if not attr_name.startswith('_'):
                        attr = getattr(criterion, attr_name, None)
                        if isinstance(attr, torch.Tensor):
                            setattr(criterion, attr_name, attr.to(images.device))

            # Ensure targets tensor is on the same device
            targets = targets.to(images.device)

            # Create items dict for RT-DETR loss
            items = {}
            # RT-DETR requires batch_idx and cls to be long/int for indexing
            items["batch_idx"] = targets[:, 0].long().to(images.device)
            items["bboxes"] = targets[:, 2:6].to(images.device)
            items["cls"] = targets[:, 1].long().to(images.device)
            items["img"] = images.to(images.device)

            # RT-DETR requires gt_groups for contrastive denoising (CDN)
            # gt_groups indicates which ground truth boxes belong to which image in the batch
            # Calculate gt_groups from batch_idx
            batch_idx = items["batch_idx"]  # Already on correct device and dtype
            unique_batch_idx = torch.unique(batch_idx, sorted=True)
            gt_groups = []
            for idx in unique_batch_idx:
                count = (batch_idx == idx).sum().item()
                gt_groups.append(count)
            # Ensure gt_groups is on the same device as images and is long type
            items["gt_groups"] = torch.tensor(gt_groups, dtype=torch.long, device=images.device)

            # RT-DETR head needs batch info during forward pass
            # Store batch in head layer so our patched forward can use it
            if hasattr(self.model, 'model') and hasattr(self.model.model, '__getitem__'):
                try:
                    head_layer = self.model.model[-1]
                    # Make sure head layer and all its parameters are on the correct device
                    head_layer.to(images.device)

                    # Also ensure all embeddings and learned parameters are on correct device
                    # RT-DETR head has learnable query embeddings
                    for param in head_layer.parameters():
                        if param.device != images.device:
                            param.data = param.data.to(images.device)
                    for buffer in head_layer.buffers():
                        if buffer.device != images.device:
                            buffer.data = buffer.data.to(images.device)

                    head_layer._stored_batch = items
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to move RT-DETR head to device: {e}")
                    pass

            # Initialize criterion if not exists
            if not hasattr(self.model, "criterion"):
                if hasattr(self.model, "init_criterion"):
                    self.model.criterion = self.model.init_criterion()

            # Prepare targets dict for RT-DETR
            # RT-DETR returns 5 values: dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
            bs = images.shape[0]
            batch_idx_tensor = items["batch_idx"]
            gt_groups_list = [(batch_idx_tensor == i).sum().item() for i in range(bs)]
            targets_dict = {
                "cls": items["cls"].to(images.device, dtype=torch.long).view(-1),
                "bboxes": items["bboxes"].to(device=images.device),
                "batch_idx": items["batch_idx"].to(images.device, dtype=torch.long).view(-1),
                "gt_groups": gt_groups_list,
            }

            # Get predictions by calling model's forward pass directly
            # The batch is already stored in the head layer via _stored_batch
            # Set model to training mode to get the 5-value output
            original_training = self.model.training
            self.model.train()

            try:
                preds = self.model(images)
            except Exception as e:
                import logging
                logging.error(f"Failed to get predictions from RT-DETR model: {e}")
                logging.error(f"Model type: {type(self.model)}")
                logging.error(f"Model training mode: {self.model.training}")
                raise
            finally:
                # Restore original training state
                if not original_training:
                    self.model.eval()

            # RT-DETR returns 5 values during training: dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
            try:
                dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds
            except (ValueError, TypeError) as e:
                import logging
                logging.error(f"Failed to unpack RT-DETR predictions: {e}")
                logging.error(f"preds type: {type(preds)}")
                if isinstance(preds, (tuple, list)):
                    logging.error(f"preds length: {len(preds)}")
                    for i, p in enumerate(preds):
                        if hasattr(p, 'shape'):
                            logging.error(f"preds[{i}] shape: {p.shape}")
                        else:
                            logging.error(f"preds[{i}] type: {type(p)}")
                raise

            # Process predictions for loss computation
            if dn_meta is None:
                dn_bboxes, dn_scores = None, None
            else:
                dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
                dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

            # Concatenate encoder and decoder outputs
            dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
            dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

            # Call criterion with properly formatted predictions
            loss = self.model.criterion(
                (dec_bboxes, dec_scores), targets_dict, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
            )

            # Clean up batch reference
            if hasattr(self.model, 'model') and hasattr(self.model.model, '__getitem__'):
                try:
                    head_layer = self.model.model[-1]
                    head_layer._stored_batch = None
                except Exception:
                    pass

            # Create loss dict - RT-DETR returns dict with various loss components
            loss_components_dict = {"loss_total": sum(loss.values())}
            # RT-DETR returns loss_giou, loss_class (not loss_cls), and loss_bbox
            if "loss_giou" in loss:
                loss_components_dict["loss_giou"] = loss["loss_giou"]
            if "loss_class" in loss:
                loss_components_dict["loss_cls"] = loss["loss_class"]
            if "loss_bbox" in loss:
                loss_components_dict["loss_l1"] = loss["loss_bbox"]

            return loss_components_dict
        else:
            # Inference mode
            preds = self.model(images)

            # If attack loss mode, return raw predictions
            if self.return_raw_predictions:
                # RT-DETR returns a single tensor during inference (after head processing)
                # Shape: (batch_size, num_queries, 84) where 84 = 4 (bbox) + 80 (classes)
                # IMPORTANT: RT-DETR outputs NORMALIZED cxcywh coordinates (0~1 range)
                # We need to convert to pixel coordinates (xywh) for consistency
                # Note: RT-DETR doesn't have objectness score (DETR-based, not anchor-based)

                if isinstance(preds, torch.Tensor):
                    # Already in good format: (batch, num_queries, features)
                    # RT-DETR typically has 300 queries
                    # Extract bbox and class scores
                    bbox_norm = preds[..., :4]  # (batch, num_queries, 4) - normalized cxcywh
                    class_scores = preds[..., 4:]  # (batch, num_queries, num_classes)

                    # Convert normalized coordinates to pixel coordinates
                    # RT-DETR outputs normalized cxcywh (0~1), need to scale to image size
                    h, w = images.shape[2], images.shape[3]
                    scale = torch.tensor([w, h, w, h], device=preds.device, dtype=preds.dtype)
                    bbox_pixels = bbox_norm * scale  # (batch, num_queries, 4) - pixel cxcywh (xywh)

                    # Combine: (batch, num_queries, 4 + num_classes)
                    scaled_preds = torch.cat([bbox_pixels, class_scores], dim=-1)
                    return scaled_preds
                else:
                    # If preds is tuple/list, it might be from training mode
                    # Extract decoder predictions
                    if isinstance(preds, (tuple, list)) and len(preds) >= 2:
                        # Training format: (dec_bboxes, dec_scores, ...)
                        # dec_scores shape: (num_decoder_layers, batch, num_queries, num_classes)
                        # dec_bboxes shape: (num_decoder_layers, batch, num_queries, 4) - normalized cxcywh
                        dec_bboxes, dec_scores = preds[0], preds[1]

                        # Use last decoder layer predictions
                        final_bboxes_norm = dec_bboxes[-1]  # (batch, num_queries, 4) - normalized
                        final_scores = dec_scores[-1]  # (batch, num_queries, num_classes)

                        # Convert normalized coordinates to pixel coordinates
                        h, w = images.shape[2], images.shape[3]
                        scale = torch.tensor([w, h, w, h], device=final_bboxes_norm.device, dtype=final_bboxes_norm.dtype)
                        final_bboxes_pixels = final_bboxes_norm * scale  # pixel cxcywh (xywh)

                        # Concatenate: (batch, num_queries, 4 + num_classes)
                        combined = torch.cat([final_bboxes_pixels, final_scores], dim=-1)
                        return combined
                    else:
                        raise ValueError(
                            f"Unexpected RT-DETR prediction format: {type(preds)}"
                        )

            # Normal inference mode: Apply NMS and postprocessing
            self.predictor.model = self.model
            self.predictor.batch = [images]
            preds = self.predictor.postprocess(preds, images, images)

            # Convert to ART format
            items = []
            for pred in preds:
                items.append({
                    "boxes": pred.boxes.xyxy,
                    "scores": pred.boxes.conf,
                    "labels": pred.boxes.cls.type(torch.int)
                })
            return items
