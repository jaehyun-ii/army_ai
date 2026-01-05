"""
Bounding box utility functions for attack loss calculations.
Ported from AEGIS framework.
"""

import torch
import numpy as np


def xywh2xyxy(x):
    """
    Converts bbox format from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        x: Bounding boxes in [x_center, y_center, width, height] format.
           Supports both torch.Tensor and np.ndarray.

    Returns:
        Bounding boxes in [x1, y1, x2, y2] format (top-left, bottom-right).
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    """
    Converts bbox format from [x1, y1, x2, y2] to [x, y, w, h].

    Args:
        x: Bounding boxes in [x1, y1, x2, y2] format (top-left, bottom-right).
           Supports both torch.Tensor and np.ndarray.

    Returns:
        Bounding boxes in [x_center, y_center, width, height] format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on: https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (Tensor[N, 4]): First set of boxes
        box2 (Tensor[M, 4]): Second set of boxes
        eps (float): Small epsilon to prevent division by zero

    Returns:
        iou (Tensor[N, M]): The NxM matrix containing the pairwise IoU values
                           for every element in box1 and box2

    Example:
        >>> box1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])  # 2 boxes
        >>> box2 = torch.tensor([[0, 0, 10, 10]])  # 1 box
        >>> iou = box_iou(box1, box2)
        >>> print(iou.shape)  # torch.Size([2, 1])
    """
    # Get coordinates of intersection rectangle
    # box1: (N, 4) -> (N, 1, 4) -> split into (N, 1, 2), (N, 1, 2)
    # box2: (M, 4) -> (1, M, 4) -> split into (1, M, 2), (1, M, 2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)

    # Calculate intersection area
    # inter: (N, M)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # Calculate union area
    # area1: (N, M), area2: (N, M)
    area1 = (a2 - a1).prod(2)
    area2 = (b2 - b1).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    return inter / (area1 + area2 - inter + eps)
