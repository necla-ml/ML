from torchvision.ops.boxes import *
import torch

def xywh2xyxy(x, inplace=False):
    """Convert Nx4 boxes from center, width and height to top-left and bottom-right coordinates.
    Args:
        x(Tensor[N, 4]): N boxes in the format [centerX, centerY, width, height]
        inplace(bool): whether to modify the input inplace as output or make the results a new copy

    Returns:
        boxes(Tensor[N, 4]): N boxes in the format of [x1, y1, x2, y2]
    """
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    if inplace:
        x.copy_(y)
        return x
    return y

def xyxy2xywh(x, inplace=False):
    """Convert Nx4 boxes from top-left and bottom-right coordinates to center, width and height
    Args:
        x(Tensor[N, 4]): N boxes in the format [x1, y1, x2, y2]
        inplace(bool): whether to modify the input inplace as output or make the results a new copy

    Returns:
        boxes(Tensor[N, 4]): N boxes in the format of [centerX, centerY, width, height]
    """
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2   # centerX
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2   # centerY
    y[:, 2] = x[:, 2] - x[:, 0]         # width
    y[:, 3] = x[:, 3] - x[:, 1]         # height
    if inplace:
        x.copy_(y)
        return x
    return y