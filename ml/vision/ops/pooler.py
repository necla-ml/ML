import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import *

from ml import logging
from ml.vision.ops import roi_align

class MultiScaleFusionRoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=-1):
        """Multi-scale fusion RoIAlign pooling fuses multi-scale feature maps for consistent RoI algin.
        Args:
            featmap_names (List[str]): names of feature maps used for the pooling.
            output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, x, boxes, shapes):
        """
        Args:
            x (List[List[Tensor]]): list of feature map lists.
            boxes (List[Tensor[N, 4]]): boxes for the pooling operation in (x1, y1, x2, y2) w.r.t. image size.
            shapes (List[Tuple[height, width]]): the sizes of each image input.
        Returns:
            features (List[Tensor[N, C]]): list of pooled RoI features in a batch
        """
        pooled = []
        for (features, shape) in zip(x, shapes):
            size = features[0].shape[2:]
            resampled = [features[0]]
            for i, feats in enumerate(features[1:], 1):
                interpolated = F.interpolate(feats, scale_factor=2 ** i, mode='bilinear', align_corners=False)
                resampled.append(interpolated)
                # logging.info(f"interploation: from {tuple(feats.shape)} to {tuple(interpolated.shape)}")
            features = torch.cat(resampled, 1)
            # logging.info(f"pooled shape: {tuple(pooled[-1].shape)}")
            aligned = roi_align(features, boxes, self.output_size, spatial_scale=(size[1]/shape[1], size[0]/shape[0]))
            pooled.append(aligned)
        return pooled