from torch import nn
import torch

from .activation import Mish
from ... import logging

ACTIVATIONS = dict(
    linear=nn.Identity(),
    leaky=nn.LeakyReLU(0.1, inplace=True),
    mish=Mish(),
)

class MaxPool2d(nn.Module):
    @classmethod
    def create(cls, cfg):
        k = cfg['size']
        stride = cfg['stride']
        return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)

class Upsample(nn.Module):
    @classmethod
    def create(cls, cfg):
        return nn.Upsample(scale_factor=cfg['stride'])

class Conv(nn.Sequential):
    """
    [convolutional]
    batch_normalize=1
    filters in [32, 64, ]
    size in [1, 3 ]
    stride in [1, 2, ]
    pad in [1, ]
    activation in [mish, linear, leaky]
    """
    @classmethod
    def create(cls, cfg, in_channels):
        out_channels = cfg['filters']
        kernel_size = cfg['size']
        stride = cfg['stride']
        padding = cfg.get('pad', cfg['size'] // 2)
        activation = cfg['activation']
        with_bn = cfg['batch_normalize'] == 1
        return cls(in_channels, out_channels, kernel_size, stride=stride, padding=padding, activation=activation, with_bn=with_bn)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='mish', with_bn=True):
        self.with_bn = with_bn
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=not with_bn),
            nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4),
        )
        if activation != 'linear':
            self.add_module(str(len(self)), ACTIVATIONS[activation])

class Route(nn.Module):
    """Concatenate output features from one or more previous layers
    [route]
    layers = -1,-10
    """
    @classmethod
    def create(cls, cfg):
        layers = cfg['layers']
        return cls(layers)
    
    def __init__(self, layers):
        super(Route, self).__init__()
        self.layers = layers  # layer indices

    def extra_repr(self):
        return f"layers={tuple(self.layers)}"

    def forward(self, x, outputs):
        if len(self.layers) > 1:
            return torch.cat([outputs[i] for i in self.layers], 1) 
        else: 
            return outputs[self.layers[0]]

# class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
class Shortcut(nn.Module):
    """Multi-input Weighted Residual Connections (MiWRC)
    [shortcut]
    from=-3
    activation=linear
    """
    @classmethod
    def create(cls, cfg):
        layers = cfg['from']
        return cls(layers, weighted='weights_type' in cfg)
    
    def __init__(self, layers, weighted=False):
        super(Shortcut, self).__init__()
        self.layers = layers      # layer indices
        self.weighted = weighted  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weighted:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def extra_repr(self):
        return f"layers={tuple(self.layers)}, weighted={self.weighted}"

    def forward(self, x, outputs):
        # Weights
        if self.weighted:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weighted else outputs[self.layers[i]]
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                logging.warning(f"Inconsistent input channels for WRC between {nx} and {na} from layer[{self.layers[i]}]")
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]
        return x

class YOLOHead(nn.Module):
    """Predict xywh, anchor confidence and class scores for selected anchors.
    [yolo]
    mask = 6,7,8
    anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
    classes=80
    
    scale_x_y = 1.05
    iou_thresh=0.213
    cls_normalizer=1.0
    iou_normalizer=0.07
    nms_kind=greedynms
    beta_nms=0.6

    num=9
    jitter=.3
    ignore_thresh = .7
    truth_thresh = 1
    random=1
    iou_loss=ciou
    max_delta=5
    """
    @classmethod
    def create(cls, mdef, index):
        strides = [8, 16, 32]  # P3, P4, P5 strides
        stride  = strides[index]
        anchors = mdef['anchors'][mdef['mask']]
        return cls(index, stride,
                   anchors,             # anchors in (w, h)+
                   mdef['classes'],)    # number of classes

    def __init__(self, index, stride, anchors, nc):
        super(YOLOHead, self).__init__()
        self.index = index                                          # index of yolo layer
        self.stride = stride                                        # layer stride
        self.anchors = torch.Tensor(anchors)
        self.anchor_vec = self.anchors / self.stride                # in strides
        self.anchor_wh = self.anchor_vec.view(1, len(anchors), 1, 1, 2)  # w.r.t. stride scale
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc            # number of classes (80)
        self.no = 5 + nc        # number of outputs (4+1+80)
        self.grid = None
    
    def extra_repr(self):
        return f"index={self.index}, stride={self.stride}, anchors={self.anchors.tolist()}, nc={self.nc}"

    def forward(self, x):
        # (bs, 255, 13, 13) -> (bs, 3, 85, 13, 13) -> (bs, 3, 13, 13, 85)  
        # (bs, anchors, gridx, gridy, xywh+conf+classes)
        bs, _, ny, nx = x.shape
        if self.grid is None or self.grid.shape[-3:-1] != (ny, nx):
            dev = x.device
            gy, gx = torch.meshgrid([torch.arange(ny, device=dev), torch.arange(nx, device=dev)])
            self.grid = torch.stack((gx, gy), 2).view(1, 1, ny, nx, 2).float()
            self.anchor_wh = self.anchor_wh.to(dev)
            # logging.debug(f"Created inference grid {tuple(self.grid.shape)}")
        x = x.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return x
        else:
            y = x.clone()                                           # inference output
            y[..., :2]  = torch.sigmoid(y[..., :2]) + self.grid     # xy
            y[..., 2:4] = torch.exp(y[..., 2:4]) * self.anchor_wh   # wh
            y[..., :4] *= self.stride                               # scale back
            torch.sigmoid_(y[..., 4:])                              # anchor and class scores?
            # return y.view(bs, -1, self.no), x                       # view [1, 3, 13, 13, 85] as [1, 507, 85]
            return y.view(bs, -1, self.no)                          # view [1, 3, 13, 13, 85] as [1, 507, 85]