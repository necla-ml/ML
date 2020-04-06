from math import exp

import torch
from torch.nn.functional import *

import numpy as np

r"""
    https://github.com/Po-Hsun-Su/pytorch-ssim
"""

def _logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: a Variable with any shape
        dim: an integer
        keepdim: a boolean

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
        
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, nc):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(nc, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, nc, size_average=True, reduce=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = nc)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = nc)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = nc) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = nc) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = nc) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)) # Nx3x32x32
    #print(f'ssim_map.size={ssim_map.size()}')
    if reduce:
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.sum()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True, reduce=True):
    nc = img1.size(1)
    window = create_window(window_size, nc)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, nc, size_average, reduce)

def ssim_loss(x1, x2, window_size=11, size_average=True, reduce=True):
    return 1 - ssim(x1, x2, window_size, size_average, reduce)
