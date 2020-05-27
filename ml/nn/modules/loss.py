import warnings

from torch.autograd import Variable
from torch.nn import Module
import torch
from .. import functional as F

class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

class SSIMLoss(_Loss):
    r"""
        https://github.com/Po-Hsun-Su/pytorch-ssim
    """

    def __init__(self, window_size = 11, size_average=True, reduce=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.reduce = reduce
        self.nc = 1
        self.window = F.create_window(window_size, self.nc)

    def forward(self, img1, img2, size_average=None, reduce=None):
        nc = img1.size(1)
        if nc == self.nc and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = F.create_window(self.window_size, nc)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())

            window = window.type_as(img1)
            self.window = window
            self.nc = nc
        size_average = self.size_average if size_average is None else size_average
        reduce = self.reduce if reduce is None else reduce
        return 1 - F._ssim(img1, img2, window, self.window_size, nc, size_average, reduce)