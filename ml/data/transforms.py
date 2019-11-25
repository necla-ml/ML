import os
import torch
import numbers
from ml import cv
from ml.utils import *

class Resize(object):
    def __init__(self, scale=1, width=0, height=0, interpolation=cv.INTER_LINEAR):
        self.scale, self.width, self.height, self.interpolation = scale, width, height, interpolation

    def __call__(self, img):
        return cv.resize(img, self.scale, self.width, self.height, self.interpolation)

class Crop(object):
    def __init__(self, x, y, w=None, h=None):
        self.x, self.y, self.w, self.h = x, y, w, h

    def __call__(self, img):
        x, y, w, h = self.x, self.y, self.w, self.h
        yh = y + (h if h else img.shape[0])
        xw = x + (w if w else img.shape[1])
        return img[y:yh, x:xw]

class CenterCrop(object):
    def __init__(self, *size):
        self.h, self.w = (int(size[0]), int(size[0])) if len(size) == 1 else size

    def __call__(self, img):
        h, w, _ = img.shape
        return cv.crop(img, int(round((w - self.w) / 2)), int(round((h - self.h) / 2)), self.w, self.h)

class ContourCrop(object):
    def __init__(self, w=0, h=0):
        self.w = w
        self.h = h

    def __call__(self, img):
        x, y, w, h = cv.contour(img)
        return cv.crop(img, x, y, w, h, self.w, self.h)

class CV2Torch(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return cv.toTorch(img)

# input is tensor       
class MaskFill(object):
    r"""Single context mask or multi-ctx masks of slices
    """
    def __init__(self, masks, value=0):
        self.masks = [torch.ones(mask.size()).byte() - mask for mask in iterator(masks)]
        self.value = value

    def __call__(self, img):
        filled = [img.clone().masked_fill_(mask, self.value) for mask in self.masks]
        return filled[0] if len(filled) == 1 else filled

class MaskStack(object):
    r"""Single context mask or multi-ctx masks of slices
    """
    def __init__(self, masks):
        self.masks = [masks] if isinstance(masks[0], slice) else masks

    def __call__(self, img):
        stacks = []
        for mask in self.masks:
            stack = []
            for m in mask:
                stack.append(img[:, m[0], m[1]])
            stacks.append(torch.stack(stack).view(-1, stack[0].size(1), stack[0].size(2)).contiguous())
        return stacks[0] if len(stacks) == 1 else stacks
