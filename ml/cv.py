import os
import math
from pathlib import Path

from cv2 import *
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

torchvision.set_image_backend('accimage')

## Essential OpenCV

BLACK    = (  0,   0,   0)
BLUE     = (255,   0,   0)
GREEN    = (  0, 255,   0)
RED      = (  0,   0, 255)
MAROON   = (  0,   0, 128)
YELLOW   = (  0, 255, 255)
WHITE    = (255, 255, 255)
FG       = GREEN
BG       = BLACK


def pts(pts):
    r"""
    Args:
        pts list of x and y or (x, y) tuples
    """
    if type(pts[0]) is int:
        pts = [[[pts[2*i], pts[2*i+1]]]for i in range(len(pts) // 2)]
    elif type(pts[0]) is tuple:
        pts = [[list(p)] for p in pts]
    return np.array(pts)


def isTorch(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def save(src, path, q=95):
    if isTorch(src):
        src = fromTorch(src)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(path), src, [IMWRITE_JPEG_QUALITY, q])

def loadGrayscale(path):
    src = imread(str(path), IMREAD_GRAYSCALE)
    return None if src.ndim == 0 else src

def loadBGR(path):
    #src = imread(str(path))
    src = imread(str(path), IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION)
    return None if src.ndim == 0 else src

def imread(path, nc=3):
    return loadGrayscale(path) if nc == 1 else loadBGR(path)

def fromTorch(src):
    r"""
        Converts a Torch tensor to a CV2 image in BGR format
    """
    if type(src) is list:
        for i, s in enumerate(src):
            src[i] = fromTorch(s)
        return src
    else:
        src = src.permute(1,2,0)
        src = (src * 255).byte().numpy()
        return src[:,:,::-1] if src.ndim == 3 else np.squeeze(src)

def toTorch(src, device='cpu'):
    # BGR2RGB, permute
    src = src[:,:,::-1] if src.ndim == 3 else src
    src = src.transpose(2, 0, 1).astype(np.float32)
    t = torch.from_numpy(src / 255).to(device)
    return t

def resize(img, scale=1, width=0, height=0, interpolation=INTER_LINEAR, **kwargs):
    if isTorch(img):
        img = fromTorch(img)

    if isinstance(img, Image.Image):
        if width > 0 and height > 0:
            return img.resize((width, height), Image.ANTIALIAS)
        else:
            size = [int(s * scale) for s in img.size]
            return img.resize(size, Image.ANTIALIAS)
    else:
        if width > 0 and height > 0:
            return resize(img, (width, height), interpolation=interpolation)
        else:
            return resize(img, None, fx=scale, fy=scale, interpolation=interpolation)

def show(img, scale=1, title='', **kwargs):
    if type(img) is list and isTorch(img[0]):
        img = torch.cat(img, 2)

    if isTorch(img):
        img = fromTorch(img)

    img = img if scale == 1 else resize(img, scale)
    if isinstance(img, Image.Image):
        img.show()
    else:
        imshow(title, img)
        waitKey(0)
        destroyAllWindows()

# OpenCV only

def drawLine(img, x1, y1, x2, y2):
    line(img, (x1, y1), (x2, y2), FG, 1)

def drawRect(img, x, y, w, h):
    rectangle(img, (x, y), (x+w, y+h), FG, 1)

def drawContour(img, contour):
    drawContours(img, [contour], 0, RED, 2)

def fillPoly(img, points, color=None):
    fillPoly(img, [pts(points)], color or RED)

def contour(src, lo=None, hi=None):
    #print(f'[P{os.getpid()}] cvtColor()')
    gray = cvtColor(src, COLOR_BGR2GRAY)
    #print(f'[P{os.getpid()}] blur()')
    blurred = blur(gray, (3, 3))
    #print(f'[P{os.getpid()}] Canny()')
    edges = Canny(blurred, lo or 90, hi or 175)
    
    x, y, xw, yh = (math.inf, math.inf, 0, 0)
    #print(f'[P{os.getpid()}] findContours()')
    contours, _ = findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)[-2:]
    for i, cnt in enumerate(contours):
        area = contourArea(cnt)
        rect = boundingRect(cnt)

        #print(f'area[{i}]={area:.3f}, ({rect[0]}, {rect[1]}, {rect[2]}, {rect[3]})')
        #drawContour(src, cnt)
        #drawRect(src, *rect)
        
        x   = min(x, rect[0])
        y   = min(y, rect[1])
        xw  = max(xw, rect[0] + rect[2] - 1)
        yh  = max(yh, rect[1] + rect[3] - 1)

    w, h = (xw - x + 1, yh - y + 1)
    #drawRect(src, x, y, w, h)
    return x, y, w, h

def crop(src, x, y, w, h, width=0, height=0):
    if width > 0 and height > 0:
        # XXX In case contour is larger than expected in width
        if w > width:
            #print('cv.crop() w > width', x, y, w, h, width, height)
            #show(src, 0.5)
            offset = (w - width) + 32
            x += offset
            w -= offset
        if h > height:
            #print(f'cv.crop(): h={h} > height={height}', x, y, w, h, width, height)
            #offset = (h - height) + 32
            #y += offset
            #h -= offset
            pass

        xw = x + w - 1
        yh = y + h - 1
        offsetX = (width - w) / 2   
        offsetY = (height - h) / 2  
        if offsetX > 0:
            x = round(x - offsetX)      
            xw = round(xw + offsetX)
        if offsetY > 0:
            y = round(y - offsetY)      
            yh = round(yh + offsetY)
        if x < 0:
            xw -= x
            x = 0
        if y < 0:
            yh -= y
            y = 0
        h = yh - y + 1
        w = xw - x + 1
        #print('cv.crop(): rounded ', x, y, w, h, xw, yh, offsetX, offsetY)
    else:
       width  = w
       height = h
    
    src = src[y:y+h, x:x+w]
    if src.shape[0] != height or src.shape[1] != width:
        src = resize(src, width=width, height=height)
    assert(src.shape[0] == height)
    assert(src.shape[1] == width)
    #show(src, 0.5)
    return src

def boundingRect(pts):
    r"""
        Args:
            pts list of points.
    """
    return boundingRect(pts(pts))
