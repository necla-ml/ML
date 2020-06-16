import os
import math
import random
from pathlib import Path

import numpy as np

from PIL import Image
import cv2

py_min, py_max = min, max
from cv2 import *

try:
    import torchvision as tv
except ImportError as e:
    pass
else:
    tv.set_image_backend('accimage')

## Image Pixel Format

'''
CV2 BGR: uint8 in HWC ndarray
PIL RGB: uinit8 in HWC ndarray
torch RGB: float in CHW tensor
'''

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
    import torch
    return torch.is_tensor(img) and img.ndimension() == 3

def save(src, path, q=95):
    if isTorch(src):
        src = fromTorch(src)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(path), src, [IMWRITE_JPEG_QUALITY, q])

def loadGrayscale(path):
    src = cv2.imread(str(path), IMREAD_GRAYSCALE)
    return None if src.ndim == 0 else src

def loadBGR(path):
    #src = imread(str(path))
    src = cv2.imread(str(path), IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION)
    return None if src.ndim == 0 else src

def imread(path, nc=3):
    """Load image(s) from path(s).
    Args:
        path(str | list[str]): path(s) to image(s) to read
    Returns:
        output(BGR | list[BGR]): an BGR image or list of images
    """
    if isinstance(path, (str, Path)):
        return loadGrayscale(path) if nc == 1 else loadBGR(path)
    elif isinstance(path, list):
        return [loadGrayscale(p) if nc == 1 else loadBGR(p) for p in path]

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
    import torch
    # BGR2RGB, permute
    src = src[:,:,::-1] if src.ndim == 3 else src
    src = src.transpose(2, 0, 1).astype(np.float32)
    t = torch.from_numpy(src / 255).to(device)
    return t

def resize(img, scale=1, width=0, height=0, interpolation=INTER_LINEAR, **kwargs):
    '''Resize input image of PIL/accimage or OpenCV BGR and convert torch tensor image if necessary
    '''
    if isinstance(img, Image.Image):
        # PIL image
        if width > 0 and height > 0:
            return img.resize((width, height), Image.ANTIALIAS)
        else:
            size = [int(s * scale) for s in img.size]
            return img.resize(size, Image.ANTIALIAS)
    else:
        if isTorch(img):
            img = fromTorch(img)

        # OpenCV BGR image
        if width > 0 and height > 0:
            return cv2.resize(img, (width, height), interpolation=interpolation)
        else:
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)

def letterbox(img, size=608, color=114, pad_w=None, pad_h=None, minimal=True, stretch=False, upscaling=True):
    """Resize and pad to the new shape.
    Args:
        img(BGR): CV2 BGR image
        size[416 | 512 | 608 | 32*]: target long side to resize to in multiples of 32
        color(tuple): Padding color
        pad_w(int): Padding along width
        pad_h(int): Padding along height
        minimal(bool): Padding up to the short side or not
        stretch(bool): Scale the short side without keeping the aspect ratio
        upscaling(bool): Allows to scale up or not
    """
    # Resize image to a multiple of 32 pixels on both sides 
    # https://github.com/ultralytics/yolov3/issues/232
    color = isinstance(color, int) and (color,) * img.shape[-1] or color
    shape = img.shape[:2]
    if isinstance(size, int):
        size = (size, size)

    r = py_min(size[0] / shape[0], size[1] / shape[1])
    if not upscaling: 
        # Only scale down but no scaling up for better test mAP
        r = py_min(r, 1.0)

    # Compute padding
    ratio = r, r
    pw = pad_w and int(round(shape[1] * r - pad_w)) or int(round(shape[1] * r))
    ph = pad_h and int(round(shape[0] * r - pad_h)) or int(round(shape[0] * r))
    new_unpad = pw, ph  # actual size to scale to (w, h)
    dw, dh = size[1] - new_unpad[0], size[0] - new_unpad[1]         # padding on sides

    if minimal: 
        # Padding up to 64 for the short side
        dw, dh = dw % 64, dh % 64
    elif stretch:  
        # Stretch the short side to the exact target size
        dw, dh = 0.0, 0.0
        new_unpad = size
        ratio = size[0] / shape[0], size[1] / shape[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = resize(img, width=new_unpad[0], height=new_unpad[1])

    # Fractional to integral padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    resized = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return resized, dict(
        shape=shape,        # HxW
        offset=(top, left), # H, W
        ratio=ratio,        # H, W
    )

def grid(images, size=608, color=114, padding=(100, 100)):
    """Load images in a grid.
    Args:
        images(list[BGR]): list of BGR images
        size(int): target grid cell resolution to resize and pad
        color(int or tuple): color to pad
        padding(tuple): custom padding on (width, height)
    """
    assert isinstance(images, list) and all([isinstance(img, np.ndarray) for img in images])
    import random
    from ml import math
    min, max = py_min, py_max
    gh, gw = math.factorize(len(images))
    tiles = np.full((size * gh, size * gw, images[0].shape[-1]), 114, dtype=np.uint8)
    metas = []
    for i, img in enumerate(images):
        # Pack img at the mosaic center
        ih = i // gw
        iw = i % gw
        y1, x1 = ih * size, iw * size
        img, meta = letterbox(img, size, minimal=False, pad_w=padding[0], pad_h=padding[1], color=color)
        tiles[y1:y1+size, x1:x1+size] = img[:, :]  # img4[ymin:ymax, xmin:xmax]
        top, left = meta['offset']
        meta['offset'] = (y1+top, x1+left)
        metas.append(meta)

    return tiles, metas

def imshow(img, scale=1, title='', **kwargs):
    import torch
    if type(img) is list and isTorch(img[0]):
        img = torch.cat(img, 2)

    if isTorch(img):
        img = fromTorch(img)

    img = img if scale == 1 else resize(img, scale)
    if isinstance(img, Image.Image):
        img.show()
    else:
        cv2.imshow(title, img)
        waitKey(0)
        destroyAllWindows()

# OpenCV only

def drawBox(img, xyxy, color=None, label=None, thickness=None):
    tl = thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = py_max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def drawBoxes(img, boxes, colors=None, labels=None, scores=None):
    from .vision.ops import clip_boxes_to_image
    for i, box in enumerate(boxes):
        label = None
        if labels:
            label = labels[i] if scores is None else f"{labels[i]} {scores[i]:.2f}"
        drawBox(img, box, color=colors and colors[i] or None, label=label)

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
