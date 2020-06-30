import os
import math
import random
from typing import *
from pathlib import Path

import numpy as np

from PIL import Image
import cv2

py_min, py_max = min, max
irange = range
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

def letterbox(img, size=736, color=114, minimal=True, stretch=False, upscaling=True):
    """Resize and pad to the new shape.
    Args:
        img(BGR): CV2 BGR image
        size[416 | 512 | 608 | 32*]: target long side to resize to in multiples of 32
        color(tuple): Padding color
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
    pw = int(round(shape[1] * r))
    ph = int(round(shape[0] * r))
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

def grid(images, size=736, color=114):
    """Load images in a grid.
    Args:
        images(list[BGR]): list of BGR images
        size(int): target grid cell resolution to resize and pad
        color(int or tuple): color to pad
    """
    assert isinstance(images, list) and all([isinstance(img, np.ndarray) for img in images])
    import random
    from . import math
    min, max = py_min, py_max
    gh, gw = math.factorize(len(images))
    tiles = np.full((size * gh, size * gw, images[0].shape[-1]), 114, dtype=np.uint8)
    metas = []
    for i, img in enumerate(images):
        # Pack img at the mosaic center
        ih = i // gw
        iw = i % gw
        y1, x1 = ih * size, iw * size
        img, meta = letterbox(img, size, minimal=False, color=color)
        tiles[y1:y1+size, x1:x1+size] = img[:, :]  # img4[ymin:ymax, xmin:xmax]
        top, left = meta['offset']
        meta['offset'] = (y1+top, x1+left)
        metas.append(meta)

    return tiles, metas

def ungrid(grid_image, meta_lst, index_only=True):
    """Get images from grid
    Args:
        grid_image(np.ndarray): grid image 
        meta_lst(list): list of meta data
        index_only(bool): return entire image or image co-ordinates only
    Returns:
        list of image indexes for respective images or numpy image
    """
    assert isinstance(grid_image, np.ndarray) and isinstance(meta_lst, list)
    import random
    from . import math

    no_of_images = len(meta_lst)
    
    gh, gw = math.factorize(no_of_images)

    images = []

    y2_meta = meta_lst[0]['offset'][0]
    x2_meta = meta_lst[0]['offset'][1]

    for i, meta in enumerate(meta_lst):
        size = meta['shape'][0] * no_of_images
        
        ih = i // gw
        iw = i % gw

        y2, x2 = ih * size, iw * size
        y1 = meta['offset'][0]
        y2 = y2 + size - y2_meta
        x1 = meta['offset'][1]
        x2 = x2 + size - x2_meta
        if index_only:
            images.append((y1, y2, x1, x2))
        else:
            img = grid_image[y1:y2, x1:x2, :]
            images.append(img)

    return images

def clip_boxes_to_coord(boxes, coord, size=None):
    """
    Clip boxes so that they lie within coordinates of an image and optionally shift them to size.

    Params:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        coord (Tensor[N, 4]): image coordinates in (x1, y1, x2, y2) format
        size (tuple(height, width)): shift boxes to size
    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    import torch

    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    coord_x = coord[..., 0::2]
    coord_y = coord[..., 1::2]

    boxes_x = boxes_x.clamp(min=coord_x[0], max=coord_x[1])
    boxes_y = boxes_y.clamp(min=coord_y[0], max=coord_y[1])

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    # reshape boxes as earlier shape
    clipped_final = clipped_boxes.reshape(boxes.shape)
    if size:
        diff = coord - torch.Tensor([0, 0, size[1], size[0]]) 
        clipped_final = clipped_final - diff
    
    return clipped_final

def make_grid(tensor, nrow: int = 1, padding: int = 50, normalize: bool = False, range: Optional[Tuple[int, int]] = None, scale_each: bool = False, pad_value: int = 0) -> tuple:
    """
    Make a grid of images.

    Params:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``1``.
        padding (int, optional): amount of padding. Default: ``50``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    Returns:
        tuple of grid image and list of coordinates of individual images (tuple(Tensor, list(tuple)))
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    import torch
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # NOTE: if uncommented, list with single image will not be padded
    # if tensor.size(0) == 1:
    #     return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = py_min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    coordinates = []
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            x1, y1 = x * width + padding,  y * height + padding
            x2, y2 =  x1 + tensor.size(3), y1 + tensor.size(2)
            coordinates.append((x1, y1, x2, y2))
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
            
    return grid, coordinates

def split_boxes(img_coordinates, boxes, boxes_scores=None):
    """
    Split boxes based on IOU of image coordinates and boxes and optionally scores.

    Params:
        img_coordinates (Tensor[N, 4]): image coordinates in (x1, y1, x2, y2) format 
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        boxes_scores (Tensor[N, 1], Optional): box scores
    Returns:
        split boxes based on image coordinates (List(Tupe(Tensor[N,4], Tensor[N,1])))
    """
    import torch
    if not torch.is_tensor(img_coordinates) and torch.is_tensor(boxes):
        raise TypeError('Input arguments must be torch tensors')

    from torchvision.ops.boxes import box_iou

    iou = box_iou(img_coordinates, boxes)

    results = []
    for i, img_coordinate in enumerate(img_coordinates):
        non_zero = (iou[i] != 0).nonzero()
        flattened_non_zero = torch.flatten(non_zero)
        final_boxes = torch.index_select(boxes, 0, flattened_non_zero)
        if not isinstance(boxes_scores, type(None)):
            final_boxes_scores = torch.index_select(boxes_scores, 0, flattened_non_zero)
            results.append((final_boxes, final_boxes_scores))
        else:
            results.append((final_boxes))

    return results
       

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
        cv2.imshow(str(title), img)
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
