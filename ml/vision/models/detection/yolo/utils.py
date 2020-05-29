from pathlib import Path
import torch

from ..... import io, logging

def parse(cfg):
    import re
    with open(cfg, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    lines = [x for x in lines if x and not x.startswith('#')]
    mdefs = []
    for line in lines:
        # print(f"line: {line}")
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].strip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # default to 0 for None
        else:
            key, val = line.split("=")
            key = key.strip()
            val = val.strip()
            if key == 'anchors':  # return nparray
                mdefs[-1][key] = torch.tensor([float(x) for x in re.split(",\s*", val)]).view(-1, 2)
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                mdefs[-1][key] = [int(x) for x in re.split(",\s*", val)]
            else:
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val

    # Check all fields are supported
    supported = {'type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'max_delta'}
    unsupported = set()

    f = set()
    for m in mdefs[1:]:
        [f.add(k) for k in m]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not u, f"Unsupported fields {u} in {cfg}. See https://github.com/ultralytics/yolov3/issues/631"
    return mdefs

import time
from ....ops import *
# from utils.utils import *
def postprocess(prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False, multi_label=False, classes=None):
    """Perform NMS on inference results
    Args:
        prediction(B, AG, 4+1+80): center, width and height refinement, plus anchor and class scores 
                                   per anchor and grid combination
        conf_thres(float): anchor confidence threshold
        iou_thres(float): NMS IoU threshold
        agnostic(bool): class agnostic NMS or per class NMS
        multi_label(bool): whether to select mulitple class labels above the threshold or just the max
        classes(list | tuple): class ids of interest to retain
    Returns:
        output(B, N, 6): list of detections per image in (x1, y1, x2, y2, conf, cls)
    """
    # Settings
    merge = True                # merge for best mAP
    min_wh, max_wh = 2, 4096    # (pixels) minimum and maximum box width and height
    time_limit = 10.0           # seconds to quit after

    t = time.time()
    B, _, nc = prediction.shape
    nc -= 5
    multi_label &= nc > 1                                           # multiple labels per box if nc > 1 too
    output = [None] * B
    for b, x in enumerate(prediction):                              # image index and inference
        x = x[x[:, 4] > conf_thres]                                 # Threshold anchors by confidence
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)] # width/height constraints
        if x.numel() == 0:
            continue

        # Compute resulting class scores = anchor * class
        x[..., 5:] *= x[..., 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(x[:, :4])

        # Single or multi-label boxes
        if multi_label:
            # Combinations of repeated boxes with different classes: [x1, y1, x2, y2, conf, class]
            keep, cls = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((boxes[keep], x[keep, 5 + cls].unsqueeze(1), cls.float().unsqueeze(1)), 1)
        else:  
            # Best class only: [x1, y1, x2, y2, conf, class]
            conf, cls = x[:, 5:].max(1)
            x = torch.cat((boxes, conf.unsqueeze(1), cls.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter out boxes not in any specified classses
        if classes:
            x = x[(cls.view(-1, 1) == torch.tensor(classes, device=cls.device)).any(1)]
        if x.numel() == 0:
            continue

        # Batched NMS
        scores = x[:, 4]
        boxes = x[:, :4]
        cls = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes = boxes + cls.view(-1, 1) * boxes.max()  # boxes (offset by class), scores
        keep = nms(boxes, scores, iou_thres)
        if merge and (1 < x.shape[0] < 3e3):
            # Weighted NMS box merge by IoU * scoress
            try:
                iou = box_iou(boxes[keep], boxes) > iou_thres   # Filtered IoU
                weights = iou * scores[None]                    # weighted IoU by class scores
                x[keep, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            except Exception as e:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                logging.error(f"Failed to merge NMS boxes by weighted IoU: {e}")
                print(x, x.shape, keep, keep.shape)
        output[b] = x[keep]
        if (time.time() - t) > time_limit:
            logging.warning("NMS timeout for {time_limit}s")
            break
    return output