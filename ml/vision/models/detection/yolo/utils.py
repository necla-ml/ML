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