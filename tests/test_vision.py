from pathlib import Path
import pytest
import torch

from ml import cv, logging
from ml.vision.models import yolo4
from ml.vision.datasets.coco import COCO80_CLASSES
from ml.vision.ops import MultiScaleFusionRoIAlign
from ml.vision.ops import xyxys2xyxysc, xcycwh2xyxy, xcycwh2xywh, xyxy2xcycwh

## Test Box format conversions

@pytest.fixture
def xyxy():
    boxes = torch.randint(100, (4, 4))
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes

@pytest.fixture
def xcycwh(xyxy):
    w = xyxy[:, 2] - xyxy[:, 0] + 1 
    h = xyxy[:, 3] - xyxy[:, 1] + 1 
    xyxy[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) // 2
    xyxy[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) // 2
    xyxy[:, 2] = w
    xyxy[:, 3] = h
    return xyxy

@pytest.fixture
def xyxys():
    dets = [torch.randn(3, 5) for c in range(4)]
    dets[0][:, 2:4] += dets[0][:, 0:2]
    dets[2][:, 2:4] += dets[2][:, 0:2]
    dets[1] = None
    dets[3] = torch.randn(0, 5)
    return dets

@pytest.fixture
def xyxysc():
    dets = []
    for c in range(4):
        boxes = torch.randn(3, 6) * 1280
        boxes[:, -1] = c
        boxes[:, 2:4] += boxes[:, 0:2]
        dets.append(boxes)
    dets[1] = torch.randn(0, 6)
    dets[3] = torch.randn(0, 6)
    return torch.cat(dets)

@pytest.mark.essential
def test_xyxys2xyxysc(xyxys):
    xyxysc = xyxys2xyxysc(xyxys)
    assert torch.is_tensor(xyxysc)
    assert xyxysc[xyxysc[:, -1] == 0].shape[0] == 3
    assert xyxysc[xyxysc[:, -1] == 1].shape[0] == 0
    assert xyxysc[xyxysc[:, -1] == 2].shape[0] == 3
    assert xyxysc[xyxysc[:, -1] == 3].shape[0] == 0
    for c, dets in enumerate(xyxys):
        assert dets is None or len(dets) == 0 or (dets == xyxysc[xyxysc[:, -1] == c][:, :5]).all()
    print()
    print('xyxys:', xyxys)
    print('xyxysc:', xyxysc)

@pytest.mark.essential
def test_xyxysc2xyxys(xyxysc):
    from ml.vision.ops.utils import xyxysc2xyxys
    xyxys = xyxysc2xyxys(xyxysc, 4)
    assert len(xyxys) == 4
    assert len(xyxys[1]) == 0
    assert len(xyxys[3]) == 0
    assert all(map(torch.is_tensor, xyxys))
    print()
    for c, dets in enumerate(xyxys):
        if c in [0, 2]:
            print(c, dets, xyxysc[xyxysc[:, -1] == c][:, :5])
            assert (dets == xyxysc[xyxysc[:, -1] == c][:, :5]).all()
        elif c in [1, 3]:
            assert len(dets) == 0
        else:
            assert False

@pytest.mark.essential
def test_xcycwh2xyxy(xcycwh):
    print()
    print("xcycwh:", xcycwh)
    xyxy = xcycwh2xyxy(xcycwh)
    print("xyxy:", xyxy)
    assert (xcycwh[:, 2] == xyxy[:, 2] - xyxy[:, 0] + 1).all()
    assert (xcycwh[:, 3] == xyxy[:, 3] - xyxy[:, 1] + 1).all()
    assert (xcycwh[:, 0] == (xyxy[:, 0] + xyxy[:, 2]) // 2).all()
    assert (xcycwh[:, 1] == (xyxy[:, 1] + xyxy[:, 3]) // 2).all()
    xcycwh2xyxy(xcycwh, inplace=True)
    assert (xyxy == xcycwh).all()
    
@pytest.mark.essential
def test_xcycwh2xywh(xcycwh):
    print()
    print("xcycwh:", xcycwh)
    xywh = xcycwh2xywh(xcycwh)
    print("xywh:", xywh)
    assert (xcycwh[:, 2] == xywh[:, 2]).all()
    assert (xcycwh[:, 3] == xywh[:, 3]).all()
    x2 = xywh[:, 0] + xywh[:, 2] - 1
    y2 = xywh[:, 1] + xywh[:, 3] - 1
    assert (xcycwh[:, 0] == (xywh[:, 0] + x2) // 2).all()
    assert (xcycwh[:, 1] == (xywh[:, 1] + y2) // 2).all()
    xcycwh2xywh(xcycwh, inplace=True)
    assert (xcycwh == xywh).all()

@pytest.mark.essential
def test_xyxy2xcycwh(xyxy):
    print()
    print("xyxy:", xyxy)
    xcycwh = xyxy2xcycwh(xyxy)
    print("xcycwh:", xcycwh)
    w = xyxy[:, 2] - xyxy[:, 0] + 1
    h = xyxy[:, 3] - xyxy[:, 1] + 1
    print("h:", h)
    assert (xcycwh[:, 2] == w).all()
    assert (xcycwh[:, 3] == h).all()
    xc = (xyxy[:, 0] + xyxy[:, 2]) // 2
    yc = (xyxy[:, 1] + xyxy[:, 3]) // 2
    assert (xcycwh[:, 0] == xc).all()
    assert (xcycwh[:, 1] == yc).all()
    xyxy2xcycwh(xyxy, inplace=True)
    assert (xcycwh == xyxy).all()

## Test YOLO

@pytest.fixture
def path():
    return '../yolov3/data/samples/bus.jpg'
    return '../yolov3/data/samples/tiles.jpg'

@pytest.mark.essential
def test_multiscale_fusion_align():
    from ml.vision import ops
    pooler = MultiScaleFusionRoIAlign(3)
    features = [
       torch.randn(2, 256, 76, 60),
       torch.randn(2, 512, 38, 30), 
       torch.randn(2, 1024, 19, 15)
    ]
    metas = [dict(
        shape=(1080, 810),
        offset=(0, (608-810/1080*608) % 64),
        ratio=(608/1080,)*2,
    ), dict(
        shape=(540, 405),
        offset=(0, (608-405/540*608) % 64),
        ratio=(608/540,)*2,
    )]

    boxes = torch.rand(6, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    rois = [boxes, boxes * 2]
    pooled = pooler(features, rois, metas)
    logging.info(f"RoI aligned features: {tuple(feats.shape for feats in pooled)}")
    assert list(pooled[0].shape) == [len(rois[0]), 1024+512+256, 3, 3]

@pytest.mark.essential
def test_yolo(path):
    path = Path(path)
    img = cv.imread(path)
    img2 = cv.resize(img, scale=0.5)
    detector = yolo4(fuse=True, pooling=True)
    dets, features = detector.detect([img, img2], size=608)
    print([det.shape for det in dets], [feats.shape for feats in features])
    assert len(dets) == 2
    assert dets[0].shape[1] == 4+1+1
    detector.render(img, dets[0], path=f"export/{path.name}")
    detector.render(img2, dets[1], path=f"export/{path.name[:-4]}2.jpg")

## Test Tracking

@pytest.fixture
def video():
    import os
    return os.path.join(os.environ['HOME'], 'Videos', 'store720p-short.264')
    return os.path.join(os.environ['HOME'], 'Videos', 'calstore-concealing.mp4')

def test_deep_sort(video):
    import numpy as np
    from ml.vision.models.tracking.dsort import DeepSort
    detector = yolo4(fuse=True, pooling=True)
    pooler = MultiScaleFusionRoIAlign(3)
    tracker = DeepSort(max_feat_dist=0.2,
                       nn_budget=100, 
                       max_iou_dist=0.7,    # 0.7
                       max_age=15,          # 30 (FPS)
                       n_init=3)            # 3

    from ml import av
    s = av.open(video)
    v = s.decode()
    video = Path(video)

    media = av.open(f"export/{video.stem}/{video.stem}-tracking.mp4", 'w')
    stream = media.add_stream('h264', 15)
    stream.bit_rate = 2000000
    for i, frame in enumerate(v):
        frame = frame.to_rgb().to_ndarray()[:,:,::-1]
        dets, features = detector.detect([frame], size=608)
        if True:
            person = dets[0][:, -1] == 0
            dets[0] = dets[0][person]
            features[0] = features[0][person]

        assert len(dets) == 1
        assert len(dets[0]) == features[0].shape[0]
        assert dets[0].shape[1] == 4+1+1
        assert features[0].shape[1] == 256+512+1024

        if len(dets[0]) > 0:
            D = 1
            for s in features[0].shape[1:]:
                D *= s
            tracker.update(dets[0], features[0].view(len(features[0]), D))
            #if i == 60:
            #    break
            logging.info(f"[{i}] dets[0]: {dets[0].shape}, features[0]: {features[0].shape}")
            #detector.render(frame, dets[0], path=f"export/{video.stem}/dets/frame{i:03d}.jpg")
        
        snapshot = tracker.snapshot()
        logging.info(f"[{i}] snapshot[0]: {snapshot and list(zip(*snapshot))[0] or len(snapshot)}")
        # detector.render(frame, snapshot, path=f"export/{video.stem}/tracking/frame{i:03d}.jpg")
        frame = detector.render(frame, snapshot)

        if media is not None:
            frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            packet = stream.encode(frame)
            media.mux(packet)
    if media is not None:
        packet = stream.encode(None)
        media.mux(packet)
        media.close()