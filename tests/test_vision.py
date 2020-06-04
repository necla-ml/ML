from pathlib import Path
import pytest
import torch

from ml.vision.models import yolo4
from ml.vision.ops import MultiScaleFusionRoIAlign
from ml import cv, logging


@pytest.fixture
def path():
    return '../yolov3/data/samples/tiles.jpg'
    return '../yolov3/data/samples/bus.jpg'

def test_pooler():
    features = [[
       torch.randn(1, 256, 80, 152),
       torch.randn(1, 512, 40, 76), 
       torch.randn(1, 1024, 20, 38)
    ]]
    shape = [(608, 1216)]
    boxes = torch.rand(6, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    pooler = MultiScaleFusionRoIAlign(3)
    pooled = pooler(features, [boxes], shape)
    logging.info(f"RoI aligned features: {tuple(pooled[0].shape)}")

def test_yolo(path):
    path = Path(path)
    img = cv.imread(path)
    detector = yolo4(fuse=True)
    results = detector.detect([img], size=1216)
    assert len(results) == 1
    assert results[0].shape[1] == 4+1+1
    detector.render(img, results[0], path=f"export/{path.name}")
    assert len(detector.features) == 3
    assert detector.features[0].shape[1] == 256
    assert detector.features[1].shape[1] == 256 * 2
    assert detector.features[2].shape[1] == 256 * 4
    shapes = tuple(tuple(feats.shape) for feats in detector.features)
    logging.info(f"Feature map shapes: {shapes}")
    #for i, (name, m) in enumerate(detector.module.named_modules()):
    #    logging.info(f"[{i}] {name} {hasattr(m, 'pooled') and 'pooled' or ''}")

def test_tracking(video):
    pass