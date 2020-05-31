from pathlib import Path
import pytest

from ml.vision.models import yolo4
from ml import cv

@pytest.fixture
def path():
    return '../yolov3/data/samples/bus.jpg'

def test_yolo(path):
    path = Path(path)
    img = cv.imread(path)
    detector = yolo4()
    results = detector.detect(img)
    assert len(results) == 1
    assert results[0].shape[1] == 4+1+1
    detector.render(img, results[0], path=f"export/{path.name}")