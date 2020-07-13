from pathlib import Path
import pytest
import torch

from ml import cv

SKU110K = dict(
    images=[
        '../../datasets/SKU110K/images/train_0.jpg',
        '../../datasets/SKU110K/images/val_0.jpg',
        '../../datasets/SKU110K/images/test_0.jpg',
    ],
    labels=[
        '../../datasets/SKU110K/labels/train_0.txt',
        '../../datasets/SKU110K/labels/val_0.txt',
        '../../datasets/SKU110K/labels/test_0.txt',
    ]
)

WP = dict(
    images=[
        '../../datasets/WiderPerson/images/000109.jpg',
        '../../datasets/WiderPerson/images/000079.jpg',
    ],
    labels=[
        '../../datasets/WiderPerson/labels/000109.txt',
        '../../datasets/WiderPerson/labels/000079.txt',
    ]
)

@pytest.fixture
def images():
    return WP['images']
    return SKU110K['images']

@pytest.fixture
def labels():
    return WP['labels']
    return SKU110K['labels']

@pytest.fixture
def classes():
    from ml.vision.datasets import sku110k
    return sku110k.SKU110K_CLASSES
    from ml.vision.datasets import widerperson
    return widerperson.WIDERPERSON_CLASSES

@pytest.fixture
def suffix():
    return 'gt'

def test_render_yolo(images, labels, suffix, classes=None, output=None):
    if not isinstance(images, list):
        images = [images]
    if not isinstance(labels, list):
        labels = [labels]
    if output is None:
        output = '.'

    for img, label in zip(images, labels):
        with open(label) as f:
            cxyxy = torch.Tensor([tuple(map(float, line.split())) for line in f.read().splitlines()])
            xyxysc = torch.cat([cxyxy[:, 1:], torch.ones(len(cxyxy), 1), cxyxy[:,0:1]], dim=1)
            path = Path(output, f"{Path(img).stem}-{suffix}.jpg")
            img = cv.imread(img)
            h, w = img.shape[:2]
            xyxysc[:, [0, 2]] *= w
            xyxysc[:, [1, 3]] *= h
            cv.render(img, xyxysc, classes=classes, path=path)