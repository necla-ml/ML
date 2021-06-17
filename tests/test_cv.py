from pathlib import Path
import pytest
import torch as th

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

# @pytest.mark.essential
def test_pil_to_cv():
    from PIL import Image
    from ml import cv
    import numpy as np
    pic = Image.new('RGB', (320, 240))
    src = np.zeros((320, 240, 3), dtype=np.uint8)
    src[:, :, 0] = 0
    src[:, :, 1] = 1
    src[:, :, 2] = 2
    pic = Image.fromarray(src)
    img = cv.pil_to_cv(pic, format='RGB')
    assert np.all(img[:, :, 0] == 0)
    assert np.all(img[:, :, 1] == 1)
    assert np.all(img[:, :, 2] == 2)
    img = cv.pil_to_cv(pic, format='BGR')
    assert np.all(img[:, :, 0] == 2)
    assert np.all(img[:, :, 1] == 1)
    assert np.all(img[:, :, 2] == 0)

@pytest.mark.essential
def test_resize_tensor():
    from ml import cv
    H, W, size = 720, 1280, 256
    img = th.randn(3, H, W)
    resized = cv.resize(img, size)
    h, w = resized.shape[-2:]
    assert (h, w) == (size, int(W / H * size)), f"mismatched after resize: (h, w) == {(h, w)} but {(size, int(W / H * size))} expected"
    resized = cv.resize(img, 256, constraint='shorter')
    h, w = resized.shape[-2:]
    assert (h, w) == (size, int(W / H * size)), f"mismatched after resize: (h, w) == {(h, w)} but {(size, int(W / H * size))} expected"
    resized = cv.resize(img, 256, constraint='longer')
    h, w = resized.shape[-2:]
    assert (h, w) == (int(H / W * size), size), f"mismatched after resize: (h, w) == {(h, w)} but {(H / W * size, size)} expected"

@pytest.mark.essential
def test_resize_cv2():
    from ml import cv
    H, W, size = 720, 1280, 256
    img = th.randn(3, H, W)
    img = cv.fromTorch(img)
    assert img.shape == (H, W, 3)
    resized = cv.resize(img, size)
    h, w = resized.shape[-3:-1]
    assert (h, w) == (size, int(W / H * size)), f"mismatched after resize: (h, w) == {(h, w)} but {(size, int(W / H * size))} expected"
    resized = cv.resize(img, 256, constraint='shorter')
    h, w = resized.shape[-3:-1]
    assert (h, w) == (size, int(W / H * size)), f"mismatched after resize: (h, w) == {(h, w)} but {(size, int(W / H * size))} expected"
    resized = cv.resize(img, 256, constraint='longer')
    h, w = resized.shape[-3:-1]
    assert (h, w) == (int(H / W * size), size), f"mismatched after resize: (h, w) == {(h, w)} but {(H / W * size, size)} expected"

@pytest.mark.essential
def test_resize_pil():
    from ml import cv
    from PIL import Image
    H, W, size = 720, 1280, 256
    img = th.randn(3, H, W)
    img = cv.fromTorch(img)
    img = Image.fromarray(img)
    assert img.size == (W, H)
    assert img.mode == 'RGB'
    resized = cv.resize(img, size)
    w, h = resized.size
    assert (h, w) == (size, int(W / H * size)), f"mismatched after resize: (h, w) == {(h, w)} but {(size, int(W / H * size))} expected"
    resized = cv.resize(img, 256, constraint='shorter')
    w, h = resized.size
    assert (h, w) == (size, int(W / H * size)), f"mismatched after resize: (h, w) == {(h, w)} but {(size, int(W / H * size))} expected"
    resized = cv.resize(img, 256, constraint='longer')
    w, h = resized.size
    assert (h, w) == (int(H / W * size), size), f"mismatched after resize: (h, w) == {(h, w)} but {(H / W * size, size)} expected"


def test_render_yolo(images, labels, suffix, classes=None, output=None):
    if not isinstance(images, list):
        images = [images]
    if not isinstance(labels, list):
        labels = [labels]
    if output is None:
        output = '.'

    for img, label in zip(images, labels):
        with open(label) as f:
            cxyxy = th.Tensor([tuple(map(float, line.split())) for line in f.read().splitlines()])
            xyxysc = th.cat([cxyxy[:, 1:], torch.ones(len(cxyxy), 1), cxyxy[:,0:1]], dim=1)
            path = Path(output, f"{Path(img).stem}-{suffix}.jpg")
            img = cv.imread(img)
            h, w = img.shape[:2]
            xyxysc[:, [0, 2]] *= w
            xyxysc[:, [1, 3]] *= h
            cv.render(img, xyxysc, classes=classes, path=path)