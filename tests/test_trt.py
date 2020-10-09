import pytest
import torch as th

@pytest.fixture
def dev():
    import torch as th
    return 'cuda' if th.cuda.is_available() else 'cpu'

@pytest.fixture
def model(dev):
    from ml.vision.models.backbone import resnext101
    model = resnext101(pretrained=True, groups=32, width_per_group=8)
    model.eval()
    model.to(dev)
    return model

def test_onnx_trt(model, dev):
    from ml import deploy
    from ml.deploy import trt
    path = 'resnext101.onnx'
    dummy = th.randn(1, 3, 720, 1280, device=dev)
    deploy.export(model, dummy, path)
    engine = trt.build(path, height=dummy.shape[-2], width=dummy.shape[-1])
    assert engine is not None
    print(engine)
