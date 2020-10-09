import pytest
import torch as th

@pytest.fixture
def dev():
    import torch as th
    return 'cuda' if th.cuda.is_available() else 'cpu'

@pytest.fixture
def args(dev):
    return th.randn(1, 3, 720, 1280, device=dev)

@pytest.fixture
def model(dev):
    from ml.vision.models.backbone import resnext101
    model = resnext101(pretrained=True, groups=32, width_per_group=8)
    model.eval()
    model.to(dev)
    return model

def test_deploy_onnx(model, dev, args):
    from ml import deploy
    import onnxruntime
    height, width = args.shape[-2:]
    engine = deploy.build('resnext101_32x8d_wsl', model, args,
                          height=height, width=width, backend='onnx', reload=True,
                          input_names=['image'],
                          dynamic_axes={'input': {0: 'batch_size'}})
    assert isinstance(engine, onnxruntime.InferenceSession)

def test_deploy_trt(model, dev, args):
    from ml import deploy
    import tensorrt as trt
    height, width = args.shape[-2:]
    engine = deploy.build('resnext101_32x8d_wsl', model, args, 
                          height=height, width=width, backend='trt', reload=True,
                          input_names=['image'],
                          dynamic_axes={'input': {0: 'batch_size'}})
    assert isinstance(engine, trt.ICudaEngine)