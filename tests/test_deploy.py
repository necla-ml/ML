import pytest
import numpy as np
import torch as th
from ml import (
    deploy, 
    logging
)

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def shape():
    return 3, 720, 1280

@pytest.fixture
def dev():
    return 'cuda' if th.cuda.is_available() else 'cpu'

@pytest.fixture
def args(shape, dev):
    return th.rand(1, *shape, device=dev)

@pytest.fixture
def batch(batch_size, shape):
    return th.rand(batch_size, *shape)

@pytest.fixture
def model(dev):
    from ml.vision.models.backbone import resnext101
    model = resnext101(pretrained=True, groups=32, width_per_group=8)
    model.eval()
    model.to(dev)
    return model

def test_deploy_onnx(batch, model, dev):
    B = batch.shape[0]
    engine = deploy.build('resnext101_32x8d_wsl',
                            model,
                            [batch.shape[1:]],
                            backend='onnx', 
                            reload=True)
    for B in [1, 2]:
        outputs = engine.predict(batch[:B])
        spatial_feats, scene_feats = outputs[-2][:B], outputs[-1][:B]
        assert spatial_feats.shape == (B, 2048, 23, 40)
        assert scene_feats.shape == (B, 2048)
        with th.no_grad():
            torch_outputs = model(batch[:B].to(dev))
        for i, (torch_output, output) in enumerate(zip(torch_outputs, outputs)):
            # logging.info(f"output[{i}] shape={tuple(output.shape)}")
            np.testing.assert_allclose(torch_output.cpu().numpy(), output[:B], rtol=1e-03, atol=3e-04)
            th.testing.assert_allclose(torch_output, th.from_numpy(output[:B]).to(dev), rtol=1e-03, atol=3e-04)

@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("amp", [True, False])
def test_deploy_trt(benchmark, batch, model, dev, amp, B):
    engine = deploy.build(f"x101_32x8d_wsl-bs{batch.shape[0]}{amp and '-amp' or ''}",
                          model,
                          [batch.shape[1:]],
                          backend='trt', 
                          reload=not True,
                          batch_size=batch.shape[0],
                          amp=amp)

    outputs = benchmark(engine.predict, batch[:B])
    # outputs = engine.predict(batch[:B])
    for i, (input, shape) in enumerate(zip(engine.inputs, engine.input_shapes)):
        # logging.info(f"inputs[{i}] {engine.input(i).shape} == {tuple(shape)}")
        assert engine.input(i).shape == shape
        assert input.host.size == np.prod(shape)
    spatial_feats, scene_feats = outputs[-2][:B], outputs[-1][:B]
    assert spatial_feats.shape == (B, 2048, 23, 40)
    assert scene_feats.shape == (B, 2048)
    with th.no_grad():
        # with th.cuda.amp.autocast(enabled=amp):
        torch_outputs = model(batch[:B].to(dev))
    for i, (torch_output, output) in enumerate(zip(torch_outputs, outputs)):
        # logging.info(f"output[{i}] shape={tuple(output.shape)}@{output.dtype}, torch shape={tuple(torch_output.shape)}@{torch_output.dtype}")
        logging.info(f"output[{i}] trt norm={np.linalg.norm(output)}, torch norm={torch_output.norm()}")
        if amp:
            pass
            #np.testing.assert_allclose(torch_output.cpu().numpy(), output[:B], rtol=1e-02, atol=3e-02)
            #th.testing.assert_allclose(torch_output, th.from_numpy(output[:B]).to(dev), rtol=1e-02, atol=3e-02)
        else:
            np.testing.assert_allclose(torch_output.cpu().numpy(), output[:B], rtol=1e-03, atol=3e-04)
            th.testing.assert_allclose(torch_output, th.from_numpy(output[:B]).to(dev), rtol=1e-03, atol=3e-04)