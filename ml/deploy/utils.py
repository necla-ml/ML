import os
import errno
from pathlib import Path

import torch as th
from ml import logging

def export(model, spec, path, **kwargs):
    """
    Args:
        spec(List[Tuple[...]], Tuple): list of input shapes to export the model in ONNX
    Kwargs:
        export_params=True
        training=None
        input_names=None
        output_names=None
        dynamic_axes=None
        fixed_batch_size=False
        onnx_shape_inference=False
        verbose=False
    """
    path = Path(path)
    if path.suffix == '.onnx':
        from torch import onnx
        logging.info(f"Exporting pytorch model to {path}")
        device = next(model.parameters()).device
        args = tuple([th.rand(1, *shape, device=device) for shape in spec])
        input_names = [f'input{i}' for i in range(len(spec))]
        dynamic_axes = {input_name: {0: 'batch_size'} for input_name in input_names}
        onnx.export(model, args, str(path), 
                    input_names=input_names, 
                    dynamic_axes=dynamic_axes,
                    opset_version=kwargs.pop('opset_version', 11),
                    **kwargs)
    else:
        raise ValueError(f"Unknown suffix `{path.suffix}` to export")

def build(name, model, spec, model_dir=None, backend='trt', reload=False, **kwargs):
    from ml import hub
    if model_dir is None:
        hub_dir = hub.get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise e

    from time import time
    t = time()
    onnx = Path(f"{model_dir}/{name}.onnx")
    if backend in ['trt', 'tensorrt']:
        trt = Path(f"{model_dir}/{name}.trt")
        if trt.exists() and not reload:
            from ml.deploy import trt as backend
            logging.info(f"Deploying TensorRT inference engine at {trt}...")
            engine = backend.build(trt)
        else:
            # pytorch -> ONNX -> TRT
            batch_size = kwargs.pop('batch_size', 1)
            workspace_size = kwargs.pop('workspace_size', 1 << 30)
            amp = kwargs.pop('amp', False)
            if not onnx.exists() or reload:
                export(model, spec, onnx, **kwargs)
            from ml.deploy import trt as backend
            engine = backend.build(onnx, batch_size=batch_size, workspace_size=workspace_size, amp=amp)
        from ml.deploy.trt import TensorRTPredictor
        predictor = TensorRTPredictor(engine)
        logging.info(f"Deployed TensorRT inference engine at {trt} for {time() - t:.3f}s")
    elif backend == 'onnx':
        batch_size = kwargs.pop('batch_size', 1)
        workspace_size = kwargs.pop('workspace_size', 1 << 30)
        amp = kwargs.pop('amp', False)
        if not onnx.exists() or reload:
            print(spec, onnx, kwargs)
            export(model, spec, onnx, **kwargs)
        import onnxruntime
        from ml.deploy.onnx import ONNXPredictor
        engine = onnxruntime.InferenceSession(str(onnx))
        predictor = ONNXPredictor(engine)
        logging.info(f"Deployed ONNX inference engine at {onnx} for {time() - t:.3f}s")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return predictor
