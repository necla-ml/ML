import os
import errno
from pathlib import Path

from ml import logging

def export(model, args, path, **kwargs):
    """
    Args:
        args(Tuple[...], Tensor): dumpy input to the model to export
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
        onnx.export(model, args, str(path), **kwargs)
    else:
        raise ValueError(f"Unknown suffix `{path.suffix}` to export")

def build(name, model, args, height=720, width=1280, model_dir=None, backend='trt', reload=False, **kwargs):
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
        from ml.deploy import trt as backend
        trt = Path(f"{model_dir}/{name}.trt")
        if trt.exists() and not reload:
            logging.info(f"Deploying TensorRT inference engine at {trt}...")
            engine = backend.build(trt)
        else:
            # pytorch -> ONNX -> TRT
            if not onnx.exists() or reload:
                export(model, args, onnx, **kwargs)
            engine = backend.build(onnx)
        logging.info(f"Deployed TensorRT inference engine at {trt} for {time() - t:.3f}s")
    elif backend == 'onnx':
        if not onnx.exists() or reload:
            export(model, args, onnx, **kwargs)
        import onnxruntime
        engine = onnxruntime.InferenceSession(str(onnx))
        logging.info(f"Deployed ONNX inference engine at {onnx} for {time() - t:.3f}s")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return engine
