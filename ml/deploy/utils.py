import os
import errno
from pathlib import Path

import torch as th
from ml import io, logging

def GiB(val):
    return val * 1 << 30

def export(model, spec, path, **kwargs):
    """
    Args:
        spec(List[Tuple[int,...]]): list of input shapes to export the model in ONNX
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
        input_names = [f'input_{i}' for i in range(len(spec))]
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
    if backend in ['trt', 'tensorrt']:
        # XXX No intermmediate ONNX archive
        from . import trt as backend
        chkpt_path = Path(f"{model_dir}/{name}.pth")
        if chkpt_path.exists() and not reload:
            logging.info(f"Loading torch2trt checkpoint from {chkpt_path}")
            chkpt = th.load(chkpt_path)
            predictor = backend.TRTPredictor()
            predictor.load_state_dict(chkpt)
            return predictor

        trt = Path(f"{model_dir}/{name}.trt")
        input_names = kwargs.pop('input_names', None)
        output_names = kwargs.pop('output_names', None)
        if trt.exists() and not reload:
            logging.info(f"Building TensorRT inference engine from {trt}")
            engine = backend.build(trt)
            predictor = backend.TRTPredictor(engine=engine, 
                                             input_names=input_names,
                                             output_names=output_names)
        else:
            batch_size = kwargs.pop('batch_size', 1)
            workspace_size = kwargs.pop('workspace_size', GiB(1))
            amp = kwargs.pop('amp', False)
            device = next(model.parameters()).device
            inputs = tuple([th.rand(1, *shape, device=device) for shape in spec])
            predictor = backend.torch2trt(model,
                                          inputs,
                                          max_batch_size=batch_size,
                                          max_workspace_size=workspace_size,
                                          input_names=input_names,
                                          output_names=output_names,
                                          fp16_mode=amp,
                                          use_onnx=True)
        logging.info(f"Saving TensorRT checkpoint to {chkpt_path}")
        io.save(predictor.state_dict(), chkpt_path)
        logging.info(f"Built TensorRT inference engine for {time() - t:.3f}s")
        return predictor                
    elif backend == 'onnx':
        onnx_path = Path(f"{model_dir}/{name}.onnx")
        batch_size = kwargs.pop('batch_size', 1)
        workspace_size = kwargs.pop('workspace_size', GiB(1))
        amp = kwargs.pop('amp', False)
        if not onnx_path.exists() or reload:
            # print(spec, onnx_path, kwargs)
            export(model, spec, onnx_path, **kwargs)
        import onnxruntime
        from .onnx import ONNXPredictor
        engine = onnxruntime.InferenceSession(str(onnx_path))
        predictor = ONNXPredictor(engine)
        logging.info(f"Built ONNX inference engine at {onnx_path} for {time() - t:.3f}s")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return predictor
