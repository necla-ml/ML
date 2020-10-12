import numpy as np
import tensorrt as trt
from tensorrt import volume, Dims
from ml import logging

from ._cuda import (
    HostDeviceMem,
    EXPLICIT_BATCH,
    GiB,
    add_help,
    allocate_buffers,
    do_inference, 
    do_inference_v2,
    cuda,
)

TRT_LOGGER = trt.Logger()

def allocate_buffers(engine, batch_size=1):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for i, binding in enumerate(engine):
        shape, dtype = engine.get_binding_shape(binding), engine.get_binding_dtype(binding)
        size = trt.volume(shape[1:]) * batch_size
        dtype = trt.nptype(dtype)
        logging.info(f"binding[{i}][{binding}] shape={shape}, dtype={dtype}, size={size}")
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

class TensorRTPredictor(object):
    def __init__(self, engine):
        from ml.deploy import trt
        for var in ['has_implicit_batch_dimension', 'max_batch_size', 'max_workspace_size', 'num_layers', 'num_bindings', 'num_optimization_profiles', 'device_memory_size', 'refittable']:
            logging.info(f"engine.{var}={getattr(engine, var)}")
        self.input_shapes = []
        self.output_shapes = []
        pid = 0
        batch_size = 1
        for i, binding in enumerate(engine):
            if engine.binding_is_input(binding):
                self.input_shapes.append((1, *engine.get_binding_shape(binding)[1:]))
                minimum, optimum, maximum = engine.get_profile_shape(pid, binding)
                logging.info(f"binding[{i}][{binding}] input shape={engine.get_binding_shape(binding)}, min={minimum}, opt={optimum}, max={maximum}")
                assert self.input_shapes[-1] == minimum
                pid += 1
                batch_size = minimum[0]
                # logging.info(f"max input binding shape={self.input_shapes[-1]}")
            else:
                self.output_shapes.append((1, *engine.get_binding_shape(binding)[1:]))
                logging.info(f"binding[{i}][{binding}] output shape={self.output_shapes[-1]}")
        
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(engine, batch_size=batch_size)
        for i, (input, shape) in enumerate(zip(self.inputs, self.input_shapes)):
            logging.info(f"inputs[{i}] host={input.host.shape[0]}{(batch_size, *shape[1:])}, device={input.device}")
        for i, (output, shape) in enumerate(zip(self.outputs, self.output_shapes)):
            logging.info(f"outputs[{i}] host={output.host.shape[0]}{(batch_size, *shape[1:])}, device={output.device}")

    def input(self, i):
        return self.inputs[i].host.reshape(self.input_shapes[i])

    def predict(self, *args, **kwargs):
        from ml.deploy import trt
        batch_size = args[0].shape[0]
        if batch_size != self.input_shapes[0][0]:
            logging.info(f"Reallocate buffers for batch size change from {self.input_shapes[0][0]} to {batch_size}")
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, batch_size=batch_size)
        for arg, input in zip(args, self.inputs):
            if arg is not input:
                arg = arg.detach().view(-1).cpu().numpy() if arg.requires_grad else arg.view(-1).cpu().numpy()
                np.copyto(input.host[:arg.size], arg)
        
        # Conclude dynamic input shapes (only batch dim)
        idx = 0
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                bix = self.engine.get_binding_index(binding)
                binding_shape = tuple(self.ctx.get_binding_shape(bix))
                arg_shape = tuple(args[idx].shape)
                if binding_shape != arg_shape:
                    logging.info(f"binding[{binding}][arg{idx}] shape{binding_shape} -> {arg_shape}")
                    self.ctx.set_binding_shape(bix, Dims(args[idx].shape))
                    self.input_shapes[idx] = args[idx].shape
                idx += 1
        
        # outputs = trt.do_inference(self.ctx, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, batch_size=batch_size)
        outputs = trt.do_inference_v2(self.ctx, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        return [output.reshape(-1, *shape[1:])[:batch_size] for output, shape in zip(outputs, self.output_shapes)]

def build(path, batch_size=1, workspace_size=GiB(1), amp=False, strict=False):
    """Build an inference engine from a serialized model.
    Args:
        path: path to a saved onnx or serialized trt model
    """
    path = str(path)
    if path.endswith('onnx'):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = workspace_size
            builder.max_batch_size = batch_size
            if amp:
                builder.fp16_mode = amp
                builder.strict_type_constraints = strict
            logging.info(f"builder.max_batch_size={builder.max_batch_size}")
            logging.info(f"builder.max_workspace_size={builder.max_workspace_size}")
            logging.info(f"builder.platform_has_fast_int8={builder.platform_has_fast_int8}")
            logging.info(f"builder.platform_has_fast_fp16={builder.platform_has_fast_fp16}")
            logging.info(f"builder.fp16_mode={builder.fp16_mode}")
            logging.info(f"builder.strict_type_constraints={builder.strict_type_constraints}")
            logging.info(f'Loading ONNX file from path {path}...')
            with open(path, 'rb') as onnx:
                logging.info(f"Parsing ONNX model at {path}...")
                if not parser.parse(onnx.read()):
                    raise ValueError(f'Failed to parse ONNX at {path}')

            logging.info(f"Building the TensorRT engine from {path}")
            if not network.has_implicit_batch_dimension:
                cfg = builder.create_builder_config()
                if amp:
                    cfg.flags |= 1 << int(trt.BuilderFlag.FP16)
                    # cfg.flags |= 1 << int(trt.BuilderFlag.INT8)
                    if strict:
                        cfg.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)
                for i in range(network.num_inputs):
                    input = network.get_input(i)
                    logging.info(f"network.get_input({i}).shape={input.shape}")
                    if input.shape[0] < 1:
                        profile = builder.create_optimization_profile()
                        min = (1, *input.shape[1:])
                        opt = (batch_size, *input.shape[1:])
                        max = (batch_size, *input.shape[1:])
                        profile.set_shape(f"input{i}", min=trt.Dims(min), opt=trt.Dims(opt), max=trt.Dims(max))
                        cfg.add_optimization_profile(profile)
                        logging.info(f"Set dynamic input{i} shape to min={min}, opt={opt}, max={max}")
                        # input.shape = [batch_size, *input.shape[1:]]
                        # logging.info(f"Reset dynamic input.shape to {tuple(input.shape)} for dynamic batch dim")
                for i in range(network.num_outputs):
                    output = network.get_output(i)
                    logging.info(f"network.get_output({i}).shape={output.shape}")

            flags = [flag for flag in [trt.BuilderFlag.FP16, trt.BuilderFlag.INT8, trt.BuilderFlag.STRICT_TYPES] if 1 << int(flag) & cfg.flags]
            logging.info(f"Builder config.flags={flags}({cfg.flags:b})")
            logging.info(f"Builder config.profile_stream={cfg.profile_stream}")
            logging.info(f"Builder config.num_optimization_profiles={cfg.num_optimization_profiles}")
            logging.info(f"Builder config.max_workspace_size={cfg.max_workspace_size}")
            if cfg.num_optimization_profiles > 0:
                engine = builder.build_engine(network, cfg)
            else:
                engine = builder.build_cuda_engine(network, cfg)
            logging.info("Built the TensorRT inference engine")
            with open(path.replace('onnx', 'trt'), "wb") as f:
                f.write(engine.serialize())
            return engine

    logging.info("Reading the TensorRT engine from {}".format(path))
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())