import tensorrt as trt
from ml import logging

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

TRT_LOGGER = trt.Logger()

def build(path, height=640, width=640, batch_size=1, workspace_size=1 << 28):
    """Build an inference engine from a serialized model.
    Args:
        path(str): path to a saved onnx or serialized trt model
    """
    if path.endswith('onnx'):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = workspace_size  # 256MiB
            builder.max_batch_size = batch_size
            logging.info(f'Loading ONNX file from path {path}...')
            with open(path, 'rb') as onnx:
                logging.info(f"Parsing ONNX model at {path}...")
                if not parser.parse(onnx.read()):
                    raise ValueError(f'tyrFailed to parse ONNX at {path}')

            logging.info(f"Building the TensorRT engine from file {path}")
            network.get_input(0).shape = [batch_size, 3, height, width]
            engine = builder.build_cuda_engine(network)
            logging.info("Built the TensorRT inference engine")
            with open(path.replace('onnx', 'trt'), "wb") as f:
                f.write(engine.serialize())
            return engine

    logging.info("Reading the TensorRT engine from {}".format(path))
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())