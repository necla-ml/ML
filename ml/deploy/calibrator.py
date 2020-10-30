import os
from uuid import uuid4

import torch
import numpy as np
from PIL import Image
import tensorrt as trt

from ml import logging

logging.Logger(name='Calibrator').setLevel('INFO')

def preprocess(image, channels=3, height=640, width=640):
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.BILINEAR)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logging.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    img_data /= 255.0

    return torch.from_numpy(img_data)

# https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html
class Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator

    Calibration data is randomly generated based on given input shapes and batch size

    Parameters
    ----------
    batch_size: int
        Number of images to pass through in one batch during calibration
    input_shape: Tuple[int]
        Tuple of integers defining the shape of input to the model (Default: (3, 224, 224))
    cache_file: str
        Name of file to read/write calibration cache from/to.
    device: str or int 
        Device for calibration data (Default: 0 ==> cuda:0)
    max_calid_data: int
        Maximum caliration dataset size (Default: 512)
    """
    def __init__(self,
                 batch_size=32,
                 inputs=[],
                 cache_file=None,
                 device=0,
                 calibration_files=[],
                 max_calib_data=512,
                 preprocess_func=None,
                 algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2):
        super().__init__()

        self.inputs = inputs
        if not inputs:
            raise ValueError('Input shapes is required to generate calibration dataset')

        # unique cache file name in case mutliple engines are built in parallel
        self.cache_file = cache_file or f'{uuid4().hex}.cache'
        self.batch_size = batch_size
        self.max_calib_data = max_calib_data
        self.algorithm = algorithm

        self.files = calibration_files

        self.buffers = [torch.empty((batch_size, *input_shape), dtype=torch.float32, device=device)
                        for input_shape in inputs]

        # Pad the list so it is a multiple of batch_size
        if self.files and len(self.files) % self.batch_size != 0:
            logging.info("Padding # calibration files to be a multiple of batch_size {:}".format(self.batch_size))
            self.files += calibration_files[(len(calibration_files) % self.batch_size):self.batch_size]

        self.batches = self.load_batches()

        if not preprocess_func:
            logging.warning('Using a general preprocess function that will resize to input shape and normalize between 0 and 1. \
                            If you need to apply any other transformation, please specify a custom preprocess function')
        self.preprocess_func = preprocess_func or preprocess

    def get_algorithm(self):
        return self.algorithm

    def load_batches(self):
        if self.files:
            # Populates a persistent buffer with images.
            for index in range(0, len(self.files), self.batch_size):
                for offset in range(self.batch_size):
                    image = Image.open(self.files[index + offset])
                    for i, input_shape in enumerate(self.inputs):
                        self.buffers[i][offset] = self.preprocess_func(image, *input_shape).contiguous()
                logging.info("Calibration images pre-processed: {:}/{:}".format(index+self.batch_size, len(self.files)))
                yield
        else:
            for index in range(0, self.max_calib_data, self.batch_size):
                for offset in range(self.batch_size):
                    for i, input_shape in enumerate(self.inputs):
                        rand_batch = torch.rand((self.batch_size, *input_shape), dtype=torch.float32).contiguous()
                        self.buffers[i].copy_(rand_batch)
                logging.info(f'Generated random calibration data batch: {index + self.batch_size}/{self.max_calib_data}')
                yield

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            next(self.batches)
            # Pass buffer pointer to tensorrt calibration algorithm
            return [int(buffer.data_ptr()) for buffer in self.buffers]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logging.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logging.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)
