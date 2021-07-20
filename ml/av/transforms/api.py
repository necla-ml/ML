from typing import Union
from ml import logging

try:
    from ml.vision.transforms import *
    from ml.vision.transforms import functional
except Exception as e:
    logging.warn(f"{e}, ml.vision.transforms unavailable, run `mamba install ml-vision -c NECLA-ML` to install")
else:
    from pathlib import Path
    from ml.vision import transforms as trans
    import torch as th