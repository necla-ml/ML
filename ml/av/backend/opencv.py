from ml import logging

try:
    from cv2 import *
except Exception as e:
    logging.warn("opencv unavailable, run `mamba install opencv -c conda-forge` to install") 