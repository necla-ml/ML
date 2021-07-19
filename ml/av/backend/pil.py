from ml import logging

try:
    from PIL import *
except Exception as e:
    logging.warn("pillow unavailable, run `mamba install pillow` to install") 