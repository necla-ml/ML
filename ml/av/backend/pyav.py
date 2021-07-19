from ml import logging

try:
    from av import *
except Exception as e:
    logging.warn("PyAV unavailable, run `mamba install av -c conda-forge` to install") 