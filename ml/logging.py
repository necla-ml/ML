import sys
import logging
from logging import *


FMT = "[%(asctime)s] %(levelname)s: %(message)s"
DATEFMT = "%m/%d/%Y %H:%M:%S"


def basicConfig(**kwargs):
    rank = kwargs.pop('rank', -1)
    ws = kwargs.pop('world_size', -1)
    level = kwargs.pop('level', logging.INFO)
    format = kwargs.pop('format', FMT if rank < 0 else f"[%(asctime)s][{rank}/{ws}] %(levelname)s: %(message)s")
    datefmt = kwargs.pop('datefmt', DATEFMT)
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=level, format=format, datefmt=datefmt, **kwargs)

# XXX Noisy logging from every imported module
import numexpr
import pytorch_pretrained_bert
basicConfig()