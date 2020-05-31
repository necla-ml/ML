from random import *
import random

def seed(s, deterministic=True):
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except ImportError:
        pass

    try:
        import ml
        import torch as th   
        th.manual_seed(s)
        # XXX no use of torch.cuda to check CUDA availability before MXNet in case of SegFault
        if ml.cuda.is_available():
            th.cuda.manual_seed_all(s)
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = not deterministic
            cudnn.deterministic = deterministic

    except ImportError:
        pass