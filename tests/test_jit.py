from numba import njit, jitclass, types, typed
import numpy as np
from array import array

@njit
def test_memoryview():
    buf = np.array([1,2,3,4])
    #view = memoryview(buf)
    n = len(buf)
    print(n)
    if n != 3:
        raise Exception('bad')

import torch

@torch.jit.script
def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r

@torch.jit.script
def sum(arr):
    s = 0
    for e in arr:
        s += e
    return s

def test_torch_jit():
    # buf = torch.tensor([1,2,3])
    buf = torch.ByteTensor([1,2,3])
    print(sum(buf))