from numba import njit, jitclass, types, typed
import numpy as np
from array import array

@njit
def method(bitstream, workaround=False):
    start = 0
    header = (1,2,3)
    buf = len(bitstream[0:4])
    buf = bitstream[4] 
    # return (start, *header), buf
    return bitstream

def test_memoryview():
    a = bytearray(8)
    ma = memoryview(a)
    output = method(ma, workaround=True)
    print(output)

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