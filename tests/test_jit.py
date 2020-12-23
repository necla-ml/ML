from numba import njit

from torch import ByteTensor
import torch
import pytest

# FIXME not supported yet
@njit
# @pytest.mark.essential
def method(bitstream, workaround=False):
    start = 0
    header = (1, 2, 3)
    buf = bitstream[4]
    return (start, *header), buf

# @pytest.mark.essential
def test_memoryview():
    a = bytearray(range(8))
    ma = memoryview(a)
    output = method(ma, workaround=True)
    output = method(a, workaround=True)
    print(output)

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

# @pytest.mark.essential
def test_torch_jit():
    buf = ByteTensor([1, 2, 3])
    print(sum(buf))
