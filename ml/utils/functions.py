from collections.abc import Sequence, Mapping, Iterable
from collections import defaultdict
from itertools import tee, chain
import sys
import pprint

def rewrite(labels, rewrites, inplace=False):
    """
    Args:
        labels(List[str]): list of class labels
        rewrites(dict): mapping from target to source labels to rewrite
    """
    if rewrites:
        labels = inplace and labels or list(labels)
        for target, sources in rewrites.items():
            if not isinstance(sources, list):
                sources = [sources]
            for src in sources:
                idx = labels.index(src)
                labels[idx] = target
    return labels

def printable(v):
    if callable(v):
        return False
    elif isinstance(v, Iterable) or isinstance(v, Mapping):
        return False
    else:
        return True

def iterator(x):
    r"""
        - tuple: Sequence: Iterable
        - Tensor: Iterable
        x: tuple or Sequence
        x: sequence
    """
    if isinstance(x, Sequence):
        return iter(x)
    if torch.is_tensor(x) or isinstance(x, Variable):
        return iter([x])
    if isinstance(x, Iterable) :
        return x
    return iter([x])

def flatten(x):
    r"""Flatten one level of nesting.
    """
    if isinstance(x, Sequence) or isinstance(x, Iterable):
        return list(chain.from_iterable(iterator(v) for v in x))
    else:
        return x

def pairwise(iterable):
    for i, x in enumerate(iterable):
        for j, y in enumerate(iterable):
            if i < j:
                yield (x, y) 

def error(msg, code=-1):
    print(msg)
    sys.exit(code)