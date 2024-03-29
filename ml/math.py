from math import floor, sqrt
from decimal import Decimal, ROUND_HALF_UP

from ml import logging

_round = round

def round(n, digits=None):
    r"""Round to the specified number of digits by input types.
    """
    if isinstance(n, float):
        if digits:
            return _round(n, digits)
        else:
            return int(Decimal(n).to_integral_value(rounding=ROUND_HALF_UP))
    else:
        try:
            import torch as th
            if th.is_tensor(n):
                n.apply_(lambda x: round(x, digits))
                return n
        except ImportError as e:
            logging.error(f"No pytorch to round potential tensor input of type {type(n)}")
            raise e

def factorize(n):
    ceiling = floor(sqrt(n))
    for i in range(ceiling, 0, -1):
        if n % i == 0:
            return i, n // i
    return None