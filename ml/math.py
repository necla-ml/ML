from decimal import Decimal, ROUND_HALF_UP
from math import *
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
