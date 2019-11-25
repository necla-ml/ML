from decimal import Decimal, ROUND_HALF_UP
from math import *

import torch as th

_round = round

def round(n, digits=None):
    if th.is_tensor(n):
        n.apply_(lambda x: round(x, digits))
        return n

    if digits:
        return _round(n, digits)
    else:
        return int(Decimal(n).to_integral_value(rounding=ROUND_HALF_UP))