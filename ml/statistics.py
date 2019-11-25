import torch as th
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option('display.expand_frame_repr', False)
pd.set_option('precision', 2)

def fscore(precision, recall, beta=1):
    product   = precision * recall
    return round(product > 0 and (1 + beta**2) * (product) / (beta**2 * precision + recall) or 0, 2)

def stats(TP, FP, FN):
    pre = TP > 0 and round(TP / (TP+FP), 2) or 0
    recall = TP > 0 and round(TP / (TP+FN), 2) or 0
    return pre, recall, round(fscore(pre, recall), 2)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
