import torch
from torch.utils.data.sampler import Sampler

class SubsetSampler(Sampler):
    """Samples elements sequentially or randomly from a given list of indices, without replacement.
    Args:
        indices (list): a list of indices
    """

    def __init__(self, indices, shuffle=True):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices))) if self.shuffle else iter(self.indices)

    def __len__(self):
        return len(self.indices)
