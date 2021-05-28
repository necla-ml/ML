import torch
import torch.distributed as dist
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

class DistributedNoExtraSampler(Sampler):
    r"""Very similar to regular DistributedSampler from pytorch 
    but does not add extra samples if the dataset is not divisible by replicas
    """

    def __init__(self, dataset, num_replicas = None,
                 rank = None, shuffle= True,
                 seed= 0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_size = len(self.dataset)
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)   

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
