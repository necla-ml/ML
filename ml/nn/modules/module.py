from torch import nn

class Module(nn.Module):
    r"""
    Module life cycle:
        - Creation
        - Parallelize/cuda()
        - Load states from checkpoint

    Optim life cycle:
        - Optim creation from model parameters
        - Load states from checkpoint

    Training life cycle:
        - save model and optim states to checkpoint

    """

    def __init__(self, cfg):
        super(Module, self).__init__()
        self.cfg = cfg
    
    def parallelize(self, cuda=None, distributed=None):
        r"""
        .. note::
            Default to parallelize the entire module as a whole.
            Subclass should override for parallel partitioning.
        """

        cuda = cuda if cuda is not None else self.cfg.nGPU > 0
        return parallelize(self, cuda, distributed)

    def forward(self, *input):
        raise NotImplementedError

    def loss(self, *input, **kwargs):
        raise NotImplementedError