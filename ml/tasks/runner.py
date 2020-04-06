from . import Engine
from .. import logging

class Runner(object):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def setup(self):
        pass

    def run(self):
        pass

def _step(engine, batch):
    return engine.step(batch)

class StepRunner(Runner, Engine):
    r"""Abstract class wrapper of ignite engine for general iterative processing.

    state:
        batch, iteration, output
        additional: dataloader, epoch=0, max_epochs, metrics={}
        
    """
    def __init__(self, cfg):
        Runner.__init__(self, cfg)
        Engine.__init__(self, _step)
    
    def step(self, batch):
        pass

class Trainer(StepRunner):
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        '''
        For each epoch:
            1. step batch by batch
            2. evaluate after an epoch
        '''
    
    def step(self, batch):
        pass

    def evaluate(self):
        pass
