'''
A general task requires a `runner` to execute.
Specialized tasks such for ML reqire a `trainer` to fit` a `model` over a `dataset`.

The task may be executed in a distributed parallel setting.
The output of a task may be evaluated and/or visualized incrementally.
The states of a task may be saved to resume by loading from a checkpoint.
'''

class Task():
    def __init__(self, *args, **kwargs):
        pass

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
    
    def setup(self):
        pass