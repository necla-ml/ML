import torch
from torch import nn

def nan_hook(self, inp, out):
    """
    Check for NaN inputs or outputs at each layer in the model
    Usage:
        # forward hook
        for submodule in model.modules():
            submodule.register_forward_hook(nan_hook)
    """

    outputs = isinstance(out, tuple) and out or [out]
    inputs = isinstance(inp, tuple) and inp or [inp]

    contains_nan = lambda x: torch.isnan(x).any()
    layer = self.__class__.__name__

    for i, inp in enumerate(inputs):
        if inp is not None and contains_nan(inp):
            raise RuntimeError(f'Found NaN input at index: {i} in layer: {layer}')

    for i, out in enumerate(outputs):
        if out is not None and contains_nan(out):
            raise RuntimeError(f'Found NaN output at index: {i} in layer: {layer}')

def find_modules(nn_module, type):
    """
    Find and return modules of the input `type`
    """
    return [module for module in nn_module.modules() if isinstance(module, type)]

class LayerRecorder(nn.Module):
    """
    Collect outputs from all the layers in `record_layers` of the `model`
    """
    def __init__(self, model, record_layers):
        super().__init__()
        self.model = model

        self.recordings = []
        self.record_layers = record_layers

        self._register_hook()

    def _hook(self, layer, input, output):
        layer_name = layer.__class__.__name__
        self.recordings.append((layer_name, output))

    def _register_hook(self):
        for layer in self.record_layers:
            modules = find_modules(self.model, layer)
            for module in modules:
                module.register_forward_hook(self._hook)

    def forward(self, *args, **kwargs):
        # clear any previous recordings
        self.recordings.clear()
        # forward
        pred = self.model(*args, **kwargs)
        return pred, self.recordings