__all__ = [
    'build',
]

def build(arch, pretrained=False, *args, **kwargs):
    '''Build a model by architecture and configurations
    
    Args:
        arch(str): supported architecture name
            - resnet50
            - resnet101
            - resnext101
            - ae
            - cae
            - dae
            - densenet101
            - grounding

        args:
            - num_classes(int):
            - ...

        kwargs:
    '''

    from .vision import models
    model = models.build(arch, pretrained, *args, **kwargs)
    if model is not None:
        return model

    from .text import models
    model = models.build(arch, pretrained, num_classes, *args, **kwargs)
    if model is not None:
        return model
    
    return None