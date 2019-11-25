#from .ae import *
#from .cae import *
#from .dae import *

import math
import types
import torch
import torch.nn as nn

from . import (
    #bert,
    detection,
    backbone,
)

from .backbone import *
from ml.utils import Config

AE   = 'AE'
AE3  = '3AE'
AE4  = '4AE'
UAE  = 'UAE'
UAE4 = '4UAE'
DAE  = 'DAE'
DAE3 = '3DAE'

__all__ = [
    'Model', 
    'DataParallel', 
    'DistributedDataParallel',
    'build',

    'resnet101',
    'resnext101',
    'Backbone',
]


def build(arch, pretrained=False, num_classes=None, **kwargs):
    model = None
    if arch == 'resnet50':
        from torchvision import models
        model = models.__dict__[arch](pretrained)
        if num_classes is not None and model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet101':
        from torchvision import models
        model = models.__dict__[arch](pretrained)
        layers = [
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        ]

        for i in range(3): # 56x56, 28x28, 14x14, 7x7
            name = 'layer%d' % (i + 1)
            layers.append(getattr(model, name))
        
        model.features = torch.nn.Sequential(*layers)
        model = models.__dict__[arch](pretrained)
        if num_classes is not None and model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnext101': # 32x4d | 64x4d
        import pretrainedmodels
        cardinality = kwargs.get('cardinality', None)
        width = kwargs.get('width', None)
        arch = f"{arch}_{cardinality}x{width}d"
        model = pretrainedmodels.__dict__[arch](num_classes=num_classes or 1000, pretrained=pretrained and 'imagenet' or None)

    """
    elif arch == 'desenet121':
        if pretrained:
            print(f"=> using pre-trained model '{arch}'")
        else:
            print(f"=> creating model '{arch}'")
        model = models.__dict__[arch](pretrained)

        # XXX replace the last FC layer if necessary
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.arch == 'alexnet':
        if args.pretrained == 1:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        # replace the last FC layer
        model._modules['classifier']._modules['6'] = nn.Linear(model.classifier[6].in_features, args.num_classes)
    elif args.arch == 'squeezenet1_0':
        if args.pretrained == 1:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        # replace the last FC layer
        model._modules['classifier']._modules['1'] = nn.Conv2d(512, args.num_classes, (1,1), stride=1)
    """
    return model


class Model(nn.Module):
    r"""
    Model life cycle:
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
        super(Model, self).__init__()
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
