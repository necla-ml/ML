"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network of (Dropout, Linear, ReLU)+
    """
    def __init__(self, dims, dropout=0, act=nn.ReLU, gated=False):
        super(FCNet, self).__init__()
        if gated:
            y_layers = []
            g_layers = []
            for i in range(len(dims)-2):
                in_dim = dims[i]
                out_dim = dims[i+1]

                y_layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
                g_layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))

                if dropout > 0:
                    y_layers.append(nn.Dropout(p=dropout))
                    g_layers.append(nn.Dropout(p=dropout))

                # use nn.Tanh()
                y_layers.append(nn.Tanh())
                # use nn.Sigmoid()
                g_layers.append(nn.Sigmoid())

            y_layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
            g_layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
            if dropout > 0:
                y_layers.append(nn.Dropout(p=dropout))
                g_layers.append(nn.Dropout(p=dropout))

            y_layers.append(nn.Tanh())
            g_layers.append(nn.Sigmoid())
            self.y_layers = nn.Sequential(*y_layers)
            self.g_layers = nn.Sequential(*g_layers)
        else:
            layers = []
            for i in range(len(dims)-2):
                in_dim = dims[i]
                out_dim = dims[i+1]
                if 0 < dropout:
                    layers.append(nn.Dropout(dropout))
                
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
                if act:
                    layers.append(act())

            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            
            layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
            if act:
                layers.append(act())

            self.main = nn.Sequential(*layers)

        self.gated = gated

    def forward(self, x):
        if self.gated:
            return self.y_layers(x)*self.g_layers(x)
        else:
            return self.main(x)


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)
