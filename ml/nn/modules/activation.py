import torch
from torch import nn
from torch.nn import functional as F

class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))

class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * F.softplus(x).tanh()
