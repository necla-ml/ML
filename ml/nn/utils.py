from collections import OrderedDict

def trim(stat, prefix='module'):
    r"""Remove prefix of state_dict keys.
    """

    stat_new = OrderedDict()
    for k, v in stat.items():
        if k.startswith(prefix):
            stat_new[k[len(prefix)+1:]] = v

    return stat_new if stat_new else stat

def fuse_conv_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    from torch import nn
    import torch
    with torch.no_grad():
        fused = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          bias=True)

        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fused.weight.copy_(torch.mm(w_bn, w_conv).view_as(fused.weight))

        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.shape[0])
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fused