def norm(params, p=2):
    import torch
    if isinstance(params, torch.Tensor):
        params = [params]
    
    params = list(filter(lambda param: param.grad is not None, params))
    total = 0
    for param in params:
        norm = param.grad.norm(p)
        total += norm.item() ** p
    
    return total ** (1. / p)

