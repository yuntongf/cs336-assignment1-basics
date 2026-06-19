import torch
from torch import Tensor


def softmax(x: Tensor, i: int) -> Tensor:
    shifted_x = x - x.amax(dim=i, keepdim=True)
    exp = torch.exp(shifted_x)
    exp_norm = exp / torch.sum(exp, dim=i, keepdim=True)
    return exp_norm
