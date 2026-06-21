import torch
from torch import Tensor


def softmax(x: Tensor, i: int = 0, temp: float = 1.0) -> Tensor:
    shifted_x = x - x.amax(dim=i, keepdim=True)
    exp = torch.exp(shifted_x)
    exp_norm = (exp / temp) / torch.sum(exp / temp, dim=i, keepdim=True)
    return exp_norm
