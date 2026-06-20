import torch
from torch import Tensor

As = [Tensor([[1, 2]]), Tensor([[10, 20]])]

concat = torch.stack(As, dim=-1).flatten(-2, -1)
