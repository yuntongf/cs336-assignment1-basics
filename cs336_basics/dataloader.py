import random

import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    num_toks = len(dataset)
    start_indices = [random.randint(0, num_toks - context_length - 1) for _ in range(batch_size)]
    samples = []
    next = []
    for i in start_indices:
        samples.append(torch.tensor(dataset[i : i + context_length], device=device, dtype=dtype))
        next.append(torch.tensor(dataset[i + 1 : i + context_length + 1], device=device, dtype=dtype))
    return torch.stack(samples, 0), torch.stack(next, 0)
