import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_dict = model.state_dict()
    opt_dict = optimizer.state_dict()
    data = {"model_dict": model_dict, "opt_dict": opt_dict, "it": iteration}
    torch.save(data, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    data = torch.load(src)
    model.load_state_dict(data["model_dict"])
    optimizer.load_state_dict(data["opt_dict"])
    return data["it"]
