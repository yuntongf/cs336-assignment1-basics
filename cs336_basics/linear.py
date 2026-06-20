import numpy as np
import torch


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        w = torch.empty([out_features, in_features], device=device, dtype=dtype)
        std = np.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(w, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W: torch.nn.Parameter = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T

    def init_weights(self, w: torch.Tensor):
        self.W = torch.nn.Parameter(w)
