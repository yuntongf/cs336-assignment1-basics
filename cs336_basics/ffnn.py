import torch

from cs336_basics.linear import Linear


class FFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.l1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.l3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.l2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.l1.forward(x)
        w3x = self.l3.forward(x)
        return self.l2.forward(self.silu(w1x) * w3x)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
