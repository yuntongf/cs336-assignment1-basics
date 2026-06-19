import torch
from jaxtyping import Float


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: Float[torch.Tensor, " d_model"] | None = None,
    ):
        super().__init__()
        if weights is not None:
            self.g = torch.nn.Parameter(weights)
        else:
            w = torch.ones([d_model], device=device, dtype=dtype)
            self.g = torch.nn.Parameter(w)

        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        sq = torch.einsum("...i,...i->...", x, x).unsqueeze(-1)
        rms = torch.sqrt(sq / self.d_model + self.eps)
        res = x * self.g / rms

        return res.to(in_dtype)
