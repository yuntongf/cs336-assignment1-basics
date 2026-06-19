import numpy as np
import torch
from torch import Tensor


class RoPe(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k

        # Θ^((2𝑘−2)/𝑑)
        rot_freq = np.array([1 / pow(theta, (2 * k - 2) / d_k) for k in range(1, d_k // 2 + 1)])  # (k, )
        rot_ang = np.array([rot_freq * i for i in range(max_seq_len)])  # (seq_len, k)

        self.register_buffer("rot_sin", Tensor(np.sin(rot_ang)), persistent=False)
        self.register_buffer("rot_cos", Tensor(np.cos(rot_ang)), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """(..., seq_len, d_k), (..., seq_len) -> (..., seq_len, d_k)"""
        rot_sin = self.get_buffer("rot_sin")[token_positions]  # (..., seq_len, k)
        rot_cos = self.get_buffer("rot_cos")[token_positions]

        x_unflat = x.unflatten(-1, (x.shape[-1] // 2, 2))  # (..., seq_len, k, 2)
        pair1, pair2 = x_unflat[..., 0], x_unflat[..., 1]  # (..., seq_len, k)
        res_unflat = torch.empty_like(x_unflat)

        res_unflat[..., 0] = pair1 * rot_cos - pair2 * rot_sin
        res_unflat[..., 1] = pair1 * rot_sin + pair2 * rot_cos
        return res_unflat.flatten(-2, -1)
