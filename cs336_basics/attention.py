import torch
from jaxtyping import Bool, Float
from sympy.calculus.accumulationbounds import AccumBounds
from torch import Tensor

from cs336_basics.linear import Linear
from cs336_basics.rope import RoPe
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    qk = torch.einsum("...qd,...kd->...qk", Q, K)
    scaled_qk = qk / (d_k**0.5)  # (..., num_q, num_k)
    if mask is not None:
        scaled_qk = torch.masked_fill(scaled_qk, ~mask, -torch.inf)

    return torch.einsum("...qk,...kd->...qd", softmax(scaled_qk, -1), V)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None):
        super().__init__()
        d_q = d_k = d_v = d_model // num_heads
        self.Wq = Linear(d_model, num_heads * d_q)
        self.Wk = Linear(d_model, num_heads * d_k)
        self.Wv = Linear(d_model, num_heads * d_v)
        self.Wo = Linear(num_heads * d_v, d_model)
        self.rope = RoPe(10000, d_k, max_seq_len) if max_seq_len is not None else None
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_q, self.d_k, self.d_v = d_q, d_k, d_v

    def forward(self, x: Tensor):
        Q = self.Wq.forward(x)  # (..., queries, num_heads * d_q)
        K = self.Wk.forward(x)
        V = self.Wv.forward(x)  # (..., d_model, num_heads * d_v)
        As = []

        for i in range(self.num_heads):
            Qi = Q[..., i * self.d_q : (i + 1) * self.d_q]  # ... queries, d_k
            Ki = K[..., i * self.d_k : (i + 1) * self.d_k]  # ... keys, d_k
            Vi = V[..., i * self.d_v : (i + 1) * self.d_v]  # ... keys, d_v

            mask_q = torch.arange(Qi.shape[-2]).unsqueeze(-1)
            mask_k = torch.arange(Ki.shape[-2]).unsqueeze(-2)
            mask = mask_q >= mask_k  # queries cannot see future keys so q always >= k

            if self.rope is not None:
                Qi = self.rope.forward(Qi, torch.arange(Qi.shape[-2]))
                Ki = self.rope.forward(Ki, torch.arange(Ki.shape[-2]))

            a = scaled_dot_product_attention(Qi, Ki, Vi, mask)  # ... queries, d_v
            As.append(a)

        A = torch.concat(As, -1)

        return self.Wo.forward(A)
