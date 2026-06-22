import torch
from jaxtyping import Float


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: Float[torch.Tensor, " num_embeddings embedding_dim"] | None = None,
    ):
        super().__init__()
        if weights is None:
            w = torch.empty([num_embeddings, embedding_dim], device=device, dtype=dtype)
            torch.nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3)
        else:
            w = weights
        self.W = torch.nn.Parameter(w)
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor(token_ids, device=self.device, dtype=torch.int64)
        return self.W[indices]

    def init_weights(self, weights: Float[torch.Tensor, " num_embeddings embedding_dim"]):
        self.W = torch.nn.Parameter(weights)
