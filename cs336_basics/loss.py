import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    indices = targets.unsqueeze(-1)
    neg_log_probs = -logits.gather(-1, indices).squeeze(-1) + torch.logsumexp(logits, -1)
    return torch.mean(neg_log_probs, dim=-1)
