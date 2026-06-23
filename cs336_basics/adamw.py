import math
from typing import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
        if lr < 0:
            raise ValueError("Invalid learning rate")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "decay": weight_decay}
        super().__init__(params, defaults)

        for g in self.param_groups:
            for p in g["params"]:
                self.state[p]["first_moments"] = torch.zeros_like(p)
                self.state[p]["second_moments"] = torch.zeros_like(p)

    def update_lr(self, lr: float):
        for g in self.param_groups:
            g["lr"] = lr

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for g in self.param_groups:
            lr, decay, (b1, b2), eps = g["lr"], g["decay"], g["betas"], g["eps"]

            for p in g["params"]:
                state = self.state[p]
                t = state.get("t", 1)

                grad = p.grad.data

                # weight decay
                p.data = p.data - lr * decay * p.data

                state["first_moments"] = b1 * state["first_moments"] + (1 - b1) * grad
                state["second_moments"] = b2 * state["second_moments"] + (1 - b2) * grad * grad

                adjusted_lr = lr * math.sqrt(1 - b2**t) / (1 - b1**t)

                # weight update using moments
                p.data = p.data - adjusted_lr * state["first_moments"] / (torch.sqrt(state["second_moments"]) + eps)

                state["t"] = t + 1

        return loss


def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    params = list(parameters)

    overall_l2 = math.sqrt(sum([(param.grad**2).sum() for param in params if param.grad is not None]))

    return overall_l2

    if overall_l2 > max_l2_norm:
        for param in params:
            if param.grad is not None:
                param.grad = param.grad * (max_l2_norm / (overall_l2 + eps))
        return max_l2_norm
    else:
        return overall_l2
