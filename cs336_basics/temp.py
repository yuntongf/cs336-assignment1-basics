from typing import Callable

import torch
from sympy.polys.domains.integerring import math


class Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr):
        if lr < 0:
            raise ValueError("Invalid learning rate")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


weights = torch.nn.Parameter(5 * torch.rand((10, 10)))
opt = Optimizer([weights], lr=1e3)

for t in range(10):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()
