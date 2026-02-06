import torch
from torch import nn, Tensor

from .linear import Linear


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + torch.exp(-x))


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.silu = SiLU()

    def forward(self, x):
        gate = self.silu(self.w1(x))
        up = self.w3(x)
        hidden = gate * up
        output = self.w2(hidden)
        return output
