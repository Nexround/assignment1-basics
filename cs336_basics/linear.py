import math

import torch
from torch import nn
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        # 在 PyTorch 中，函数名以 下划线（_）​ 结尾表示这是一个 原地操作（in-place operation）。
        tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std=math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3*std, b=3*std)

        self.weight = nn.Parameter(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return result
