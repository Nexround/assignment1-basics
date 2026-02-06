import torch
from torch import nn
from einops import einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embedding_matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(embedding_matrix, mean=0.0, std=1, a=-3, b=3)
        self.embedding_weight = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 原生列表：只能接受整数或切片。
        # PyTorch Tensor：接受张量作为索引，并能保持或扩展维度。

        return self.embedding_weight[token_ids]
