import torch
from torch import nn
from einops import einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embedding_matrix = torch.empty(embedding_dim, num_embeddings, device=device, dtype=dtype)
        embedding_weight = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pass
