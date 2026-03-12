import math
import torch
from torch import nn
from einops import einsum
from .softmax import softmax


def sdpa(query, key, value, mask):
    d_k = query.shape[-1]

    # scores = torch.matmul(query, key.transpose(-2, -1))
    scores = einsum(query, key, "... q d_k, ... k d_k -> ... q k")
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    act = softmax(scores / math.sqrt(d_k), -1)
    return act @ value
